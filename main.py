from __future__ import annotations

from io import BytesIO
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import sites
import town
import time


# CONFIGURATION & PARAMETERS

# Town and infrastructure settings
town_width = 5000
town_height = 5000
num_areas = 6
grid_resolution = 10
total_towers = 15
coverage_range = 1000  # metres

# Simulation settings
num_users = 1000
time_step = 5  # simulated seconds per dashboard tick

# Tower outage/recovery probabilities (per simulation tick)
outage_chance = 0.0002
recovery_chance = 0.05

# User demographics and behaviour profiles
user_type = [0.7, 0.2, 0.1]
device_type = [0.6, 0.1, 0.2, 0.1]
network_capability = [0.7, 0.3]
activities = [0.3, 0.2, 0.1, 0.1, 0.2, 0.1]

user_profiles = {
    "user_type": user_type,
    "device_type": device_type,
    "network_capability": network_capability,
    "activities": activities,
}

movement_colours = {
    "stationary": "grey",
    "walking": "green",
    "driving": "orange",
}


# CORE SIMULATION ENGINE 

def update_serving_tower_distances(users):
    """Refresh the cached distance from every user to its assigned serving tower.

    Purpose:
        Position updates change the geometry each simulation tick, but the serving tower should remain fixed until a handover or recovery event changes it. This helper recomputes the distance field without altering the current association.

    Parameters:
        users (pd.DataFrame): User table including x/y coordinates plus the current serving tower stored in the "Connected Tower" column.

    Returns:
        pd.DataFrame: A copied and updated user table with a refreshed "Distance to Tower" column.
    """
    # Work on a copy so the caller keeps its original DataFrame.
    users = users.copy()
    # An empty user table has no distances to refresh.
    if users.empty:
        return users

    if "Connected Tower" not in users.columns:
        raise KeyError("Users are missing the 'Connected Tower' column.")

    user_coords = users[["x", "y"]].to_numpy(dtype=float)
    serving_coords = np.asarray(users["Connected Tower"].tolist(), dtype=float)

    if serving_coords.shape != user_coords.shape:
        raise ValueError(
            "The 'Connected Tower' coordinates must contain one (x, y) pair "
            "for every user."
        )

    users["Distance to Tower"] = np.linalg.norm(
        user_coords - serving_coords,
        axis=1,
    )
    return users

def safe_mean(frame, column, mask=None):
    """Calculate a finite mean for a numeric column while ignoring invalid values.

    Purpose:
        This keeps dashboard metrics stable when some users have missing, non-numeric, or infinite values in the source column.

    Parameters:
        frame (pd.DataFrame): Source data table.
        column (str): Numeric column to aggregate.
        mask (pd.Series | None): Optional boolean mask limiting the rows considered in the calculation.

    Returns:
        float: The finite mean of the selected values, or np.nan when no valid data exists.
    """
    if column not in frame.columns:
        return np.nan

    values = frame[column] if mask is None else frame.loc[mask, column]
    # Convert invalid entries to NaN before filtering them out.
    values = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else np.nan

def safe_sum(frame, column):
    """Sum only finite numeric values in a column.

    Purpose:
        This keeps totals stable when some rows contain NaN, infinity, or missing values.

    Parameters:
        frame (pd.DataFrame): Source data table.
        column (str): Numeric column to aggregate.

    Returns:
        float: The total of valid numeric entries in the column, or 0.0 when no usable values exist.
    """
    if column not in frame.columns:
        return 0.0
    # Convert the column to numbers, ignoring values that cannot be parsed.
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return float(values.sum()) if len(values) else 0.0

def display_value(value, suffix=""):
    """Format a numeric KPI value for compact dashboard display.

    Purpose:
        Values that are finite are shown with one decimal place and an optional suffix; otherwise the dashboard prints a graceful fallback string.

    Parameters:
        value (float): Numeric KPI being displayed.
        suffix (str): Optional unit suffix such as " dB" or " Mbps" to append to the number.

    Returns:
        str: A formatted display string suitable for metrics cards.
    """
    return f"{value:.1f}{suffix}" if np.isfinite(value) else "n/a"

def table_display_value(value):
    """Convert mixed dataframe values into safe, display-friendly strings.

    Purpose:
        Some user records contain lists, arrays, NaN values, or non-string objects. This helper normalises them into text that can be rendered in Streamlit tables without type errors.

    Parameters:
        value: Any value drawn from the user or tower tables.

    Returns:
        str: A clean string representation of the value, or "n/a" for missing entries.
    """
    if value is None:
        return "n/a"
    if isinstance(value, (list, tuple, dict, np.ndarray)):
        return str(value)
    if isinstance(value, (float, np.floating)):
        return f"{value:.3f}" if np.isfinite(value) else "n/a"
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return str(int(value))
    try:
        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)) and missing:
            return "n/a"
    except (TypeError, ValueError):
        pass
    return str(value)

# SETUP

def initialise_simulation(
    town_width,
    town_height,
    num_areas,
    grid_resolution,
    coverage_range,
    total_towers,
    num_users,
    user_profiles,
    seed=None,
):
    """Build the synthetic town, user population and initial radio network state.

    Purpose:
        This is the simulation bootstrap. It creates the town grid, spreads the population, places towers, assigns initial connections, and calculates the first radio KPI set before the dashboard runs.

    Parameters:
        town_width (int): Width of the fictional town in metres.
        town_height (int): Height of the fictional town in metres.
        num_areas (int): Number of geographic zones used to build the town layout.
        grid_resolution (int): Spatial cell size used for the town and occupancy maps.
        coverage_range (float): Estimated tower coverage radius in metres.
        total_towers (int): Number of cell sites to deploy.
        num_users (int): Number of user devices to generate.
        user_profiles (dict): Demographic and behaviour distribution settings.
        seed (int | None): Optional random seed controlling reproducible generation.

    Returns:
        dict: A configuration bundle containing generated town data, tower state, user tables, and the first cell performance snapshot.
    """

    # Generate town
    assignments, centers, xx, yy = town.generate_town(
        town_width,
        town_height,
        num_areas=num_areas,
        grid_resolution=grid_resolution,
        seed=seed,
    )

    # Generate population
    population_grid, region_population, total_population = town.generate_population(
        assignments,
        centers,
        town_width,
        town_height,
        grid_resolution=grid_resolution,
        seed=seed,
    )

    # Allocate towers by region
    tower_coords, region_allocations = town.allocate_towers(
        assignments,
        region_population,
        coverage_range=coverage_range,
        total_towers=total_towers,
        grid_resolution=grid_resolution,
        seed=seed,
    )

    # Preserve regional allocations and create one row per individual tower
    region_allocations = region_allocations.copy()
    tower_allocations = sites.create_tower_table(region_allocations)

    # Ensure outage tracking fields exist
    if "Tower Status" not in tower_allocations.columns:
        tower_allocations["Tower Status"] = "operational"
    if "Tower Outage Count" not in tower_allocations.columns:
        tower_allocations["Tower Outage Count"] = 0
    if "Tower Recovery Count" not in tower_allocations.columns:
        tower_allocations["Tower Recovery Count"] = 0

    # Generate users
    users = town.generate_users(
        assignments=assignments,
        population_grid=population_grid,
        region_population=region_population,
        num_users=num_users,
        grid_resolution=grid_resolution,
        user_profiles=user_profiles,
        seed=seed,
    )

    # Add movement profiles
    user_movement_profiles = town.assign_movement(
        users,
        town_width,
        town_height,
        seed=seed,
    )

    # ------------------------------------------
    # INITIAL RADIO CONNECTION (ONCE ONLY)
    # ------------------------------------------
    user_movement_profiles = sites.connect_tower(
        users=user_movement_profiles,
        tower_allocations=tower_allocations,
    )

    user_movement_profiles = sites.assign_frequency(user_movement_profiles)
    user_movement_profiles = sites.path_loss(user_movement_profiles)
    user_movement_profiles = sites.shadow_fading(
        user_movement_profiles,
        seed=seed,
    )
    user_movement_profiles = sites.received_power(user_movement_profiles)
    user_movement_profiles = sites.calculate_interference(
        user_movement_profiles,
        tower_allocations,
        seed=seed,
    )
    user_movement_profiles = sites.calculate_noise(
        user_movement_profiles,
        bandwidth=10e6,
    )
    user_movement_profiles = sites.calculate_sinr(user_movement_profiles)

    # Initialise connection status and historical counters ONCE
    user_movement_profiles = sites.initialise_connection_status(
        user_movement_profiles,
        min_sinr=-5,
    )

    # Initial cell capacity and resource allocation
    user_movement_profiles, tower_allocations = sites.allocate_cell_resources(
        user_movement_profiles,
        tower_allocations,
        bandwidth_hz=10e6,
        efficiency=0.70,
    )

    # Initial congestion and service KPIs
    user_movement_profiles, tower_allocations = (
        sites.calculate_congestion_and_network_performance(
            user_movement_profiles,
            tower_allocations,
            base_latency=20,
        )
    )

    cell_performance = sites.calculate_cell_performance(
        user_movement_profiles,
        tower_allocations,
    )

    return {
        "assignments": assignments,
        "centers": centers,
        "xx": xx,
        "yy": yy,
        "population_grid": population_grid,
        "region_population": region_population,
        "total_population": total_population,
        "tower_coords": tower_coords,
        "region_allocations": region_allocations,
        "tower_allocations": tower_allocations,
        "users": users,
        "user_movement_profiles": user_movement_profiles,
        "cell_performance": cell_performance,
    }

# SIMULATION

def simulation_step(
    users,
    tower_allocations,
    time_step,
    town_width,
    town_height,
    seed=None,
    outage_chance=0.0002,
    recovery_chance=0.05,
):
    """Advance the simulation by one time step.

    Purpose:
        Each tick moves users, can trigger faults, recalculates radio conditions, runs handover logic, updates congestion, and refreshes the KPI tables that feed the dashboard.

    Parameters:
        users (pd.DataFrame): Current user table with mobility and radio state.
        tower_allocations (pd.DataFrame): Tower inventory and operational metadata.
        time_step (float): Simulation interval represented in seconds.
        town_width (int): Town width in metres.
        town_height (int): Town height in metres.
        seed (int | None): Random seed for stochastic effects.
        outage_chance (float): Probability that a tower fails in this tick.
        recovery_chance (float): Probability that an outage tower recovers in this tick.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Updated users, current cell performance table, and tower allocation table.
    """
    if time_step <= 0:
        raise ValueError("time_step must be greater than zero.")

    users = users.copy()
    tower_allocations = tower_allocations.copy()

    # Move users towards their destinations
    users = town.update_user_positions(
        users,
        time_step=time_step,
    )

    # Give arrived users new destinations
    users = town.assign_new_destinations(
        users,
        town_width,
        town_height,
        seed=seed,
    )

    # Generate tower outages and recoveries while preserving the latest tower state for the next KPI pass.
    users, tower_allocations = sites.generate_tower_outages(
        users,
        tower_allocations,
        outage_chance=outage_chance,
        recovery_chance=recovery_chance,
        seed=seed,
    )

    # Refresh distance to the CURRENT serving tower. Do not call connect_tower:
    # that would overwrite handovers and force every user onto the nearest cell.
    users = update_serving_tower_distances(users)

    # Recalculate serving-link radio conditions
    users = sites.path_loss(users)
    users = sites.shadow_fading(users, seed=seed)
    users = sites.received_power(users)
    users = sites.calculate_interference(
        users,
        tower_allocations,
        seed=seed,
    )
    users = sites.calculate_noise(users, bandwidth=10e6)
    users = sites.calculate_sinr(users)

    # Find and evaluate operational neighbouring towers
    users = sites.find_neighbouring_towers(
        users,
        tower_allocations,
        num_neighbours=5,
    )
    users = sites.evaluate_neighbouring_towers(
        users,
        tower_allocations,
        seed=seed,
    )

    # Handover before dropping users for poor serving signal
    users = sites.process_handovers(
        users,
        tower_allocations,
        time_step,
        hysteresis=3,
        time_to_trigger=1.0,
    )

    # Check users whose serving signal has become too weak
    users = sites.check_connection_drops(
        users,
        min_sinr=-8,
    )

    # Retry users with failed initial access or dropped sessions
    users = sites.process_automatic_recovery(
        users,
        tower_allocations,
        time_step=time_step,
        min_sinr=-5,
        retry_interval=2.0,
        seed=seed,
    )

    # Share finite cell capacity between connected users so the model reflects service contention and packet delay.
    users, tower_allocations = sites.allocate_cell_resources(
        users,
        tower_allocations,
        bandwidth_hz=10e6,
        efficiency=0.70,
    )

    # Add cell congestion effects to latency and packet loss
    users, tower_allocations = sites.calculate_congestion_and_network_performance(
        users,
        tower_allocations,
        base_latency=20,
    )

    # Refresh tower-level performance table, including empty/outage cells
    cell_performance = sites.calculate_cell_performance(
        users,
        tower_allocations,
    )

    return users, cell_performance, tower_allocations

def run_simulation(
    users,
    tower_allocations,
    town_width,
    town_height,
    time_step=0.2,
    realtime=True,
    max_steps=None,
    outage_chance=0.0002,
    recovery_chance=0.05,
):
    """Run the simulation loop without the dashboard.

    Purpose:
        This is the terminal or console-oriented loop used for batch testing and diagnostic runs. It repeatedly advances the model and prints summary metrics until a stop condition is reached.

    Parameters:
        users (pd.DataFrame): Initial user table.
        tower_allocations (pd.DataFrame): Initial tower state.
        town_width (int): Town width in metres.
        town_height (int): Town height in metres.
        time_step (float): Time delta applied each step.
        realtime (bool): Whether to sleep between steps to mimic live-time progression.
        max_steps (int | None): Maximum number of steps before stopping.
        outage_chance (float): Tower outage probability per tick.
        recovery_chance (float): Tower recovery probability per tick.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Final users, cell performance data, and tower state after the loop exits.
    """
    if time_step <= 0:
        raise ValueError("time_step must be greater than zero.")
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps must be a positive integer or None.")

    users = users.copy()
    tower_allocations = tower_allocations.copy()
    simulation_time = 0.0
    step_count = 0

    cell_performance = sites.calculate_cell_performance(
        users,
        tower_allocations,
    )

    print("Simulation started. Press Ctrl+C to stop.")
    try:
        while max_steps is None or step_count < max_steps:
            # seed=None allows random effects to evolve between time steps
            users, cell_performance, tower_allocations = simulation_step(
                users=users,
                tower_allocations=tower_allocations,
                time_step=time_step,
                town_width=town_width,
                town_height=town_height,
                seed=None,
                outage_chance=outage_chance,
                recovery_chance=recovery_chance,
            )

            simulation_time += time_step
            step_count += 1

            if step_count % 10 == 0:
                moving_users = (
                    (users["movement_status"] == "moving").sum()
                    if "movement_status" in users.columns
                    else 0
                )
                connected_users = (users["Connection Status"] == "connected").sum()
                print(
                    f"Simulation time: {simulation_time:.1f}s | "
                    f"Moving users: {moving_users} | "
                    f"Connected users: {connected_users}"
                )

            if realtime:
                time.sleep(time_step)

    except KeyboardInterrupt:
        print("\nSimulation stopped.")

    return users, cell_performance, tower_allocations

def save_simulation_results(
    users,
    cell_performance,
    tower_allocations,
    output_file="output.xlsx",
):
    """Export the latest simulation state into an Excel workbook.

    Purpose:
        This saves the full user table, cell metrics, tower inventory, and network summary so results can be shared or reviewed outside the dashboard.

    Parameters:
        users (pd.DataFrame): User table to export.
        cell_performance (pd.DataFrame): Cell-level KPI table.
        tower_allocations (pd.DataFrame): Tower inventory and operational status.
        output_file (str): Excel file path or workbook name used for the export.

    Returns:
        dict: The generated network performance report dictionary, including summary metrics.
    """
    report = sites.generate_network_performance_report(
        users,
        tower_allocations,
    )

    with pd.ExcelWriter(output_file) as writer:
        users.to_excel(writer, sheet_name="users", index=False)
        cell_performance.to_excel(writer, sheet_name="cell_performance", index=False)
        tower_allocations.to_excel(writer, sheet_name="towers", index=False)

        if isinstance(report.get("tower_report"), pd.DataFrame):
            report["tower_report"].to_excel(
                writer,
                sheet_name="tower_performance",
                index=False,
            )
        if isinstance(report.get("region_report"), pd.DataFrame):
            report["region_report"].to_excel(
                writer,
                sheet_name="region_performance",
                index=False,
            )
        if isinstance(report.get("user_report"), pd.DataFrame):
            report["user_report"].to_excel(
                writer,
                sheet_name="user_performance",
                index=False,
            )

        pd.DataFrame([report["network_summary"]]).to_excel(
            writer,
            sheet_name="network_summary",
            index=False,
        )

    return report


# DASHBOARD

DEFAULT_CONFIG = {
    "town_width": 8000,
    "town_height": 8000,
    "num_areas": 6,
    "grid_resolution": 10,
    "total_towers": 15,
    "coverage_range": 1000,
    "num_users": 1000,
    "time_step": 5,
    "outage_chance": 0.0002,
    "recovery_chance": 0.05,
    "seed": 42,
}

ACTIVITY_PROFILES = {
    "user_type": user_type,
    "device_type": device_type,
    "network_capability": network_capability,
    "activities": activities,
}


def finite_mean(frame, column, mask=None):
    """Return the finite mean of a column while ignoring non-numeric values.

    Purpose:
        This helper is used for summary KPIs such as mean SINR, latency, and packet loss.

    Parameters:
        frame (pd.DataFrame): Source data table.
        column (str): Numeric column name to evaluate.
        mask (pd.Series | None): Optional boolean mask selecting rows to include.

    Returns:
        float: Finite mean of the filtered values, or np.nan when nothing is valid.
    """
    if column not in frame.columns:
        return np.nan
    values = frame[column] if mask is None else frame.loc[mask, column]
    values = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else np.nan

def finite_sum(frame, column):
    """Return the finite sum of a column while ignoring invalid values.

    Purpose:
        This helper totals event counters and performance metrics without being corrupted by NaN or non-numeric entries.

    Parameters:
        frame (pd.DataFrame): Source data table.
        column (str): Numeric column name to evaluate.

    Returns:
        float: The sum of valid numeric values, or 0.0 when no values are usable.
    """
    if column not in frame.columns or frame.empty:
        return 0.0
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return float(values.sum()) if values.size else 0.0

def get_sinr_column(users):
    """Return the column name that stores SINR values.

    Purpose:
        Some data frames use "SINR" while others use "SINR (dB)". This helper standardises the lookup for metric calculations and dashboards.

    Parameters:
        users (pd.DataFrame): User table containing one of the supported SINR column names.

    Returns:
        str | None: The matching SINR column name, or None when neither exists.
    """
    if "SINR" in users.columns:
        return "SINR"
    if "SINR (dB)" in users.columns:
        return "SINR (dB)"
    return None

def network_snapshot(users, towers, simulation_time, step_count):
    """Build a compact summary snapshot of the current network state.

    Purpose:
        The dashboard uses this helper to create KPI cards and time-series data for each simulation tick without re-deriving every metric from scratch.

    Parameters:
        users (pd.DataFrame): The current user table.
        towers (pd.DataFrame): The current tower inventory state.
        simulation_time (float): Current simulated time in seconds.
        step_count (int): Current simulation tick count.

    Returns:
        dict: A summary dictionary containing connection, SINR, throughput, tower, and congestion metrics.
    """
    status = users.get(
        "Connection Status", pd.Series("connected", index=users.index)
    ).astype(str)
    connected = status.eq("connected")
    sinr_column = get_sinr_column(users)
    outage_mask = (
        towers.get("Tower Status", pd.Series("operational", index=towers.index))
        .astype(str)
        .eq("outage")
    )
    operational = (
        towers.get("Tower Status", pd.Series("operational", index=towers.index))
        .astype(str)
        .eq("operational")
    )

    total_users = int(len(users))
    connected_count = int(connected.sum())
    total_towers = int(len(towers))

    return {
        "Step": int(step_count),
        "Simulation time (s)": float(simulation_time),
        "Total users": total_users,
        "Connected users": connected_count,
        "Disconnected users": int(status.eq("disconnected").sum()),
        "Failed users": int(status.eq("failed").sum()),
        "Connection rate (%)": (100.0 * connected_count / total_users)
        if total_users
        else 0.0,
        "Mean SINR (dB)": finite_mean(users, sinr_column, connected)
        if sinr_column
        else np.nan,
        "Total throughput (Mbps)": finite_sum(
            users.loc[connected], "User Throughput (Mbps)"
        ),
        "Mean user throughput (Mbps)": finite_mean(
            users, "User Throughput (Mbps)", connected
        ),
        "Mean latency (ms)": finite_mean(users, "Latency (ms)", connected),
        "Packet loss (%)": finite_mean(users, "Packet Loss (%)", connected),
        "Capacity utilisation (%)": finite_mean(
            towers.loc[operational], "Capacity Utilisation (%)"
        ),
        "Operational towers": int(operational.sum()),
        "Tower outages": int(outage_mask.sum()),
        "Congested towers": int(
            towers.get("Congestion Level", pd.Series(index=towers.index, dtype=object))
            .isin(["high", "overloaded"])
            .sum()
        ),
        "Completed handovers": finite_sum(users, "Handover Count"),
        "Handover failures": finite_sum(users, "Handover Failure Count"),
        "Dropped sessions": finite_sum(users, "Dropped Session Count"),
    }

def append_event(message, event_type="Scenario"):
    """Add a new log event to the Streamlit session event history.

    Purpose:
        The dashboard keeps a rolling list of notable scenario changes such as manual outages, recovery events, or traffic surges.

    Parameters:
        message (str): Human-readable event description.
        event_type (str): Event category used to group messages in the report log.

    Returns:
        None: The event is appended to Streamlit session state in place.
    """
    events = st.session_state.get("events", [])
    sim = st.session_state.get("simulation")
    sim_time = float(st.session_state.get("simulation_time", 0.0))
    step_count = int(st.session_state.get("step_count", 0))
    events.insert(
        0,
        {
            "Simulation time (s)": sim_time,
            "Step": step_count,
            "Type": event_type,
            "Event": message,
            "Recorded at": datetime.now().strftime("%H:%M:%S"),
        },
    )
    st.session_state.events = events[:300]

def initialise_dashboard(config):
    """Initialise the live dashboard state and in-memory simulation snapshot.

    Purpose:
        This prepares the first network run and clears prior filters, history, and runtime variables so the dashboard starts from a clean baseline.

    Parameters:
        config (dict): Configuration settings for the simulation including town dimensions, users, towers, and probabilities.

    Returns:
        None: This function populates Streamlit session state directly.
    """
    state = initialise_simulation(
        town_width=config["town_width"],
        town_height=config["town_height"],
        num_areas=config["num_areas"],
        grid_resolution=config["grid_resolution"],
        coverage_range=config["coverage_range"],
        total_towers=config["total_towers"],
        num_users=config["num_users"],
        user_profiles=ACTIVITY_PROFILES,
        seed=config["seed"],
    )
    state["coverage_range"] = config["coverage_range"]
    st.session_state.simulation = state
    for widget_key in [
        "filter_regions",
        "filter_user_types",
        "filter_statuses",
        "target_tower",
        "traffic_region",
        "inspect_user_id",
    ]:
        st.session_state.pop(widget_key, None)
    st.session_state.pop("excel_export", None)
    st.session_state.simulation_time = 0.0
    st.session_state.step_count = 0
    st.session_state.running = False
    st.session_state.history = [
        network_snapshot(
            state["user_movement_profiles"],
            state["tower_allocations"],
            0.0,
            0,
        )
    ]
    st.session_state.events = []
    st.session_state.sim_config = dict(config)

def refresh_performance(state):
    """Recalculate resource utilisation and congestion metrics for the current state.

    Purpose:
        After manual changes or scenario events, the network can drift from its last metric refresh. This helper updates capacity and congestion calculations before the dashboard is rendered again.

    Parameters:
        state (dict): Simulation state dictionary containing current user and tower tables.

    Returns:
        None: The state dictionary is mutated in place with refreshed performance tables.
    """
    users = state["user_movement_profiles"]
    towers = state["tower_allocations"]
    users, towers = sites.allocate_cell_resources(
        users,
        towers,
        bandwidth_hz=10e6,
        efficiency=0.70,
    )
    users, towers = sites.calculate_congestion_and_network_performance(
        users,
        towers,
        base_latency=20,
    )
    state["user_movement_profiles"] = users
    state["tower_allocations"] = towers
    state["cell_performance"] = sites.calculate_cell_performance(users, towers)

def advance_one_step():
    """Advance the live dashboard simulation by one tick.

    Purpose:
        This function performs the same single-tick update used elsewhere but reads the in-memory Streamlit state so the running dashboard can progress automatically.

    Parameters:
        None: The function reads its inputs from st.session_state.

    Returns:
        None: It updates the active simulation state and dashboard history in place.
    """
    state = st.session_state.simulation
    config = st.session_state.sim_config
    old_users = state["user_movement_profiles"]
    old_towers = state["tower_allocations"]

    users, cells, towers = simulation_step(
        users=old_users,
        tower_allocations=old_towers,
        time_step=float(config["time_step"]),
        town_width=int(config["town_width"]),
        town_height=int(config["town_height"]),
        seed=None,
        outage_chance=float(config["outage_chance"]),
        recovery_chance=float(config["recovery_chance"]),
    )

    state["user_movement_profiles"] = users
    state["tower_allocations"] = towers
    state["cell_performance"] = cells
    st.session_state.pop("excel_export", None)
    st.session_state.simulation_time += float(config["time_step"])
    st.session_state.step_count += 1

    # Add concise event records for changes that occurred during this tick.
    old_status = old_towers.set_index("Tower ID").get(
        "Tower Status", pd.Series(dtype=object)
    )
    new_status = towers.set_index("Tower ID").get(
        "Tower Status", pd.Series(dtype=object)
    )
    if len(old_status) and len(new_status):
        for tower_id in new_status.index:
            before = old_status.get(tower_id, "operational")
            after = new_status.get(tower_id, "operational")
            if before != after:
                append_event(f"{tower_id} changed from {before} to {after}.", "Tower")

    for column, label in [
        ("Handover Count", "Handover completed"),
        ("Handover Failure Count", "Handover failed"),
        ("Dropped Session Count", "Session dropped"),
        ("Successful Reconnection Count", "User reconnected"),
    ]:
        delta = finite_sum(users, column) - finite_sum(old_users, column)
        if delta > 0:
            append_event(f"{int(delta)} {label.lower()} event(s).", "Network")

    st.session_state.history.append(
        network_snapshot(
            users,
            towers,
            st.session_state.simulation_time,
            st.session_state.step_count,
        )
    )
    st.session_state.history = st.session_state.history[-600:]

def set_tower_status(tower_id, new_status):
    """Manually force a tower into an outage or operational state.

    Purpose:
        This supports dashboard fault injection and lets the operator simulate cell failures and recoveries without reinitialising the full network.

    Parameters:
        tower_id: The unique identifier for the affected tower.
        new_status (str): Target tower state, typically "outage" or "operational".

    Returns:
        None: The tower record and affected user statuses are updated directly in session state.
    """
    state = st.session_state.simulation
    users = state["user_movement_profiles"].copy()
    towers = state["tower_allocations"].copy()
    row_index = towers.index[towers["Tower ID"].eq(tower_id)]
    if len(row_index) == 0:
        return
    index = row_index[0]
    current = towers.at[index, "Tower Status"]
    if current == new_status:
        return

    towers.at[index, "Tower Status"] = new_status
    counter = "Tower Outage Count" if new_status == "outage" else "Tower Recovery Count"
    if counter not in towers.columns:
        towers[counter] = 0
    towers.at[index, counter] = int(towers.at[index, counter]) + 1

    if new_status == "outage":
        affected = users["Connected Tower ID"].eq(tower_id) & users[
            "Connection Status"
        ].eq("connected")
        users.loc[affected, "Connection Status"] = "disconnected"
        if "Session Status" in users.columns:
            users.loc[affected, "Session Status"] = "dropped"
        if "Dropped Session Count" not in users.columns:
            users["Dropped Session Count"] = 0
        users.loc[affected, "Dropped Session Count"] += 1
        if "Handover Candidate" in users.columns:
            users.loc[affected, "Handover Candidate"] = None
        if "Handover Timer" in users.columns:
            users.loc[affected, "Handover Timer"] = 0.0
        append_event(
            f"Manual outage applied to {tower_id}; {int(affected.sum())} connected user(s) disconnected.",
            "Manual fault",
        )
    else:
        append_event(
            f"{tower_id} manually restored; disconnected users can reconnect on subsequent simulation ticks.",
            "Manual recovery",
        )

    state["user_movement_profiles"] = users
    state["tower_allocations"] = towers
    st.session_state.pop("excel_export", None)
    refresh_performance(state)
    st.session_state.history.append(
        network_snapshot(
            state["user_movement_profiles"],
            state["tower_allocations"],
            st.session_state.simulation_time,
            st.session_state.step_count,
        )
    )

def apply_traffic_surge(region_id):
    """Increase demand in a selected zone by switching users to a streaming activity profile.

    Purpose:
        This simulates a traffic hotspot by reclassifying users in a region so they generate heavier service demand and congestion.

    Parameters:
        region_id: The region identifier to which the high-traffic scenario will be applied.

    Returns:
        None: The user activity profile and state are updated in place.
    """
    state = st.session_state.simulation
    users = state["user_movement_profiles"].copy()
    selected = users["region_id"].eq(region_id)
    if not selected.any():
        return
    if "activity" not in users.columns:
        raise KeyError("The users table does not contain an 'activity' column.")
    users["activity"] = users["activity"].astype(str)
    if "Activity before dashboard surge" not in users.columns:
        users["Activity before dashboard surge"] = users["activity"]
    users.loc[selected, "activity"] = "streaming"
    users.loc[selected, "Traffic Scenario"] = "High traffic"
    state["user_movement_profiles"] = users
    st.session_state.pop("excel_export", None)
    refresh_performance(state)
    append_event(
        f"High-traffic scenario applied to region {region_id} ({int(selected.sum())} users switched to streaming demand).",
        "Traffic scenario",
    )

def clear_traffic_surge():
    """Remove the active traffic surge and restore the generated user activity profile.

    Purpose:
        This helper reverses a manual hotspot scenario and returns the simulation to its baseline utilisation pattern.

    Parameters:
        None: The function reads the current simulation state from st.session_state.

    Returns:
        None: The user table is reverted and the state is refreshed.
    """
    state = st.session_state.simulation
    users = state["user_movement_profiles"].copy()
    backup = "Activity before dashboard surge"
    if backup not in users.columns:
        return
    selected = users[backup].notna()
    users.loc[selected, "activity"] = users.loc[selected, backup]
    users = users.drop(columns=[backup])
    if "Traffic Scenario" in users.columns:
        users = users.drop(columns=["Traffic Scenario"])
    state["user_movement_profiles"] = users
    st.session_state.pop("excel_export", None)
    refresh_performance(state)
    append_event(
        "Traffic surge cleared; users returned to their generated activity profiles.",
        "Traffic scenario",
    )

def build_excel_export(state):
    """Export the current simulation state to an in-memory Excel workbook.

    Purpose:
        This is used by the dashboard download button to generate a workbook without writing a file to disk.

    Parameters:
        state (dict): Simulation state bundle including users, towers, and metrics.

    Returns:
        bytes: Excel workbook content in memory, ready to download.
    """
    users = state["user_movement_profiles"]
    towers = state["tower_allocations"]
    cells = sites.calculate_cell_performance(users, towers)
    report = sites.generate_network_performance_report(users, towers)
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        users.to_excel(writer, sheet_name="users", index=False)
        cells.to_excel(writer, sheet_name="cell_performance", index=False)
        towers.to_excel(writer, sheet_name="towers", index=False)
        for key, sheet in [
            ("tower_report", "tower_performance"),
            ("region_report", "region_performance"),
            ("user_report", "user_performance"),
        ]:
            if isinstance(report.get(key), pd.DataFrame):
                report[key].to_excel(writer, sheet_name=sheet, index=False)
        pd.DataFrame([report["network_summary"]]).to_excel(
            writer, sheet_name="network_summary", index=False
        )
    output.seek(0)
    return output.getvalue()


# DASHBOARD RENDERING

def make_network_map(
    state, users, towers, metric="SINR", show_coverage=True, show_handovers=True
):
    """Build the Plotly map that visualises town, towers, users and handover state.

    Purpose:
        This function creates the main geographic dashboard view and encodes the selected metric so connected users can be coloured by SINR, throughput, power, motion, or connection state.

    Parameters:
        state (dict): Simulation state containing the town layout and coverage details.
        users (pd.DataFrame): User table to render on the network map.
        towers (pd.DataFrame): Tower table to render on the network map.
        metric (str): Metric used to colour connected users.
        show_coverage (bool): Whether to draw tower coverage circles.
        show_handovers (bool): Whether to draw pending handover links.

    Returns:
        plotly.graph_objects.Figure: A configured map figure ready to be rendered by Streamlit.
    """
    fig = go.Figure()
    assignments = np.asarray(state["assignments"])
    xx = np.asarray(state["xx"])
    yy = np.asarray(state["yy"])
    stride = max(1, int(np.ceil(max(assignments.shape) / 110)))
    x_values = (
        xx[0, ::stride] if xx.ndim == 2 else np.arange(assignments.shape[1])[::stride]
    )
    y_values = (
        yy[::stride, 0] if yy.ndim == 2 else np.arange(assignments.shape[0])[::stride]
    )

    fig.add_trace(
        go.Heatmap(
            x=x_values,
            y=y_values,
            z=assignments[::stride, ::stride],
            uid="town-region-heatmap",
            colorscale="Viridis",
            opacity=0.24,
            showscale=False,
            hoverinfo="skip",
            name="Town regions",
        )
    )

    centers = np.asarray(state["centers"], dtype=float)
    if centers.size:
        fig.add_trace(
            go.Scatter(
                x=centers[:, 0],
                y=centers[:, 1],
                mode="markers",
                name="Region centres",
                uid="region-centres",
                marker=dict(symbol="x", size=11, color="#e2e8f0", line=dict(width=2)),
                hovertemplate="Region centre<extra></extra>",
            )
        )

    tower_status = towers.get(
        "Tower Status", pd.Series("operational", index=towers.index)
    ).astype(str)
    connected_counts = (
        users.loc[users["Connection Status"].eq("connected"), "Connected Tower ID"]
        .value_counts()
        .to_dict()
    )

    # Coverage circles are split by operating state so the legend is meaningful.
    if show_coverage:
        theta = np.linspace(0, 2 * np.pi, 65)
        for _, tower in towers.iterrows():
            is_operational = str(tower_status.loc[tower.name]) == "operational"
            radius = float(
                state.get("coverage_range", DEFAULT_CONFIG["coverage_range"])
            )
            circle_x = float(tower["x"]) + radius * np.cos(theta)
            circle_y = float(tower["y"]) + radius * np.sin(theta)
            fig.add_trace(
                go.Scatter(
                    x=circle_x,
                    y=circle_y,
                    mode="lines",
                    uid=f"coverage-{tower['Tower ID']}",
                    line=dict(
                        color="#f87171" if is_operational else "#94a3b8",
                        width=1.2,
                        dash="dash" if is_operational else "dot",
                    ),
                    opacity=0.48,
                    name="Coverage radius"
                    if is_operational
                    else "Outage coverage footprint",
                    legendgroup="coverage-operational"
                    if is_operational
                    else "coverage-outage",
                    showlegend=(tower.name == towers.index[0]),
                    hoverinfo="skip",
                )
            )

    # Connected users carry the selected radio, throughput or movement metric.
    status = users["Connection Status"].astype(str)
    connected = users.loc[status.eq("connected")]
    disconnected = users.loc[status.eq("disconnected")]
    failed = users.loc[status.eq("failed")]

    def add_user_trace(frame, name, color, symbol="circle", size=6, extra_custom=None):
        """Add a single user-layer trace to the network map.

        Purpose:
            This helper handles the repeated drawing pattern for connected, disconnected and failed users so each group can use its own colour, symbol and hover data.

        Parameters:
            frame (pd.DataFrame): User subset being drawn.
            name (str): Display name for the trace legend.
            color: Colour or colour scale definition used for the markers.
            symbol (str): Marker symbol type such as circle or diamond.
            size (int): Marker size in pixels.
            extra_custom: Additional custom hover metadata if the caller supplies any.

        Returns:
            None: The trace is added directly to the Plotly figure in place.
        """

        if frame.empty:
            return
        custom_columns = [
            c
            for c in [
                "user_id",
                "Connected Tower ID",
                "region_id",
                "movement_mode",
                "activity",
            ]
            if c in frame.columns
        ]
        custom = frame[custom_columns].to_numpy() if custom_columns else None
        hover_lines = []
        for col in custom_columns:
            hover_lines.append(f"{col}: %{{customdata[{custom_columns.index(col)}]}}")
        hover = "<br>".join(hover_lines) + "<extra>%{fullData.name}</extra>"
        marker = dict(size=size, symbol=symbol, color=color, opacity=0.86)
        if metric == "SINR" and name == "Connected users" and "SINR" in frame.columns:
            marker.update(
                color=pd.to_numeric(frame["SINR"], errors="coerce"),
                colorscale="RdYlGn",
                cmin=-10,
                cmax=30,
                colorbar=dict(title="SINR (dB)"),
                showscale=True,
            )
        elif (
            metric == "Throughput"
            and name == "Connected users"
            and "User Throughput (Mbps)" in frame.columns
        ):
            marker.update(
                color=pd.to_numeric(frame["User Throughput (Mbps)"], errors="coerce"),
                colorscale="Blues",
                colorbar=dict(title="Mbps"),
                showscale=True,
            )
        elif (
            metric == "Received power"
            and name == "Connected users"
            and "Received Power" in frame.columns
        ):
            marker.update(
                color=pd.to_numeric(frame["Received Power"], errors="coerce"),
                colorscale="Turbo",
                colorbar=dict(title="dBm"),
                showscale=True,
            )
        trace_uid = "user-" + name.lower().replace(" ", "-")
        point_ids = (
            frame["user_id"].astype(str).tolist()
            if "user_id" in frame.columns
            else None
        )
        fig.add_trace(
            go.Scatter(
                x=frame["x"],
                y=frame["y"],
                mode="markers",
                name=name,
                uid=trace_uid,
                ids=point_ids,
                marker=marker,
                customdata=custom,
                hovertemplate=hover,
            )
        )

    movement_column = (
        "movement_mode"
        if "movement_mode" in users.columns
        else ("movement_status" if "movement_status" in users.columns else None)
    )
    if metric == "Movement mode" and movement_column is not None:
        movement_colors = {
            "stationary": "#94a3b8",
            "walking": "#22c55e",
            "driving": "#f59e0b",
            "moving": "#f59e0b",
        }
        for mode in sorted(users[movement_column].dropna().astype(str).unique()):
            frame = users.loc[
                users[movement_column].astype(str).eq(mode) & status.eq("connected")
            ]
            add_user_trace(
                frame, f"{mode.title()} users", movement_colors.get(mode, "#60a5fa")
            )
    elif metric == "Connection state":
        add_user_trace(connected, "Connected users", "#22c55e")
    else:
        add_user_trace(connected, "Connected users", "#22c55e")

    add_user_trace(disconnected, "Disconnected users", "#f8fafc", symbol="x", size=8)
    add_user_trace(failed, "Failed initial access", "#e879f9", symbol="diamond", size=8)

    # Tower symbols scale with the current number of connected users.
    for desired_status, symbol, color, label in [
        ("operational", "triangle-up", "#fb4b4e", "Operational cell sites"),
        ("outage", "x", "#020617", "Tower outages"),
    ]:
        frame = towers.loc[tower_status.eq(desired_status)]
        if frame.empty:
            continue
        sizes = [
            13 + min(15, np.sqrt(connected_counts.get(tid, 0)) * 1.4)
            for tid in frame["Tower ID"]
        ]
        custom = np.column_stack(
            [
                frame["Tower ID"].astype(str).to_numpy(),
                frame.get("Tower region type", pd.Series("unknown", index=frame.index))
                .astype(str)
                .to_numpy(),
                [connected_counts.get(tid, 0) for tid in frame["Tower ID"]],
            ]
        )
        fig.add_trace(
            go.Scatter(
                x=frame["x"],
                y=frame["y"],
                mode="markers",
                name=label,
                uid=f"tower-state-{desired_status}",
                ids=frame["Tower ID"].astype(str).tolist(),
                marker=dict(
                    symbol=symbol,
                    size=sizes,
                    color=color,
                    line=dict(color="#f8fafc", width=1.2),
                ),
                customdata=custom,
                hovertemplate="Tower: %{customdata[0]}<br>Region type: %{customdata[1]}<br>Connected users: %{customdata[2]}<extra></extra>",
            )
        )

    # Draw pending handover links from user location to candidate cell.
    if (
        show_handovers
        and "Handover Candidate" in users.columns
        and "Handover Status" in users.columns
    ):
        candidates = towers.set_index("Tower ID")
        pending = users.loc[users["Handover Status"].eq("preparing")]
        line_x, line_y = [], []
        for _, user in pending.iterrows():
            tower_id = user.get("Handover Candidate")
            if tower_id not in candidates.index:
                continue
            candidate = candidates.loc[tower_id]
            line_x.extend([float(user["x"]), float(candidate["x"]), None])
            line_y.extend([float(user["y"]), float(candidate["y"]), None])
        if line_x:
            fig.add_trace(
                go.Scatter(
                    x=line_x,
                    y=line_y,
                    mode="lines",
                    name="Pending handovers",
                    uid="pending-handover-links",
                    line=dict(color="#fbbf24", width=1.4, dash="dash"),
                    hoverinfo="skip",
                )
            )

    fig.update_layout(
        height=680,
        margin=dict(l=10, r=10, t=24, b=10),
        paper_bgcolor="#0b1220",
        plot_bgcolor="#101b2d",
        font=dict(color="#e5edf7"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            x=0,
            uirevision="network-map-legend-v1",
        ),
        hovermode="closest",
        dragmode="pan",
        # Keep the Plotly component mounted and smoothly interpolate supported
        # point/marker changes instead of visibly snapping between each tick.
        uirevision="telecom-network-map-v1",
        transition=dict(duration=650, easing="cubic-in-out"),
    )
    fig.update_xaxes(
        title="X position (m)",
        range=[0, DEFAULT_CONFIG["town_width"]],
        gridcolor="#263449",
        zeroline=False,
    )
    fig.update_yaxes(
        title="Y position (m)",
        range=[0, DEFAULT_CONFIG["town_height"]],
        gridcolor="#263449",
        zeroline=False,
        scaleanchor="x",
        scaleratio=1,
    )
    return fig

def render_dashboard():
    """Render the full Streamlit dashboard layout.

    Purpose:
        This is the main dashboard assembly function that creates KPI cards, trend plots, user explorer tables, tower fault panels and export buttons from the current simulation state.

    Parameters:
        None: The function reads all current state from st.session_state.

    Returns:
        None: It writes the dashboard UI into the active Streamlit page.
    """

    state = st.session_state.simulation
    users = state["user_movement_profiles"]
    towers = state["tower_allocations"]
    cells = state["cell_performance"]
    history = pd.DataFrame(st.session_state.history)
    events = pd.DataFrame(st.session_state.events)
    config = st.session_state.sim_config

    # Filters are selected in the sidebar and affect the map and user explorer, letting the operator narrow the active network view.
    filtered = users.copy()
    selected_regions = st.session_state.get("filter_regions", [])
    selected_types = st.session_state.get("filter_user_types", [])
    selected_statuses = st.session_state.get("filter_statuses", [])
    if selected_regions:
        filtered = filtered.loc[filtered["region_id"].isin(selected_regions)]
    if selected_types and "user_type" in filtered.columns:
        filtered = filtered.loc[filtered["user_type"].isin(selected_types)]
    if selected_statuses and "Connection Status" in filtered.columns:
        filtered = filtered.loc[filtered["Connection Status"].isin(selected_statuses)]

    # The summary strip reports whether the simulation is live or paused and includes the current time and tick count.
    st.markdown(
        f"<div class='topline'><span class='live-dot'></span> {'SIMULATION RUNNING' if st.session_state.running else 'SIMULATION PAUSED'} "
        f"<span class='muted'>· STEP {st.session_state.step_count:,} · T+{st.session_state.simulation_time:,.0f}s · {len(users):,} simulated devices</span></div>",
        unsafe_allow_html=True,
    )

    snapshot = network_snapshot(
        users, towers, st.session_state.simulation_time, st.session_state.step_count
    )
    status_cols = st.columns(6)
    kpis = [
        (
            "Connected users",
            f"{snapshot['Connected users']:,}",
            f"{snapshot['Connection rate (%)']:.1f}% of total",
        ),
        (
            "Mean SINR",
            display_value(snapshot["Mean SINR (dB)"], " dB"),
            "Connected users",
        ),
        (
            "Delivered throughput",
            display_value(snapshot["Total throughput (Mbps)"], " Mbps"),
            "Aggregate cell delivery",
        ),
        (
            "Mean latency",
            display_value(snapshot["Mean latency (ms)"], " ms"),
            "Connected users",
        ),
        (
            "Tower outages",
            f"{snapshot['Tower outages']}/{len(towers)}",
            f"{snapshot['Operational towers']} operational",
        ),
        (
            "Capacity utilisation",
            display_value(snapshot["Capacity utilisation (%)"], "%"),
            "Mean operational cell",
        ),
    ]
    for col, (label, value, delta) in zip(status_cols, kpis):
        col.metric(label, value, delta)

    overview, ran, users_tab, incidents = st.tabs(
        ["Network overview", "RAN & cell capacity", "User explorer", "Faults & reports"]
    )

    with overview:
        left, right = st.columns([3.2, 1])
        with left:
            map_metric = st.session_state.get("map_metric", "SINR")
            map_figure = make_network_map(
                state,
                filtered,
                towers,
                metric=map_metric,
                show_coverage=st.session_state.get("show_coverage", True),
                show_handovers=st.session_state.get("show_handovers", True),
            )
            # Use the configured town extent, rather than the default, if changed.
            map_figure.update_xaxes(range=[0, config["town_width"]])
            map_figure.update_yaxes(range=[0, config["town_height"]])
            st.plotly_chart(
                map_figure,
                width="stretch",
                key="telecom_network_map",
                theme=None,
                config={"scrollZoom": True, "displaylogo": False, "responsive": True},
            )
            st.caption(
                f"Map filters currently show {len(filtered):,} of {len(users):,} users. Drag to pan, scroll to zoom, and toggle traces in the legend."
            )
        with right:
            st.subheader("Live network state")
            st.metric("Disconnected", f"{snapshot['Disconnected users']:,}")
            st.metric("Failed initial access", f"{snapshot['Failed users']:,}")
            st.metric("Congested cells", f"{snapshot['Congested towers']:,}")
            st.metric("Completed handovers", f"{snapshot['Completed handovers']:.0f}")
            st.metric("Dropped sessions", f"{snapshot['Dropped sessions']:.0f}")
            st.metric("Handover failures", f"{snapshot['Handover failures']:.0f}")
            st.divider()
            st.write("**Map key**")
            st.caption(
                "Connected users are coloured by the selected metric. Disconnected users use crosses; initial access failures use diamonds. Cell-site marker size reflects connected-user load."
            )

        st.subheader("Network performance over time")
        if not history.empty:
            metric_choices = [
                "Connection rate (%)",
                "Mean SINR (dB)",
                "Total throughput (Mbps)",
                "Mean latency (ms)",
                "Packet loss (%)",
                "Capacity utilisation (%)",
            ]
            trend_metric = st.selectbox(
                "Trend metric", metric_choices, key="trend_metric"
            )
            trend = px.line(
                history, x="Simulation time (s)", y=trend_metric, markers=True
            )
            trend.update_layout(
                height=320,
                margin=dict(l=10, r=10, t=20, b=10),
                paper_bgcolor="#0b1220",
                plot_bgcolor="#101b2d",
                font_color="#e5edf7",
                uirevision="network-trend-v1",
                transition=dict(duration=500, easing="cubic-in-out"),
            )
            trend.update_xaxes(gridcolor="#263449")
            trend.update_yaxes(gridcolor="#263449")
            st.plotly_chart(
                trend, width="stretch", key="network_performance_trend", theme=None
            )

    with ran:
        st.subheader("Cell performance")
        cell_table = cells.copy()
        if "Tower Status" not in cell_table.columns:
            cell_table["Tower Status"] = "operational"
        chart_metric = st.selectbox(
            "Compare cells by",
            [
                c
                for c in [
                    "Capacity Utilisation (%)",
                    "Cell Load (%)",
                    "Cell Throughput (Mbps)",
                    "Connected_Users",
                    "Average_SINR_dB",
                ]
                if c in cell_table.columns
            ],
            key="cell_metric",
        )
        bar = px.bar(
            cell_table,
            x="Tower ID",
            y=chart_metric,
            color="Tower Status",
            hover_data=[
                c
                for c in [
                    "Tower region type",
                    "Connected_Users",
                    "Cell Capacity (Mbps)",
                    "Cell Demand (Mbps)",
                ]
                if c in cell_table.columns
            ],
        )
        bar.update_layout(
            height=380,
            margin=dict(l=10, r=10, t=25, b=10),
            paper_bgcolor="#0b1220",
            plot_bgcolor="#101b2d",
            font_color="#e5edf7",
            uirevision="cell-performance-v1",
            transition=dict(duration=500, easing="cubic-in-out"),
        )
        st.plotly_chart(bar, width="stretch", key="cell_performance_chart", theme=None)
        table_columns = [
            c
            for c in [
                "Tower ID",
                "Tower region",
                "Tower region type",
                "Tower Status",
                "Connected_Users",
                "Cell Capacity (Mbps)",
                "Cell Demand (Mbps)",
                "Cell Throughput (Mbps)",
                "Cell Load (%)",
                "Capacity Utilisation (%)",
                "Congestion Level",
                "Average_SINR_dB",
                "Average_Latency_ms",
                "Average_Packet_Loss_Percent",
            ]
            if c in cell_table.columns
        ]
        st.dataframe(cell_table[table_columns], width="stretch", hide_index=True)

    with users_tab:
        st.subheader("Device-level explorer")
        user_ids = (
            users["user_id"].tolist()
            if "user_id" in users.columns
            else users.index.tolist()
        )
        selected_user_id = (
            st.selectbox("Inspect user", user_ids, key="inspect_user_id")
            if user_ids
            else None
        )
        if selected_user_id is not None:
            selected_rows = (
                users.loc[users["user_id"].eq(selected_user_id)]
                if "user_id" in users.columns
                else users.loc[[selected_user_id]]
            )
            if not selected_rows.empty:
                row = selected_rows.iloc[0]
                detail_cols = st.columns(5)
                detail_cols[0].metric(
                    "Connection", str(row.get("Connection Status", "n/a"))
                )
                detail_cols[1].metric(
                    "Serving cell", str(row.get("Connected Tower ID", "n/a"))
                )
                detail_cols[2].metric(
                    "SINR",
                    display_value(
                        pd.to_numeric(
                            pd.Series([row.get("SINR", row.get("SINR (dB)", np.nan))]),
                            errors="coerce",
                        ).iloc[0],
                        " dB",
                    ),
                )
                detail_cols[3].metric(
                    "Throughput",
                    display_value(
                        pd.to_numeric(
                            pd.Series([row.get("User Throughput (Mbps)", np.nan)]),
                            errors="coerce",
                        ).iloc[0],
                        " Mbps",
                    ),
                )
                detail_cols[4].metric(
                    "Latency",
                    display_value(
                        pd.to_numeric(
                            pd.Series([row.get("Latency (ms)", np.nan)]),
                            errors="coerce",
                        ).iloc[0],
                        " ms",
                    ),
                )
                detail_fields = [
                    "user_id",
                    "user_type",
                    "device_type",
                    "network_capability",
                    "activity",
                    "movement_mode",
                    "movement_status",
                    "speed",
                    "x",
                    "y",
                    "region_id",
                    "Connected Tower ID",
                    "Distance to Tower",
                    "Received Power",
                    "SINR",
                    "User Throughput (Mbps)",
                    "Latency (ms)",
                    "Packet Loss (%)",
                    "Handover Count",
                    "Dropped Session Count",
                ]
                detail_rows = [
                    (field, table_display_value(row[field]))
                    for field in detail_fields
                    if field in row.index
                ]
                # The Value column intentionally contains display strings only.
                # User records mix strings, numbers, NaN values and coordinate
                # objects; converting them prevents PyArrow type-inference errors.
                detail_table = pd.DataFrame(detail_rows, columns=["Field", "Value"])
                st.dataframe(detail_table, width="stretch", hide_index=True)
        st.subheader("Filtered user records")
        table_columns = [
            c
            for c in [
                "user_id",
                "region_id",
                "user_type",
                "device_type",
                "activity",
                "movement_mode",
                "Connection Status",
                "Connected Tower ID",
                "SINR",
                "Received Power",
                "User Throughput (Mbps)",
                "Latency (ms)",
                "Packet Loss (%)",
            ]
            if c in filtered.columns
        ]
        st.dataframe(
            filtered[table_columns], width="stretch", hide_index=True, height=420
        )
        st.download_button(
            "Download filtered users (CSV)",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="filtered_users.csv",
            mime="text/csv",
        )

    with incidents:
        st.subheader("Tower faults and recovery")
        outage_table = towers[
            [
                c
                for c in [
                    "Tower ID",
                    "Tower region",
                    "Tower region type",
                    "Tower Status",
                    "Tower Outage Count",
                    "Tower Recovery Count",
                ]
                if c in towers.columns
            ]
        ].copy()
        st.dataframe(outage_table, width="stretch", hide_index=True)
        st.subheader("Simulation event log")
        if events.empty:
            st.info(
                "No events recorded yet. Start the simulation or trigger a scenario from the sidebar."
            )
        else:
            st.dataframe(events, width="stretch", hide_index=True, height=300)
        st.subheader("Export project results")
        if st.button("Prepare latest Excel workbook"):
            with st.spinner("Preparing report workbook..."):
                st.session_state.excel_export = build_excel_export(state)
        if st.session_state.get("excel_export"):
            st.download_button(
                "Download full Excel report",
                data=st.session_state.excel_export,
                file_name="telecommunications_digital_twin_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help="Includes user, cell, tower, region and network-summary sheets.",
            )
        st.download_button(
            "Download all towers (CSV)",
            data=towers.to_csv(index=False).encode("utf-8"),
            file_name="tower_status.csv",
            mime="text/csv",
        )

# STREAMLIT APPLICATIONS 

def main():
    """Run the telecommunications digital twin application entry point.

    Purpose:
        This function configures the page theme, builds the sidebar controls, initialises the simulation, and keeps the live dashboard updating while the app is running.

    Parameters:
        None: Streamlit widgets and session state provide the runtime inputs.

    Returns:
        None: This is the application entry point and typically runs until the browser session ends.
    """
    st.set_page_config(
        page_title="Telecommunications Digital Twin",
        page_icon="📡",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
    <style>
      .stApp { background: #080f1c; color: #e5edf7; }
      [data-testid="stSidebar"] { background: #0d1727; }
      .block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1800px; }
      h1, h2, h3 { color: #edf4ff; }
      .topline { background: linear-gradient(90deg,#111e31,#0c1524); border: 1px solid #26364c; border-radius: 12px; padding: 12px 16px; margin-bottom: 14px; font-weight: 700; letter-spacing: .06em; }
      .live-dot { display:inline-block; width:9px; height:9px; border-radius:50%; background:#22c55e; margin-right:8px; box-shadow:0 0 10px #22c55e; }
      .muted { color:#93a4ba; font-weight:400; letter-spacing:0; }
      [data-testid="stMetric"] { background:#101b2d; border:1px solid #23344b; border-radius:10px; padding:12px 14px; }
      [data-testid="stMetricLabel"] { color:#9fb0c6; }
      div[data-testid="stTabs"] button { color:#b8c7d9; }
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.title("📡 Telecommunications Digital Twin")
    st.caption(
        "Fictional town · Multi-cell RAN · Mobility · Handovers · Faults · Service KPIs"
    )

    with st.sidebar:
        st.header("Simulation controls")
        st.caption(
            "Town, user and tower settings take effect when you initialise or reset the simulation."
        )
        c1, c2 = st.columns(2)
        cfg = dict(DEFAULT_CONFIG)
        cfg["num_users"] = c1.number_input(
            "Users",
            min_value=100,
            max_value=10000,
            value=DEFAULT_CONFIG["num_users"],
            step=100,
        )
        cfg["total_towers"] = c2.number_input(
            "Target cell sites",
            min_value=1,
            max_value=100,
            value=DEFAULT_CONFIG["total_towers"],
            step=1,
        )
        c1, c2 = st.columns(2)
        cfg["num_areas"] = c1.number_input(
            "Town regions",
            min_value=2,
            max_value=20,
            value=DEFAULT_CONFIG["num_areas"],
            step=1,
        )
        cfg["grid_resolution"] = c2.select_slider(
            "Grid resolution (m)",
            options=[10, 20, 25, 50, 100],
            value=DEFAULT_CONFIG["grid_resolution"],
        )
        cfg["coverage_range"] = st.slider(
            "Cell radius (m)",
            min_value=250,
            max_value=2500,
            value=DEFAULT_CONFIG["coverage_range"],
            step=50,
        )
        cfg["time_step"] = st.slider(
            "Simulated seconds per tick",
            min_value=1,
            max_value=30,
            value=DEFAULT_CONFIG["time_step"],
            step=1,
        )
        cfg["outage_chance"] = st.slider(
            "Outage probability per tick",
            min_value=0.0,
            max_value=0.05,
            value=float(DEFAULT_CONFIG["outage_chance"]),
            step=0.0005,
            format="%.4f",
        )
        cfg["recovery_chance"] = st.slider(
            "Recovery probability per tick",
            min_value=0.0,
            max_value=0.5,
            value=float(DEFAULT_CONFIG["recovery_chance"]),
            step=0.01,
            format="%.2f",
        )
        refresh_seconds = st.slider(
            "Dashboard refresh (seconds)",
            min_value=0.75,
            max_value=3.0,
            value=1.0,
            step=0.25,
        )
        st.caption(
            "For smooth map movement, keep the refresh interval at 0.75–1.0 seconds or slower."
        )
        initialise_clicked = st.button(
            "⟲ Initialise / reset network", width="stretch", type="primary"
        )

    # Build the initial network on first launch or after a reset so the dashboard starts from a clean simulation state.
    if "simulation" not in st.session_state or initialise_clicked:
        with st.spinner(
            "Generating town, users, cell sites and initial radio connections..."
        ):
            initialise_dashboard(cfg)
        st.success("Simulation initialised.")

    # Live controls are separate from the initialise-only settings.
    with st.sidebar:
        st.divider()
        st.subheader("Playback")
        c1, c2 = st.columns(2)
        start_clicked = c1.button(
            "▶ Start", width="stretch", disabled=st.session_state.running
        )
        pause_clicked = c2.button(
            "Ⅱ Pause", width="stretch", disabled=not st.session_state.running
        )
        step_clicked = st.button(
            "Step one tick", width="stretch", disabled=st.session_state.running
        )
        if start_clicked:
            st.session_state.running = True
        if pause_clicked:
            st.session_state.running = False
        if step_clicked:
            try:
                advance_one_step()
            except Exception as exc:
                st.error(f"Simulation step failed: {exc}")

        st.divider()
        st.subheader("Map & data filters")
        users_now = st.session_state.simulation["user_movement_profiles"]
        region_options = (
            sorted(users_now["region_id"].dropna().unique().tolist())
            if "region_id" in users_now.columns
            else []
        )
        user_type_options = (
            sorted(users_now["user_type"].dropna().unique().tolist())
            if "user_type" in users_now.columns
            else []
        )
        st.multiselect(
            "Regions",
            region_options,
            key="filter_regions",
            help="Leave empty to show every region.",
        )
        st.multiselect(
            "User types",
            user_type_options,
            key="filter_user_types",
            help="Leave empty to show every type.",
        )
        st.multiselect(
            "Connection states",
            ["connected", "disconnected", "failed"],
            key="filter_statuses",
            help="Leave empty to show all states.",
        )
        st.selectbox(
            "Colour connected users by",
            [
                "SINR",
                "Throughput",
                "Received power",
                "Movement mode",
                "Connection state",
            ],
            key="map_metric",
        )
        st.checkbox("Show coverage circles", value=True, key="show_coverage")
        st.checkbox("Show pending handovers", value=True, key="show_handovers")

        st.divider()
        st.subheader("Live fault injection")
        current_towers = st.session_state.simulation["tower_allocations"]
        tower_ids = current_towers["Tower ID"].astype(str).tolist()
        target_tower = (
            st.selectbox("Target cell site", tower_ids, key="target_tower")
            if tower_ids
            else None
        )
        if target_tower:
            target_row = current_towers.loc[
                current_towers["Tower ID"].eq(target_tower)
            ].iloc[0]
            st.caption(
                f"Current state: **{target_row.get('Tower Status', 'unknown')}**"
            )
            f1, f2 = st.columns(2)
            if f1.button("Force outage", width="stretch"):
                set_tower_status(target_tower, "outage")
            if f2.button("Restore cell", width="stretch"):
                set_tower_status(target_tower, "operational")

        region_for_surge = (
            st.selectbox("Traffic surge region", region_options, key="traffic_region")
            if region_options
            else None
        )
        if st.button(
            "Apply high-traffic scenario",
            width="stretch",
            disabled=region_for_surge is None,
        ):
            apply_traffic_surge(region_for_surge)
        if st.button("Clear traffic scenario", width="stretch"):
            clear_traffic_surge()

        st.divider()
        st.caption(
            "Simplified synthetic RAN model for learning and demonstration. Not a standards-compliant LTE/5G simulator."
        )

    # Persist the chosen dashboard settings without rebuilding the town or user population, keeping the simulation stable while controls change.
    st.session_state.sim_config = dict(cfg)
    st.session_state.sim_config["town_width"] = DEFAULT_CONFIG["town_width"]
    st.session_state.sim_config["town_height"] = DEFAULT_CONFIG["town_height"]
    st.session_state.sim_config["seed"] = DEFAULT_CONFIG["seed"]

    @st.fragment(run_every=refresh_seconds if st.session_state.running else None)
    def live_dashboard_fragment():
        """Refresh the dashboard fragment on a periodic timer while the simulation is running.

        Purpose:
            This helper advances the model when live playback is active and then re-renders the dashboard, giving the user a continuously updating view of the network.

        Parameters:
            None: It uses the active Streamlit session state and refresh interval.

        Returns:
            None: It triggers the dashboard update and advances the simulation state in place.
        """
        if st.session_state.running:
            try:
                advance_one_step()
            except Exception as exc:
                st.session_state.running = False
                st.error(f"Simulation paused because a step failed: {exc}")
        render_dashboard()

    live_dashboard_fragment()

if __name__ == "__main__":
    main()
