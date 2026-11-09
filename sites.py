import numpy as np
import pandas as pd


# CONSTANTS

_SHADOW_SIGMA = {
    "urban": 8.0,
    "residential": 7.0,
    "suburban": 5.0,
    "rural": 2.0,
}

_DEVICE_GAIN = {
    "smartphone": 1.0,
    "tablet": 1.0,
    "laptop": 2.0,
    "laptops": 2.0,
    "IoT": 3.0,
}

_ACTIVITY_DEMAND = {
    "idle": 0.02,
    "browsing": 1.5,
    "streaming": 5.0,
    "voice_call": 0.15,
    "uploading": 3.0,
}


def _get_sinr(users):
    """Return the project's SINR values, accepting either historical column."""
    if "SINR" in users.columns:
        return users["SINR"].astype(float)

    if "SINR (dB)" in users.columns:
        return users["SINR (dB)"].astype(float)

    raise ValueError("Users DataFrame must contain 'SINR' or 'SINR (dB)'.")


def _operational_towers(tower_allocations):
    """Return operational rows from the individual-tower table."""
    if tower_allocations is None or tower_allocations.empty:
        columns = [] if tower_allocations is None else tower_allocations.columns
        return pd.DataFrame(columns=columns)

    if "Tower Status" not in tower_allocations.columns:
        return tower_allocations.copy()

    return tower_allocations.loc[
        tower_allocations["Tower Status"].eq("operational")
    ].copy()


def _tower_index(tower_allocations):
    """Create a Tower ID -> row-position lookup for an individual-tower table."""
    return {
        tower_id: i
        for i, tower_id in enumerate(tower_allocations["Tower ID"].tolist())
    }


def _radio_link_matrices(
    users,
    tower_allocations,
    seed=None,
    bandwidth=10e6,
    noise_figure=7
):
    """Calculate link powers, distances, per-tower interference and SINR.

    Matrix rows are users and columns are operational towers. The same link
    realisation is used for every candidate tower within this calculation.
    """

    # Keep only operational towers.
    towers = _operational_towers(tower_allocations).reset_index(drop=True)

    # Return correctly shaped empty matrices when no towers are available.
    if len(towers) == 0:
        empty = np.empty((len(users), 0))
        noise_dbm = -174 + 10 * np.log10(bandwidth) + noise_figure

        return (
            towers,
            empty,
            empty,
            empty,
            empty,
            noise_dbm,
            empty
        )

    # Extract user and tower coordinates.
    user_coords = users[["x", "y"]].to_numpy(dtype=float)
    tower_coords = towers[["x", "y"]].to_numpy(dtype=float)

    # Extract radio properties from the tower table.
    frequencies = towers["Tower frequency"].to_numpy(dtype=float)
    powers = towers["Tower power"].to_numpy(dtype=float)
    antenna_gains = towers["Antenna gain"].to_numpy(dtype=float)

    # Assign shadow-fading variation based on the tower's region type.
    tower_sigma = towers["Tower region type"].map(
        _SHADOW_SIGMA
    ).to_numpy(dtype=float)

    if np.any(frequencies <= 0):
        raise ValueError("Tower frequencies must be greater than zero.")

    # Calculate the distance between every user and operational tower.
    differences = (
        user_coords[:, np.newaxis, :]
        - tower_coords[np.newaxis, :, :]
    )

    distances = np.sqrt(np.sum(differences ** 2, axis=2))

    # Prevent zero-distance calculations.
    distances = np.maximum(distances, 1.0)

    # Calculate frequency-dependent reference path loss.
    c = 3e8
    reference_loss = 20 * np.log10((4 * np.pi * frequencies) / c)

    # Add distance-dependent attenuation.
    path_losses = (
        reference_loss[np.newaxis, :]
        + 30 * np.log10(distances)
    )

    # Generate shadow fading for each user-to-tower link.
    rng = np.random.default_rng(seed)

    shadow = rng.normal(
        0.0,
        tower_sigma[np.newaxis, :],
        size=distances.shape
    )

    # Retrieve receiver gain where it is available.
    receiver_gain = users.get(
        "Receiver antenna gain",
        pd.Series(0.0, index=users.index)
    ).to_numpy(dtype=float)

    # Calculate received power in dBm.
    received_dbm = (
        powers[np.newaxis, :]
        + antenna_gains[np.newaxis, :]
        + receiver_gain[:, np.newaxis]
        - path_losses
        + shadow
    )

    # Convert received power from dBm to milliwatts.
    received_mw = 10 ** (received_dbm / 10)

    # Identify co-channel towers operating on the same frequency.
    cochannel = np.isclose(
        frequencies[:, np.newaxis],
        frequencies[np.newaxis, :],
        rtol=1e-9,
        atol=0
    )

    # A tower does not interfere with itself.
    np.fill_diagonal(cochannel, False)

    # Calculate interference from other co-channel towers.
    interference_mw = received_mw @ cochannel.T

    # Calculate thermal noise power.
    noise_dbm = -174 + 10 * np.log10(bandwidth) + noise_figure
    noise_mw = 10 ** (noise_dbm / 10)

    # Calculate SINR for every possible user-to-tower link.
    sinr_db = 10 * np.log10(
        received_mw
        / np.maximum(interference_mw + noise_mw, 1e-300)
    )

    return (
        towers,
        distances,
        path_losses,
        received_dbm,
        interference_mw,
        noise_dbm,
        sinr_db
    )


def _set_connection_to_tower(
    users,
    row_index,
    tower,
    distance,
    received_dbm=None,
    path_loss_value=None,
    interference_mw=None,
    noise_dbm=None,
    sinr_db=None
):
    """Update the serving-tower columns for one user."""

    # Update the user's serving tower identity and coordinates.
    users.at[row_index, "Connected Tower ID"] = tower["Tower ID"]
    users.at[row_index, "Connected Tower"] = tuple(tower["Tower coordinates"])

    # Update the distance and regional information.
    users.at[row_index, "Distance to Tower"] = float(distance)
    users.at[row_index, "Tower region"] = tower["Tower region"]
    users.at[row_index, "Tower region type"] = tower["Tower region type"]

    # Update serving tower radio properties.
    users.at[row_index, "Tower frequency"] = float(tower["Tower frequency"])
    users.at[row_index, "Tower power"] = float(tower["Tower power"])
    users.at[row_index, "Antenna gain"] = float(tower["Antenna gain"])

    # Update path loss when supplied.
    if path_loss_value is not None:
        users.at[row_index, "Path loss"] = float(path_loss_value)

    # Update received power when supplied.
    if received_dbm is not None:
        users.at[row_index, "Received Power"] = float(received_dbm)

    # Update interference values when supplied.
    if interference_mw is not None:
        users.at[row_index, "Interference (mW)"] = float(interference_mw)

        users.at[row_index, "Interference (dBm)"] = (
            10 * np.log10(interference_mw)
            if interference_mw > 0
            else -np.inf
        )

    # Update noise values when supplied.
    if noise_dbm is not None:
        users.at[row_index, "Noise (dBm)"] = float(noise_dbm)
        users.at[row_index, "Noise (mW)"] = 10 ** (noise_dbm / 10)

    # Maintain both SINR column conventions used in the project.
    if sinr_db is not None:
        users.at[row_index, "SINR"] = float(sinr_db)
        users.at[row_index, "SINR (dB)"] = float(sinr_db)


# TOWER SETUP

def connect_tower(users, tower_allocations):
    """Connect each user to the nearest operational tower.

    Pass the individual-tower table returned by create_tower_table(). The
    legacy region-allocation table is also accepted for backwards compatibility.
    Call this once during initialisation, not during every simulation step.
    """

    # Copy the user table before updating serving tower information.
    users = users.copy()

    # Check whether the input contains individual tower records.
    if (
        "Tower ID" in tower_allocations.columns
        and {"x", "y"}.issubset(tower_allocations.columns)
    ):
        # Retrieve operational towers only.
        towers = _operational_towers(
            tower_allocations
        ).reset_index(drop=True)

        if towers.empty:
            raise ValueError("No operational towers are available.")

        # Extract tower and user coordinates.
        tower_coords = towers[["x", "y"]].to_numpy(dtype=float)
        user_coords = users[["x", "y"]].to_numpy(dtype=float)

        # Calculate user-to-tower distances.
        differences = (
            user_coords[:, np.newaxis, :]
            - tower_coords[np.newaxis, :, :]
        )

        distances = np.sqrt(np.sum(differences ** 2, axis=2))

        # Find the nearest operational tower for each user.
        nearest_indices = np.argmin(distances, axis=1)
        nearest_towers = towers.iloc[nearest_indices]

        # Assign the nearest tower information to each user.
        users["Connected Tower ID"] = nearest_towers["Tower ID"].to_numpy()

        users["Connected Tower"] = [
            tuple(x)
            for x in nearest_towers["Tower coordinates"]
        ]

        users["Distance to Tower"] = distances[
            np.arange(len(users)),
            nearest_indices
        ]

        users["Tower region"] = nearest_towers["Tower region"].to_numpy()
        users["Tower region type"] = nearest_towers["Tower region type"].to_numpy()

    else:
        # Legacy input: regional rows with a list of tower_locations.
        tower_coords_list = []
        tower_regions = []
        tower_region_types = []

        # Extract all tower locations from the regional allocation table.
        for _, row in tower_allocations.iterrows():
            locations = row["tower_locations"]

            if len(locations) > 0:
                tower_coords_list.append(
                    np.asarray(locations, dtype=float)
                )

                tower_regions.extend(
                    [row["region_id"]] * len(locations)
                )

                tower_region_types.extend(
                    [row["region_type"]] * len(locations)
                )

        if not tower_coords_list:
            raise ValueError("No tower locations available.")

        # Combine regional tower coordinates into one array.
        tower_coords = np.concatenate(tower_coords_list, axis=0)
        user_coords = users[["x", "y"]].to_numpy(dtype=float)

        # Calculate the distance from each user to each tower.
        differences = (
            user_coords[:, np.newaxis, :]
            - tower_coords[np.newaxis, :, :]
        )

        distances = np.sqrt(np.sum(differences ** 2, axis=2))

        # Find the closest tower for each user.
        nearest_indices = np.argmin(distances, axis=1)
        nearest_towers = tower_coords[nearest_indices]

        # Assign tower information to each user.
        users["Connected Tower ID"] = [
            f"tower_{i:03d}"
            for i in nearest_indices
        ]

        users["Connected Tower"] = [
            tuple(coord)
            for coord in nearest_towers
        ]

        users["Distance to Tower"] = distances[
            np.arange(len(users)),
            nearest_indices
        ]

        users["Tower region"] = np.asarray(
            tower_regions
        )[nearest_indices]

        users["Tower region type"] = np.asarray(
            tower_region_types
        )[nearest_indices]

    return users


def assign_frequency(users):
    """Assign serving-cell RF properties and receiver gain to each user."""

    # Copy the user table before adding radio properties.
    users = users.copy()

    # Define frequency, transmit power and antenna gain by region type.
    frequencies = {
        "urban": 6e9,
        "residential": 2e9,
        "suburban": 3e9,
        "rural": 900e6
    }

    powers = {
        "urban": 40,
        "residential": 20,
        "suburban": 43,
        "rural": 46
    }

    antenna_gains = {
        "urban": 10,
        "residential": 5,
        "suburban": 15,
        "rural": 18
    }

    # Assign receiver gain according to the user's device type.
    receiver_gain = users["device_type"].map(_DEVICE_GAIN)

    if receiver_gain.isna().any():
        unknown = users.loc[
            receiver_gain.isna(),
            "device_type"
        ].unique().tolist()

        raise ValueError(f"Unknown device type(s): {unknown}")

    # Add receiver and serving tower radio properties.
    users["Receiver antenna gain"] = receiver_gain.astype(float)
    users["Antenna gain"] = users["Tower region type"].map(antenna_gains)
    users["Tower power"] = users["Tower region type"].map(powers)
    users["Tower frequency"] = users["Tower region type"].map(frequencies)

    # Check that every user has been assigned valid radio properties.
    if users[
        ["Antenna gain", "Tower power", "Tower frequency"]
    ].isna().any().any():
        raise ValueError(
            "Unknown tower region type detected when assigning RF properties."
        )

    return users


def create_tower_table(tower_allocations):
    """Convert regional allocations into one row per tower."""

    # Prepare a list for the individual tower records.
    tower_data = []

    # Define radio properties according to each region type.
    frequencies = {
        "urban": 6e9,
        "residential": 2e9,
        "suburban": 3e9,
        "rural": 900e6
    }

    powers = {
        "urban": 40,
        "residential": 20,
        "suburban": 43,
        "rural": 46
    }

    antenna_gains = {
        "urban": 10,
        "residential": 5,
        "suburban": 15,
        "rural": 18
    }

    # Start numbering individual towers from zero.
    tower_number = 0

    # Loop through each region in the allocation table.
    for _, row in tower_allocations.iterrows():
        region_id = row["region_id"]
        region_type = row["region_type"]
        locations = row["tower_locations"]

        # Create a separate record for every tower location.
        for location in locations:
            x, y = location

            tower_data.append({
                "Tower ID": f"tower_{tower_number:03d}",
                "x": float(x),
                "y": float(y),
                "Tower coordinates": (float(x), float(y)),
                "Tower region": region_id,
                "Tower region type": region_type,
                "Tower frequency": frequencies[region_type],
                "Tower power": powers[region_type],
                "Antenna gain": antenna_gains[region_type],
                "Tower Status": "operational",
                "Tower Outage Count": 0,
                "Tower Recovery Count": 0,
                "Bandwidth (Hz)": 10e6,
            })

            # Increment the tower ID for the next record.
            tower_number += 1

    # Convert the generated tower records into a DataFrame.
    towers = pd.DataFrame(tower_data)

    if towers.empty:
        raise ValueError("Tower allocation produced no individual towers.")

    return towers


def path_loss(users):
    # Copy the input before adding the path-loss calculation.
    users = users.copy()

    # Define the reference distance, speed of light and path-loss exponent.
    d0, c, n = 1.0, 3e8, 3.0

    # Extract frequency and distance values.
    frequency = users["Tower frequency"].to_numpy(dtype=float)
    distance = users["Distance to Tower"].to_numpy(dtype=float)

    # Validate the input values.
    if np.any(frequency <= 0):
        raise ValueError("Frequencies must be greater than zero.")

    if np.any(distance < 0):
        raise ValueError("Distances cannot be negative.")

    # Prevent logarithms of zero.
    distance = np.maximum(distance, d0)

    # Calculate reference loss, then add distance-dependent attenuation.
    reference_loss = 20 * np.log10(
        (4 * np.pi * d0 * frequency) / c
    )

    users["Path loss"] = (
        reference_loss
        + 10 * n * np.log10(distance / d0)
    )

    return users


def shadow_fading(users, seed=None):
    # Copy user records before adding the random shadowing term.
    users = users.copy()

    # Retrieve the shadow-fading standard deviation for each region.
    sigma = users["Tower region type"].map(_SHADOW_SIGMA)

    if sigma.isna().any():
        raise ValueError("Unknown tower region type detected.")

    # Use NumPy random generation so a seed can make results reproducible.
    rng = np.random.default_rng(seed)

    users["Shadow fading"] = rng.normal(
        0,
        sigma.to_numpy(dtype=float),
        len(users)
    )

    return users


def received_power(users):
    # Keep the original input unchanged while calculating received power.
    users = users.copy()

    # Calculate received power from transmit power, gains and path loss.
    users["Received Power"] = (
        users["Tower power"]
        + users["Antenna gain"]
        + users["Receiver antenna gain"]
        - users["Path loss"]
        + users["Shadow fading"]
    )

    return users


def calculate_interference(users, tower_allocations=None, seed=None):
    """Calculate co-channel interference from every operational tower.

    Supplying the individual tower table is strongly recommended because this
    includes operational towers even when they currently have no connected users.
    """

    # Copy the input user table.
    users = users.copy()

    # Return empty interference columns when no users exist.
    if users.empty:
        users["Interference (mW)"] = pd.Series(dtype=float)
        users["Interference (dBm)"] = pd.Series(dtype=float)
        return users

    # Use the supplied tower table where available.
    if tower_allocations is None:
        # Backwards-compatible fallback: reconstruct only towers seen in users.
        tower_columns = [
            "Connected Tower ID",
            "Connected Tower",
            "Tower region type",
            "Tower frequency",
            "Tower power",
            "Antenna gain"
        ]

        towers = users[tower_columns].drop_duplicates(
            "Connected Tower ID"
        ).copy()

        # Recover tower coordinates from the serving tower column.
        coords = np.asarray(
            towers["Connected Tower"].tolist(),
            dtype=float
        )

        towers["x"] = coords[:, 0]
        towers["y"] = coords[:, 1]
        towers["Tower coordinates"] = list(towers["Connected Tower"])

        # Reconstruct the tower region information.
        towers["Tower region"] = (
            users.groupby("Connected Tower ID")["Tower region"]
            .first()
            .reindex(towers["Connected Tower ID"])
            .to_numpy()
        )

        towers["Tower Status"] = "operational"

    else:
        towers = tower_allocations

    # Calculate radio link conditions for all operational towers.
    (
        op_towers,
        distances,
        path_losses,
        received_dbm,
        interference_matrix,
        noise_dbm,
        _
    ) = _radio_link_matrices(
        users,
        towers,
        seed=seed
    )

    # Handle the situation where all towers are unavailable.
    if op_towers.empty:
        users["Received Power"] = -np.inf
        users["Interference (mW)"] = 0.0
        users["Interference (dBm)"] = -np.inf
        return users

    # Create a lookup between tower IDs and their matrix positions.
    lookup = _tower_index(op_towers)

    # Prepare arrays for the interference and serving-link values.
    interference = np.zeros(len(users), dtype=float)
    serving_signal = np.full(len(users), -np.inf, dtype=float)
    serving_path_loss = np.full(len(users), np.nan, dtype=float)

    # Preserve the current serving link values where available.
    current_received = (
        users["Received Power"].to_numpy(dtype=float)
        if "Received Power" in users
        else None
    )

    current_path_loss = (
        users["Path loss"].to_numpy(dtype=float)
        if "Path loss" in users
        else None
    )

    frequencies = op_towers["Tower frequency"].to_numpy(dtype=float)

    # Calculate interference for each user's serving tower.
    for user_idx, tower_id in enumerate(
        users["Connected Tower ID"].tolist()
    ):
        serving_idx = lookup.get(tower_id)

        if serving_idx is None:
            continue

        # Identify other operational towers using the same frequency.
        same_frequency = np.isclose(
            frequencies,
            frequencies[serving_idx],
            rtol=1e-9,
            atol=0
        )

        # Exclude the serving tower itself from interference.
        same_frequency[serving_idx] = False

        interference[user_idx] = np.sum(
            10 ** (received_dbm[user_idx, same_frequency] / 10)
        )

        # Preserve the user's original serving signal when possible.
        if (
            current_received is not None
            and np.isfinite(current_received[user_idx])
        ):
            serving_signal[user_idx] = current_received[user_idx]
        else:
            serving_signal[user_idx] = received_dbm[
                user_idx,
                serving_idx
            ]

        # Preserve the original serving path loss when available.
        if current_path_loss is not None:
            serving_path_loss[user_idx] = current_path_loss[user_idx]
        else:
            serving_path_loss[user_idx] = path_losses[
                user_idx,
                serving_idx
            ]

    # Update the user DataFrame with the calculated results.
    users["Received Power"] = serving_signal
    users["Path loss"] = serving_path_loss
    users["Interference (mW)"] = interference

    # Convert interference from milliwatts to dBm.
    with np.errstate(divide="ignore"):
        users["Interference (dBm)"] = np.where(
            interference > 0,
            10 * np.log10(interference),
            -np.inf
        )

    return users


def calculate_noise(users, bandwidth=10e6, noise_figure=7):
    # Add the thermal-noise estimate to a copied user table.
    users = users.copy()

    if bandwidth <= 0:
        raise ValueError("Bandwidth must be greater than zero.")

    # Calculate noise power in dBm, then convert it to milliwatts.
    noise_power_dbm = (
        -174
        + 10 * np.log10(bandwidth)
        + noise_figure
    )

    users["Noise (dBm)"] = noise_power_dbm
    users["Noise (mW)"] = 10 ** (noise_power_dbm / 10)

    return users


def calculate_sinr(users):
    # Calculate SINR without changing the supplied DataFrame.
    users = users.copy()

    # Convert received signal power from dBm to milliwatts.
    signal_mw = 10 ** (
        users["Received Power"].astype(float) / 10
    )

    # Retrieve interference and noise in milliwatts.
    interference_mw = users["Interference (mW)"].astype(float)
    noise_mw = users["Noise (mW)"].astype(float)

    # SINR is the serving signal divided by interference plus noise.
    denominator = interference_mw + noise_mw
    sinr_linear = signal_mw / denominator

    # Convert the linear SINR value into decibels.
    with np.errstate(divide="ignore", invalid="ignore"):
        sinr_db = 10 * np.log10(sinr_linear)

    # Maintain both SINR column conventions used in the project.
    users["SINR"] = sinr_db
    users["SINR (dB)"] = sinr_db

    return users


# HANDOVER 

def find_neighbouring_towers(users, tower_allocations, num_neighbours=5):
    # Copy users before adding neighbouring tower information.
    users = users.copy()

    # Only operational towers can be handover candidates.
    towers = _operational_towers(
        tower_allocations
    ).reset_index(drop=True)

    # Handle the absence of operational towers.
    if towers.empty:
        users["Neighbour Tower IDs"] = [
            []
            for _ in range(len(users))
        ]

        users["Neighbour Tower Distances"] = [
            []
            for _ in range(len(users))
        ]

        return users

    # Extract tower and user coordinates.
    tower_coords = towers[["x", "y"]].to_numpy(dtype=float)
    tower_ids = towers["Tower ID"].tolist()
    user_coords = users[["x", "y"]].to_numpy(dtype=float)

    # Calculate user-to-tower distances.
    differences = (
        user_coords[:, np.newaxis, :]
        - tower_coords[np.newaxis, :, :]
    )

    distances = np.sqrt(np.sum(differences ** 2, axis=2))

    # Exclude each user's currently connected tower.
    current_ids = users["Connected Tower ID"].to_numpy()

    for i, tower_id in enumerate(tower_ids):
        distances[current_ids == tower_id, i] = np.inf

    # Limit the number of neighbours to the available tower count.
    k = min(max(int(num_neighbours), 0), len(tower_ids))

    if k == 0:
        users["Neighbour Tower IDs"] = [
            []
            for _ in range(len(users))
        ]

        users["Neighbour Tower Distances"] = [
            []
            for _ in range(len(users))
        ]

        return users

    # Sort distances and select the nearest neighbouring towers.
    indices = np.argsort(distances, axis=1)[:, :k]

    # Store neighbouring tower IDs and their distances for each user.
    users["Neighbour Tower IDs"] = [
        [
            tower_ids[i]
            for i in row
            if np.isfinite(distances[r, i])
        ]
        for r, row in enumerate(indices)
    ]

    users["Neighbour Tower Distances"] = [
        [
            float(distances[r, i])
            for i in row
            if np.isfinite(distances[r, i])
        ]
        for r, row in enumerate(indices)
    ]

    return users


def evaluate_neighbouring_towers(users, tower_allocation, seed=None):
    # Copy the user table before evaluating candidate links.
    users = users.copy()

    if users.empty:
        users["Neighbour Tower Results"] = [[] for _ in range(0)]
        return users

    # Calculate radio conditions for the available towers.
    (
        towers,
        distances,
        path_losses,
        received_dbm,
        interference_mw,
        noise_dbm,
        sinr
    ) = _radio_link_matrices(
        users,
        tower_allocation,
        seed=seed
    )

    # Return empty results when no operational tower is available.
    if towers.empty:
        users["Neighbour Tower Results"] = [
            []
            for _ in range(len(users))
        ]

        return users

    # Create a lookup between tower IDs and their matrix positions.
    tower_ids = towers["Tower ID"].tolist()
    lookup = {
        tower_id: i
        for i, tower_id in enumerate(tower_ids)
    }

    # Convert thermal noise from dBm to milliwatts.
    noise_mw = 10 ** (noise_dbm / 10)
    results = []

    # Evaluate each user's neighbouring towers.
    for user_idx, neighbouring_ids in enumerate(
        users.get(
            "Neighbour Tower IDs",
            [[] for _ in range(len(users))]
        )
    ):
        user_results = []

        for tower_id in neighbouring_ids:
            tower_idx = lookup.get(tower_id)

            if tower_idx is None:
                continue

            # Store the candidate tower's radio measurements.
            user_results.append({
                "Tower ID": tower_id,
                "Distance": float(distances[user_idx, tower_idx]),
                "Path loss": float(path_losses[user_idx, tower_idx]),
                "Received Power": float(received_dbm[user_idx, tower_idx]),
                "Interference (mW)": float(interference_mw[user_idx, tower_idx]),
                "Interference (dBm)": (
                    float(
                        10 * np.log10(
                            interference_mw[user_idx, tower_idx]
                        )
                    )
                    if interference_mw[user_idx, tower_idx] > 0
                    else -np.inf
                ),
                "Noise (mW)": float(noise_mw),
                "Noise (dBm)": float(noise_dbm),
                "SINR": float(sinr[user_idx, tower_idx]),
            })

        results.append(user_results)

    users["Neighbour Tower Results"] = results

    return users


def process_handovers(
    users,
    tower_allocation,
    time_step,
    hysteresis=3,
    time_to_trigger=1.0
):
    # Copy users so handover updates do not alter the original input.
    users = users.copy()

    # Index towers by ID for quick candidate lookups.
    towers = tower_allocation.set_index(
        "Tower ID",
        drop=False
    )

    # Create default handover tracking fields when they do not exist.
    defaults = {
        "Handover Candidate": None,
        "Handover Timer": 0.0,
        "Handover Status": "connected",
        "Handover Count": 0,
        "Handover Failure Count": 0,
    }

    for column, default in defaults.items():
        if column not in users.columns:
            users[column] = default

    # Check that the required serving and neighbour information exists.
    required = [
        "Connected Tower ID",
        "Neighbour Tower Results"
    ]

    missing = [
        column
        for column in required
        if column not in users.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns for handover processing: {missing}"
        )

    # Extract SINR values for the current serving connections.
    sinr_values = _get_sinr(users).to_numpy(dtype=float)

    # Evaluate handover conditions for each user.
    for idx in users.index:

        # Skip users that are not currently connected.
        if (
            "Connection Status" in users.columns
            and users.at[idx, "Connection Status"] != "connected"
        ):
            users.at[idx, "Handover Candidate"] = None
            users.at[idx, "Handover Timer"] = 0.0
            users.at[idx, "Handover Status"] = "not_connected"
            continue

        # Retrieve serving SINR and neighbouring tower measurements.
        current_sinr = sinr_values[users.index.get_loc(idx)]
        neighbour_results = users.at[idx, "Neighbour Tower Results"]

        # Keep only neighbouring towers with finite SINR.
        valid = [
            result
            for result in neighbour_results
            if np.isfinite(result.get("SINR", np.nan))
        ]

        if not valid or not np.isfinite(current_sinr):
            users.at[idx, "Handover Candidate"] = None
            users.at[idx, "Handover Timer"] = 0.0
            users.at[idx, "Handover Status"] = "connected"
            continue

        # Select the neighbour with the strongest SINR.
        best = max(
            valid,
            key=lambda result: result["SINR"]
        )

        candidate_id = best["Tower ID"]
        candidate_sinr = float(best["SINR"])

        # Apply the hysteresis margin before attempting handover.
        if candidate_sinr <= current_sinr + hysteresis:
            users.at[idx, "Handover Candidate"] = None
            users.at[idx, "Handover Timer"] = 0.0
            users.at[idx, "Handover Status"] = "connected"
            continue

        # Confirm the candidate tower is operational.
        candidate_available = (
            candidate_id in towers.index
            and towers.at[candidate_id, "Tower Status"] == "operational"
        )

        # Reject unavailable towers or candidates with very weak SINR.
        if not candidate_available or candidate_sinr < -5:
            users.at[idx, "Handover Failure Count"] += 1
            users.at[idx, "Handover Candidate"] = None
            users.at[idx, "Handover Timer"] = 0.0
            users.at[idx, "Handover Status"] = "failed"
            continue

        # Update the time-to-trigger timer for the current candidate.
        previous_candidate = users.at[idx, "Handover Candidate"]

        if previous_candidate == candidate_id:
            timer = float(users.at[idx, "Handover Timer"]) + time_step
        else:
            timer = time_step

        users.at[idx, "Handover Candidate"] = candidate_id
        users.at[idx, "Handover Timer"] = timer
        users.at[idx, "Handover Status"] = "preparing"

        # Wait until the time-to-trigger requirement has been satisfied.
        if timer < time_to_trigger:
            continue

        # Update the user's serving tower after successful handover.
        new_tower = towers.loc[candidate_id]

        _set_connection_to_tower(
            users,
            idx,
            new_tower,
            best["Distance"],
            received_dbm=best["Received Power"],
            path_loss_value=best.get("Path loss"),
            interference_mw=best["Interference (mW)"],
            noise_dbm=best["Noise (dBm)"],
            sinr_db=candidate_sinr
        )

        # Reset the candidate timer and record the completed handover.
        users.at[idx, "Handover Candidate"] = None
        users.at[idx, "Handover Timer"] = 0.0
        users.at[idx, "Handover Status"] = "completed"
        users.at[idx, "Handover Count"] += 1

    return users


# FAILURES

def initialise_connection_status(users, min_sinr=-5):
    # Copy the user table before setting initial connection states.
    users = users.copy()

    # Retrieve SINR values from the available column.
    sinr = _get_sinr(users)

    # Connect users whose SINR meets the minimum threshold.
    users["Connection Status"] = np.where(
        sinr >= min_sinr,
        "connected",
        "failed"
    )

    # Initialise connection and failure counters.
    users["Connection Attempts"] = 1

    users["Failed Connection Count"] = (
        users["Connection Status"] == "failed"
    ).astype(int)

    users["Dropped Session Count"] = 0

    # Initialise session state based on the connection result.
    users["Session Status"] = np.where(
        users["Connection Status"] == "connected",
        "active",
        "not_started"
    )

    # Initialise automatic recovery tracking.
    users["Recovery Timer"] = 0.0
    users["Recovery Attempts"] = 0
    users["Successful Recovery Count"] = 0

    # Initialise handover tracking.
    users["Handover Candidate"] = None
    users["Handover Timer"] = 0.0
    users["Handover Status"] = "connected"
    users["Handover Count"] = 0
    users["Handover Failure Count"] = 0

    return users


def check_connection_drops(users, min_sinr=-8):
    # Copy the user table before checking connection quality.
    users = users.copy()

    if "Connection Status" not in users.columns:
        raise ValueError(
            "Initialise connection status before checking for drops."
        )

    # Retrieve serving-link SINR values.
    sinr = _get_sinr(users)

    # Identify connected users with poor or invalid signal quality.
    connected = users["Connection Status"].eq("connected")
    poor_signal = sinr.lt(min_sinr) | ~np.isfinite(sinr)

    dropped = connected & poor_signal

    # Update connection and session states for affected users.
    users.loc[dropped, "Connection Status"] = "disconnected"
    users.loc[dropped, "Session Status"] = "dropped"

    # Update the drop counter and restart the recovery timer.
    users.loc[dropped, "Dropped Session Count"] += 1
    users.loc[dropped, "Recovery Timer"] = 0.0

    return users


def generate_tower_outages(
    users,
    tower_allocation,
    outage_chance=0.0002,
    recovery_chance=0.05,
    seed=None,
    time_step=1.0
):
    """Generate tower outages/recoveries and drop users served by failed towers.

    Probabilities are interpreted per simulated second and converted to the
    supplied time_step. The low default outage rate avoids excessive failures
    when the simulation is running at many frames per second.
    """

    # Copy both DataFrames before updating outage and connection information.
    users = users.copy()
    tower_allocation = tower_allocation.copy()

    # Validate time and probability settings.
    if time_step <= 0:
        raise ValueError("time_step must be greater than zero.")

    if (
        not 0 <= outage_chance <= 1
        or not 0 <= recovery_chance <= 1
    ):
        raise ValueError(
            "Outage and recovery probabilities must be between 0 and 1."
        )

    # Create a random number generator for outage and recovery events.
    rng = np.random.default_rng(seed)

    # Ensure tower status and event counters exist.
    if "Tower Status" not in tower_allocation.columns:
        tower_allocation["Tower Status"] = "operational"

    if "Tower Outage Count" not in tower_allocation.columns:
        tower_allocation["Tower Outage Count"] = 0

    if "Tower Recovery Count" not in tower_allocation.columns:
        tower_allocation["Tower Recovery Count"] = 0

    # Convert per-second probabilities into probabilities for this time step.
    outage_step_probability = (
        1 - (1 - outage_chance) ** time_step
    )

    recovery_step_probability = (
        1 - (1 - recovery_chance) ** time_step
    )

    # Evaluate the outage or recovery state of every tower.
    for idx, tower in tower_allocation.iterrows():

        if tower["Tower Status"] == "operational":
            # Generate a possible outage for operational towers.
            if rng.random() < outage_step_probability:
                tower_allocation.at[idx, "Tower Status"] = "outage"
                tower_allocation.at[idx, "Tower Outage Count"] += 1

        elif tower["Tower Status"] == "outage":
            # Generate a possible recovery for unavailable towers.
            if rng.random() < recovery_step_probability:
                tower_allocation.at[idx, "Tower Status"] = "operational"
                tower_allocation.at[idx, "Tower Recovery Count"] += 1

    # Identify all towers currently experiencing an outage.
    unavailable = tower_allocation.loc[
        tower_allocation["Tower Status"].eq("outage"),
        "Tower ID"
    ]

    # Disconnect users currently served by unavailable towers.
    if "Connection Status" in users.columns:
        affected = (
            users["Connected Tower ID"].isin(unavailable)
            & users["Connection Status"].eq("connected")
        )

        users.loc[affected, "Connection Status"] = "disconnected"
        users.loc[affected, "Session Status"] = "dropped"

        # Record the dropped sessions.
        users.loc[affected, "Dropped Session Count"] += 1

        # Restart their recovery timers where available.
        if "Recovery Timer" in users.columns:
            users.loc[affected, "Recovery Timer"] = 0.0

    return users, tower_allocation


def process_automatic_recovery(
    users,
    tower_allocation,
    time_step=1.0,
    min_sinr=-5,
    retry_interval=2.0,
    bandwidth=10e6,
    noise_figure=7,
    seed=None
):
    """Retry failed/disconnected users on the best suitable operational tower."""

    # Copy user records before updating recovery status.
    users = users.copy()

    # Ensure all recovery tracking columns exist.
    for column, default in {
        "Recovery Timer": 0.0,
        "Recovery Attempts": 0,
        "Successful Recovery Count": 0,
        "Connection Attempts": 0,
        "Failed Connection Count": 0,
        "Dropped Session Count": 0,
    }.items():

        if column not in users.columns:
            users[column] = default

    # Check that connection status has been initialised.
    if "Connection Status" not in users.columns:
        raise ValueError(
            "Initialise connection status before automatic recovery."
        )

    # Identify users who are eligible for reconnection attempts.
    retry_mask = users["Connection Status"].isin(
        ["failed", "disconnected"]
    )

    # Increase the recovery timer for eligible users.
    users.loc[retry_mask, "Recovery Timer"] += time_step

    # Find users whose retry interval has elapsed.
    due_indices = users.index[
        retry_mask
        & users["Recovery Timer"].ge(retry_interval)
    ].tolist()

    if not due_indices:
        return users

    # Retrieve towers that can accept new connections.
    operational = _operational_towers(
        tower_allocation
    ).reset_index(drop=True)

    # Reset timers and update attempt counters.
    users.loc[due_indices, "Recovery Timer"] = 0.0
    users.loc[due_indices, "Recovery Attempts"] += 1
    users.loc[due_indices, "Connection Attempts"] += 1

    # Record failure when no operational towers are available.
    if operational.empty:
        users.loc[due_indices, "Failed Connection Count"] += 1
        return users

    # Create a subset containing only users due for reconnection.
    retry_users = users.loc[due_indices].copy()

    # Calculate radio conditions against all operational towers.
    (
        towers,
        distances,
        path_losses,
        received_dbm,
        interference_mw,
        noise_dbm,
        sinr
    ) = _radio_link_matrices(
        retry_users,
        operational,
        seed=seed,
        bandwidth=bandwidth,
        noise_figure=noise_figure
    )

    # Record failure if no candidate towers are available.
    if towers.empty:
        users.loc[due_indices, "Failed Connection Count"] += 1
        return users

    # Identify the strongest candidate tower for each retrying user.
    best_indices = np.argmax(sinr, axis=1)

    for position, user_idx in enumerate(due_indices):
        tower_idx = int(best_indices[position])
        candidate_sinr = float(sinr[position, tower_idx])

        # Reject candidates with invalid or insufficient signal quality.
        if (
            not np.isfinite(candidate_sinr)
            or candidate_sinr < min_sinr
        ):
            users.at[user_idx, "Failed Connection Count"] += 1
            continue

        # Retrieve the selected operational tower.
        tower = towers.iloc[tower_idx]

        # Update serving tower and radio measurements.
        _set_connection_to_tower(
            users,
            user_idx,
            tower,
            distances[position, tower_idx],
            received_dbm=received_dbm[position, tower_idx],
            path_loss_value=path_losses[position, tower_idx],
            interference_mw=interference_mw[position, tower_idx],
            noise_dbm=noise_dbm,
            sinr_db=candidate_sinr
        )

        # Record successful reconnection.
        users.at[user_idx, "Connection Status"] = "connected"
        users.at[user_idx, "Session Status"] = "active"
        users.at[user_idx, "Successful Recovery Count"] += 1

        # Reset handover tracking after successful recovery.
        users.at[user_idx, "Handover Candidate"] = None
        users.at[user_idx, "Handover Timer"] = 0.0
        users.at[user_idx, "Handover Status"] = "connected"

    return users


# PERFORMANCE 

def calculate_achievable_data_rate(users, bandwidth_hz=10e6):
    # Copy the user table before calculating achievable data rate.
    users = users.copy()

    # Retrieve SINR and convert it into a linear ratio.
    sinr = _get_sinr(users).to_numpy(dtype=float)
    sinr_linear = 10 ** (sinr / 10)

    # Apply the Shannon capacity formula and convert to Mbps.
    users["Achievable Data Rate (Mbps)"] = (
        bandwidth_hz * np.log2(1 + sinr_linear) / 1e6
    )

    # Replace invalid results and prevent negative data rates.
    users["Achievable Data Rate (Mbps)"] = (
        users["Achievable Data Rate (Mbps)"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .clip(lower=0)
    )

    return users


def calculate_user_throughput(users, efficiency=0.70):
    # Copy the user table before calculating throughput.
    users = users.copy()

    # Apply the assumed efficiency to the achievable data rate.
    users["User Throughput (Mbps)"] = (
        users["Achievable Data Rate (Mbps)"] * efficiency
    )

    return users


def _weighted_capacity_allocation(demand, radio_limit, capacity):
    """Allocate cell capacity proportionally to demand, respecting radio caps."""

    # Convert inputs to NumPy arrays for numerical calculations.
    demand = np.asarray(demand, dtype=float)

    # Apply the lower of demand and individual radio capacity.
    limits = np.minimum(
        demand,
        np.asarray(radio_limit, dtype=float)
    )

    # Initialise allocations and remaining capacity.
    allocation = np.zeros(len(demand), dtype=float)
    remaining = max(float(capacity), 0.0)

    # Select users who still require capacity.
    active = np.flatnonzero(limits > 0)

    # Distribute remaining capacity until no capacity or demand remains.
    while remaining > 1e-10 and len(active) > 0:
        weights = demand[active]

        if weights.sum() <= 0:
            break

        # Calculate the proportional capacity proposal.
        proposed = remaining * weights / weights.sum()

        # Restrict allocations to the remaining individual user limits.
        room = limits[active] - allocation[active]
        granted = np.minimum(proposed, room)

        # Update allocations and remaining capacity.
        allocation[active] += granted
        used = float(granted.sum())
        remaining -= used

        # Keep only users who still need additional capacity.
        still_needing = (
            limits[active] - allocation[active] > 1e-10
        )

        active = active[still_needing]

        if used <= 1e-12:
            break

    return allocation


def allocate_cell_resources(
    users,
    tower_allocations,
    bandwidth_hz=10e6,
    efficiency=0.70,
    max_spectral_efficiency=6.0
):
    """Estimate each cell's capacity and share it between connected users.

    Cell capacity is based on the mean connected-user SINR. Each user's
    allocation is limited by both its activity demand and its own radio rate.
    """

    # Copy both DataFrames before adding resource allocation results.
    users = users.copy()
    tower_allocations = tower_allocations.copy()

    # Validate bandwidth and efficiency settings.
    if bandwidth_hz <= 0 or not 0 < efficiency <= 1:
        raise ValueError(
            "Bandwidth must be positive and efficiency must be in (0, 1]."
        )

    # Retrieve each user's activity and map it to a traffic demand.
    activity = users.get(
        "activity",
        pd.Series("browsing", index=users.index)
    ).astype(str)

    demand = (
        activity.map(_ACTIVITY_DEMAND)
        .fillna(_ACTIVITY_DEMAND["browsing"])
        .to_numpy(dtype=float)
    )

    users["Traffic Demand (Mbps)"] = demand

    # Calculate the maximum radio data rate for each user.
    if "SINR" in users.columns or "SINR (dB)" in users.columns:
        users = calculate_achievable_data_rate(
            users,
            bandwidth_hz=bandwidth_hz
        )

        radio_limit = (
            users["Achievable Data Rate (Mbps)"].to_numpy(dtype=float)
            * efficiency
        )

    else:
        users["Achievable Data Rate (Mbps)"] = 0.0
        radio_limit = np.zeros(len(users), dtype=float)

    # Initialise user throughput and capacity fields.
    users["Radio Capacity (Mbps)"] = radio_limit
    users["Allocated Throughput (Mbps)"] = 0.0
    users["User Throughput (Mbps)"] = 0.0

    # Prepare tower output columns, retaining all existing tower information.
    tower_defaults = {
        "Available Capacity (Mbps)": 0.0,
        "Cell Demand (Mbps)": 0.0,
        "Total Cell Throughput (Mbps)": 0.0,
        "Average User Throughput (Mbps)": 0.0,
        "Capacity Utilisation (%)": 0.0,
        "Demand Load (%)": 0.0,
        "Remaining Capacity (Mbps)": 0.0,
        "Spectral Efficiency (bits/s/Hz)": 0.0,
    }

    for column, default in tower_defaults.items():
        tower_allocations[column] = default

    # Identify connected users.
    if "Connection Status" in users.columns:
        connected_mask = users["Connection Status"].eq("connected")
    else:
        connected_mask = pd.Series(True, index=users.index)

    # Allocate resources independently for each operational tower.
    for tower_idx, tower in tower_allocations.iterrows():

        # Skip towers that are currently experiencing an outage.
        if tower.get("Tower Status", "operational") != "operational":
            continue

        tower_id = tower["Tower ID"]

        # Identify connected users assigned to this tower.
        assigned = users.index[
            connected_mask
            & users["Connected Tower ID"].eq(tower_id)
        ]

        if len(assigned) == 0:
            continue

        # Retrieve SINR values for the assigned users.
        sinr_db = _get_sinr(
            users.loc[assigned]
        ).to_numpy(dtype=float)

        # Ignore non-finite SINR values when calculating cell capacity.
        finite_sinr = sinr_db[np.isfinite(sinr_db)]

        if len(finite_sinr):
            average_sinr_linear = np.mean(10 ** (finite_sinr / 10))
        else:
            average_sinr_linear = 0.0

        # Calculate the cell's spectral efficiency.
        spectral_efficiency = (
            min(
                np.log2(1 + average_sinr_linear),
                max_spectral_efficiency
            )
            * efficiency
        )

        # Convert available cell capacity into Mbps.
        available_capacity = (
            bandwidth_hz * spectral_efficiency / 1e6
        )

        # Calculate total requested traffic from connected users.
        cell_demand = users.loc[
            assigned,
            "Traffic Demand (Mbps)"
        ].sum()

        # Share the cell capacity across assigned users.
        allocation = _weighted_capacity_allocation(
            users.loc[
                assigned,
                "Traffic Demand (Mbps)"
            ].to_numpy(dtype=float),
            users.loc[
                assigned,
                "Radio Capacity (Mbps)"
            ].to_numpy(dtype=float),
            available_capacity
        )

        # Update individual user throughput.
        users.loc[
            assigned,
            "Allocated Throughput (Mbps)"
        ] = allocation

        users.loc[
            assigned,
            "User Throughput (Mbps)"
        ] = allocation

        # Calculate total throughput and capacity utilisation.
        total_throughput = float(allocation.sum())

        if available_capacity > 0:
            utilisation = (
                100 * total_throughput / available_capacity
            )
        else:
            utilisation = 0.0

        if available_capacity > 0:
            demand_load = (
                100 * cell_demand / available_capacity
            )
        elif cell_demand > 0:
            demand_load = 100.0
        else:
            demand_load = 0.0

        # Save calculated capacity metrics to the tower table.
        tower_allocations.at[
            tower_idx,
            "Available Capacity (Mbps)"
        ] = available_capacity

        tower_allocations.at[
            tower_idx,
            "Cell Demand (Mbps)"
        ] = float(cell_demand)

        tower_allocations.at[
            tower_idx,
            "Total Cell Throughput (Mbps)"
        ] = total_throughput

        tower_allocations.at[
            tower_idx,
            "Average User Throughput (Mbps)"
        ] = (
            float(np.mean(allocation))
            if len(allocation)
            else 0.0
        )

        tower_allocations.at[
            tower_idx,
            "Capacity Utilisation (%)"
        ] = min(utilisation, 100.0)

        tower_allocations.at[
            tower_idx,
            "Demand Load (%)"
        ] = demand_load

        tower_allocations.at[
            tower_idx,
            "Remaining Capacity (Mbps)"
        ] = max(
            available_capacity - total_throughput,
            0.0
        )

        tower_allocations.at[
            tower_idx,
            "Spectral Efficiency (bits/s/Hz)"
        ] = spectral_efficiency

    return users, tower_allocations


def calculate_latency(users, base_latency=20):
    # Copy users before calculating latency.
    users = users.copy()

    # Retrieve SINR values for each user.
    sinr = _get_sinr(users)
    sinr_values = sinr.to_numpy(dtype=float)

    # Calculate additional latency caused by poor signal quality.
    quality_penalty = np.where(
        np.isfinite(sinr_values),
        np.maximum(0, 10 - sinr_values) * 2,
        np.nan
    )

    users["Latency (ms)"] = base_latency + quality_penalty

    return users


def calculate_packet_loss(users, threshold=5, scale=2):
    # Copy users before calculating packet loss.
    users = users.copy()

    # Retrieve SINR values.
    sinr = _get_sinr(users)

    # Estimate packet loss using a logistic relationship with SINR.
    users["Packet Loss (%)"] = (
        100
        / (
            1
            + np.exp(
                np.clip(
                    (sinr - threshold) / scale,
                    -50,
                    50
                )
            )
        )
    )

    return users


def calculate_congestion_and_network_performance(
    users,
    tower_allocations,
    base_latency=20,
    queue_delay_factor=15
):
    """Add congestion, queue delay, latency and packet-loss effects."""

    # Copy input DataFrames before updating performance metrics.
    users = users.copy()
    tower_allocations = tower_allocations.copy()

    # Allocate cell resources when throughput fields do not yet exist.
    if (
        "Traffic Demand (Mbps)" not in users.columns
        or "User Throughput (Mbps)" not in users.columns
    ):
        users, tower_allocations = allocate_cell_resources(
            users,
            tower_allocations
        )

    # Calculate radio-quality effects on latency and packet loss.
    users = calculate_latency(
        users,
        base_latency=base_latency
    )

    users = calculate_packet_loss(users)

    radio_packet_loss = users[
        "Packet Loss (%)"
    ].to_numpy(dtype=float)

    # Retrieve cell demand load for each tower.
    demand_load = (
        tower_allocations.set_index("Tower ID")["Demand Load (%)"]
        .to_dict()
    )

    # Prepare dictionaries for congestion results.
    congestion_levels = {}
    queue_delays = {}
    congestion_losses = {}

    # Calculate congestion effects for every tower.
    for idx, tower in tower_allocations.iterrows():
        tower_id = tower["Tower ID"]
        load = float(demand_load.get(tower_id, 0.0)) / 100

        # Outage towers cannot provide normal service.
        if tower.get("Tower Status", "operational") != "operational":
            level = "outage"
            queue_delay = 0.0
            congestion_loss = 0.0

        # Classify low congestion.
        elif load < 0.70:
            level = "low"
            queue_delay = 0.0
            congestion_loss = 0.0

        # Classify moderate congestion.
        elif load < 0.90:
            level = "moderate"

            queue_delay = (
                queue_delay_factor
                * (load - 0.70)
                / max(1 - load, 0.05)
            )

            congestion_loss = 0.0

        # Classify high congestion.
        elif load <= 1.0:
            level = "high"

            queue_delay = min(
                250.0,
                queue_delay_factor
                * (load - 0.70)
                / max(1 - load, 0.01)
            )

            congestion_loss = max(
                0.0,
                (load - 0.85) * 100
            )

        # Classify overloaded cells.
        else:
            level = "overloaded"

            queue_delay = min(
                500.0,
                250.0 + (load - 1.0) * 100.0
            )

            congestion_loss = min(
                50.0,
                15.0 + (load - 1.0) * 50.0
            )

        # Store congestion calculations for later user mapping.
        congestion_levels[tower_id] = level
        queue_delays[tower_id] = float(queue_delay)
        congestion_losses[tower_id] = float(congestion_loss)

        # Update tower-level congestion metrics.
        tower_allocations.at[idx, "Congestion Level"] = level
        tower_allocations.at[idx, "Queue Delay (ms)"] = float(queue_delay)

    # Map tower congestion levels to individual users.
    users["Congestion Level"] = (
        users["Connected Tower ID"]
        .map(congestion_levels)
        .fillna("unavailable")
    )

    users["Queue Delay (ms)"] = (
        users["Connected Tower ID"]
        .map(queue_delays)
        .fillna(0.0)
    )

    # Add queueing delay to the existing radio latency.
    users["Latency (ms)"] = (
        users["Latency (ms)"]
        + users["Queue Delay (ms)"]
    )

    # Map additional packet loss caused by congestion.
    congestion_loss = (
        users["Connected Tower ID"]
        .map(congestion_losses)
        .fillna(0.0)
        .to_numpy(dtype=float)
    )

    # Combine independent radio and congestion loss probabilities.
    users["Packet Loss (%)"] = (
        100
        * (
            1
            - (1 - radio_packet_loss / 100)
            * (1 - congestion_loss / 100)
        )
    )

    users["Packet Loss (%)"] = (
        users["Packet Loss (%)"].clip(0, 100)
    )

    return users, tower_allocations


def calculate_cell_throughput(users):
    # Copy the user table before calculating cell throughput.
    users = users.copy()

    # Use connected users when connection status is available.
    active = users

    if "Connection Status" in users.columns:
        active = users.loc[
            users["Connection Status"].eq("connected")
        ]

    # Aggregate user throughput by serving tower.
    cell_metrics = (
        active.groupby("Connected Tower ID")
        .agg(
            Connected_Users=("user_id", "count"),
            Cell_Throughput_Mbps=("User Throughput (Mbps)", "sum"),
            Average_User_Throughput_Mbps=(
                "User Throughput (Mbps)",
                "mean"
            )
        )
        .reset_index()
    )

    return users, cell_metrics


def calculate_cell_performance(users, tower_allocation):
    # Restrict cell-level metrics to connected users.
    active = users

    if "Connection Status" in users.columns:
        active = users.loc[
            users["Connection Status"].eq("connected")
        ]

    # Create an empty metrics table when no users are connected.
    if active.empty:
        cell_metrics = pd.DataFrame(
            columns=[
                "Connected Tower ID",
                "Connected_Users",
                "Total_Cell_Throughput_Mbps",
                "Average_User_Throughput_Mbps",
                "Average_SINR_dB",
                "Average_Latency_ms",
                "Average_Packet_Loss_Percent"
            ]
        )

    else:
        # Define the metrics to aggregate for each serving tower.
        metrics = {
            "Connected_Users": ("user_id", "count"),
            "Total_Cell_Throughput_Mbps": (
                "User Throughput (Mbps)",
                "sum"
            ),
            "Average_User_Throughput_Mbps": (
                "User Throughput (Mbps)",
                "mean"
            ),
            "Average_SINR_dB": ("SINR", "mean"),
            "Average_Latency_ms": ("Latency (ms)", "mean"),
            "Average_Packet_Loss_Percent": ("Packet Loss (%)", "mean"),
        }

        # Accept the old SINR column without changing the rest of the DataFrame.
        active = active.copy()

        if "SINR" not in active.columns and "SINR (dB)" in active.columns:
            active["SINR"] = active["SINR (dB)"]

        # Aggregate performance metrics by connected tower.
        cell_metrics = (
            active.groupby("Connected Tower ID")
            .agg(**metrics)
            .reset_index()
        )

    # Select available tower columns for the cell performance table.
    columns = [
        "Tower ID",
        "Tower region",
        "Tower region type",
        "Tower Status"
    ]

    columns = [
        column
        for column in columns
        if column in tower_allocation.columns
    ]

    # Merge tower information with connected-user metrics.
    cell_performance = tower_allocation[columns].merge(
        cell_metrics,
        left_on="Tower ID",
        right_on="Connected Tower ID",
        how="left"
    )

    # Replace missing user counts with zero.
    cell_performance["Connected_Users"] = (
        cell_performance["Connected_Users"]
        .fillna(0)
        .astype(int)
    )

    # Replace missing KPI values with zero.
    for column in [
        "Total_Cell_Throughput_Mbps",
        "Average_User_Throughput_Mbps",
        "Average_SINR_dB",
        "Average_Latency_ms",
        "Average_Packet_Loss_Percent"
    ]:
        if column in cell_performance.columns:
            cell_performance[column] = (
                cell_performance[column].fillna(0.0)
            )

    # Attach capacity and congestion information to the main cell KPI table.
    extra = [
        "Available Capacity (Mbps)",
        "Cell Demand (Mbps)",
        "Capacity Utilisation (%)",
        "Demand Load (%)",
        "Remaining Capacity (Mbps)",
        "Congestion Level",
        "Queue Delay (ms)",
    ]

    extra = [
        column
        for column in extra
        if column in tower_allocation.columns
    ]

    if extra:
        cell_performance = cell_performance.merge(
            tower_allocation[["Tower ID"] + extra],
            on="Tower ID",
            how="left"
        )

    return cell_performance


# SUMMARY 

def calculate_network_performance(users):
    # Define the network performance metrics to retrieve.
    metrics = [
        "User Throughput (Mbps)",
        "Achievable Data Rate (Mbps)",
        "Latency (ms)",
        "Packet Loss (%)",
        "SINR",
    ]

    # Copy user data so the original DataFrame remains unchanged.
    working = users.copy()

    # Support both SINR column conventions.
    if "SINR" not in working.columns and "SINR (dB)" in working.columns:
        working["SINR"] = working["SINR (dB)"]

    # Check that all required performance fields are available.
    missing = [
        column
        for column in metrics
        if column not in working.columns
    ]

    if missing:
        raise ValueError(f"Missing performance columns: {missing}")

    # Restrict tower performance calculations to connected users.
    if "Connection Status" in working.columns:
        active = working.loc[
            working["Connection Status"].eq("connected")
        ].copy()
    else:
        active = working.copy()

    # Calculate average performance for each serving tower.
    tower_performance = (
        active.groupby("Connected Tower ID")[metrics]
        .mean()
        .reset_index()
    )

    # Count connected users per tower.
    tower_users = (
        active.groupby("Connected Tower ID")
        .size()
        .reset_index(name="Connected Users")
    )

    # Add connected-user counts to the tower performance data.
    tower_performance = tower_performance.merge(
        tower_users,
        on="Connected Tower ID",
        how="left"
    )

    # Use region ID when available, otherwise use the serving tower region.
    region_column = (
        "region_id"
        if "region_id" in active.columns
        else "Tower region"
    )

    # Calculate average network performance per region.
    if region_column in active.columns:
        region_performance = (
            active.groupby(region_column)[metrics]
            .mean()
            .reset_index()
        )
    else:
        region_performance = pd.DataFrame()

    # Select user-level performance columns for the report.
    user_columns = [
        column
        for column in [
            "user_id",
            "Connected Tower ID",
            "region_id",
            "Connection Status"
        ]
        if column in working.columns
    ]

    user_performance = working[
        user_columns + metrics
    ].copy()

    return (
        user_performance,
        tower_performance,
        region_performance
    )


def calculate_network_failures(users, tower_allocation):
    # Helper to total a counter when the column exists.
    def user_sum(column):
        if column in users.columns:
            return int(users[column].sum())

        return 0

    # Retrieve user connection statuses.
    status = users.get(
        "Connection Status",
        pd.Series("connected", index=users.index)
    )

    # Retrieve tower operating statuses.
    tower_status = tower_allocation.get(
        "Tower Status",
        pd.Series("operational", index=tower_allocation.index)
    )

    # Compile network failure and recovery counters.
    failure_summary = {
        "Total Users": int(len(users)),
        "Connected Users": int(status.eq("connected").sum()),
        "Disconnected Users": int(status.eq("disconnected").sum()),
        "Failed Users": int(status.eq("failed").sum()),
        "Failed Connection Attempts": user_sum("Failed Connection Count"),
        "Connection Attempts": user_sum("Connection Attempts"),
        "Dropped Sessions": user_sum("Dropped Session Count"),
        "Handover Count": user_sum("Handover Count"),
        "Handover Failures": user_sum("Handover Failure Count"),
        "Recovery Attempts": user_sum("Recovery Attempts"),
        "Successful Recoveries": user_sum("Successful Recovery Count"),
        "Operational Towers": int(tower_status.eq("operational").sum()),
        "Towers Currently Down": int(tower_status.eq("outage").sum()),
        "Total Tower Outages": (
            int(tower_allocation["Tower Outage Count"].sum())
            if "Tower Outage Count" in tower_allocation.columns
            else 0
        ),
        "Total Tower Recoveries": (
            int(tower_allocation["Tower Recovery Count"].sum())
            if "Tower Recovery Count" in tower_allocation.columns
            else 0
        ),
    }

    return failure_summary


def generate_network_performance_report(users, tower_allocations):
    """Return network summary, per-tower, per-region and per-user reports."""

    # Generate detailed performance reports.
    (
        user_performance,
        tower_performance,
        region_performance
    ) = calculate_network_performance(users)

    # Calculate connection, handover and outage statistics.
    failures = calculate_network_failures(
        users,
        tower_allocations
    )

    # Restrict average network KPIs to connected users.
    active = users

    if "Connection Status" in users.columns:
        active = users.loc[
            users["Connection Status"].eq("connected")
        ]

    # Helper to calculate an average when the column is available.
    def safe_mean(column):
        if column in active.columns and len(active):
            return float(active[column].mean())

        return 0.0

    # Compile overall network performance metrics.
    network_summary = {
        **failures,
        "Average User Throughput (Mbps)": safe_mean(
            "User Throughput (Mbps)"
        ),
        "Total Network Throughput (Mbps)": (
            float(active["User Throughput (Mbps)"].sum())
            if "User Throughput (Mbps)" in active.columns
            else 0.0
        ),
        "Average SINR (dB)": (
            safe_mean("SINR")
            if "SINR" in active.columns
            else safe_mean("SINR (dB)")
        ),
        "Average Latency (ms)": safe_mean("Latency (ms)"),
        "Average Packet Loss (%)": safe_mean("Packet Loss (%)"),
        "Average Capacity Utilisation (%)": (
            float(tower_allocations["Capacity Utilisation (%)"].mean())
            if (
                "Capacity Utilisation (%)" in tower_allocations.columns
                and len(tower_allocations)
            )
            else 0.0
        ),
        "Congested Towers": (
            int(
                tower_allocations["Congestion Level"]
                .isin(["high", "overloaded"])
                .sum()
            )
            if "Congestion Level" in tower_allocations.columns
            else 0
        ),
    }

    # Return the combined network and detailed performance reports.
    return {
        "network_summary": network_summary,
        "tower_report": (
            tower_allocations.merge(
                tower_performance,
                left_on="Tower ID",
                right_on="Connected Tower ID",
                how="left"
            )
            if "Tower ID" in tower_allocations.columns
            else tower_performance
        ),
        "region_report": region_performance,
        "user_report": user_performance,
    }
