import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Function to build the infrastructure
def generate_town(town_width, town_height, num_areas=6, grid_resolution=25, seed=None):
    
    if town_width <= 0 or town_height <= 0:
        raise ValueError("Town dimensions must be positive.")

    if grid_resolution <= 0:
        raise ValueError("Grid resolution must be positive.")

    if town_width % grid_resolution != 0:
        raise ValueError("Town width must be divisible by grid_resolution.")

    if town_height % grid_resolution != 0:
        raise ValueError("Town height must be divisible by grid_resolution.")

    if num_areas < 1:
        raise ValueError("num_areas must be at least 1.")

    rng = np.random.default_rng(seed)

    # Calculate grid dimensions
    cols = town_width // grid_resolution
    rows = town_height // grid_resolution

    # Generate grid-cell centre coordinates
    x = (np.arange(cols) + 0.5) * grid_resolution
    y = (np.arange(rows) + 0.5) * grid_resolution

    xx, yy = np.meshgrid(x, y)

    # Generate random region centres
    margin = min(town_width, town_height) * 0.03

    centers_x = rng.uniform(margin, town_width - margin, num_areas)

    centers_y = rng.uniform(margin, town_height - margin, num_areas)

    centers = np.column_stack((centers_x, centers_y))

    # Assign each grid cell to its nearest region centre
    grid_assignments = np.empty((rows, cols), dtype=int)

    for row in range(rows):
        distances = (xx[row, :, None] - centers[:, 0]) ** 2 + (
            yy[row, :, None] - centers[:, 1]
        ) ** 2

        grid_assignments[row] = np.argmin(distances, axis=1)

    return grid_assignments, centers, xx, yy

# Function to create the population in the town
def generate_population(
    assignments, centers, town_width, town_height, grid_resolution=25, seed=None
):
    """
    Generate population density and population estimates for each region.
    """

    rng = np.random.default_rng(seed)

    density_ranges = {
        "urban": (3000, 8000),
        "residential": (1000, 4000),
        "suburban": (500, 2500),
        "rural": (100, 1000),
    }

    population_grid = np.zeros(assignments.shape, dtype=np.float32)

    region_data = []

    # Calculate area of each grid cell in km²
    cell_area_km2 = grid_resolution**2 / 1_000_000

    # Define the geographical town centre
    town_centre = np.array([town_width / 2, town_height / 2])

    max_distance = np.linalg.norm(town_centre)

    for region_id in np.unique(assignments):
        region_mask = assignments == region_id

        cell_count = np.count_nonzero(region_mask)

        # Calculate region area
        area_km2 = cell_count * cell_area_km2

        # Calculate distance from region centre to town centre
        region_centre = centers[int(region_id)]

        distance = np.linalg.norm(region_centre - town_centre)

        relative_distance = distance / max_distance

        # Assign region type based on geographical position
        if relative_distance <= 0.25:
            category = "urban"

        elif relative_distance <= 0.50:
            category = "residential"

        elif relative_distance <= 0.75:
            category = "suburban"

        else:
            category = "rural"

        # Generate population density
        min_density, max_density = density_ranges[category]

        density = rng.integers(min_density, max_density + 1)

        # Calculate estimated population
        population = round(area_km2 * density)

        population_grid[region_mask] = density

        region_data.append(
            {
                "region_id": int(region_id),
                "region_type": category,
                "area_km2": round(area_km2, 3),
                "population_density": int(density),
                "estimated_population": int(population),
            }
        )

    region_population = pd.DataFrame(region_data)

    total_population = int(region_population["estimated_population"].sum())

    return (population_grid, region_population, total_population)

# Function to position towers
def allocate_towers(
    assignments,
    region_population,
    coverage_range, 
    total_towers=25,
    grid_resolution=25,
    seed=None,
):
    """
    Allocate towers proportionally to regional population.

    Tower locations are selected to maximise coverage of currently
    uncovered areas by placing subsequent towers as far as possible 
    from existing ones, without enforcing a strict minimum distance constraint.

    Uses the largest-remainder method for exact tower allocation.
    Includes region_type in the tower_allocations DataFrame.
    """

    if total_towers < 1:
        raise ValueError("total_towers must be at least 1.")

    if coverage_range < 0:
        raise ValueError("coverage_range must be non-negative.")

    rng = np.random.default_rng(seed)

    # Retrieve regional population and region types efficiently
    region_ids = np.unique(assignments)
    pop_indexed = region_population.set_index("region_id").reindex(region_ids)
    
    populations = pop_indexed["estimated_population"].values.astype(float)
    region_types = pop_indexed["region_type"].values

    total_population = populations.sum()

    if total_population <= 0:
        raise ValueError("Total population must be greater than zero.")

    # Calculate proportional tower allocations
    population_shares = populations / total_population
    exact_allocations = population_shares * total_towers

    tower_allocations = np.floor(exact_allocations).astype(int)

    remaining_towers = total_towers - tower_allocations.sum()
    remainders = exact_allocations - tower_allocations

    if remaining_towers > 0:
        largest_remainders = np.argsort(-remainders)
        tower_allocations[largest_remainders[:remaining_towers]] += 1

    all_tower_coords = []
    allocation_data = []

    # Tower placement loop
    for region_id, population, region_type, share, num_towers in zip(
        region_ids,
        populations,
        region_types,
        population_shares,
        tower_allocations
    ):

        y_indices, x_indices = np.where(assignments == region_id)
        area_coords = np.column_stack((x_indices, y_indices))
        candidate_coords = (area_coords + 0.5) * grid_resolution

        # Handle regions allocated zero towers
        if num_towers == 0:
            allocation_data.append({
                "region_id": int(region_id),
                "region_type": region_type,
                "population": int(population),
                "population_share": round(share * 100, 2),
                "allocated_towers": 0,
                "placed_towers": 0,
                "tower_locations": [],
                "Tower Status": "operational"
            })
            continue

        if len(candidate_coords) < num_towers:
            raise ValueError(
                f"Region {region_id} does not have enough grid cells."
            )

        # Shuffle to randomise ties
        rng.shuffle(candidate_coords)

        region_tower_coords = []
        chosen_indices = set()

        for _ in range(num_towers):
            placed_towers = all_tower_coords + region_tower_coords

            # Exclude locations already selected in this region
            valid_mask = np.ones(len(candidate_coords), dtype=bool)
            if chosen_indices:
                valid_mask[list(chosen_indices)] = False

            # Calculate distances to existing towers to maximize regional coverage
            if placed_towers:
                placed_arr = np.asarray(placed_towers, dtype=float)
                distances = np.linalg.norm(
                    candidate_coords[:, None, :] - placed_arr[None, :, :],
                    axis=2,
                )
                min_dists = distances.min(axis=1)
            else:
                min_dists = None

            available_indices = np.where(valid_mask)[0]

            if len(available_indices) == 0:
                raise ValueError(
                    f"Region {region_id} does not have enough available grid cells "
                    f"to place {num_towers} towers."
                )

            # ---------------------------------------------
            # COVERAGE-MAXIMIZING PLACEMENT (Furthest Point)
            # ---------------------------------------------
            if placed_towers:
                best_local_idx = np.argmax(min_dists[available_indices])
                best_idx = available_indices[best_local_idx]
            else:
                best_idx = available_indices[rng.integers(len(available_indices))]

            chosen_indices.add(best_idx)
            region_tower_coords.append(candidate_coords[best_idx].tolist())

        region_tower_coords = np.asarray(
            region_tower_coords,
            dtype=float
        )

        all_tower_coords.extend(region_tower_coords.tolist())

        # Save regional allocation information
        allocation_data.append({
            "region_id": int(region_id),
            "region_type": region_type,
            "population": int(population),
            "population_share": round(share * 100, 2),
            "allocated_towers": int(num_towers),
            "placed_towers": len(region_tower_coords),
            "tower_locations": region_tower_coords.tolist(),
            "Tower Status": "operational"
        })

    return (
        np.asarray(all_tower_coords, dtype=float).reshape(-1, 2),
        pd.DataFrame(allocation_data),
    )

# Function Generate users from the population grid
def generate_users(assignments, population_grid, region_population, num_users, grid_resolution, user_profiles, seed=None,):

    # Unpack User profiles which is a dict
    user_type = user_profiles["user_type"]
    device_type = user_profiles["device_type"]
    network_capability = user_profiles["network_capability"]
    activities_prob = user_profiles["activities"]

    # Check the number of users cannot be negative
    if num_users < 1:
        raise ValueError("num_users must be at least 1")

    # Check the grid resolution is not negative
    if grid_resolution < 1:
        raise ValueError("Grid resolution must be positive")

    # Ensure the assignments and population grid is the same
    if assignments.shape != population_grid.shape:
        raise ValueError("Assignments and population grid must match")

    rng = np.random.default_rng(seed)

    # Using the weighted summation, calculate cell selection probability
    weights = population_grid.flatten().astype(float)
    total_weights = weights.sum()
    probabilities = weights / total_weights

    # Given the number of users, weights and probs, random select cells
    selected_cells = rng.choice(
        len(weights), size=num_users, replace=True, p=probabilities
    )

    row_indices, col_indices = np.unravel_index(selected_cells, population_grid.shape)

    # Now within this cell generate a random position
    x_coords = col_indices * grid_resolution + rng.uniform(
        0, grid_resolution, num_users
    )
    y_coords = row_indices * grid_resolution + rng.uniform(
        0, grid_resolution, num_users
    )
    user_regions = assignments[row_indices, col_indices]

    # Divide the population based on the user profiles
    user_types = rng.choice(
        ["resident", "commuter", "visitor"], size=num_users, p=user_type
    )

    device_types = rng.choice(
        ["smartphone", "tablet", "laptops", "IoT"], size=num_users, p=device_type
    )

    network_capabilities = rng.choice(
        ["4G_only", "4G_5G"], size=num_users, p=network_capability
    )
    activities = rng.choice(
        ["idle", "browsing", "streaming", "voice call", "video call", "uploading"], size=num_users, p=activities_prob
    )

    # Get the region type for every user 
    region_types = region_population.set_index('region_id')['region_type']
    user_region_types = [region_types.loc[int(region_id)] for region_id in user_regions]

    # Build the output dataframe 
    users = pd.DataFrame({
    "user_id": [
        f"user_{i:06d}"
        for i in range(1, num_users + 1)
    ],
    "region_id": user_regions,
    "region_type": user_region_types,
    "x": x_coords,
    "y": y_coords,
    "user_type": user_types,
    "device_type": device_types,
    "network_capability": network_capabilities,
    "activity": activities
    })

    return users

# A Function to add movement to the users
def assign_movement(users, town_width, town_height, seed=None):
    
    rng = np.random.default_rng(seed)

     # Movement mode probabilities by user type
    movement_profiles = {
        "resident": {
            "modes": ["stationary", "walking", "driving"],
            "probabilities": [0.50, 0.35, 0.15]
        },
        "commuter": {
            "modes": ["stationary", "walking", "driving"],
            "probabilities": [0.10, 0.20, 0.70]
        },
        "visitor": {
            "modes": ["stationary", "walking", "driving"],
            "probabilities": [0.20, 0.60, 0.20]
        }
    }

    # Assign movement modes to each of the users 
    movement_modes = []

    for user_type in users['user_type']:
        profile = movement_profiles[user_type]

        mode = rng.choice(profile['modes'], p = profile['probabilities'])
        movement_modes.append(mode)

    users['movement_mode'] = movement_modes

    # Assign speeds to each type in metres per second 
    speed_mapping = {
        'stationary': 0.0,
        'walking': 1.4, 
        'driving': 8.3
    }
    users['speed'] = users['movement_mode'].map(speed_mapping)

    # Maximum distances that each type can travel 
    distance_mapping = {
        'resident': 500, 
        'commuter': 3000, 
        'visitor': 2000
    }

    destination_distances = []

    for user_type in users['user_type']:
        max_distance = distance_mapping[user_type]

        distance = rng.uniform(100, max_distance)

        destination_distances.append(distance)

    # Generate the random directions 
    angles = rng.uniform(0, 2 * np.pi, len(users))

    destination_distances = np.array(destination_distances)

    # Calculate destination coordinates
    destination_x = (
        users["x"].to_numpy()
        + destination_distances * np.cos(angles)
    )

    destination_y = (
        users["y"].to_numpy()
        + destination_distances * np.sin(angles)
    )

    # Keep destinations inside the town
    users["destination_x"] = np.clip(
        destination_x, 0, town_width
    )

    users["destination_y"] = np.clip(
        destination_y, 0, town_height
    )

    # Stationary users have no destination movement
    stationary = users["movement_mode"] == "stationary"

    users.loc[stationary, "destination_x"] = users.loc[stationary, "x"]
    users.loc[stationary, "destination_y"] = users.loc[stationary, "y"]

    users["movement_status"] = np.where(
        stationary, "stationary", "moving"
    )

    return users

# A function to move the users
def update_user_positions(users, time_step):
    users = users.copy()

    # Calculate distance between current position and destination
    dx = users["destination_x"] - users["x"]
    dy = users["destination_y"] - users["y"]

    distance_to_destination = np.sqrt(dx**2 + dy**2)

    # Calculate maximum distance each user can travel
    travel_distance = users["speed"] * time_step

    # Identify users that can reach their destination
    reached_destination = (
        distance_to_destination <= travel_distance
    )

    # Identify users that are still moving
    still_moving = (
        (distance_to_destination > travel_distance)
        & (users["speed"] > 0)
    )

    # Move users that can reach their destination
    users.loc[reached_destination, "x"] = (
        users.loc[reached_destination, "destination_x"]
    )

    users.loc[reached_destination, "y"] = (
        users.loc[reached_destination, "destination_y"]
    )

    # Move users that have not yet reached their destination
    users.loc[still_moving, "x"] += (
        dx[still_moving]
        / distance_to_destination[still_moving]
        * travel_distance[still_moving]
    )

    users.loc[still_moving, "y"] += (
        dy[still_moving]
        / distance_to_destination[still_moving]
        * travel_distance[still_moving]
    )

    # Update movement status
    users.loc[reached_destination, "movement_status"] = "stationary"

    return users

# A Function to assign new locations
def assign_new_destinations(users, town_width, town_height, seed=None):

    """Assign fresh destinations to mobile users who have stopped moving.

    Purpose:
        Users who reach a destination pause briefly, then receive a new
        destination so their movement can continue during the simulation.

    Parameters:
        users (pd.DataFrame): User table containing movement and position data.
        town_width (int): Width of the town in metres.
        town_height (int): Height of the town in metres.
        seed (int | None): Optional seed for reproducible destination choices.

    Returns:
        pd.DataFrame: A copied user table with updated destinations and status.
    """
    # Copy the table so callers retain their previous state.
    users = users.copy()
    if users.empty:
        return users

    # Only users with a mobile movement mode need a new destination.
    eligible = users["movement_status"].eq("stationary")
    eligible &= users["movement_mode"].ne("stationary")
    eligible &= users["speed"].gt(0)
    if not eligible.any():
        return users

    # Map each eligible user's profile to its maximum travel distance.
    distance_mapping = {"resident": 500, "commuter": 3000, "visitor": 2000}
    max_distances = users.loc[eligible, "user_type"].map(distance_mapping)
    if max_distances.isna().any():
        raise ValueError("Unknown user type found when assigning destinations.")

    # Generate a random travel distance and direction for each eligible user.
    rng = np.random.default_rng(seed)
    distances = rng.uniform(100.0, max_distances.to_numpy(dtype=float))
    angles = rng.uniform(0.0, 2 * np.pi, len(max_distances))

    # Convert each distance and direction into destination coordinates.
    destination_x = users.loc[eligible, "x"].to_numpy() + distances * np.cos(angles)
    destination_y = users.loc[eligible, "y"].to_numpy() + distances * np.sin(angles)

    # Keep the new destinations within the town boundaries.
    users.loc[eligible, "destination_x"] = np.clip(destination_x, 0, town_width)
    users.loc[eligible, "destination_y"] = np.clip(destination_y, 0, town_height)

    # Mark the users as moving towards their newly assigned destinations.
    users.loc[eligible, "movement_status"] = "moving"
    return users
