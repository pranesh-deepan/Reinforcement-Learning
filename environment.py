import random


class WasteCollectionEnvironment:

    # ============================================================
    # INITIALIZATION
    # ============================================================

    def __init__(
        self,
        grid_size=(10, 10),
        depot=(0, 0),
        bins=None,
        obstacles=None,
        initial_fill_levels=None
    ):

        # --------------------------------------------------------
        # Grid
        # --------------------------------------------------------

        self.grid_size = tuple(grid_size)

        # --------------------------------------------------------
        # Depot
        # --------------------------------------------------------

        self.depot = tuple(depot)

        # --------------------------------------------------------
        # Bins
        # --------------------------------------------------------

        if bins is None:

            self.bins = {
                "B1": (1, 8),
                "B2": (3, 4),
                "B3": (5, 8),
                "B4": (7, 2),
                "B5": (8, 7)
            }

        else:

            self.bins = {
                str(name): tuple(position)
                for name, position in bins.items()
            }

        # --------------------------------------------------------
        # Initial fill levels
        # --------------------------------------------------------

        if initial_fill_levels is None:

            self.initial_fill_levels = {
                name: random.choice(
                    [30, 50, 60, 75, 80, 90]
                )
                for name in self.bins
            }

        else:

            self.initial_fill_levels = {
                str(name): int(level)
                for name, level
                in initial_fill_levels.items()
            }

        # --------------------------------------------------------
        # Obstacles
        # --------------------------------------------------------

        if obstacles is None:

            self.obstacles = {

                (1, 3),
                (1, 4),

                (2, 3),
                (2, 4),

                (3, 3),

                (4, 6),
                (4, 7),

                (5, 6),

                (6, 2),
                (6, 3),

                (7, 6),

                (8, 3),
                (8, 4)

            }

        else:

            self.obstacles = {
                tuple(position)
                for position in obstacles
            }

        # --------------------------------------------------------
        # Validate environment
        # --------------------------------------------------------

        self.validate_environment()

        # --------------------------------------------------------
        # Actions
        #
        # 0 = UP
        # 1 = DOWN
        # 2 = LEFT
        # 3 = RIGHT
        # --------------------------------------------------------

        self.actions = {

            0: (-1, 0),

            1: (1, 0),

            2: (0, -1),

            3: (0, 1)

        }

        # --------------------------------------------------------
        # Maximum steps
        # --------------------------------------------------------

        self.max_steps = max(
            300,
            self.grid_size[0]
            * self.grid_size[1]
            * 3
        )

        # --------------------------------------------------------
        # Runtime variables
        # --------------------------------------------------------

        self.vehicle_position = self.depot

        self.bin_fill_levels = {}

        self.collected_bins = set()

        self.steps = 0

        self.total_reward = 0

        self.done = False

        # --------------------------------------------------------
        # Reset
        # --------------------------------------------------------

        self.reset()


    # ============================================================
    # VALIDATE ENVIRONMENT
    # ============================================================

    def validate_environment(self):

        rows, cols = self.grid_size

        # --------------------------------------------------------
        # Validate grid
        # --------------------------------------------------------

        if rows < 2 or cols < 2:

            raise ValueError(
                "Grid size must be at least 2 x 2."
            )

        # --------------------------------------------------------
        # Validate depot
        # --------------------------------------------------------

        if not self.position_inside_grid(
            self.depot
        ):

            raise ValueError(
                f"Depot {self.depot} is outside the grid."
            )

        # --------------------------------------------------------
        # Depot cannot be obstacle
        # --------------------------------------------------------

        if self.depot in self.obstacles:

            raise ValueError(
                "Depot cannot be placed on an obstacle."
            )

        # --------------------------------------------------------
        # Validate bins
        # --------------------------------------------------------

        positions = set()

        for name, position in self.bins.items():

            if not self.position_inside_grid(
                position
            ):

                raise ValueError(
                    f"Bin {name} {position} "
                    f"is outside the grid."
                )

            if position == self.depot:

                raise ValueError(
                    f"Bin {name} cannot be placed "
                    f"on the depot."
                )

            if position in self.obstacles:

                raise ValueError(
                    f"Bin {name} cannot be placed "
                    f"on an obstacle."
                )

            if position in positions:

                raise ValueError(
                    "Two bins cannot occupy "
                    "the same position."
                )

            positions.add(position)

        # --------------------------------------------------------
        # Validate obstacles
        # --------------------------------------------------------

        for obstacle in self.obstacles:

            if not self.position_inside_grid(
                obstacle
            ):

                raise ValueError(
                    f"Obstacle {obstacle} "
                    f"is outside the grid."
                )

            if obstacle == self.depot:

                raise ValueError(
                    "Obstacle cannot be placed "
                    "on the depot."
                )

        # --------------------------------------------------------
        # Validate fill levels
        # --------------------------------------------------------

        for name in self.bins:

            if name not in self.initial_fill_levels:

                raise ValueError(
                    f"Missing fill level for {name}."
                )

            level = self.initial_fill_levels[name]

            if level < 0 or level > 100:

                raise ValueError(
                    f"Invalid fill level for {name}."
                )


    # ============================================================
    # CHECK POSITION INSIDE GRID
    # ============================================================

    def position_inside_grid(
        self,
        position
    ):

        row, col = position

        rows, cols = self.grid_size

        return (
            0 <= row < rows
            and
            0 <= col < cols
        )


    # ============================================================
    # RESET ENVIRONMENT
    # ============================================================

    def reset(
        self,
        start_position=None
    ):

        # --------------------------------------------------------
        # Reset fill levels
        # --------------------------------------------------------

        self.bin_fill_levels = {

            name: level

            for name, level
            in self.initial_fill_levels.items()

        }

        # --------------------------------------------------------
        # Reset collected bins
        # --------------------------------------------------------

        self.collected_bins = set()

        # --------------------------------------------------------
        # Reset counters
        # --------------------------------------------------------

        self.steps = 0

        self.total_reward = 0

        self.done = False

        # --------------------------------------------------------
        # Starting position
        # --------------------------------------------------------

        if start_position is None:

            self.vehicle_position = self.depot

        else:

            start_position = tuple(
                start_position
            )

            if not self.is_valid_position(
                start_position
            ):

                raise ValueError(
                    f"Invalid starting position: "
                    f"{start_position}"
                )

            self.vehicle_position = (
                start_position
            )

        # --------------------------------------------------------
        # Return initial state
        # --------------------------------------------------------

        return self.get_state()


    # ============================================================
    # CHECK VALID POSITION
    # ============================================================

    def is_valid_position(
        self,
        position
    ):

        position = tuple(position)

        if not self.position_inside_grid(
            position
        ):

            return False

        if position in self.obstacles:

            return False

        return True


    # ============================================================
    # CHECK WHETHER POSITION IS A BIN
    # ============================================================

    def get_bin_at_position(
        self,
        position
    ):

        position = tuple(position)

        for name, bin_position in self.bins.items():

            if position == bin_position:

                return name

        return None


    # ============================================================
    # GET STATE
    # ============================================================

    def get_state(self):

        row, col = self.vehicle_position

        state = [

            row,
            col

        ]

        # --------------------------------------------------------
        # Add fill level of every bin
        # --------------------------------------------------------

        for name in sorted(
            self.bins.keys()
        ):

            state.append(
                self.bin_fill_levels[name]
            )

        # --------------------------------------------------------
        # Add collected status
        # --------------------------------------------------------

        for name in sorted(
            self.bins.keys()
        ):

            if name in self.collected_bins:

                state.append(1)

            else:

                state.append(0)

        # --------------------------------------------------------
        # Convert to tuple
        # --------------------------------------------------------

        return tuple(state)


    # ============================================================
    # MANHATTAN DISTANCE
    # ============================================================

    def manhattan_distance(
        self,
        position1,
        position2
    ):

        return (

            abs(
                position1[0]
                -
                position2[0]
            )

            +

            abs(
                position1[1]
                -
                position2[1]
            )

        )


    # ============================================================
    # FIND NEAREST UNCOLLECTED BIN
    # ============================================================

    def nearest_uncollected_bin(self):

        remaining_bins = [

            name

            for name in self.bins

            if name not in self.collected_bins

        ]

        if not remaining_bins:

            return None

        nearest_bin = min(

            remaining_bins,

            key=lambda name:

                self.manhattan_distance(

                    self.vehicle_position,

                    self.bins[name]

                )

        )

        return nearest_bin


    # ============================================================
    # COLLECTION REWARD
    # ============================================================

    def collection_reward(
        self,
        bin_name
    ):

        fill_level = self.bin_fill_levels[
            bin_name
        ]

        if fill_level >= 90:

            return 40

        elif fill_level >= 75:

            return 30

        elif fill_level >= 50:

            return 20

        else:

            return 10


    # ============================================================
    # TAKE ACTION
    # ============================================================

    def step(
        self,
        action
    ):

        # --------------------------------------------------------
        # Episode already finished
        # --------------------------------------------------------

        if self.done:

            return (

                self.get_state(),

                0,

                True

            )

        # --------------------------------------------------------
        # Validate action
        # --------------------------------------------------------

        if action not in self.actions:

            raise ValueError(
                f"Invalid action: {action}"
            )

        # --------------------------------------------------------
        # Current position
        # --------------------------------------------------------

        current_position = (
            self.vehicle_position
        )

        row, col = current_position

        delta_row, delta_col = (
            self.actions[action]
        )

        new_position = (

            row + delta_row,

            col + delta_col

        )

        reward = -1

        # --------------------------------------------------------
        # Invalid movement
        # --------------------------------------------------------

        if not self.is_valid_position(
            new_position
        ):

            reward = -5

            self.steps += 1

            self.total_reward += reward

            if self.steps >= self.max_steps:

                self.done = True

            return (

                self.get_state(),

                reward,

                self.done

            )

        # --------------------------------------------------------
        # Move vehicle
        # --------------------------------------------------------

        self.vehicle_position = (
            new_position
        )

        self.steps += 1

        # --------------------------------------------------------
        # Check bin
        # --------------------------------------------------------

        bin_name = self.get_bin_at_position(
            new_position
        )

        # --------------------------------------------------------
        # Collect bin
        # --------------------------------------------------------

        if (

            bin_name is not None

            and

            bin_name not in self.collected_bins

        ):

            reward += self.collection_reward(
                bin_name
            )

            self.collected_bins.add(
                bin_name
            )

            self.bin_fill_levels[
                bin_name
            ] = 0

        # --------------------------------------------------------
        # Check all bins
        # --------------------------------------------------------

        all_bins_collected = (

            len(self.collected_bins)

            ==

            len(self.bins)

        )

        # --------------------------------------------------------
        # All bins collected and returned
        # to depot
        # --------------------------------------------------------

        if (

            all_bins_collected

            and

            self.vehicle_position
            == self.depot

        ):

            reward += 100

            self.done = True

        # --------------------------------------------------------
        # Maximum steps
        # --------------------------------------------------------

        elif self.steps >= self.max_steps:

            self.done = True

        # --------------------------------------------------------
        # Total reward
        # --------------------------------------------------------

        self.total_reward += reward

        # --------------------------------------------------------
        # Return
        # --------------------------------------------------------

        return (

            self.get_state(),

            reward,

            self.done

        )


    # ============================================================
    # GET ENVIRONMENT INFORMATION
    # ============================================================

    def get_environment_info(self):

        return {

            "grid_size":
                list(self.grid_size),

            "depot":
                list(self.depot),

            "bins": {

                name:
                    list(position)

                for name, position
                in self.bins.items()

            },

            "bin_fill_levels": {

                name: level

                for name, level
                in self.bin_fill_levels.items()

            },

            "initial_fill_levels": {

                name: level

                for name, level
                in self.initial_fill_levels.items()

            },

            "obstacles": [

                list(position)

                for position
                in self.obstacles

            ],

            "vehicle_position":
                list(self.vehicle_position),

            "collected_bins":
                list(self.collected_bins),

            "steps":
                self.steps,

            "total_reward":
                self.total_reward,

            "done":
                self.done

        }


# ============================================================
# SIMPLE TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("========================================")
    print("DYNAMIC ENVIRONMENT TEST")
    print("========================================")

    # --------------------------------------------------------
    # Create a custom environment
    # --------------------------------------------------------

    custom_environment = (
        WasteCollectionEnvironment(

            grid_size=(12, 12),

            depot=(1, 1),

            bins={

                "B1": (3, 8),

                "B2": (6, 5),

                "B3": (9, 10),

                "B4": (10, 2)

            },

            obstacles={

                (2, 2),
                (2, 3),
                (2, 4),

                (4, 6),
                (5, 6),
                (6, 6),

                (8, 3),
                (8, 4)

            },

            initial_fill_levels={

                "B1": 80,

                "B2": 60,

                "B3": 90,

                "B4": 50

            }

        )
    )

    print()

    print(
        "Grid:",
        custom_environment.grid_size
    )

    print(
        "Depot:",
        custom_environment.depot
    )

    print(
        "Bins:",
        custom_environment.bins
    )

    print(
        "Obstacles:",
        custom_environment.obstacles
    )

    print(
        "Fill levels:",
        custom_environment.initial_fill_levels
    )

    print()

    # --------------------------------------------------------
    # Test default start
    # --------------------------------------------------------

    state = custom_environment.reset()

    print(
        "Starting position:",
        custom_environment.vehicle_position
    )

    print(
        "Initial state:",
        state
    )

    # --------------------------------------------------------
    # Test custom start
    # --------------------------------------------------------

    test_position = (5, 5)

    if custom_environment.is_valid_position(
        test_position
    ):

        state = custom_environment.reset(
            test_position
        )

        print()

        print(
            "Custom starting position:",
            custom_environment.vehicle_position
        )

        print(
            "Custom state:",
            state
        )

    else:

        print(
            "Invalid test position:",
            test_position
        )

    print()
    print(
        "Dynamic environment test completed."
    )