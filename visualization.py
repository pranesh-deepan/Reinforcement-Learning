import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from q_learning import train_q_learning
from sarsa import train_sarsa
from environment import WasteCollectionEnvironment


# ============================================================
# SETTINGS
# ============================================================

TRAINING_EPISODES = 1000

# Environment itself allows maximum 300 steps
MAX_VISUALIZATION_STEPS = 300

# Animation speed
ANIMATION_INTERVAL = 200


# ============================================================
# TRAIN Q-LEARNING
# ============================================================

print("=" * 65)
print("TRAINING Q-LEARNING")
print("=" * 65)

q_agent, q_rewards, q_steps = train_q_learning(
    episodes=TRAINING_EPISODES
)


# ============================================================
# TRAIN SARSA
# ============================================================

print("\n")
print("=" * 65)
print("TRAINING SARSA")
print("=" * 65)

sarsa_agent, sarsa_rewards, sarsa_steps = train_sarsa(
    episodes=TRAINING_EPISODES
)


# ============================================================
# GET BEST ACTION
# ============================================================

def get_best_action(agent, state):

    q_values = agent.get_q_values(state)

    return int(q_values.argmax())


# ============================================================
# GENERATE ROUTE
# ============================================================

def generate_route(agent, algorithm_name):

    env = WasteCollectionEnvironment()

    state = env.reset()

    route = []

    total_reward = 0

    done = False

    print("\n" + "=" * 65)
    print(f"GENERATING {algorithm_name.upper()} ROUTE")
    print("=" * 65)

    # --------------------------------------------------------
    # Run trained policy
    # --------------------------------------------------------

    for step in range(MAX_VISUALIZATION_STEPS):

        # Current information

        route.append({
            "position": env.vehicle_position,
            "bin_levels": env.bin_fill_levels.copy(),
            "collected": env.collected_bins.copy(),
            "reward": 0,
            "total_reward": total_reward,
            "step": step,
            "completed": False
        })

        # ----------------------------------------------------
        # Choose learned action
        # ----------------------------------------------------

        action = get_best_action(
            agent,
            state
        )

        # ----------------------------------------------------
        # Perform action
        # ----------------------------------------------------

        next_state, reward, done = env.step(action)

        total_reward += reward

        # ----------------------------------------------------
        # Store reward
        # ----------------------------------------------------

        route[-1]["reward"] = reward

        route[-1]["total_reward"] = total_reward

        # ----------------------------------------------------
        # Move to next state
        # ----------------------------------------------------

        state = next_state

        # ----------------------------------------------------
        # Check completion
        # ----------------------------------------------------

        if done:

            completed = (
                len(env.collected_bins) == len(env.bins)
                and env.vehicle_position == env.depot
            )

            route[-1]["completed"] = completed

            # Add final frame
            route.append({
                "position": env.vehicle_position,
                "bin_levels": env.bin_fill_levels.copy(),
                "collected": env.collected_bins.copy(),
                "reward": 0,
                "total_reward": total_reward,
                "step": step + 1,
                "completed": completed
            })

            break

    # ========================================================
    # FINAL STATUS
    # ========================================================

    all_bins_collected = (
        len(env.collected_bins) == len(env.bins)
    )

    returned_to_depot = (
        env.vehicle_position == env.depot
    )

    completed = (
        all_bins_collected
        and returned_to_depot
    )

    # --------------------------------------------------------
    # Print result
    # --------------------------------------------------------

    print("\nFinal Position:")
    print(env.vehicle_position)

    print("\nCollected Bins:")
    print(
        f"{len(env.collected_bins)} / "
        f"{len(env.bins)}"
    )

    print("\nCollected:")
    print(
        sorted(env.collected_bins)
    )

    print("\nTotal Reward:")
    print(
        f"{total_reward:.2f}"
    )

    print("\nTotal Steps:")
    print(
        env.steps
    )

    print("\nReturned to Depot:")
    print(
        "YES" if returned_to_depot else "NO"
    )

    print("\nTask Completed:")
    print(
        "YES" if completed else "NO"
    )

    return route, completed


# ============================================================
# GENERATE ROUTES
# ============================================================

q_route, q_completed = generate_route(
    q_agent,
    "Q-Learning"
)

sarsa_route, sarsa_completed = generate_route(
    sarsa_agent,
    "SARSA"
)


# ============================================================
# ANIMATION
# ============================================================

def animate_route(
    route,
    algorithm_name,
    completed
):

    env = WasteCollectionEnvironment()

    # ========================================================
    # FIGURE
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(12, 8)
    )

    # ========================================================
    # GRID
    # ========================================================

    ax.set_xlim(
        -0.5,
        env.grid_size - 0.5
    )

    ax.set_ylim(
        env.grid_size - 0.5,
        -0.5
    )

    ax.set_xticks(
        range(env.grid_size)
    )

    ax.set_yticks(
        range(env.grid_size)
    )

    ax.grid(
        True,
        linewidth=0.8,
        alpha=0.6
    )

    # ========================================================
    # OBSTACLES
    # ========================================================

    for row, col in env.obstacles:

        ax.add_patch(
            plt.Rectangle(
                (
                    col - 0.5,
                    row - 0.5
                ),
                1,
                1,
                alpha=0.8
            )
        )

    # ========================================================
    # DEPOT
    # ========================================================

    depot_row, depot_col = env.depot

    ax.scatter(
        depot_col,
        depot_row,
        s=450,
        marker="s",
        label="Depot",
        zorder=5
    )

    ax.text(
        depot_col,
        depot_row,
        "D",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        zorder=6
    )

    # ========================================================
    # BINS
    # ========================================================

    bin_scatter = {}
    bin_text = {}

    for bin_name, position in env.bins.items():

        row, col = position

        bin_scatter[bin_name] = ax.scatter(
            col,
            row,
            s=350,
            marker="D",
            zorder=5
        )

        bin_text[bin_name] = ax.text(
            col,
            row,
            bin_name,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            zorder=6
        )

    # ========================================================
    # VEHICLE
    # ========================================================

    first_row, first_col = route[0]["position"]

    vehicle, = ax.plot(
        first_col,
        first_row,
        marker="o",
        markersize=18,
        linestyle="None",
        label="Collection Vehicle",
        zorder=10
    )

    # ========================================================
    # ROUTE LINE
    # ========================================================

    route_line, = ax.plot(
        [],
        [],
        linewidth=2.5,
        alpha=0.8,
        zorder=3
    )

    route_x = []
    route_y = []

    # ========================================================
    # TITLE
    # ========================================================

    ax.set_title(
        f"{algorithm_name} - Smart Waste Collection",
        fontsize=17,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Column"
    )

    ax.set_ylabel(
        "Row"
    )

    # ========================================================
    # INFORMATION PANEL
    # ========================================================

    information = ax.text(
        1.02,
        0.95,
        "",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round",
            alpha=0.1
        )
    )

    # ========================================================
    # STATUS TEXT
    # ========================================================

    status_text = ax.text(
        1.02,
        0.25,
        "",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        verticalalignment="top"
    )

    # ========================================================
    # UPDATE FUNCTION
    # ========================================================

    def update(frame):

        data = route[frame]

        # ----------------------------------------------------
        # Vehicle position
        # ----------------------------------------------------

        row, col = data["position"]

        vehicle.set_data(
            [col],
            [row]
        )

        # ----------------------------------------------------
        # Route
        # ----------------------------------------------------

        route_x.append(col)
        route_y.append(row)

        route_line.set_data(
            route_x,
            route_y
        )

        # ----------------------------------------------------
        # Update bins
        # ----------------------------------------------------

        collected = data["collected"]

        for bin_name in env.bins:

            bin_row, bin_col = env.bins[bin_name]

            fill_level = data["bin_levels"][bin_name]

            # ------------------------------------------------
            # Collected
            # ------------------------------------------------

            if bin_name in collected:

                bin_scatter[bin_name].set_sizes(
                    [130]
                )

                bin_text[bin_name].set_text(
                    f"{bin_name}\n✓"
                )

            # ------------------------------------------------
            # Not collected
            # ------------------------------------------------

            else:

                bin_scatter[bin_name].set_sizes(
                    [350]
                )

                bin_text[bin_name].set_text(
                    f"{bin_name}\n{fill_level:.0f}%"
                )

        # ----------------------------------------------------
        # Number of collected bins
        # ----------------------------------------------------

        collected_count = len(
            collected
        )

        total_bins = len(
            env.bins
        )

        # ----------------------------------------------------
        # Information
        # ----------------------------------------------------

        information.set_text(
            f"Algorithm: {algorithm_name}\n\n"
            f"Step: {data['step']}\n"
            f"Vehicle: ({row}, {col})\n\n"
            f"Bins Collected: "
            f"{collected_count}/{total_bins}\n\n"
            f"Current Reward: "
            f"{data['reward']:+.1f}\n\n"
            f"Total Reward: "
            f"{data['total_reward']:+.1f}"
        )

        # ----------------------------------------------------
        # Completion status
        # ----------------------------------------------------

        if (
            collected_count == total_bins
            and (row, col) == env.depot
        ):

            status_text.set_text(
                "TASK COMPLETED\n"
                "✓ All bins collected\n"
                "✓ Returned to depot"
            )

        elif collected_count == total_bins:

            status_text.set_text(
                "ALL BINS COLLECTED\n"
                "Vehicle returning to depot..."
            )

        else:

            status_text.set_text(
                "COLLECTING BINS..."
            )

        return (
            vehicle,
            route_line,
            information,
            status_text
        )

    # ========================================================
    # CREATE ANIMATION
    # ========================================================

    animation = FuncAnimation(
        fig,
        update,
        frames=len(route),
        interval=ANIMATION_INTERVAL,
        repeat=False
    )

    # ========================================================
    # LAYOUT
    # ========================================================

    plt.tight_layout()

    plt.show()


# ============================================================
# Q-LEARNING VISUALIZATION
# ============================================================

print("\n")
print("=" * 65)
print("STARTING Q-LEARNING ANIMATION")
print("=" * 65)

animate_route(
    q_route,
    "Q-Learning",
    q_completed
)


# ============================================================
# SARSA VISUALIZATION
# ============================================================

print("\n")
print("=" * 65)
print("STARTING SARSA ANIMATION")
print("=" * 65)

animate_route(
    sarsa_route,
    "SARSA",
    sarsa_completed
)


# ============================================================
# FINAL COMPARISON
# ============================================================

print("\n")
print("=" * 65)
print("FINAL VISUALIZATION RESULTS")
print("=" * 65)

print(
    f"Q-Learning completed: "
    f"{'YES' if q_completed else 'NO'}"
)

print(
    f"SARSA completed: "
    f"{'YES' if sarsa_completed else 'NO'}"
)

print("=" * 65)