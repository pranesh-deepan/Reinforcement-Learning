import matplotlib.pyplot as plt

from q_learning import train_q_learning
from sarsa import train_sarsa
from environment import WasteCollectionEnvironment


# ============================================================
# SETTINGS
# ============================================================

TRAINING_EPISODES = 1000
EVALUATION_EPISODES = 10
MAX_STEPS = 300


# ============================================================
# TRAINING
# ============================================================

print("=" * 70)
print("TRAINING Q-LEARNING")
print("=" * 70)

q_agent, q_training_rewards, q_training_steps = train_q_learning(
    episodes=TRAINING_EPISODES
)


print("\n")
print("=" * 70)
print("TRAINING SARSA")
print("=" * 70)

sarsa_agent, sarsa_training_rewards, sarsa_training_steps = train_sarsa(
    episodes=TRAINING_EPISODES
)


# ============================================================
# GET BEST ACTION
# ============================================================

def get_best_action(agent, state):

    q_values = agent.get_q_values(state)

    return int(q_values.argmax())


# ============================================================
# EVALUATE ONE EPISODE
# ============================================================

def evaluate_agent(agent):

    env = WasteCollectionEnvironment()

    state = env.reset()

    total_reward = 0

    route = [env.vehicle_position]

    done = False

    steps = 0

    # --------------------------------------------------------
    # Run greedy policy
    # --------------------------------------------------------

    while not done and steps < MAX_STEPS:

        action = get_best_action(
            agent,
            state
        )

        next_state, reward, done = env.step(
            action
        )

        total_reward += reward

        route.append(
            env.vehicle_position
        )

        state = next_state

        steps += 1

    # ========================================================
    # CALCULATE DISTANCE
    # ========================================================

    distance = 0

    for i in range(1, len(route)):

        previous_row, previous_col = route[i - 1]

        current_row, current_col = route[i]

        distance += (
            abs(current_row - previous_row)
            +
            abs(current_col - previous_col)
        )

    # ========================================================
    # COLLECTION STATUS
    # ========================================================

    bins_collected = len(
        env.collected_bins
    )

    total_bins = len(
        env.bins
    )

    all_bins_collected = (
        bins_collected == total_bins
    )

    returned_to_depot = (
        env.vehicle_position == env.depot
    )

    completed = (
        all_bins_collected
        and returned_to_depot
    )

    # ========================================================
    # RETURN RESULTS
    # ========================================================

    return {
        "reward": total_reward,
        "steps": steps,
        "distance": distance,
        "bins_collected": bins_collected,
        "total_bins": total_bins,
        "returned_to_depot": returned_to_depot,
        "completed": completed
    }


# ============================================================
# EVALUATE MULTIPLE EPISODES
# ============================================================

def evaluate_multiple(agent, algorithm_name):

    results = []

    print("\n")
    print("=" * 70)
    print(f"EVALUATING {algorithm_name.upper()}")
    print("=" * 70)

    for episode in range(EVALUATION_EPISODES):

        result = evaluate_agent(
            agent
        )

        results.append(result)

        print(
            f"Episode {episode + 1:02d} | "
            f"Reward: {result['reward']:8.2f} | "
            f"Steps: {result['steps']:3d} | "
            f"Distance: {result['distance']:3d} | "
            f"Bins: "
            f"{result['bins_collected']}/"
            f"{result['total_bins']} | "
            f"Depot: "
            f"{'YES' if result['returned_to_depot'] else 'NO'} | "
            f"Complete: "
            f"{'YES' if result['completed'] else 'NO'}"
        )

    return results


# ============================================================
# RUN EVALUATION
# ============================================================

q_results = evaluate_multiple(
    q_agent,
    "Q-Learning"
)

sarsa_results = evaluate_multiple(
    sarsa_agent,
    "SARSA"
)


# ============================================================
# CALCULATE AVERAGES
# ============================================================

def calculate_average(results, key):

    values = [
        result[key]
        for result in results
    ]

    return sum(values) / len(values)


# ------------------------------------------------------------
# Q-Learning averages
# ------------------------------------------------------------

q_avg_reward = calculate_average(
    q_results,
    "reward"
)

q_avg_steps = calculate_average(
    q_results,
    "steps"
)

q_avg_distance = calculate_average(
    q_results,
    "distance"
)

q_avg_bins = calculate_average(
    q_results,
    "bins_collected"
)


# ------------------------------------------------------------
# SARSA averages
# ------------------------------------------------------------

sarsa_avg_reward = calculate_average(
    sarsa_results,
    "reward"
)

sarsa_avg_steps = calculate_average(
    sarsa_results,
    "steps"
)

sarsa_avg_distance = calculate_average(
    sarsa_results,
    "distance"
)

sarsa_avg_bins = calculate_average(
    sarsa_results,
    "bins_collected"
)


# ============================================================
# COMPLETION COUNTS
# ============================================================

q_completed_count = sum(
    result["completed"]
    for result in q_results
)

sarsa_completed_count = sum(
    result["completed"]
    for result in sarsa_results
)


# ============================================================
# PRINT FINAL TABLE
# ============================================================

print("\n")
print("=" * 70)
print("FINAL PERFORMANCE COMPARISON")
print("=" * 70)

print(
    f"{'Metric':<25}"
    f"{'Q-Learning':>18}"
    f"{'SARSA':>18}"
)

print("-" * 61)

print(
    f"{'Average Reward':<25}"
    f"{q_avg_reward:>18.2f}"
    f"{sarsa_avg_reward:>18.2f}"
)

print(
    f"{'Average Steps':<25}"
    f"{q_avg_steps:>18.2f}"
    f"{sarsa_avg_steps:>18.2f}"
)

print(
    f"{'Average Distance':<25}"
    f"{q_avg_distance:>18.2f}"
    f"{sarsa_avg_distance:>18.2f}"
)

print(
    f"{'Average Bins Collected':<25}"
    f"{q_avg_bins:>18.2f}"
    f"{sarsa_avg_bins:>18.2f}"
)

print(
    f"{'Completed Episodes':<25}"
    f"{q_completed_count:>18}"
    f"{sarsa_completed_count:>18}"
)

print(
    f"{'Total Evaluation Episodes':<25}"
    f"{EVALUATION_EPISODES:>18}"
    f"{EVALUATION_EPISODES:>18}"
)

print("=" * 70)


# ============================================================
# VISUALIZATION 1
# AVERAGE REWARD
# ============================================================

algorithms = [
    "Q-Learning",
    "SARSA"
]

average_rewards = [
    q_avg_reward,
    sarsa_avg_reward
]

plt.figure(
    figsize=(8, 5)
)

plt.bar(
    algorithms,
    average_rewards
)

plt.title(
    "Average Reward Comparison"
)

plt.ylabel(
    "Average Reward"
)

plt.xlabel(
    "Algorithm"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.show()


# ============================================================
# VISUALIZATION 2
# AVERAGE STEPS
# ============================================================

average_steps = [
    q_avg_steps,
    sarsa_avg_steps
]

plt.figure(
    figsize=(8, 5)
)

plt.bar(
    algorithms,
    average_steps
)

plt.title(
    "Average Steps Comparison"
)

plt.ylabel(
    "Average Steps"
)

plt.xlabel(
    "Algorithm"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.show()


# ============================================================
# VISUALIZATION 3
# AVERAGE ROUTE DISTANCE
# ============================================================

average_distances = [
    q_avg_distance,
    sarsa_avg_distance
]

plt.figure(
    figsize=(8, 5)
)

plt.bar(
    algorithms,
    average_distances
)

plt.title(
    "Average Route Distance Comparison"
)

plt.ylabel(
    "Distance"
)

plt.xlabel(
    "Algorithm"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.show()


# ============================================================
# VISUALIZATION 4
# AVERAGE BINS COLLECTED
# ============================================================

average_bins = [
    q_avg_bins,
    sarsa_avg_bins
]

plt.figure(
    figsize=(8, 5)
)

plt.bar(
    algorithms,
    average_bins
)

plt.title(
    "Average Bins Collected Comparison"
)

plt.ylabel(
    "Number of Bins"
)

plt.xlabel(
    "Algorithm"
)

plt.ylim(
    0,
    len(WasteCollectionEnvironment().bins) + 0.5
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.show()


# ============================================================
# VISUALIZATION 5
# COMPLETED EPISODES
# ============================================================

completed_counts = [
    q_completed_count,
    sarsa_completed_count
]

plt.figure(
    figsize=(8, 5)
)

plt.bar(
    algorithms,
    completed_counts
)

plt.title(
    "Completed Episodes Comparison"
)

plt.ylabel(
    "Number of Completed Episodes"
)

plt.xlabel(
    "Algorithm"
)

plt.ylim(
    0,
    EVALUATION_EPISODES + 1
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.show()


# ============================================================
# END
# ============================================================

print("\n")
print("=" * 70)
print("PERFORMANCE COMPARISON COMPLETED")
print("=" * 70)