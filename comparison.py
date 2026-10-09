import matplotlib.pyplot as plt

from q_learning import train_q_learning
from sarsa import train_sarsa


# ============================================================
# TRAIN BOTH ALGORITHMS
# ============================================================

print("=" * 60)
print("TRAINING Q-LEARNING")
print("=" * 60)

q_agent, q_rewards, q_steps = train_q_learning(
    episodes=1000
)


print("\n")
print("=" * 60)
print("TRAINING SARSA")
print("=" * 60)

sarsa_agent, sarsa_rewards, sarsa_steps = train_sarsa(
    episodes=1000
)


# ============================================================
# MOVING AVERAGE FUNCTION
# ============================================================

def moving_average(values, window=50):

    averages = []

    for i in range(len(values)):

        start = max(0, i - window + 1)

        average = sum(
            values[start:i + 1]
        ) / (i - start + 1)

        averages.append(average)

    return averages


# ============================================================
# CALCULATE SMOOTHED VALUES
# ============================================================

q_reward_avg = moving_average(
    q_rewards,
    window=50
)

sarsa_reward_avg = moving_average(
    sarsa_rewards,
    window=50
)


q_steps_avg = moving_average(
    q_steps,
    window=50
)

sarsa_steps_avg = moving_average(
    sarsa_steps,
    window=50
)


# ============================================================
# REWARD COMPARISON
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    q_rewards,
    alpha=0.25,
    label="Q-Learning Raw"
)

plt.plot(
    sarsa_rewards,
    alpha=0.25,
    label="SARSA Raw"
)

plt.plot(
    q_reward_avg,
    linewidth=2,
    label="Q-Learning Moving Average"
)

plt.plot(
    sarsa_reward_avg,
    linewidth=2,
    label="SARSA Moving Average"
)

plt.xlabel("Episode")

plt.ylabel("Total Reward")

plt.title(
    "Q-Learning vs SARSA - Reward per Episode"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# STEPS COMPARISON
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    q_steps,
    alpha=0.25,
    label="Q-Learning Raw"
)

plt.plot(
    sarsa_steps,
    alpha=0.25,
    label="SARSA Raw"
)

plt.plot(
    q_steps_avg,
    linewidth=2,
    label="Q-Learning Moving Average"
)

plt.plot(
    sarsa_steps_avg,
    linewidth=2,
    label="SARSA Moving Average"
)

plt.xlabel("Episode")

plt.ylabel("Number of Steps")

plt.title(
    "Q-Learning vs SARSA - Steps per Episode"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# FINAL STATISTICS
# ============================================================

print("\n")
print("=" * 60)
print("FINAL TRAINING COMPARISON")
print("=" * 60)

print(
    f"Q-Learning average reward "
    f"(last 100 episodes): "
    f"{sum(q_rewards[-100:]) / 100:.2f}"
)

print(
    f"SARSA average reward "
    f"(last 100 episodes): "
    f"{sum(sarsa_rewards[-100:]) / 100:.2f}"
)

print()

print(
    f"Q-Learning average steps "
    f"(last 100 episodes): "
    f"{sum(q_steps[-100:]) / 100:.2f}"
)

print(
    f"SARSA average steps "
    f"(last 100 episodes): "
    f"{sum(sarsa_steps[-100:]) / 100:.2f}"
)

print()

print(
    "Q-Learning learned states:",
    len(q_agent.q_table)
)

print(
    "SARSA learned states:",
    len(sarsa_agent.q_table)
)