import random
import numpy as np

from environment import WasteCollectionEnvironment


class SARSAAgent:

    def __init__(
        self,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.05
    ):

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        # Exploration probability
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Q-table
        self.q_table = {}


    # ========================================================
    # GET Q VALUES
    # ========================================================

    def get_q_values(self, state):

        if state not in self.q_table:

            self.q_table[state] = np.zeros(4)

        return self.q_table[state]


    # ========================================================
    # CHOOSE ACTION - EPSILON GREEDY
    # ========================================================

    def choose_action(self, state, explore=True):

        q_values = self.get_q_values(state)

        # Exploration
        if explore and random.random() < self.epsilon:

            return random.randint(0, 3)

        # Exploitation with random tie-breaking
        max_q = np.max(q_values)
        best_actions = [a for a in range(4) if q_values[a] == max_q]
        return random.choice(best_actions)


    # ========================================================
    # SARSA UPDATE
    # ========================================================

    def update(
        self,
        state,
        action,
        reward,
        next_state,
        next_action,
        done
    ):

        current_q = (
            self.get_q_values(state)[action]
        )


        if done:

            target = reward

        else:

            # IMPORTANT:
            # SARSA uses the Q-value of the
            # ACTUALLY SELECTED next action.

            next_q = (
                self.get_q_values(next_state)[next_action]
            )

            target = (
                reward
                +
                self.discount_factor * next_q
            )


        # SARSA update equation

        new_q = (
            current_q
            +
            self.learning_rate
            *
            (target - current_q)
        )


        self.q_table[state][action] = new_q


    # ========================================================
    # DECAY EPSILON
    # ========================================================

    def decay_epsilon(self):

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )


# ============================================================
# FIND VALID STARTING POSITION
# ============================================================

def get_random_start_position(env):

    valid_positions = []


    for row in range(
        env.grid_size[0]
    ):

        for col in range(
            env.grid_size[1]
        ):

            position = (
                row,
                col
            )


            # Cannot start on obstacle

            if position in env.obstacles:

                continue


            # Cannot start directly on a bin

            if position in env.bins.values():

                continue


            valid_positions.append(
                position
            )


    # Return any valid position including depot
    return random.choice(valid_positions)


# ============================================================
# TRAIN SARSA
# ============================================================

def train_sarsa(
    env,
    episodes=1000
):

    agent = SARSAAgent()


    rewards_per_episode = []

    steps_per_episode = []


    print(
        "\n========================================"
    )

    print(
        "STARTING SARSA TRAINING"
    )

    print(
        "========================================"
    )


    print(
        f"Grid Size: {env.grid_size}"
    )

    print(
        f"Depot: {env.depot}"
    )

    print(
        f"Bins: {len(env.bins)}"
    )

    print(
        f"Obstacles: {len(env.obstacles)}"
    )

    print()


    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for episode in range(
        episodes
    ):

        # ----------------------------------------------------
        # Start from a random valid position
        # ----------------------------------------------------

        start_position = (
            get_random_start_position(env)
        )


        state = env.reset(
            start_position
        )


        total_reward = 0

        steps = 0

        done = False


        # ----------------------------------------------------
        # Choose FIRST ACTION
        # ----------------------------------------------------

        action = agent.choose_action(
            state
        )


        # ----------------------------------------------------
        # EPISODE
        # ----------------------------------------------------

        while not done:

            # ------------------------------------------------
            # Take action
            # ------------------------------------------------

            next_state, reward, done = (
                env.step(action)
            )


            total_reward += reward

            steps += 1


            # ------------------------------------------------
            # Choose NEXT ACTION
            # ------------------------------------------------

            if not done:

                next_action = (
                    agent.choose_action(
                        next_state
                    )
                )

            else:

                # No next action exists after terminal state.

                next_action = 0


            # ------------------------------------------------
            # SARSA UPDATE
            # ------------------------------------------------

            agent.update(
                state,
                action,
                reward,
                next_state,
                next_action,
                done
            )


            # ------------------------------------------------
            # Move to next state/action
            # ------------------------------------------------

            state = next_state

            action = next_action


        # ----------------------------------------------------
        # Reduce exploration
        # ----------------------------------------------------

        agent.decay_epsilon()


        rewards_per_episode.append(
            total_reward
        )


        steps_per_episode.append(
            steps
        )


        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            episode + 1
        ) % 100 == 0:

            print(
                f"Episode {episode + 1}/{episodes} | "
                f"Reward: {total_reward:7.1f} | "
                f"Steps: {steps:3d} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )


    print(
        "\nSARSA training completed."
    )


    print(
        "Number of learned states:",
        len(agent.q_table)
    )


    return (
        agent,
        rewards_per_episode,
        steps_per_episode
    )


# ============================================================
# TEST SARSA DIRECTLY
# ============================================================

if __name__ == "__main__":

    env = WasteCollectionEnvironment()


    agent, rewards, steps = (
        train_sarsa(
            env=env,
            episodes=1000
        )
    )