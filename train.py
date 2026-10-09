import os
import pickle
import random

from environment import WasteCollectionEnvironment
from q_learning import QLearningAgent
from sarsa import SARSAAgent


# ============================================================
# SETTINGS
# ============================================================

EPISODES_PER_START = 100

MODEL_FOLDER = "models"


# ============================================================
# GET VALID POSITIONS
# ============================================================

def get_valid_positions(env):

    positions = []

    for row in range(env.grid_size):

        for col in range(env.grid_size):

            position = (
                row,
                col
            )

            if env.is_valid_position(position):

                positions.append(position)

    return positions


# ============================================================
# TRAIN Q-LEARNING
# ============================================================

def train_q_learning():

    print("\n")
    print("=" * 60)
    print("Q-LEARNING TRAINING")
    print("=" * 60)


    env = WasteCollectionEnvironment()

    agent = QLearningAgent()

    valid_positions = get_valid_positions(env)


    total_episodes = (
        len(valid_positions)
        *
        EPISODES_PER_START
    )


    episode_count = 0


    for start_position in valid_positions:

        for _ in range(EPISODES_PER_START):

            episode_count += 1


            state = env.reset(
                start_position
            )


            done = False

            steps = 0


            while not done:

                action = agent.choose_action(
                    state
                )


                next_state, reward, done = (
                    env.step(action)
                )


                agent.update(

                    state,
                    action,
                    reward,
                    next_state,
                    done

                )


                state = next_state

                steps += 1


            agent.decay_epsilon()


        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            start_position == (4, 5)
        ):

            print(
                "\n*** (4,5) START POSITION TRAINED ***"
            )


        if (
            len(valid_positions)
            == 0
        ):

            break


    print("\nQ-Learning completed.")

    print(
        "Learned states:",
        len(agent.q_table)
    )


    return agent


# ============================================================
# TRAIN SARSA
# ============================================================

def train_sarsa():

    print("\n")
    print("=" * 60)
    print("SARSA TRAINING")
    print("=" * 60)


    env = WasteCollectionEnvironment()

    agent = SARSAAgent()

    valid_positions = get_valid_positions(env)


    total_episodes = (
        len(valid_positions)
        *
        EPISODES_PER_START
    )


    episode_count = 0


    for start_position in valid_positions:

        for _ in range(EPISODES_PER_START):

            episode_count += 1


            state = env.reset(
                start_position
            )


            action = agent.choose_action(
                state
            )


            done = False

            steps = 0


            while not done:

                next_state, reward, done = (
                    env.step(action)
                )


                if done:

                    next_action = 0

                else:

                    next_action = (
                        agent.choose_action(
                            next_state
                        )
                    )


                agent.update(

                    state,
                    action,
                    reward,
                    next_state,
                    next_action,
                    done

                )


                state = next_state

                action = next_action

                steps += 1


            agent.decay_epsilon()


        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            start_position == (4, 5)
        ):

            print(
                "\n*** (4,5) START POSITION TRAINED ***"
            )


    print("\nSARSA completed.")

    print(
        "Learned states:",
        len(agent.q_table)
    )


    return agent


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(agent, filename):

    os.makedirs(
        MODEL_FOLDER,
        exist_ok=True
    )


    path = os.path.join(
        MODEL_FOLDER,
        filename
    )


    with open(
        path,
        "wb"
    ) as file:

        pickle.dump(
            agent,
            file
        )


    print(
        "Saved:",
        path
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 60)
    print("SMART WASTE COLLECTION")
    print("REINFORCEMENT LEARNING TRAINING")
    print("=" * 60)


    # --------------------------------------------------------
    # Q-LEARNING
    # --------------------------------------------------------

    q_agent = train_q_learning()

    save_model(
        q_agent,
        "q_learning.pkl"
    )


    # --------------------------------------------------------
    # SARSA
    # --------------------------------------------------------

    sarsa_agent = train_sarsa()

    save_model(
        sarsa_agent,
        "sarsa.pkl"
    )


    print("\n")
    print("=" * 60)
    print("ALL TRAINING COMPLETED")
    print("=" * 60)