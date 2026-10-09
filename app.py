from flask import Flask, render_template, request, jsonify

from environment import WasteCollectionEnvironment
from q_learning import QLearningAgent, train_q_learning
from sarsa import SARSAAgent, train_sarsa


app = Flask(__name__)


# ============================================================
# GLOBAL VARIABLES
# ============================================================

env = None

q_agent = None
sarsa_agent = None

q_training_rewards = []
sarsa_training_rewards = []

current_algorithm = None
current_state = None
current_action = None

total_reward = 0
steps = 0

training_completed = False


# ============================================================
# PATH STORAGE
# ============================================================

q_path = []
sarsa_path = []


# ============================================================
# RESULT STORAGE
# ============================================================

q_metrics = {
    "reward": 0,
    "steps": 0,
    "path_length": 0,
    "bins_collected": 0,
    "completed": False
}


sarsa_metrics = {
    "reward": 0,
    "steps": 0,
    "path_length": 0,
    "bins_collected": 0,
    "completed": False
}


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# ENVIRONMENT DATA
# ============================================================

def get_environment_data(environment):

    return {

        "grid_size":
            list(
                environment.grid_size
            ),

        "depot":
            list(
                environment.depot
            ),

        "bins": {

            name:
                list(position)

            for name, position
            in environment.bins.items()

        },

        "obstacles": [

            list(position)

            for position
            in environment.obstacles

        ],

        "bin_fill_levels":
            environment.initial_fill_levels
    }


# ============================================================
# CONFIGURE ENVIRONMENT
# ============================================================

@app.route(
    "/configure",
    methods=["POST"]
)
def configure_environment():

    global env

    global q_agent
    global sarsa_agent

    global q_training_rewards
    global sarsa_training_rewards

    global training_completed

    global q_path
    global sarsa_path

    global q_metrics
    global sarsa_metrics

    try:

        data = request.get_json()


        # ----------------------------------------------------
        # GRID
        # ----------------------------------------------------

        rows = int(
            data.get("rows")
        )

        cols = int(
            data.get("cols")
        )


        # ----------------------------------------------------
        # DEPOT
        # ----------------------------------------------------

        depot_data = data.get(
            "depot"
        )


        if depot_data is None:

            return jsonify({

                "success": False,

                "message":
                    "Depot is required."

            }), 400


        depot = tuple(
            depot_data
        )


        # ----------------------------------------------------
        # BINS
        # ----------------------------------------------------

        bins_data = data.get(
            "bins",
            []
        )


        bins = {}


        for index, position in enumerate(
            bins_data
        ):

            bins[
                f"B{index + 1}"
            ] = tuple(
                position
            )


        # ----------------------------------------------------
        # OBSTACLES
        # ----------------------------------------------------

        obstacles_data = data.get(
            "obstacles",
            []
        )


        obstacles = [

            tuple(position)

            for position
            in obstacles_data

        ]


        # ----------------------------------------------------
        # CREATE ENVIRONMENT
        # ----------------------------------------------------

        env = WasteCollectionEnvironment(

            grid_size=(
                rows,
                cols
            ),

            depot=depot,

            bins=bins,

            obstacles=obstacles

        )


        # ----------------------------------------------------
        # CLEAR OLD TRAINING
        # ----------------------------------------------------

        q_agent = None
        sarsa_agent = None

        q_training_rewards = []
        sarsa_training_rewards = []

        training_completed = False


        # ----------------------------------------------------
        # CLEAR OLD PATHS
        # ----------------------------------------------------

        q_path = []
        sarsa_path = []


        # ----------------------------------------------------
        # CLEAR OLD METRICS
        # ----------------------------------------------------

        q_metrics = {

            "reward": 0,
            "steps": 0,
            "path_length": 0,
            "bins_collected": 0,
            "completed": False

        }


        sarsa_metrics = {

            "reward": 0,
            "steps": 0,
            "path_length": 0,
            "bins_collected": 0,
            "completed": False

        }


        print(
            "\n========================================"
        )

        print(
            "NEW USER ENVIRONMENT CREATED"
        )

        print(
            "========================================"
        )

        print(
            "Grid:",
            env.grid_size
        )

        print(
            "Depot:",
            env.depot
        )

        print(
            "Bins:",
            env.bins
        )

        print(
            "Obstacles:",
            len(
                env.obstacles
            )
        )


        return jsonify({

            "success": True,

            "message":
                "Environment created successfully.",

            "environment":
                get_environment_data(
                    env
                )

        })


    except Exception as e:

        print(
            "Configuration error:",
            e
        )


        return jsonify({

            "success": False,

            "message":
                str(e)

        }), 400


# ============================================================
# TRAIN BOTH ALGORITHMS
# ============================================================

@app.route(
    "/train",
    methods=["POST"]
)
def train():

    global q_agent
    global sarsa_agent

    global q_training_rewards
    global sarsa_training_rewards

    global training_completed


    if env is None:

        return jsonify({

            "success": False,

            "message":
                "Please create an environment first."

        }), 400


    try:

        print(
            "\n========================================"
        )

        print(
            "TRAINING ON USER CREATED ENVIRONMENT"
        )

        print(
            "========================================"
        )


        # ====================================================
        # Q-LEARNING
        # ====================================================

        (
            q_agent,
            q_training_rewards,
            q_steps
        ) = train_q_learning(

            env=env,

            episodes=1000

        )


        # ====================================================
        # SARSA
        # ====================================================

        (
            sarsa_agent,
            sarsa_training_rewards,
            sarsa_steps
        ) = train_sarsa(

            env=env,

            episodes=1000

        )


        # ====================================================
        # GREEDY SIMULATION
        # ====================================================

        q_agent.epsilon = 0

        sarsa_agent.epsilon = 0


        training_completed = True


        print(
            "\n========================================"
        )

        print(
            "TRAINING COMPLETED"
        )

        print(
            "========================================"
        )


        return jsonify({

            "success": True,

            "message":
                "Q-Learning and SARSA training completed.",

            "q_learning": {

                "episodes":
                    len(
                        q_training_rewards
                    ),

                "rewards":
                    q_training_rewards,

                "states_learned":
                    len(q_agent.q_table)

            },

            "sarsa": {

                "episodes":
                    len(
                        sarsa_training_rewards
                    ),

                "rewards":
                    sarsa_training_rewards,

                "states_learned":
                    len(sarsa_agent.q_table)

            }

        })


    except Exception as e:

        print(
            "Training error:",
            e
        )


        return jsonify({

            "success": False,

            "message":
                str(e)

        }), 500


# ============================================================
# START SIMULATION
# ============================================================

@app.route(
    "/start",
    methods=["POST"]
)
def start_simulation():

    global current_algorithm
    global current_state
    global current_action

    global total_reward
    global steps

    global q_path
    global sarsa_path


    if env is None:

        return jsonify({

            "success": False,

            "message":
                "Create an environment first."

        }), 400


    if not training_completed:

        return jsonify({

            "success": False,

            "message":
                "Train the algorithms first."

        }), 400


    data = request.get_json()


    algorithm = data.get(
        "algorithm"
    )


    row = int(
        data.get("row")
    )

    col = int(
        data.get("col")
    )


    start_position = (
        row,
        col
    )


    if algorithm not in [
        "q_learning",
        "sarsa"
    ]:

        return jsonify({

            "success": False,

            "message":
                "Invalid algorithm."

        }), 400


    # --------------------------------------------------------
    # Validate position
    # --------------------------------------------------------

    if not env.is_valid_position(
        start_position
    ):

        return jsonify({

            "success": False,

            "message":
                "Selected position is not valid."

        }), 400


    # --------------------------------------------------------
    # Reset environment
    # --------------------------------------------------------

    current_state = env.reset(
        start_position
    )


    current_algorithm = algorithm

    total_reward = 0

    steps = 0


    # --------------------------------------------------------
    # IMPORTANT:
    # Do NOT clear the other algorithm's path.
    #
    # This allows:
    #
    # Q-Learning path
    # +
    # SARSA path
    #
    # to remain visible for comparison.
    # --------------------------------------------------------

    if algorithm == "q_learning":

        q_path = [
            list(start_position)
        ]

    else:

        sarsa_path = [
            list(start_position)
        ]


    # --------------------------------------------------------
    # Select first action
    # --------------------------------------------------------

    if algorithm == "q_learning":

        current_action = (
            q_agent.choose_action(
                current_state,
                explore=False
            )
        )

    else:

        current_action = (
            sarsa_agent.choose_action(
                current_state,
                explore=False
            )
        )


    return jsonify({

        "success": True,

        "algorithm":
            current_algorithm,

        "position":
            list(
                env.vehicle_position
            ),

        "reward": 0,

        "total_reward": 0,

        "steps": 0,

        "collected_bins":
            list(
                env.collected_bins
            ),

        "bin_fill_levels":
            env.bin_fill_levels,

        "done": False,

        "q_path":
            q_path,

        "sarsa_path":
            sarsa_path,

        "q_metrics":
            q_metrics,

        "sarsa_metrics":
            sarsa_metrics

    })


# ============================================================
# PERFORM ONE STEP
# ============================================================

@app.route(
    "/step",
    methods=["POST"]
)
def simulation_step():

    global current_state
    global current_action

    global total_reward
    global steps

    global q_path
    global sarsa_path

    global q_metrics
    global sarsa_metrics


    if (
        env is None
        or current_state is None
    ):

        return jsonify({

            "success": False,

            "message":
                "Start a simulation first."

        }), 400


    # --------------------------------------------------------
    # Execute action
    # --------------------------------------------------------

    next_state, reward, done = (
        env.step(
            current_action
        )
    )


    total_reward += reward

    steps += 1


    # --------------------------------------------------------
    # Current position
    # --------------------------------------------------------

    current_position = list(
        env.vehicle_position
    )


    # --------------------------------------------------------
    # Store path
    # --------------------------------------------------------

    if current_algorithm == "q_learning":

        q_path.append(
            current_position
        )

    else:

        sarsa_path.append(
            current_position
        )


    # --------------------------------------------------------
    # Next action
    # --------------------------------------------------------

    if not done:

        if current_algorithm == "q_learning":

            current_action = (
                q_agent.choose_action(
                    next_state,
                    explore=False
                )
            )

        else:

            current_action = (
                sarsa_agent.choose_action(
                    next_state,
                    explore=False
                )
            )

    else:

        current_action = None


    # --------------------------------------------------------
    # Update state
    # --------------------------------------------------------

    current_state = next_state


    # --------------------------------------------------------
    # Update metrics
    # --------------------------------------------------------

    current_path = (

        q_path

        if current_algorithm == "q_learning"

        else sarsa_path

    )


    current_metrics = {

        "reward":
            total_reward,

        "steps":
            steps,

        "path_length":
            max(
                0,
                len(current_path) - 1
            ),

        "bins_collected":
            len(
                env.collected_bins
            ),

        "completed":
            done

    }


    if current_algorithm == "q_learning":

        q_metrics = current_metrics

    else:

        sarsa_metrics = current_metrics


    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    return jsonify({

        "success": True,

        "position":
            list(
                env.vehicle_position
            ),

        "reward":
            reward,

        "total_reward":
            total_reward,

        "steps":
            steps,

        "collected_bins":
            list(
                env.collected_bins
            ),

        "bin_fill_levels":
            env.bin_fill_levels,

        "done":
            done,

        "action":
            current_action,

        "q_path":
            q_path,

        "sarsa_path":
            sarsa_path,

        "q_metrics":
            q_metrics,

        "sarsa_metrics":
            sarsa_metrics

    })


# ============================================================
# RESET CURRENT SIMULATION
# ============================================================

@app.route(
    "/reset",
    methods=["POST"]
)
def reset_simulation():

    global current_state
    global current_action

    global total_reward
    global steps


    if env is None:

        return jsonify({

            "success": False,

            "message":
                "Environment does not exist."

        }), 400


    current_state = env.reset(
        env.depot
    )


    current_action = None

    total_reward = 0

    steps = 0


    return jsonify({

        "success": True,

        "position":
            list(
                env.vehicle_position
            ),

        "reward": 0,

        "total_reward": 0,

        "steps": 0,

        "collected_bins":
            list(
                env.collected_bins
            ),

        "bin_fill_levels":
            env.bin_fill_levels,

        "done": False,

        "q_path":
            q_path,

        "sarsa_path":
            sarsa_path,

        "q_metrics":
            q_metrics,

        "sarsa_metrics":
            sarsa_metrics

    })


# ============================================================
# CLEAR PATHS
# ============================================================

@app.route(
    "/clear_paths",
    methods=["POST"]
)
def clear_paths():

    global q_path
    global sarsa_path

    global q_metrics
    global sarsa_metrics


    q_path = []

    sarsa_path = []


    q_metrics = {

        "reward": 0,
        "steps": 0,
        "path_length": 0,
        "bins_collected": 0,
        "completed": False

    }


    sarsa_metrics = {

        "reward": 0,
        "steps": 0,
        "path_length": 0,
        "bins_collected": 0,
        "completed": False

    }


    return jsonify({

        "success": True,

        "q_path":
            q_path,

        "sarsa_path":
            sarsa_path,

        "q_metrics":
            q_metrics,

        "sarsa_metrics":
            sarsa_metrics

    })


# ============================================================
# ENVIRONMENT INFORMATION
# ============================================================

@app.route(
    "/environment",
    methods=["GET"]
)
def environment_information():

    if env is None:

        return jsonify({

            "success": False,

            "message":
                "No environment created."

        }), 400


    return jsonify({

        "success": True,

        "environment":
            get_environment_data(
                env
            ),

        "q_path":
            q_path,

        "sarsa_path":
            sarsa_path,

        "q_metrics":
            q_metrics,

        "sarsa_metrics":
            sarsa_metrics

    })


# ============================================================
# RUN FLASK
# ============================================================

if __name__ == "__main__":

    print(
        "\n========================================"
    )

    print(
        "SMART WASTE COLLECTION RL WEB APP"
    )

    print(
        "========================================"
    )

    print(
        "\nOpen:"
    )

    print(
        "http://127.0.0.1:5000"
    )


    app.run(
        debug=True
    )