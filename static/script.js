// ============================================================
// SMART WASTE COLLECTION RL — COMPLETE FRONTEND
// ============================================================


// ============================================================
// GLOBAL STATE
// ============================================================

let rows = 10;
let cols = 10;

let selectedTool = "depot";

let depot = null;
let bins = [];
let obstacles = [];

let environmentCreated = false;
let algorithmsTrained = false;

// Selected start position for simulation
let selectedStart = null;

// Simulation state
let simulationRunning = false;
let simulationFinished = false;
let autoRunInterval = null;

// Training results stored for comparison
let trainingData = {
    q_learning: { episodes: 0, final_reward: 0, states: 0 },
    sarsa: { episodes: 0, final_reward: 0, states: 0 }
};

// Local path tracking (mirrors backend)
let localQPath = [];
let localSarsaPath = [];


// ============================================================
// DOM ELEMENTS
// ============================================================

const mapElement =
    document.getElementById("map");

const simulationMapElement =
    document.getElementById("simulation-map");

const gridSizeSelect =
    document.getElementById("grid-size");

const createEnvironmentButton =
    document.getElementById("create-environment");

const trainButton =
    document.getElementById("train-button");

const startButton =
    document.getElementById("start-button");

const stepButton =
    document.getElementById("step-button");

const autoButton =
    document.getElementById("auto-button");

const resetButton =
    document.getElementById("reset-button");

const clearPathsButton =
    document.getElementById("clear-paths-button");

const algorithmSelect =
    document.getElementById("algorithm");

const builderMessage =
    document.getElementById("builder-message");

const trainingStatusBadge =
    document.getElementById("training-status");

const envBadge =
    document.getElementById("env-badge");

const simulationInstruction =
    document.getElementById("simulation-instruction");

const globalMessage =
    document.getElementById("global-message");

const trainingResultsPanel =
    document.getElementById("training-results");

const trainingMessage =
    document.getElementById("training-message");


// ============================================================
// INITIALIZE
// ============================================================

document.addEventListener("DOMContentLoaded", () => {
    createBuilderMap();
    updateBuilderStats();
    setToolActive("depot");
});


// ============================================================
// GRID SIZE CHANGE
// ============================================================

gridSizeSelect.addEventListener("change", () => {

    rows = parseInt(gridSizeSelect.value);
    cols = rows;

    // Reset everything
    depot = null;
    bins = [];
    obstacles = [];

    environmentCreated = false;
    algorithmsTrained = false;
    selectedStart = null;

    // Reset buttons
    trainButton.disabled = true;
    startButton.disabled = true;
    stepButton.disabled = true;
    autoButton.disabled = true;

    stopAutoRun();

    // Reset badges
    setEnvBadge("none");
    setTrainingBadge("none");

    trainingMessage.textContent =
        "Create an environment first, then click Train.";

    simulationInstruction.textContent =
        "Train the algorithms, then click a valid cell on the map to set the starting position.";

    trainingResultsPanel.style.display = "none";

    builderMessage.textContent =
        "Select a tool and click the grid to build your environment.";

    updateBuilderStats();

    createBuilderMap();

    // Clear simulation map
    simulationMapElement.innerHTML = "";

    document.getElementById("grid-count").textContent =
        `${rows} x ${cols}`;

});


// ============================================================
// TOOL SELECTION
// ============================================================

const toolButtons =
    document.querySelectorAll(".tool");

toolButtons.forEach(button => {
    button.addEventListener("click", () => {
        setToolActive(button.dataset.tool);
        updateBuilderMessage();
    });
});

function setToolActive(tool) {
    toolButtons.forEach(btn => btn.classList.remove("active"));
    selectedTool = tool;
    const activeBtn = document.querySelector(`.tool[data-tool="${tool}"]`);
    if (activeBtn) activeBtn.classList.add("active");
}


// ============================================================
// BUILDER MAP
// ============================================================

function createBuilderMap() {

    mapElement.innerHTML = "";
    mapElement.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;

    for (let row = 0; row < rows; row++) {
        for (let col = 0; col < cols; col++) {

            const cell = document.createElement("div");
            cell.classList.add("cell");
            cell.dataset.row = row;
            cell.dataset.col = col;

            cell.addEventListener("click", () => {
                handleBuilderCell(row, col, cell);
            });

            mapElement.appendChild(cell);
        }
    }

    renderBuilderMap();
}


// ============================================================
// HANDLE BUILDER CELL
// ============================================================

function handleBuilderCell(row, col, cell) {

    const position = [row, col];

    // DEPOT
    if (selectedTool === "depot") {
        depot = null;

        // Remove position from bins
        bins = bins.filter(bin => !samePosition(bin, position));

        // Remove position from obstacles
        obstacles = obstacles.filter(obs => !samePosition(obs, position));

        depot = position;
        builderMessage.textContent = `Depot placed at (${row}, ${col}).`;
        renderBuilderMap();
        updateBuilderStats();
        return;
    }

    // BIN
    if (selectedTool === "bin") {

        if (depot !== null && samePosition(depot, position)) {
            showMessage("A bin cannot be placed on the depot.");
            return;
        }

        if (containsPosition(obstacles, position)) {
            showMessage("A bin cannot be placed on an obstacle.");
            return;
        }

        if (containsPosition(bins, position)) {
            bins = bins.filter(bin => !samePosition(bin, position));
        } else {
            bins.push(position);
        }

        builderMessage.textContent = `${bins.length} waste bin(s) placed.`;
        renderBuilderMap();
        updateBuilderStats();
        return;
    }

    // OBSTACLE
    if (selectedTool === "obstacle") {

        if (depot !== null && samePosition(depot, position)) {
            showMessage("An obstacle cannot be placed on the depot.");
            return;
        }

        if (containsPosition(bins, position)) {
            showMessage("An obstacle cannot be placed on a waste bin.");
            return;
        }

        if (containsPosition(obstacles, position)) {
            obstacles = obstacles.filter(obs => !samePosition(obs, position));
        } else {
            obstacles.push(position);
        }

        builderMessage.textContent = `${obstacles.length} obstacle(s) placed.`;
        renderBuilderMap();
        updateBuilderStats();
        return;
    }

    // ERASE
    if (selectedTool === "erase") {
        if (depot !== null && samePosition(depot, position)) depot = null;
        bins = bins.filter(bin => !samePosition(bin, position));
        obstacles = obstacles.filter(obs => !samePosition(obs, position));
        builderMessage.textContent = `Cell (${row}, ${col}) cleared.`;
        renderBuilderMap();
        updateBuilderStats();
    }
}


// ============================================================
// RENDER BUILDER MAP
// ============================================================

function renderBuilderMap() {

    const cells = mapElement.querySelectorAll(".cell");

    cells.forEach(cell => {

        const row = parseInt(cell.dataset.row);
        const col = parseInt(cell.dataset.col);
        const position = [row, col];

        cell.className = "cell";
        cell.textContent = "";

        if (depot !== null && samePosition(depot, position)) {
            cell.classList.add("depot");
            cell.textContent = "🏭";
            return;
        }

        if (containsPosition(bins, position)) {
            cell.classList.add("bin");
            cell.textContent = "🗑️";
            return;
        }

        if (containsPosition(obstacles, position)) {
            cell.classList.add("obstacle");
            cell.textContent = "🧱";
        }

    });
}


// ============================================================
// UPDATE BUILDER STATS
// ============================================================

function updateBuilderStats() {
    document.getElementById("depot-count").textContent =
        depot === null ? "Not Set" : `(${depot[0]}, ${depot[1]})`;
    document.getElementById("bin-count").textContent = bins.length;
    document.getElementById("obstacle-count").textContent = obstacles.length;
    document.getElementById("grid-count").textContent = `${rows} x ${cols}`;
}


// ============================================================
// BUILDER TOOL MESSAGE
// ============================================================

function updateBuilderMessage() {
    const messages = {
        depot: "Click a cell to place the depot (starting area).",
        bin: "Click cells to add or remove waste bins.",
        obstacle: "Click cells to add or remove obstacles.",
        erase: "Click any cell to erase it."
    };
    builderMessage.textContent = messages[selectedTool] || "";
}


// ============================================================
// CREATE ENVIRONMENT
// ============================================================

createEnvironmentButton.addEventListener("click", async () => {

    if (depot === null) {
        showMessage("Please place a depot first.");
        return;
    }

    if (bins.length === 0) {
        showMessage("Please place at least one waste bin.");
        return;
    }

    createEnvironmentButton.disabled = true;
    createEnvironmentButton.textContent = "CREATING...";

    try {

        const response = await fetch("/configure", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                rows: rows,
                cols: cols,
                depot: depot,
                bins: bins,
                obstacles: obstacles
            })
        });

        const data = await response.json();

        if (!response.ok || !data.success) {
            throw new Error(data.message || "Failed to create environment.");
        }

        environmentCreated = true;
        algorithmsTrained = false;

        // Reset training
        trainingResultsPanel.style.display = "none";
        trainingMessage.textContent =
            "Environment created successfully! Click Train to train the algorithms.";

        trainButton.disabled = false;
        startButton.disabled = true;
        stepButton.disabled = true;
        autoButton.disabled = true;

        localQPath = [];
        localSarsaPath = [];

        setEnvBadge("ready");
        setTrainingBadge("none");

        simulationInstruction.textContent =
            "Environment created. Train the algorithms first.";

        showMessage(`Environment created: ${rows}x${cols} grid, ${bins.length} bins, ${obstacles.length} obstacles.`);

        // Build simulation map from server response
        createSimulationMap(data.environment);

    } catch (error) {
        console.error(error);
        showMessage(error.message);
    } finally {
        createEnvironmentButton.disabled = false;
        createEnvironmentButton.textContent = "CREATE ENVIRONMENT";
    }

});


// ============================================================
// TRAIN
// ============================================================

trainButton.addEventListener("click", async () => {

    if (!environmentCreated) {
        showMessage("Create an environment first.");
        return;
    }

    stopAutoRun();

    trainButton.disabled = true;
    trainButton.textContent = "Training... Please wait";
    trainingMessage.textContent = "Training Q-Learning and SARSA (1000 episodes each). This may take a moment...";

    showMessage("Training started. Please wait...");

    try {

        const response = await fetch("/train", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({})
        });

        const data = await response.json();

        if (!response.ok || !data.success) {
            throw new Error(data.message || "Training failed.");
        }

        algorithmsTrained = true;

        // Store training data
        const qRewards = data.q_learning.rewards || [];
        const sarsaRewards = data.sarsa.rewards || [];

        trainingData.q_learning = {
            episodes: data.q_learning.episodes || 0,
            final_reward: qRewards.length > 0
                ? Math.round(qRewards[qRewards.length - 1] * 10) / 10
                : 0,
            states: data.q_learning.states_learned || 0
        };

        trainingData.sarsa = {
            episodes: data.sarsa.episodes || 0,
            final_reward: sarsaRewards.length > 0
                ? Math.round(sarsaRewards[sarsaRewards.length - 1] * 10) / 10
                : 0,
            states: data.sarsa.states_learned || 0
        };

        // Update training results display
        document.getElementById("q-episodes").textContent = trainingData.q_learning.episodes;
        document.getElementById("q-final-reward").textContent = trainingData.q_learning.final_reward;
        document.getElementById("q-states").textContent = trainingData.q_learning.states;

        document.getElementById("sarsa-episodes").textContent = trainingData.sarsa.episodes;
        document.getElementById("sarsa-final-reward").textContent = trainingData.sarsa.final_reward;
        document.getElementById("sarsa-states").textContent = trainingData.sarsa.states;

        trainingResultsPanel.style.display = "grid";

        trainingMessage.textContent =
            "Training completed! Select an algorithm, click a cell on the map, then start the simulation.";

        setTrainingBadge("done");

        simulationInstruction.textContent =
            "Click a valid cell on the simulation map to select the starting position.";

        // Update comparison table with training data
        updateComparisonTableTraining();

        showMessage("Q-Learning and SARSA training completed successfully.");

        // Start button will enable after position is selected
        startButton.disabled = true;

    } catch (error) {
        console.error(error);
        trainingMessage.textContent = "Training failed. Check the console.";
        showMessage(error.message);
    } finally {
        trainButton.disabled = false;
        trainButton.textContent = "Train Q-Learning + SARSA";
    }

});


// ============================================================
// SIMULATION MAP
// ============================================================

function createSimulationMap(environment) {

    const gridSize = environment.grid_size;
    const simRows = gridSize[0];
    const simCols = gridSize[1];
    const envDepot = environment.depot;
    const envBins = environment.bins;
    const envObstacles = environment.obstacles;

    simulationMapElement.innerHTML = "";
    simulationMapElement.style.gridTemplateColumns =
        `repeat(${simCols}, 1fr)`;

    for (let row = 0; row < simRows; row++) {
        for (let col = 0; col < simCols; col++) {

            const cell = document.createElement("div");
            cell.classList.add("cell");
            cell.dataset.row = row;
            cell.dataset.col = col;

            const position = [row, col];

            // Depot
            if (samePosition(envDepot, position)) {
                cell.classList.add("depot");
                cell.textContent = "🏭";
            }

            // Bin
            const binEntry = Object.entries(envBins).find(
                ([name, pos]) => samePosition(pos, position)
            );
            if (binEntry) {
                cell.classList.add("bin");
                cell.textContent = "🗑️";
                cell.dataset.binName = binEntry[0];
            }

            // Obstacle
            const isObstacle = envObstacles.some(
                obs => samePosition(obs, position)
            );
            if (isObstacle) {
                cell.classList.add("obstacle");
                cell.textContent = "🧱";
            }

            // Click to select start
            cell.addEventListener("click", () => {
                selectStartingPosition(row, col);
            });

            simulationMapElement.appendChild(cell);
        }
    }

}


// ============================================================
// SELECT STARTING POSITION
// ============================================================

function selectStartingPosition(row, col) {

    if (!algorithmsTrained) {
        showMessage("Train the algorithms first.");
        return;
    }

    const position = [row, col];

    // Cannot start on obstacle
    if (containsPosition(obstacles, position)) {
        showMessage("You cannot start on an obstacle.");
        return;
    }

    selectedStart = position;

    renderSelectedStart();

    document.getElementById("position-status").textContent =
        `(${row}, ${col})`;

    simulationInstruction.textContent =
        `Starting position selected: (${row}, ${col}). Click Start Simulation.`;

    startButton.disabled = false;

    showMessage(`Starting position selected: (${row}, ${col})`);
}


// ============================================================
// RENDER SELECTED START
// ============================================================

function renderSelectedStart() {

    const cells = simulationMapElement.querySelectorAll(".cell");

    cells.forEach(cell => {
        cell.classList.remove("start-position");

        if (
            selectedStart !== null &&
            parseInt(cell.dataset.row) === selectedStart[0] &&
            parseInt(cell.dataset.col) === selectedStart[1]
        ) {
            cell.classList.add("start-position");
        }
    });
}


// ============================================================
// ALGORITHM SELECT CHANGE
// ============================================================

algorithmSelect.addEventListener("change", () => {
    const alg = algorithmSelect.value;
    document.getElementById("algorithm-status").textContent =
        alg === "q_learning" ? "Q-Learning" : "SARSA";
});


// ============================================================
// START SIMULATION
// ============================================================

startButton.addEventListener("click", async () => {

    if (selectedStart === null) {
        showMessage("Please select a starting position on the map.");
        return;
    }

    stopAutoRun();

    const algorithm = algorithmSelect.value;

    try {

        const response = await fetch("/start", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                algorithm: algorithm,
                row: selectedStart[0],
                col: selectedStart[1]
            })
        });

        const data = await response.json();

        if (!response.ok || !data.success) {
            throw new Error(data.message || "Could not start simulation.");
        }

        simulationRunning = true;
        simulationFinished = false;

        // Update local path tracking from backend
        localQPath = data.q_path || [];
        localSarsaPath = data.sarsa_path || [];

        stepButton.disabled = false;
        autoButton.disabled = false;
        startButton.disabled = true;

        // Update algorithm status
        document.getElementById("algorithm-status").textContent =
            algorithm === "q_learning" ? "Q-Learning" : "SARSA";

        updateSimulationDisplay(data);

        simulationInstruction.textContent =
            `${algorithm === "q_learning" ? "Q-Learning" : "SARSA"} simulation started from (${selectedStart[0]}, ${selectedStart[1]}).`;

        showMessage("Simulation started. Use Next Step or Auto Run.");

    } catch (error) {
        console.error(error);
        showMessage(error.message);
    }

});


// ============================================================
// NEXT STEP
// ============================================================

stepButton.addEventListener("click", async () => {
    await performStep();
});

async function performStep() {

    if (!simulationRunning) {
        showMessage("Start the simulation first.");
        return false;
    }

    if (simulationFinished) {
        showMessage("Simulation has already finished. Reset to run again.");
        stopAutoRun();
        return false;
    }

    try {

        const response = await fetch("/step", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({})
        });

        const data = await response.json();

        if (!response.ok || !data.success) {
            throw new Error(data.message || "Step failed.");
        }

        // Update local path tracking
        localQPath = data.q_path || [];
        localSarsaPath = data.sarsa_path || [];

        updateSimulationDisplay(data);

        if (data.done) {
            simulationFinished = true;
            simulationRunning = false;

            stepButton.disabled = true;
            autoButton.disabled = true;

            simulationInstruction.textContent =
                "Simulation completed! Reset to run again or switch algorithms.";

            showMessage("Simulation completed!");
            stopAutoRun();

            // Update comparison table with simulation results
            updateComparisonTableSim(data);

            return false;
        }

        return true;

    } catch (error) {
        console.error(error);
        showMessage(error.message);
        stopAutoRun();
        return false;
    }
}


// ============================================================
// AUTO RUN
// ============================================================

let autoRunning = false;

autoButton.addEventListener("click", () => {

    if (!simulationRunning) {
        showMessage("Start the simulation first.");
        return;
    }

    if (simulationFinished) {
        showMessage("Simulation already finished. Reset to run again.");
        return;
    }

    if (autoRunning) {
        stopAutoRun();
    } else {
        startAutoRun();
    }

});

function startAutoRun() {
    autoRunning = true;
    autoButton.textContent = "⏸ Pause";
    autoButton.classList.remove("btn-purple");
    autoButton.classList.add("btn-gray");
    stepButton.disabled = true;

    autoRunInterval = setInterval(async () => {
        const cont = await performStep();
        if (!cont) {
            stopAutoRun();
        }
    }, 300);
}

function stopAutoRun() {
    autoRunning = false;
    autoButton.textContent = "Auto Run";
    autoButton.classList.add("btn-purple");
    autoButton.classList.remove("btn-gray");

    if (autoRunInterval) {
        clearInterval(autoRunInterval);
        autoRunInterval = null;
    }

    if (simulationRunning && !simulationFinished) {
        stepButton.disabled = false;
    }
}


// ============================================================
// RESET SIMULATION
// ============================================================

resetButton.addEventListener("click", async () => {

    stopAutoRun();

    try {

        if (environmentCreated) {

            const response = await fetch("/reset", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({})
            });

            const data = await response.json();

            if (data.success) {
                localQPath = data.q_path || [];
                localSarsaPath = data.sarsa_path || [];
            }
        }

        selectedStart = null;
        simulationRunning = false;
        simulationFinished = false;

        startButton.disabled = algorithmsTrained ? false : true;
        stepButton.disabled = true;
        autoButton.disabled = true;

        // Re-enable start only after position selection
        startButton.disabled = true;

        document.getElementById("position-status").textContent = "Not selected";
        document.getElementById("step-status").textContent = "0";
        document.getElementById("reward-status").textContent = "0";
        document.getElementById("total-reward-status").textContent = "0";
        document.getElementById("bins-status").textContent = `0 / ${bins.length}`;
        document.getElementById("current-position-status").textContent = "-";

        renderSelectedStart();

        // Redraw simulation map without vehicle/path overlays
        if (environmentCreated) {
            rerenderSimMap();
        }

        simulationInstruction.textContent =
            algorithmsTrained
                ? "Click a valid cell on the simulation map to select the starting position."
                : "Train the algorithms first.";

        showMessage("Simulation reset. Select a starting position.");

    } catch (error) {
        console.error(error);
        showMessage(error.message);
    }

});


// ============================================================
// CLEAR PATHS
// ============================================================

clearPathsButton.addEventListener("click", async () => {

    stopAutoRun();

    try {

        const response = await fetch("/clear_paths", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({})
        });

        const data = await response.json();

        if (data.success) {
            localQPath = [];
            localSarsaPath = [];

            rerenderSimMap();
            updateComparisonTableSim(null);

            showMessage("All paths cleared.");
        }

    } catch (error) {
        console.error(error);
        showMessage(error.message);
    }

});


// ============================================================
// UPDATE SIMULATION DISPLAY
// ============================================================

function updateSimulationDisplay(data) {

    const position = data.position;

    // Update stats panel
    if (position) {
        document.getElementById("current-position-status").textContent =
            `(${position[0]}, ${position[1]})`;
    }

    if (data.steps !== undefined) {
        document.getElementById("step-status").textContent = data.steps;
    }

    if (data.reward !== undefined) {
        document.getElementById("reward-status").textContent =
            typeof data.reward === "number" ? data.reward.toFixed(1) : data.reward;
    }

    if (data.total_reward !== undefined) {
        document.getElementById("total-reward-status").textContent =
            typeof data.total_reward === "number"
                ? data.total_reward.toFixed(1)
                : data.total_reward;
    }

    const collectedBins = data.collected_bins || [];
    document.getElementById("bins-status").textContent =
        `${collectedBins.length} / ${bins.length}`;

    // Render the simulation grid with paths + vehicle
    if (position) {
        renderSimGrid(
            position,
            collectedBins,
            data.q_path || localQPath,
            data.sarsa_path || localSarsaPath
        );
    }

    // Update comparison after each step
    if (data.q_metrics || data.sarsa_metrics) {
        updateComparisonTableSim(data);
    }
}


// ============================================================
// RENDER SIMULATION GRID
// The path visualization: paths persist, vehicle moves on top.
// Q-path = blue, SARSA-path = red, both = purple
// ============================================================

function renderSimGrid(vehiclePosition, collectedBins, qPath, sarsaPath) {

    const cells = simulationMapElement.querySelectorAll(".cell");

    // Build lookup sets for O(1) checking
    const qSet = new Set(qPath.map(p => `${p[0]},${p[1]}`));
    const sarsaSet = new Set(sarsaPath.map(p => `${p[0]},${p[1]}`));

    // Build collected bins set
    const collectedSet = new Set(collectedBins);

    cells.forEach(cell => {

        const row = parseInt(cell.dataset.row);
        const col = parseInt(cell.dataset.col);
        const posKey = `${row},${col}`;

        // Preserve structural classes
        const isDepot = cell.classList.contains("depot");
        const isBin = cell.classList.contains("bin");
        const isObstacle = cell.classList.contains("obstacle");
        const binName = cell.dataset.binName;

        // Remove simulation overlay classes
        cell.classList.remove(
            "vehicle",
            "start-position",
            "q-path",
            "sarsa-path",
            "both-path",
            "collected-bin"
        );

        // Reset text for non-structural cells
        if (!isDepot && !isBin && !isObstacle) {
            cell.textContent = "";
        }

        // ─── Vehicle ────────────────────────────────────────────
        if (
            vehiclePosition &&
            row === vehiclePosition[0] &&
            col === vehiclePosition[1]
        ) {
            cell.classList.add("vehicle");
            cell.textContent = "🚛";
            return;
        }

        // ─── Collected bin ──────────────────────────────────────
        if (isBin && binName && collectedSet.has(binName)) {
            cell.classList.add("collected-bin");
            cell.textContent = "✅";
        }

        // ─── Path overlays (don't apply to obstacles or depot) ──
        if (!isObstacle && !isDepot) {
            const inQ = qSet.has(posKey);
            const inSarsa = sarsaSet.has(posKey);

            if (inQ && inSarsa) {
                cell.classList.add("both-path");
            } else if (inQ) {
                cell.classList.add("q-path");
            } else if (inSarsa) {
                cell.classList.add("sarsa-path");
            }
        }

        // ─── Depot - preserve appearance ────────────────────────
        if (isDepot) {
            cell.textContent = "🏭";
        }

        // ─── Bin - if not collected, keep emoji ──────────────────
        if (isBin && !(binName && collectedSet.has(binName))) {
            cell.textContent = "🗑️";
        }

    });
}


// ============================================================
// RERENDER SIM MAP — without vehicle or path overlays
// Used after reset / clear
// ============================================================

function rerenderSimMap() {

    const cells = simulationMapElement.querySelectorAll(".cell");

    cells.forEach(cell => {

        const isDepot = cell.classList.contains("depot");
        const isBin = cell.classList.contains("bin");
        const isObstacle = cell.classList.contains("obstacle");

        cell.classList.remove(
            "vehicle",
            "start-position",
            "q-path",
            "sarsa-path",
            "both-path",
            "collected-bin"
        );

        if (isDepot) {
            cell.textContent = "🏭";
        } else if (isBin) {
            cell.textContent = "🗑️";
        } else if (isObstacle) {
            cell.textContent = "🧱";
        } else {
            cell.textContent = "";
        }
    });

    // Rerender stored paths if any
    if (localQPath.length > 0 || localSarsaPath.length > 0) {
        renderSimGrid(null, [], localQPath, localSarsaPath);
    }
}


// ============================================================
// COMPARISON TABLE — TRAINING DATA
// ============================================================

function updateComparisonTableTraining() {

    document.getElementById("cmp-q-episodes").textContent =
        trainingData.q_learning.episodes || "-";

    document.getElementById("cmp-sarsa-episodes").textContent =
        trainingData.sarsa.episodes || "-";

    document.getElementById("cmp-q-train-reward").textContent =
        trainingData.q_learning.final_reward !== undefined
            ? trainingData.q_learning.final_reward
            : "-";

    document.getElementById("cmp-sarsa-train-reward").textContent =
        trainingData.sarsa.final_reward !== undefined
            ? trainingData.sarsa.final_reward
            : "-";
}


// ============================================================
// COMPARISON TABLE — SIMULATION RESULTS
// ============================================================

function updateComparisonTableSim(data) {

    if (!data) {
        // Clear sim columns
        const simIds = [
            "cmp-q-steps", "cmp-q-reward", "cmp-q-path",
            "cmp-q-bins", "cmp-q-completed",
            "cmp-sarsa-steps", "cmp-sarsa-reward", "cmp-sarsa-path",
            "cmp-sarsa-bins", "cmp-sarsa-completed"
        ];
        simIds.forEach(id => {
            const el = document.getElementById(id);
            if (el) el.textContent = "-";
        });
        return;
    }

    const qm = data.q_metrics;
    const sm = data.sarsa_metrics;

    if (qm) {
        document.getElementById("cmp-q-steps").textContent =
            qm.steps !== undefined ? qm.steps : "-";
        document.getElementById("cmp-q-reward").textContent =
            qm.reward !== undefined ? (Math.round(qm.reward * 10) / 10) : "-";
        document.getElementById("cmp-q-path").textContent =
            qm.path_length !== undefined ? qm.path_length : "-";
        document.getElementById("cmp-q-bins").textContent =
            qm.bins_collected !== undefined ? qm.bins_collected : "-";
        document.getElementById("cmp-q-completed").textContent =
            qm.completed !== undefined ? (qm.completed ? "Yes ✅" : "No") : "-";
    }

    if (sm) {
        document.getElementById("cmp-sarsa-steps").textContent =
            sm.steps !== undefined ? sm.steps : "-";
        document.getElementById("cmp-sarsa-reward").textContent =
            sm.reward !== undefined ? (Math.round(sm.reward * 10) / 10) : "-";
        document.getElementById("cmp-sarsa-path").textContent =
            sm.path_length !== undefined ? sm.path_length : "-";
        document.getElementById("cmp-sarsa-bins").textContent =
            sm.bins_collected !== undefined ? sm.bins_collected : "-";
        document.getElementById("cmp-sarsa-completed").textContent =
            sm.completed !== undefined ? (sm.completed ? "Yes ✅" : "No") : "-";
    }
}


// ============================================================
// BADGE HELPERS
// ============================================================

function setEnvBadge(state) {
    envBadge.className = "badge";
    if (state === "none") {
        envBadge.classList.add("badge-gray");
        envBadge.textContent = "NO ENVIRONMENT";
    } else if (state === "ready") {
        envBadge.classList.add("badge-blue");
        envBadge.textContent = "ENVIRONMENT READY";
    }
}

function setTrainingBadge(state) {
    trainingStatusBadge.className = "badge";
    if (state === "none") {
        trainingStatusBadge.classList.add("badge-gray");
        trainingStatusBadge.textContent = "NOT TRAINED";
    } else if (state === "training") {
        trainingStatusBadge.classList.add("badge-orange");
        trainingStatusBadge.textContent = "TRAINING...";
    } else if (state === "done") {
        trainingStatusBadge.classList.add("badge-green");
        trainingStatusBadge.textContent = "TRAINED";
    }
}


// ============================================================
// POSITION HELPERS
// ============================================================

function samePosition(pos1, pos2) {
    if (!pos1 || !pos2) return false;
    return pos1[0] === pos2[0] && pos1[1] === pos2[1];
}

function containsPosition(positions, target) {
    return positions.some(pos => samePosition(pos, target));
}


// ============================================================
// TOAST MESSAGE
// ============================================================

let messageTimeout = null;

function showMessage(message) {
    globalMessage.textContent = message;
    globalMessage.classList.add("show");

    if (messageTimeout) clearTimeout(messageTimeout);

    messageTimeout = setTimeout(() => {
        globalMessage.classList.remove("show");
    }, 4000);
}