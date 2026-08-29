<<<<<<< HEAD
# Network-Aware Reinforcement Learning for Autonomous Drone Route Planning

This repository contains the complete, modular Reinforcement Learning (RL) training and evaluation subsystem for planning autonomous drone routes using a preprocessed **Geo-Network Map**. 

The goal of the system is to train a Proximal Policy Optimization (PPO) agent to balance reaching its goal, avoiding physical obstacles, maintaining reliable network connectivity (minimizing latency, packet loss, and outages), and reducing travel distance and energy consumption.

---

## 1. Project Architecture

The reinforcement learning pipeline is structured as follows:

```
                  [ GEO-NETWORK MAP ] (Parquet + Metadata JSON)
                           │
                           ▼
                  [ DATA VALIDATION ]
                           │
                           ▼
                 [ FEATURE PROCESSOR ] (Standardizes & Scaler Joblib)
                           │
                           ▼
              [ MISSION SCENARIO SPLIT ] (Train / Validation / Test)
                           │
                           ▼
               [ GYMNASIUM ENVIRONMENT ] (DroneNetworkEnv)
                ├── Actions Space (8 discrete directions)
                ├── State Space (2D coordinates, Obstacles, Network Quality, Battery)
                └── Reward Engine (Multiobjective cost weights)
                           │
                           ▼
                     [ PPO AGENT ] (Stable-Baselines3)
                ├── Monitor & VecNormalize Wrappers
                └── Curriculum Learning Scheduler
                           │
                           ▼
                 [ PERIODIC CALLBACKS ] (Checkpointing, TensorBoard)
                           │
                           ▼
                    [ BEST MODEL ]
                           │
                 ┌─────────┴─────────┐
                 ▼                   ▼
         [ PPO EVALUATION ]     [ A* PATH BASELINE ]
                 └─────────┬─────────┘
                           ▼
               [ COMPARISON & REPORTS ] (CSV/JSON summaries + Heatmap Plotting)
```

---

## 2. Geo-Network Map Data Format

The training pipeline loads real-world geographic, obstacle, and cellular network data from the following locations:
* **Parquet Map**: `data/processed/geo_network_map.parquet`
* **Metadata JSON**: `data/processed/geo_network_metadata.json`

### Grid Cell Attributes
Each grid cell in the parquet map contains:
* **Spatial & Physical**: `cell_id` (int), `grid_x` (int), `grid_y` (int), `latitude` (float), `longitude` (float), `is_obstacle` (bool), `obstacle_distance` (float), `elevation` (float), `slope` (float)
* **Network Measurements**: `rssi` (float), `rsrp` (float), `sinr` (float), `latency` (float), `packet_loss` (float), `throughput` (float)
* **Infrastructure & Metadata**: `nearest_tower_distance` (float), `network_data_confidence` (float), `network_source` (str: `"measured"` or `"interpolated"`)

### Data Policy
* **Real Data Only**: The training pipeline enforces that `data/processed/geo_network_map.parquet` and `metadata.json` exist. If missing, it immediately stops and displays a clear error:
  `ERROR: Real Geo-Network Map not found. Please build data/processed/geo_network_map.parquet before training.`
* **Missing Value Handling**: Not all network metrics are available for all cells. The pipeline automatically calculates metric coverage and masks missing values (providing value + availability flag `0` / `1`) to keep the PPO input space stable.

---

## 3. State and Action Spaces

### Observation Space (State)
The agent receives a fixed-size normalized vector representing the current state of the environment:
1. **Position & Navigational**: Current `x, y` (normalized), Goal `x, y` (normalized), Deltas `dx, dy`, and direct Euclidean distance to the goal.
2. **Obstacle Proximity Radar**: A lookahead distance radar checking in 8 directions (North, South, East, West, and diagonals) for the nearest obstacle, plus the overall nearest obstacle distance.
3. **Signal Attributes**: Consolidates RSSI, RSRP, SINR, Latency, Packet Loss, and Throughput values paired with binary **availability masks** (1.0 if measured, 0.0 if missing/null).
4. **Neighbor Network Awareness**: (If `include_neighbor_network` is enabled) Appends the consolidated network quality score and data confidence for neighboring grid cells (North, South, East, West). This allows the agent to evaluate spatial trends in network strength before deciding on movement.
5. **Battery Capacity**: Remaining battery ratio `[0.0, 1.0]`.
6. **Terrain Info**: Elevation and slope metrics combined with availability masks.

### Action Space
Discrete action space of size 8 representing grid transitions:
* `0`: NORTH (0, 1)
* `1`: SOUTH (0, -1)
* `2`: EAST (1, 0)
* `3`: WEST (-1, 0)
* `4`: NORTH_EAST (1, 1)
* `5`: NORTH_WEST (-1, 1)
* `6`: SOUTH_EAST (1, -1)
* `7`: SOUTH_WEST (-1, -1)

Cardinal movements consume a base energy of `0.5`, while diagonal movements consume `0.7`. Slopes add a deterministic energy cost penalty: `slope_cost = max(0, slope) * slope_factor`.

---

## 4. Reward Engine

The `RewardCalculator` evaluates every step against multiple goals:

$$\text{Total Reward} = W_{progress} \times \text{Goal Progress} + W_{network} \times \text{Network Quality} + W_{safety} \times \text{Obstacle Proximity} - W_{movement} \times \text{Movement Cost} - W_{energy} \times \text{Energy Cost} - W_{outage} \times \text{Connectivity Outage} + W_{goal} \times \text{Goal Reached} - W_{collision} \times \text{Collision}$$

* **Goal Progress**: Compares $\text{Distance}_{\text{prev}} - \text{Distance}_{\text{curr}}$. Positive reward for moving closer; negative penalty for moving away.
* **Network Quality**: Uses a normalized consolidated quality score from `NetworkQualityCalculator` (renormalizes weights if signal, latency, packet loss, or throughput metrics are missing).
* **Connectivity Outage**: Triggers when $\text{Network Quality} < \text{Threshold}$. Outage penalty increases with duration ($\text{consecutive outage steps}$) to strongly discourage sustained communication loss.
* **Obstacle Safety Margin**: Proximity potential field penalty when the drone enters a 3-cell safety margin around obstacles.
* **Movement Cost & Energy Cost**: Standard small step penalties to minimize route distance and battery draw.
* **Goal Reward & Collision Penalty**: terminal boundary rewards.

---

## 5. Training Subsystem

The training code (`rl/training/train_ppo.py`) integrates:
* **Reproducibility**: Sets seeds across random, NumPy, PyTorch, Gym, and Stable-Baselines3.
* **Wrappers**: Vectorizes the Gymnasium environment, wraps with `Monitor` to capture stats, and wraps in `VecNormalize` to compute running means and standard deviations of observations.
* **TensorBoard logging**: Tracks rewards, success rate, collision rates, battery status, network quality, average latency, and average packet loss.
* **Custom callbacks**:
  * `CustomMetricsCallback`: Automatically records episode stats and saves them to `logs/training_metrics.csv`.
  * `CheckpointWithVecNormalizeCallback`: Periodically saves PPO models and corresponding `VecNormalize` statistics.
  * `CurriculumCallback`: Gradually increases difficulty (Stage 1: short distance; Stage 2: medium distance; Stage 3: long distance).

---

## 6. A* Baseline and Comparison

To demonstrate the benefits of network-aware RL path planning, the evaluation subsystem (`rl/evaluation/evaluate.py`) compares the trained PPO agent against a standard A* path planner on identical, unseen test missions.

The A* planner optimizes solely for the shortest obstacle-free path (using Octile distances on the 8-directional grid) and ignores all cellular parameters. PPO is expected to take slightly longer, network-aware paths to bypass cellular dead zones.

---

## 7. Configuration Details

Configurations are kept inside `rl/config/rl_config.yaml`. 

This defines:
* Battery parameters
* Reward weights (easily tunable)
* PPO parameters (epochs, clip range, learning rate, GAE lambda, etc.)
* Training steps
* Curriculum boundaries

---

## 8. Commands

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Validate Dataset Integrity
Checks cells, uniqueness, coordinates, obstacles, and network coverages:
```bash
python -m rl.data.validation
```

### Run gymnasium checks
Verifies Gym API compliance:
```bash
python -m rl.environment.drone_network_env
```

### Run Unit Tests
Executes unit tests covering state builder, reward functions, env steps, and short PPO trainings:
```bash
python -m pytest rl/tests/
```

### Train, Evaluate, and Compare
Runs the entire training, test set evaluation, A* comparison, and generates comparison plots:
```bash
python run_training.py
```

### Run TensorBoard
```bash
tensorboard --logdir rl/logs/
```

### Run Route Inference
Performs route generation for custom start/goal coordinates:
```bash
python -m rl.inference.route_inference
```
#   R L - B a s e d - D r o n e -  
 
=======
# Mission Pilot AI — Lovable project export

This ZIP contains the core source code exported from the Lovable project:

**SITUATION AWARE DRONE SYSTEM / Mission Pilot AI**

Project ID: `aca8ab7b-7a16-4e6e-9048-f3e89af5257f`

## Included

- Interactive Leaflet/OpenStreetMap mission map
- Start and destination coordinate selection
- Bidirectional map/input synchronization
- Coordinate validation
- 50 km operational-range validation
- Mission summary and direct-distance calculation
- `/mission` analysis page
- TanStack Router / TanStack Start configuration
- Tailwind CSS design system and technical dashboard styling

## Run

Requirements: Node.js 20+ (or a compatible Bun setup).

```bash
npm install
npm run dev
```

Then open the local URL printed by Vite.

The map uses OpenStreetMap tiles, so internet access is required for map tiles.

## Note

The ZIP intentionally focuses on the application-specific source and required configuration rather than the large set of unused generated shadcn/ui component files present in the Lovable repository.
>>>>>>> e196950b2079677db3dfc8852638daf1eee6ec95
