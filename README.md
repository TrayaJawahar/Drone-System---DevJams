Network-Aware Reinforcement Learning for Autonomous Drone Route Planning: 
This repository contains a modular Reinforcement Learning (RL) framework for autonomous drone route planning using a preprocessed Geo-Network Map. The system trains a Proximal Policy Optimization (PPO) agent to navigate efficiently while balancing multiple objectives:

Reaching the destination successfully
Avoiding obstacles and unsafe terrain
Maintaining reliable network connectivity
Minimizing latency, packet loss, and communication outages
Reducing travel distance and energy consumption

Project Architecture:
                 GEO-NETWORK MAP
        (Parquet Dataset + Metadata JSON)
                          │
                          ▼
                  Data Validation
                          │
                          ▼
                 Feature Processing
          (Scaling & Normalization)
                          │
                          ▼
              Mission Scenario Split
          (Train / Validation / Test)
                          │
                          ▼
              DroneNetworkEnv (Gymnasium)
        ├── 8-Direction Action Space
        ├── Network-Aware State Space
        └── Multi-Objective Reward Engine
                          │
                          ▼
                     PPO Agent
        ├── Monitor & VecNormalize
        └── Curriculum Learning
                          │
                          ▼
               Training Callbacks
      (Checkpointing, Metrics, TensorBoard)
                          │
                          ▼
                     Best Model
                          │
                ┌─────────┴─────────┐
                ▼                   ▼
          PPO Evaluation      A* Baseline
                └─────────┬─────────┘
                          ▼
              Performance Comparison
        (Reports, Metrics, Visualizations)
Geo-Network Map Dataset

The RL environment operates on a preprocessed geographic and network dataset.

Required Files:
data/processed/geo_network_map.parquet
data/processed/geo_network_metadata.json
Grid Cell Attributes
Spatial & Terrain Features
cell_id
grid_x, grid_y
latitude, longitude
is_obstacle
obstacle_distance
elevation
slope
Network Metrics
rssi
rsrp
sinr
latency
packet_loss
throughput
Infrastructure Metadata
nearest_tower_distance
network_data_confidence
network_source (measured or interpolated)
Data Policy

The training pipeline strictly requires real geo-network data.

If the dataset is unavailable, execution stops with:

ERROR: Real Geo-Network Map not found.
Please build data/processed/geo_network_map.parquet before training.

Missing network measurements are handled using:
Coverage analysis
Availability masks (1 = available, 0 = missing)
Stable observation-space representation
Environment Design
Observation Space

The agent receives a normalized state vector containing:

Navigation Features:
Current position (x, y)
Goal position (x, y)
Relative offsets (dx, dy)
Euclidean distance to goal
Obstacle Awareness
Obstacle radar in 8 directions
Nearest obstacle distance
Network Features
RSSI
RSRP
SINR
Latency
Packet loss
Throughput

Each metric is paired with an availability mask to account for missing data.

Neighbor Network Awareness (Optional)

When enabled, neighboring cells contribute:

Network quality score
Data confidence values

This allows the agent to anticipate connectivity trends before moving.

Battery Status
Remaining battery ratio [0,1]
Terrain Information
Elevation
Slope
Availability masks
Action Space

The environment uses an 8-direction discrete action space:
Action	Movement
0	North
1	South
2	East
3	West
4	North-East
5	North-West
6	South-East
7	South-West
Energy Consumption
Cardinal movement: 0.5 units
Diagonal movement: 0.7 units

Additional terrain cost:

slope_cost = max(0, slope) × slope_factor
Reward Function

The reward system balances navigation efficiency, connectivity, safety, and energy usage.

$$ Reward = W_{progress} \cdot Progress + W_{network} \cdot NetworkQuality + W_{safety} \cdot Safety - W_{movement} \cdot MovementCost - W_{energy} \cdot EnergyCost - W_{outage} \cdot OutagePenalty + W_{goal} \cdot GoalReward - W_{collision} \cdot CollisionPenalty $$
Reward Components
Component	Description
Goal Progress	Rewards movement toward the destination
Network Quality	Rewards strong and reliable connectivity
Outage Penalty	Penalizes sustained communication loss
Obstacle Safety	Penalizes movement near obstacles
Movement Cost	Encourages shorter routes
Energy Cost	Encourages battery-efficient navigation
Goal Reward	Reward for successfully reaching the target
Collision Penalty	Penalty for obstacle collisions

The network-quality term is computed using a consolidated score derived from available signal and communication metrics.

Training Pipeline

The training subsystem (rl/training/train_ppo.py) includes:

Reproducibility
Random seed initialization
NumPy seeding
PyTorch seeding
Gymnasium seeding
Stable-Baselines3 seeding
Environment Wrappers
Monitor
VecNormalize
TensorBoard Logging

Tracks:

Episode rewards
Success rate
Collision rate
Battery usage
Network quality
Latency
Packet loss
Custom Callbacks
CustomMetricsCallback

Records episode statistics and exports:

logs/training_metrics.csv
CheckpointWithVecNormalizeCallback

Saves:

PPO checkpoints
VecNormalize statistics
CurriculumCallback

Training difficulty increases progressively:

Stage	Mission Length
Stage 1	Short-range
Stage 2	Medium-range
Stage 3	Long-range
Evaluation and Baseline Comparison

The evaluation module compares the trained PPO agent against a traditional A* path planner on unseen test missions.

A* Planner
Optimizes shortest obstacle-free path
Uses octile distance heuristics
Ignores network conditions
PPO Agent
Considers connectivity, safety, battery usage, and travel efficiency
May choose longer routes to avoid network dead zones

This comparison highlights the benefits of network-aware route planning.

Configuration

All configurable parameters are stored in:

rl/config/rl_config.yaml

Configuration categories include:

Battery parameters
Reward weights
PPO hyperparameters
Training duration
Curriculum settings
Usage
Install Dependencies
pip install -r requirements.txt
Validate Dataset
python -m rl.data.validation

Validates:

Cell integrity
Coordinate consistency
Obstacle information
Network metric coverage
Verify Gymnasium Environment
python -m rl.environment.drone_network_env
Run Unit Tests
python -m pytest rl/tests/
Train, Evaluate, and Compare
python run_training.py

This command:

Trains the PPO agent
Evaluates performance on test missions
Runs A* baseline comparison
Generates metrics and visualizations
Launch TensorBoard
tensorboard --logdir rl/logs/
Route Inference

Generate routes for custom start and goal coordinates:

python -m rl.inference.route_inference
Key Objective

The primary objective of this project is to demonstrate that a network-aware reinforcement learning agent can generate safer and more reliable autonomous drone routes by jointly optimizing navigation efficiency, connectivity quality, obstacle avoidance, and energy consumption in real-world geo-network environments.
