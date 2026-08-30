import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from rl.environment.network_quality import NetworkQualityCalculator
from rl.utils.logger import setup_logger

logger = setup_logger("visualization")

def setup_matplotlib():
    """
    Configures headless Matplotlib parameters.
    """
    plt.switch_backend('Agg')  # Headless mode: no window display
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

def plot_network_heatmap(df: pd.DataFrame, save_path: str = "logs/network_heatmap.png"):
    """
    Plots a 2D spatial heatmap of cellular network quality.
    """
    setup_matplotlib()
    
    # Calculate Network Quality for all cells to display
    net_calc = NetworkQualityCalculator()
    
    # Grid sizes
    max_x = int(df["grid_x"].max())
    max_y = int(df["grid_y"].max())
    
    # Create grid arrays
    grid_net = np.zeros((max_x + 1, max_y + 1))
    grid_obs = np.zeros((max_x + 1, max_y + 1))
    
    for _, row in df.iterrows():
        gx = int(row["grid_x"])
        gy = int(row["grid_y"])
        is_obs = bool(row["is_obstacle"])
        
        if is_obs:
            grid_obs[gx, gy] = 1.0
            grid_net[gx, gy] = np.nan
        else:
            cell_data = row.to_dict()
            net_info = net_calc.calculate(cell_data)
            grid_net[gx, gy] = net_info["network_score"]
            
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Custom colormap for network quality (red-yellow-green)
    cmap = plt.cm.RdYlGn
    cmap.set_bad(color='black')  # Obstacles as black
    
    # Plot heatmap
    im = ax.imshow(grid_net.T, origin='lower', cmap=cmap, interpolation='nearest', aspect='auto')
    
    # Overlay obstacles explicitly for clarity
    obs_x, obs_y = np.where(grid_obs == 1.0)
    ax.scatter(obs_x, obs_y, color='black', marker='s', s=8, label='Obstacle')
    
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Consolidated Network Quality Score [0-1]')
    
    ax.set_title("Geo-Network Quality Map")
    ax.set_xlabel("Grid X")
    ax.set_ylabel("Grid Y")
    ax.grid(False)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved network quality heatmap to {save_path}")

def compare_routes(
    df: pd.DataFrame,
    ppo_route: list[tuple[int, int]],
    astar_route: list[tuple[int, int]],
    start: tuple[int, int],
    goal: tuple[int, int],
    save_path: str = "logs/route_comparison.png"
):
    """
    Plots the planned PPO route vs A* route overlaying the network quality and obstacle cells.
    """
    setup_matplotlib()
    
    net_calc = NetworkQualityCalculator()
    
    # Grid sizes
    max_x = int(df["grid_x"].max())
    max_y = int(df["grid_y"].max())
    
    # Setup grid representation
    grid_net = np.zeros((max_x + 1, max_y + 1))
    grid_obs = np.zeros((max_x + 1, max_y + 1))
    
    for _, row in df.iterrows():
        gx = int(row["grid_x"])
        gy = int(row["grid_y"])
        is_obs = bool(row["is_obstacle"])
        
        if is_obs:
            grid_obs[gx, gy] = 1.0
            grid_net[gx, gy] = np.nan
        else:
            cell_data = row.to_dict()
            net_info = net_calc.calculate(cell_data)
            grid_net[gx, gy] = net_info["network_score"]
            
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Colormap
    cmap = plt.cm.RdYlGn
    cmap.set_bad(color='#d3d3d3')  # Light gray background for obstacle cells
    
    # Plot heatmap background
    im = ax.imshow(grid_net.T, origin='lower', cmap=cmap, alpha=0.6, interpolation='nearest', aspect='auto')
    
    # Obstacles overlay
    obs_x, obs_y = np.where(grid_obs == 1.0)
    ax.scatter(obs_x, obs_y, color='black', marker='s', s=10, label='Obstacles')

    # A* Route (Shortest Path)
    if astar_route:
        ax_coords = np.array(astar_route)
        ax.plot(ax_coords[:, 0], ax_coords[:, 1], color='red', linestyle='--', linewidth=2.5, marker='o', markersize=4, label='A* Route (Shortest Path)')

    # PPO Route (Network-Aware)
    if ppo_route:
        ppo_coords = np.array(ppo_route)
        ax.plot(ppo_coords[:, 0], ppo_coords[:, 1], color='blue', linestyle='-', linewidth=3.0, marker='^', markersize=5, label='PPO Route (Network-Aware)')

    # Start and Goal points
    ax.scatter(start[0], start[1], color='green', marker='o', s=150, zorder=5, label='Start Location')
    ax.scatter(goal[0], goal[1], color='purple', marker='*', s=250, zorder=5, label='Goal Location')

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Network Quality (0 = Weak, 1 = Strong)')
    
    ax.set_title("A* Shortest Path vs PPO Network-Aware Route planning")
    ax.set_xlabel("Grid X Coordinate")
    ax.set_ylabel("Grid Y Coordinate")
    ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
    ax.grid(True, linestyle=':', alpha=0.5)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved route comparison plot to {save_path}")
