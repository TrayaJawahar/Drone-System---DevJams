import numpy as np
import pandas as pd

def calculate_route_metrics(route: list[tuple[int, int]], cell_lookup: dict, net_calc) -> dict:
    """
    Computes geographical, battery, and network quality metrics for a given route.
    
    Args:
        route: List of grid cell coordinates [(x1, y1), (x2, y2), ...]
        cell_lookup: Dict lookup (x, y) -> cell attributes dict
        net_calc: NetworkQualityCalculator instance
        
    Returns:
        dict: Summary of path and network statistics.
    """
    if not route:
        return {
            "success": False,
            "steps": 0,
            "distance": 0.0,
            "avg_network_quality": 0.0,
            "avg_rssi": -110.0,
            "avg_latency": 500.0,
            "avg_packet_loss": 1.0,
            "outage_steps": 0,
            "battery_used": 100.0
        }

    total_steps = len(route) - 1
    total_dist = 0.0
    total_battery_used = 0.0
    
    net_scores = []
    rssis = []
    latencies = []
    packet_losses = []
    outage_steps = 0

    # Battery consumption costs (synchronized with Env)
    h_cost = 0.5
    d_cost = 0.7
    slope_factor = 0.05

    for idx, cell in enumerate(route):
        cell_data = cell_lookup.get(cell, {})
        
        # Calculate cell network metrics
        net_info = net_calc.calculate(cell_data)
        net_score = net_info["network_score"]
        net_scores.append(net_score)
        
        if net_score < net_calc.outage_threshold:
            outage_steps += 1
            
        rssi = cell_data.get("rssi")
        if rssi is not None and not pd.isnull(rssi):
            rssis.append(rssi)
            
        latency = cell_data.get("latency")
        if latency is not None and not pd.isnull(latency):
            latencies.append(latency)
            
        packet_loss = cell_data.get("packet_loss")
        if packet_loss is not None and not pd.isnull(packet_loss):
            packet_losses.append(packet_loss)

        # Accumulate route distance and energy cost (from second step onwards)
        if idx > 0:
            p_prev = route[idx - 1]
            dx = abs(cell[0] - p_prev[0])
            dy = abs(cell[1] - p_prev[1])
            is_diag = (dx > 0 and dy > 0)
            
            # Step distance
            step_d = np.sqrt(dx**2 + dy**2)
            total_dist += step_d
            
            # Step energy
            base_cost = d_cost if is_diag else h_cost
            slope = cell_data.get("slope", 0.0)
            if pd.isnull(slope):
                slope = 0.0
            slope_cost = max(0.0, slope) * slope_factor
            
            total_battery_used += (base_cost + slope_cost)

    return {
        "success": True,
        "steps": total_steps,
        "distance": float(total_dist),
        "avg_network_quality": float(np.mean(net_scores)) if net_scores else 0.0,
        "avg_rssi": float(np.mean(rssis)) if rssis else -110.0,
        "avg_latency": float(np.mean(latencies)) if latencies else 500.0,
        "avg_packet_loss": float(np.mean(packet_losses)) if packet_losses else 1.0,
        "outage_steps": outage_steps,
        "battery_used": float(total_battery_used)
    }
