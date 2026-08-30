import os
import csv
import json
import pandas as pd
import numpy as np
from rl.evaluation.test_scenarios import load_test_scenarios
from rl.inference.route_inference import RouteInferenceEngine
from rl.evaluation.comparison import AStarPlanner, calculate_route_metrics
from rl.utils.logger import setup_logger

logger = setup_logger("evaluate")

def run_evaluation():
    logger.info("Starting final PPO vs A* evaluation...")

    # 1. Load test scenarios
    try:
        test_missions = load_test_scenarios()
    except FileNotFoundError:
        logger.error("Test scenarios not found. Please run training first to generate them.")
        return

    # 2. Load PPO Inference Engine
    try:
        engine = RouteInferenceEngine()
    except Exception as e:
        logger.error(f"Failed to load RouteInferenceEngine: {e}")
        return

    # 3. Setup A* Planner
    astar = AStarPlanner(engine.raw_env.mission_gen.grid_free_mask)
    cell_lookup = engine.raw_env.cell_lookup
    net_calc = engine.raw_env.net_calc

    results = []

    # Run evaluations
    logger.info(f"Evaluating {len(test_missions)} unseen test scenarios...")
    for idx, (start, goal) in enumerate(test_missions):
        logger.info(f"Scenario {idx + 1}/{len(test_missions)}: {start} -> {goal}")
        
        # PPO Inference
        ppo_res = engine.generate_route(start, goal)
        
        # A* Planning
        astar_route = astar.plan(start, goal)
        astar_res = calculate_route_metrics(astar_route, cell_lookup, net_calc)

        # Record Comparison
        scenario_data = {
            "scenario_index": idx + 1,
            "start": list(start),
            "goal": list(goal),
            "ppo_success": ppo_res["success"],
            "ppo_steps": ppo_res["total_steps"],
            "ppo_distance": ppo_res["total_distance"],
            "ppo_net_quality": ppo_res["average_network_quality"],
            "ppo_outage_steps": ppo_res["outage_steps"],
            "ppo_battery_used": ppo_res["battery_used"],
            "ppo_failure_reason": ppo_res["failure_reason"],
            
            "astar_success": astar_res["success"],
            "astar_steps": astar_res["steps"],
            "astar_distance": astar_res["distance"],
            "astar_net_quality": astar_res["avg_network_quality"],
            "astar_outage_steps": astar_res["outage_steps"],
            "astar_battery_used": astar_res["battery_used"]
        }
        
        # Pull detailed network averages
        if ppo_res["success"] and len(ppo_res["route_cells"]) > 0:
            calc_ppo = calculate_route_metrics(ppo_res["route_cells"], cell_lookup, net_calc)
            scenario_data["ppo_rssi"] = calc_ppo["avg_rssi"]
            scenario_data["ppo_latency"] = calc_ppo["avg_latency"]
            scenario_data["ppo_packet_loss"] = calc_ppo["avg_packet_loss"]
        else:
            scenario_data["ppo_rssi"] = -110.0
            scenario_data["ppo_latency"] = 500.0
            scenario_data["ppo_packet_loss"] = 1.0

        scenario_data["astar_rssi"] = astar_res["avg_rssi"]
        scenario_data["astar_latency"] = astar_res["avg_latency"]
        scenario_data["astar_packet_loss"] = astar_res["avg_packet_loss"]
        
        results.append(scenario_data)

    # 4. Save results to JSON & CSV
    eval_dir = "rl/evaluation"
    os.makedirs(eval_dir, exist_ok=True)
    
    json_path = os.path.join(eval_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    logger.info(f"Saved detailed results to {json_path}")

    df_results = pd.DataFrame(results)
    csv_path = os.path.join(eval_dir, "results.csv")
    df_results.to_csv(csv_path, index=False)
    logger.info(f"Saved results CSV to {csv_path}")

    # 5. Calculate Aggregate Averages
    ppo_successes = [r["ppo_success"] for r in results]
    astar_successes = [r["astar_success"] for r in results]
    
    # Filter successful runs for averages to avoid distorting path comparisons
    ppo_success_routes = [r for r in results if r["ppo_success"]]
    astar_success_routes = [r for r in results if r["astar_success"]]

    summary = {
        "PPO": {
            "Success Rate": f"{np.mean(ppo_successes) * 100.0:.1f}%",
            "Avg Distance": f"{np.mean([r['ppo_distance'] for r in ppo_success_routes]):.2f}" if ppo_success_routes else "N/A",
            "Avg Steps": f"{np.mean([r['ppo_steps'] for r in ppo_success_routes]):.1f}" if ppo_success_routes else "N/A",
            "Avg Network Quality": f"{np.mean([r['ppo_net_quality'] for r in ppo_success_routes]):.3f}" if ppo_success_routes else "N/A",
            "Avg RSSI": f"{np.mean([r['ppo_rssi'] for r in ppo_success_routes]):.1f} dBm" if ppo_success_routes else "N/A",
            "Avg Latency": f"{np.mean([r['ppo_latency'] for r in ppo_success_routes]):.1f} ms" if ppo_success_routes else "N/A",
            "Avg Packet Loss": f"{np.mean([r['ppo_packet_loss'] for r in ppo_success_routes]) * 100.0:.1f}%" if ppo_success_routes else "N/A",
            "Outage Steps": f"{np.mean([r['ppo_outage_steps'] for r in ppo_success_routes]):.1f}" if ppo_success_routes else "N/A",
            "Battery Used": f"{np.mean([r['ppo_battery_used'] for r in ppo_success_routes]):.1f}%" if ppo_success_routes else "N/A"
        },
        "A*": {
            "Success Rate": f"{np.mean(astar_successes) * 100.0:.1f}%",
            "Avg Distance": f"{np.mean([r['astar_distance'] for r in astar_success_routes]):.2f}" if astar_success_routes else "N/A",
            "Avg Steps": f"{np.mean([r['astar_steps'] for r in astar_success_routes]):.1f}" if astar_success_routes else "N/A",
            "Avg Network Quality": f"{np.mean([r['astar_net_quality'] for r in astar_success_routes]):.3f}" if astar_success_routes else "N/A",
            "Avg RSSI": f"{np.mean([r['astar_rssi'] for r in astar_success_routes]):.1f} dBm" if astar_success_routes else "N/A",
            "Avg Latency": f"{np.mean([r['astar_latency'] for r in astar_success_routes]):.1f} ms" if astar_success_routes else "N/A",
            "Avg Packet Loss": f"{np.mean([r['astar_packet_loss'] for r in astar_success_routes]) * 100.0:.1f}%" if astar_success_routes else "N/A",
            "Outage Steps": f"{np.mean([r['astar_outage_steps'] for r in astar_success_routes]):.1f}" if astar_success_routes else "N/A",
            "Battery Used": f"{np.mean([r['astar_battery_used'] for r in astar_success_routes]):.1f}%" if astar_success_routes else "N/A"
        }
    }

    # Print Comparison Table
    print("\n" + "=" * 55)
    print("A* SHORTEST PATH VS PPO NETWORK-AWARE PATH")
    print("=" * 55)
    print(f"{'Metric':<25} | {'A* Planner':<12} | {'PPO Agent':<12}")
    print("-" * 55)
    for metric in summary["PPO"].keys():
        a_val = summary["A*"][metric]
        p_val = summary["PPO"][metric]
        print(f"{metric:<25} | {a_val:<12} | {p_val:<12}")
    print("=" * 55 + "\n")

if __name__ == "__main__":
    run_evaluation()
