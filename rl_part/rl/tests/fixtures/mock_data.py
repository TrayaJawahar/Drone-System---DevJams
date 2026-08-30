import os
import tempfile
import json
import pandas as pd
import numpy as np
import pytest

@pytest.fixture(scope="session")
def mock_dataset_paths():
    """
    Creates a temporary 10x10 Geo-Network Map parquet and JSON metadata.
    Returns:
        map_path, metadata_path
    """
    # 1. Create a 10x10 grid of cells
    cells = []
    cell_id = 0
    
    # Simple layout:
    # 10x10 grid
    # Obstacles at column 4 (except row 2 and 7)
    for x in range(10):
        for y in range(10):
            # Check obstacle condition
            is_obstacle = (x == 4) and (y != 2) and (y != 7)
            
            # Cellular network measurements
            # Stronger signals on the right side, weaker on the left side
            # Let's say column 8 has RSSI -45 (strong), column 0 has -105 (weak)
            rssi = -105.0 + (x * 7.0)
            latency = 450.0 - (x * 40.0)
            packet_loss = 0.45 - (x * 0.05)
            throughput = 1.0 + (x * 5.0)
            
            # Make columns 1 and 2 missing RSSI to test availability masks
            if x in [1, 2]:
                rssi = None
                latency = None
                
            cell = {
                "cell_id": cell_id,
                "grid_x": x,
                "grid_y": y,
                "latitude": 37.7749 + (x * 0.0001),
                "longitude": -122.4194 + (y * 0.0001),
                "is_obstacle": is_obstacle,
                "obstacle_distance": float(abs(x - 4)) if not is_obstacle else 0.0,
                "elevation": float(10.0 + x + y),
                "slope": float(x * 0.5),
                "rssi": rssi,
                "rsrp": -130.0 + (x * 6.0) if x not in [1, 2] else None,
                "sinr": -5.0 + (x * 3.0) if x not in [1, 2] else None,
                "latency": latency,
                "packet_loss": packet_loss,
                "throughput": throughput,
                "nearest_tower_distance": float(100.0 - (x * 8.0)),
                "network_data_confidence": 0.8,
                "network_source": "measured" if x > 2 else "interpolated"
            }
            cells.append(cell)
            cell_id += 1
            
    df = pd.DataFrame(cells)
    
    # 2. Metadata dict
    metadata = {
        "grid_width": 10,
        "grid_height": 10,
        "resolution_meters": 10.0
    }

    # Save to temp files
    temp_dir = tempfile.mkdtemp()
    map_path = os.path.join(temp_dir, "geo_network_map.parquet")
    metadata_path = os.path.join(temp_dir, "geo_network_metadata.json")
    
    df.to_parquet(map_path, index=False)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)
        
    yield map_path, metadata_path
    
    # Cleanup temp directory after test session
    try:
        if os.path.exists(map_path):
            os.remove(map_path)
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
        os.rmdir(temp_dir)
    except Exception:
        pass
