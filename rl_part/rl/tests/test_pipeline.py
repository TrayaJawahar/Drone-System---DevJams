"""
test_pipeline.py
----------------
Unit tests for the Geo-Network Map data preparation pipeline.

Tests cover:
  - Grid builder (bounding box, cell conversion, round-trip GPS ↔ grid)
  - Obstacle processor (polygon extraction, STRtree intersection)
  - Network estimator (path-loss model, Haversine distance)
  - Map assembler (column validation, parquet output)
  - OSM query builder (correct tag separation)
"""

import math
import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, Polygon

# Project root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from data.pipeline.grid_builder import (
    compute_bounding_box, build_grid, latlon_to_grid, grid_to_latlon
)
from data.pipeline.obstacle_processor import (
    _is_obstacle_feature, extract_osm_polygons, assign_obstacles_to_grid
)
from data.pipeline.network_estimator import (
    haversine_distance, compute_path_loss_db, estimate_rssi, estimate_network_for_grid
)
from data.pipeline.osm_collector import build_overpass_query
from data.pipeline.map_assembler import assemble_geo_network_map


# ─────────────────────────────────────────────
# Grid Builder Tests
# ─────────────────────────────────────────────

class TestGridBuilder:
    def test_bounding_box_dimensions(self):
        """Bounding box should cover approximately the requested area."""
        bbox = compute_bounding_box(12.9352, 77.6245, 5.0, 5.0)
        lat_span_km = (bbox["max_lat"] - bbox["min_lat"]) * 111.32
        lon_span_km = (bbox["max_lon"] - bbox["min_lon"]) * 111.32 * math.cos(math.radians(12.9352))
        assert abs(lat_span_km - 5.0) < 0.1, f"Latitude span wrong: {lat_span_km}"
        assert abs(lon_span_km - 5.0) < 0.1, f"Longitude span wrong: {lon_span_km}"

    def test_grid_cell_count(self):
        """5km x 5km at 100m resolution should give 50x50 = 2500 cells."""
        bbox = compute_bounding_box(12.9352, 77.6245, 5.0, 5.0)
        grid_df, gw, gh = build_grid(bbox, 100)
        assert gw == 50
        assert gh == 50
        assert len(grid_df) == 2500

    def test_grid_required_columns(self):
        """Grid DataFrame must have all required columns."""
        bbox = compute_bounding_box(12.9352, 77.6245, 5.0, 5.0)
        grid_df, _, _ = build_grid(bbox, 100)
        for col in ["cell_id", "grid_x", "grid_y", "latitude", "longitude"]:
            assert col in grid_df.columns, f"Missing column: {col}"

    def test_grid_unique_cells(self):
        """cell_id and (grid_x, grid_y) pairs must be unique."""
        bbox = compute_bounding_box(12.9352, 77.6245, 5.0, 5.0)
        grid_df, _, _ = build_grid(bbox, 100)
        assert grid_df["cell_id"].is_unique
        assert not grid_df.duplicated(subset=["grid_x", "grid_y"]).any()

    def test_center_maps_to_grid_center(self):
        """Center lat/lon should map to approximately (25, 25) in a 50x50 grid."""
        bbox = compute_bounding_box(12.9352, 77.6245, 5.0, 5.0)
        _, gw, gh = build_grid(bbox, 100)
        gx, gy = latlon_to_grid(12.9352, 77.6245, bbox, gw, gh)
        assert abs(gx - 25) <= 1, f"gx={gx} expected ~25"
        assert abs(gy - 25) <= 1, f"gy={gy} expected ~25"

    def test_latlon_roundtrip(self):
        """Converting GPS → grid → GPS should give approximately the original coordinates."""
        bbox = compute_bounding_box(12.9352, 77.6245, 5.0, 5.0)
        _, gw, gh = build_grid(bbox, 100)
        gx, gy = latlon_to_grid(12.94, 77.62, bbox, gw, gh)
        lat_back, lon_back = grid_to_latlon(gx, gy, bbox, gw, gh)
        # Should be within 1 cell size (100m ≈ 0.0009 degrees)
        assert abs(lat_back - 12.94) < 0.002, f"Latitude roundtrip error: {abs(lat_back - 12.94)}"
        assert abs(lon_back - 77.62) < 0.002, f"Longitude roundtrip error: {abs(lon_back - 77.62)}"

    def test_out_of_bounds_returns_minus_one(self):
        """Coordinates outside the bounding box should return (-1, -1)."""
        bbox = compute_bounding_box(12.9352, 77.6245, 5.0, 5.0)
        _, gw, gh = build_grid(bbox, 100)
        gx, gy = latlon_to_grid(0.0, 0.0, bbox, gw, gh)
        assert gx == -1
        assert gy == -1


# ─────────────────────────────────────────────
# Obstacle Processor Tests
# ─────────────────────────────────────────────

class TestObstacleProcessor:
    def _make_config(self):
        return {"osm": {"obstacle_tags": {"buildings": True, "water": True, "industrial": False}}}

    def test_building_tag_is_obstacle(self):
        assert _is_obstacle_feature({"building": "yes"}, self._make_config()) is True

    def test_road_tag_is_not_obstacle(self):
        assert _is_obstacle_feature({"highway": "primary", "building": "yes"}, self._make_config()) is False

    def test_water_tag_is_obstacle(self):
        assert _is_obstacle_feature({"natural": "water"}, self._make_config()) is True

    def test_plain_way_not_obstacle(self):
        assert _is_obstacle_feature({"name": "Some Path"}, self._make_config()) is False

    def test_osm_polygon_extraction(self):
        """Minimal OSM data with one building way should produce one obstacle polygon."""
        osm_data = {
            "elements": [
                {
                    "type": "way",
                    "id": 1,
                    "tags": {"building": "yes"},
                    "geometry": [
                        {"lat": 12.93, "lon": 77.62},
                        {"lat": 12.935, "lon": 77.62},
                        {"lat": 12.935, "lon": 77.625},
                        {"lat": 12.93, "lon": 77.625},
                        {"lat": 12.93, "lon": 77.62},
                    ]
                }
            ]
        }
        polys, roads = extract_osm_polygons(osm_data, self._make_config())
        assert len(polys) == 1
        assert len(roads) == 0

    def test_road_way_not_in_obstacles(self):
        """Highway ways should go to road_lines, not obstacle_polys."""
        osm_data = {
            "elements": [
                {
                    "type": "way",
                    "id": 2,
                    "tags": {"highway": "primary"},
                    "geometry": [
                        {"lat": 12.93, "lon": 77.62},
                        {"lat": 12.935, "lon": 77.625},
                    ]
                }
            ]
        }
        polys, roads = extract_osm_polygons(osm_data, self._make_config())
        assert len(polys) == 0
        assert len(roads) == 1

    def test_grid_obstacle_assignment(self):
        """Cell whose center falls inside a building polygon should be marked as obstacle."""
        bbox = compute_bounding_box(12.9352, 77.6245, 5.0, 5.0)
        grid_df, gw, gh = build_grid(bbox, 100)

        # Create a building polygon around grid center
        building = Polygon([
            (77.623, 12.934), (77.627, 12.934),
            (77.627, 12.937), (77.623, 12.937),
            (77.623, 12.934)
        ])
        grid_df = assign_obstacles_to_grid(grid_df, [building], [], self._make_config())

        inside = grid_df[grid_df["is_obstacle"] == True]
        assert len(inside) > 0, "At least one cell should be inside the building polygon"
        assert "obstacle_distance" in grid_df.columns


# ─────────────────────────────────────────────
# Network Estimator Tests
# ─────────────────────────────────────────────

class TestNetworkEstimator:
    def test_haversine_known_distance(self):
        """Bangalore center to a point 1km east should be approximately 1000m."""
        lat, lon = 12.9352, 77.6245
        # 1km east (approximately 0.009 degrees at this latitude)
        lon2 = lon + 1.0 / (111.32 * math.cos(math.radians(lat)))
        d = haversine_distance(lat, lon, lat, lon2)
        assert abs(d - 1000) < 50, f"Expected ~1000m, got {d:.1f}m"

    def test_haversine_self_distance_is_zero(self):
        d = haversine_distance(12.9352, 77.6245, 12.9352, 77.6245)
        assert d == pytest.approx(0.0, abs=0.01)

    def test_path_loss_increases_with_distance(self):
        """Path loss should be greater at 1000m than at 100m."""
        pl100 = compute_path_loss_db(100, 2100.0, 3.5, 8.0)
        pl1000 = compute_path_loss_db(1000, 2100.0, 3.5, 8.0)
        assert pl1000 > pl100

    def test_rssi_decreases_with_distance(self):
        """RSSI should be lower (worse) at greater distances."""
        rssi_near = estimate_rssi(43.0, compute_path_loss_db(200, 2100, 3.5, 8.0))
        rssi_far = estimate_rssi(43.0, compute_path_loss_db(2000, 2100, 3.5, 8.0))
        assert rssi_near > rssi_far

    def test_network_estimation_with_towers(self):
        """Grid cells near towers should have non-NaN RSSI values."""
        bbox = compute_bounding_box(12.9352, 77.6245, 5.0, 5.0)
        grid_df, gw, gh = build_grid(bbox, 100)
        grid_df["is_obstacle"] = False

        towers = [
            {"radio": "LTE", "mcc": "404", "mnn": "1",
             "lat": 12.935, "lon": 77.625, "range_m": 5000, "samples": 10, "avg_signal": None}
        ]
        config = {
            "network_estimation": {
                "path_loss_exponent": 3.5,
                "noise_floor_dbm": -100.0,
                "min_rssi_dbm": -120.0,
                "tx_power": {"LTE": 43.0},
                "frequency_mhz": {"LTE": 2100.0},
                "shadow_margin_db": {"LTE": 8.0},
                "bandwidth_mhz": {"LTE": 20.0},
            },
            "towers": {"search_radius_meters": 5000},
        }
        result = estimate_network_for_grid(grid_df, towers, config)
        assert "rssi" in result.columns
        assert result["rssi"].notna().any(), "Some cells near the tower should have RSSI"
        assert result["latency"].isna().all(), "Latency must remain NaN (no measurements)"
        assert result["packet_loss"].isna().all(), "Packet loss must remain NaN"


# ─────────────────────────────────────────────
# OSM Query Builder Tests
# ─────────────────────────────────────────────

class TestOSMQueryBuilder:
    def _make_bbox(self):
        return {"min_lat": 12.91, "min_lon": 77.60, "max_lat": 12.96, "max_lon": 77.65}

    def _make_config(self):
        return {"osm": {"endpoint": "https://overpass-api.de/api/interpreter",
                        "obstacle_tags": {"buildings": True, "water": True, "industrial": False},
                        "timeout": 120, "max_retries": 3}}

    def test_query_contains_building(self):
        q = build_overpass_query(self._make_bbox(), self._make_config())
        assert '"building"' in q

    def test_query_contains_water(self):
        q = build_overpass_query(self._make_bbox(), self._make_config())
        assert '"natural"="water"' in q

    def test_query_excludes_industrial(self):
        q = build_overpass_query(self._make_bbox(), self._make_config())
        assert '"industrial"' not in q

    def test_query_bbox_correct(self):
        q = build_overpass_query(self._make_bbox(), self._make_config())
        # Python formats 77.60 as "77.6" — check both key coordinates
        assert "12.91" in q
        assert "77.6" in q
        assert "12.96" in q


# ─────────────────────────────────────────────
# Map Assembler Tests
# ─────────────────────────────────────────────

class TestMapAssembler:
    def _make_minimal_grid(self) -> pd.DataFrame:
        bbox = compute_bounding_box(12.9352, 77.6245, 1.0, 1.0)
        grid_df, _, _ = build_grid(bbox, 100)
        grid_df["is_obstacle"] = False
        grid_df["obstacle_distance"] = 9999.0
        grid_df["elevation"] = None
        grid_df["slope"] = None
        for col in ["rssi", "rsrp", "sinr", "latency", "packet_loss", "throughput",
                    "nearest_tower_distance", "network_data_confidence", "tower_count_nearby"]:
            grid_df[col] = None
        grid_df["network_source"] = "missing"
        return grid_df

    def test_assemble_creates_parquet(self, tmp_path):
        grid_df = self._make_minimal_grid()
        bbox = compute_bounding_box(12.9352, 77.6245, 1.0, 1.0)
        parquet_path = str(tmp_path / "test.parquet")
        meta_path = str(tmp_path / "test.json")
        config = {"region": {"name": "Test", "center_latitude": 12.9352, "center_longitude": 77.6245,
                              "width_km": 1.0, "height_km": 1.0}}

        result = assemble_geo_network_map(grid_df, bbox, 10, 10, 100, config, parquet_path, meta_path)

        assert os.path.exists(parquet_path)
        assert os.path.exists(meta_path)
        loaded = pd.read_parquet(parquet_path)
        assert len(loaded) == len(grid_df)
        assert "is_obstacle" in loaded.columns

    def test_metadata_json_structure(self, tmp_path):
        grid_df = self._make_minimal_grid()
        bbox = compute_bounding_box(12.9352, 77.6245, 1.0, 1.0)
        parquet_path = str(tmp_path / "test2.parquet")
        meta_path = str(tmp_path / "test2.json")
        config = {"region": {"name": "Test Region", "center_latitude": 12.9352,
                              "center_longitude": 77.6245, "width_km": 1.0, "height_km": 1.0}}

        assemble_geo_network_map(grid_df, bbox, 10, 10, 100, config, parquet_path, meta_path)

        with open(meta_path) as f:
            meta = json.load(f)

        assert "created_at" in meta
        assert "region" in meta
        assert "bounding_box" in meta
        assert "grid" in meta
        assert "data_sources" in meta
        assert meta["region"]["name"] == "Test Region"
