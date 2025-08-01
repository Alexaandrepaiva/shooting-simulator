#!/usr/bin/env python3
"""
Test script to verify multi-target functionality fixes
"""

import json
import os
import numpy as np
from typing import List, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_blob_grouping():
    """Test blob grouping logic for multiple targets"""
    print("=== Testing Blob Grouping Logic ===")
    
    # Sample blob positions (simulating 2 targets with 4 blobs each)
    # Target 1: Left side blobs
    # Target 2: Right side blobs
    sample_blobs = [
        [100, 100],  # Target 1 - top-left
        [200, 100],  # Target 1 - top-right  
        [200, 200],  # Target 1 - bottom-right
        [100, 200],  # Target 1 - bottom-left
        [400, 100],  # Target 2 - top-left
        [500, 100],  # Target 2 - top-right
        [500, 200],  # Target 2 - bottom-right
        [400, 200],  # Target 2 - bottom-left
    ]
    
    print(f"Input blobs: {sample_blobs}")
    
    # Test simple target grouping (fallback method)
    try:
        from controllers.calibration_controller import CalibrationController
        controller = CalibrationController(None)
        
        grouped_blobs = controller._simple_target_grouping(sample_blobs, 2)
        print(f"Grouped blobs: {grouped_blobs}")
        
        # Verify grouping
        target1_blobs = grouped_blobs[:4]
        target2_blobs = grouped_blobs[4:8]
        
        print(f"Target 1 blobs: {target1_blobs}")
        print(f"Target 2 blobs: {target2_blobs}")
        
        # Check that targets are separated
        target1_x = [blob[0] for blob in target1_blobs]
        target2_x = [blob[0] for blob in target2_blobs]
        
        if max(target1_x) < min(target2_x):
            print("✅ Targets are properly separated (no X overlap)")
        else:
            print("❌ Targets overlap in X direction")
            
    except Exception as e:
        print(f"❌ Error testing blob grouping: {e}")

def test_target_region_separation():
    """Test target region separation validation"""
    print("\n=== Testing Target Region Separation ===")
    
    # Test case 1: Well separated targets
    separated_blobs = [
        [100, 100], [200, 100], [200, 200], [100, 200],  # Target 1
        [400, 100], [500, 100], [500, 200], [400, 200],  # Target 2
    ]
    
    print("Test case 1: Well separated targets")
    test_region_overlap(separated_blobs, 2)
    
    # Test case 2: Overlapping targets (should trigger warning)
    overlapping_blobs = [
        [100, 100], [250, 100], [250, 200], [100, 200],  # Target 1
        [200, 100], [350, 100], [350, 200], [200, 200],  # Target 2 (overlaps)
    ]
    
    print("\nTest case 2: Overlapping targets")
    test_region_overlap(overlapping_blobs, 2)

def test_region_overlap(blob_positions: List[List[float]], number_of_targets: int):
    """Test region overlap detection"""
    try:
        from controllers.calibration_controller import CalibrationController
        controller = CalibrationController(None)
        
        controller._validate_target_separation(blob_positions, number_of_targets)
        
    except Exception as e:
        print(f"❌ Error testing region separation: {e}")

def test_shot_validation():
    """Test shot validation within target regions"""
    print("\n=== Testing Shot Validation ===")
    
    # Define target bounds
    target_bounds = {
        'min_x': 100,
        'max_x': 200,
        'min_y': 100,
        'max_y': 200
    }
    
    # Test shots
    test_shots = [
        {'absolute_position': {'x': 150, 'y': 150}},  # Inside target
        {'absolute_position': {'x': 250, 'y': 150}},  # Outside target (right)
        {'absolute_position': {'x': 150, 'y': 250}},  # Outside target (below)
        {'absolute_position': {'x': 110, 'y': 110}},  # Inside target (near edge)
    ]
    
    try:
        from models.shot_detection import ShotDetectionModel
        model = ShotDetectionModel()
        
        validated_shots = model._validate_shots_in_target_region(test_shots, target_bounds, 1)
        
        print(f"Input shots: {len(test_shots)}")
        print(f"Validated shots: {len(validated_shots)}")
        
        if len(validated_shots) == 2:  # Should validate shots 1 and 4
            print("✅ Shot validation working correctly")
        else:
            print("❌ Shot validation not working as expected")
            
    except Exception as e:
        print(f"❌ Error testing shot validation: {e}")

def check_calibration_data():
    """Check current calibration data for issues"""
    print("\n=== Checking Current Calibration Data ===")
    
    try:
        calibration_file = os.path.join("data", "calibration_data.json")
        if not os.path.exists(calibration_file):
            print("No calibration data found")
            return
            
        with open(calibration_file, 'r') as f:
            data = json.load(f)
            
        blob_positions = data.get('fixed_target_positions', [])
        num_targets = data.get('parameters', {}).get('number_of_targets', 1)
        
        print(f"Number of targets: {num_targets}")
        print(f"Number of blob positions: {len(blob_positions)}")
        print(f"Expected blob positions: {num_targets * 4}")
        
        if len(blob_positions) == num_targets * 4:
            print("✅ Correct number of blob positions")
            
            # Check target separation
            for target_idx in range(num_targets):
                start_idx = target_idx * 4
                end_idx = start_idx + 4
                target_blobs = blob_positions[start_idx:end_idx]
                
                x_coords = [blob[0] for blob in target_blobs]
                y_coords = [blob[1] for blob in target_blobs]
                
                print(f"Target {target_idx + 1} region: "
                      f"X=[{min(x_coords):.1f}, {max(x_coords):.1f}], "
                      f"Y=[{min(y_coords):.1f}, {max(y_coords):.1f}]")
                      
        else:
            print("❌ Incorrect number of blob positions")
            
    except Exception as e:
        print(f"❌ Error checking calibration data: {e}")

def main():
    """Run all tests"""
    print("Multi-Target Shooting Simulator Fix Verification")
    print("=" * 50)
    
    test_blob_grouping()
    test_target_region_separation()
    test_shot_validation()
    check_calibration_data()
    
    print("\n" + "=" * 50)
    print("Test completed! Check the logs for any issues.")

if __name__ == "__main__":
    main() 