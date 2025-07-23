import json
import os
import numpy as np
from typing import List, Dict, Any


class CalibrationModel:
    """Model to manage calibration parameters and blob detection data"""
    
    def __init__(self):
        self.default_params = {
            'min_area': 30,
            'max_area': 400,
            'min_circularity': 0.5,
            'shots_per_series': 3,
            'number_of_targets': 1
        }
        self.current_params = self.default_params.copy()
        self.fixed_target_positions = []
        self.recent_blob_detections = []
        
    def update_parameter(self, param_name: str, value: Any):
        """Update a calibration parameter"""
        if param_name in self.current_params:
            self.current_params[param_name] = value
            
    def get_parameter(self, param_name: str) -> Any:
        """Get a calibration parameter value"""
        return self.current_params.get(param_name, None)
        
    def get_all_parameters(self) -> Dict[str, Any]:
        """Get all current parameters"""
        return self.current_params.copy()
        
    def set_blob_positions(self, positions: List[List[float]]):
        """Set detected blob positions"""
        self.fixed_target_positions = positions
        
    def get_blob_positions(self) -> List[List[float]]:
        """Get current blob positions"""
        return self.fixed_target_positions
        
    def save_calibration_data(self, filename: str = "calibration_data.json"):
        """Save calibration parameters and blob positions to JSON file"""
        try:
            # Ensure data directory exists
            data_dir = "data"
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
                
            filepath = os.path.join(data_dir, filename)
            
            # Prepare data for saving
            save_data = {
                'parameters': self.current_params,
                'fixed_target_positions': self.fixed_target_positions,
                'timestamp': self._get_timestamp()
            }
            
            # Save to JSON
            with open(filepath, 'w') as f:
                json.dump(save_data, f, indent=2)
                
            return filepath
            
        except Exception as e:
            raise Exception(f"Error saving calibration data: {e}")
            
    def load_calibration_data(self, filename: str = "calibration_data.json") -> bool:
        """Load calibration parameters from JSON file"""
        try:
            data_dir = "data"
            filepath = os.path.join(data_dir, filename)
            
            if not os.path.exists(filepath):
                return False
                
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            # Load parameters
            if 'parameters' in data:
                self.current_params.update(data['parameters'])
                
            # Load blob positions
            if 'fixed_target_positions' in data:
                self.fixed_target_positions = data['fixed_target_positions']
                
            return True
            
        except Exception as e:
            print(f"Error loading calibration data: {e}")
            return False
            
    def _get_timestamp(self) -> str:
        """Get current timestamp as string"""
        from datetime import datetime
        return datetime.now().isoformat() 