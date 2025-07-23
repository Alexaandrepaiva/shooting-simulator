import cv2
import numpy as np
import json
import os
import logging
from typing import List, Tuple, Dict, Any
from datetime import datetime


class HomographyModel:
    """Model for managing homography transformations between camera and target coordinates"""
    
    def __init__(self):
        self.homographies_file = os.path.join("data", "homographies.json")
        self.homographies = {}  # target_number -> homography matrix
        
    def calculate_and_save_homographies(self, calibration_data: Dict) -> bool:
        """
        Calculate homographies for all targets and save to file
        
        Args:
            calibration_data: Calibration data containing blob positions and parameters
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            blob_positions = calibration_data.get('fixed_target_positions', [])
            parameters = calibration_data.get('parameters', {})
            number_of_targets = parameters.get('number_of_targets', 1)
            
            if len(blob_positions) < number_of_targets * 4:
                logging.error(f"Insufficient blob positions: {len(blob_positions)}, need {number_of_targets * 4}")
                return False
                
            # Load alvobase image to get dimensions
            alvobase_path = os.path.join("resources", "drawable", "alvobase.jpg")
            if not os.path.exists(alvobase_path):
                logging.error(f"Alvobase image not found: {alvobase_path}")
                return False
                
            alvobase_image = cv2.imread(alvobase_path)
            if alvobase_image is None:
                logging.error("Failed to load alvobase image")
                return False
                
            alvo_height, alvo_width = alvobase_image.shape[:2]
            
            # Define corners of alvobase image (destination points)
            alvobase_corners = np.float32([
                [0, 0],                    # Top-left
                [alvo_width, 0],           # Top-right  
                [alvo_width, alvo_height], # Bottom-right
                [0, alvo_height]           # Bottom-left
            ])
            
            homographies_data = {
                "calculation_timestamp": datetime.now().isoformat(),
                "alvobase_dimensions": {
                    "width": alvo_width,
                    "height": alvo_height
                },
                "number_of_targets": number_of_targets,
                "homographies": {}
            }
            
            # Calculate homography for each target
            for target_num in range(1, number_of_targets + 1):
                # Get the 4 blob positions for this target
                blob_start_idx = (target_num - 1) * 4
                blob_end_idx = blob_start_idx + 4
                
                if blob_end_idx <= len(blob_positions):
                    target_blobs = blob_positions[blob_start_idx:blob_end_idx]
                    
                    # Convert to numpy array (camera coordinates - source points)
                    camera_points = np.float32(target_blobs)
                    
                    # Calculate homography: camera coordinates -> alvobase coordinates
                    homography_matrix, mask = cv2.findHomography(
                        camera_points, 
                        alvobase_corners, 
                        cv2.RANSAC
                    )
                    
                    if homography_matrix is not None:
                        # Store homography (convert to list for JSON serialization)
                        homographies_data["homographies"][str(target_num)] = {
                            "camera_points": target_blobs,  # Original blob positions in camera frame
                            "alvobase_corners": alvobase_corners.tolist(),
                            "homography_matrix": homography_matrix.tolist()
                        }
                        
                        # Store in memory for immediate use
                        self.homographies[target_num] = homography_matrix
                        
                        logging.info(f"Calculated homography for target {target_num}")
                        logging.info(f"  Camera points: {target_blobs}")
                        
                    else:
                        logging.error(f"Failed to calculate homography for target {target_num}")
                        return False
                else:
                    logging.error(f"Not enough blob positions for target {target_num}")
                    return False
                    
            # Save to file
            os.makedirs("data", exist_ok=True)
            with open(self.homographies_file, 'w') as f:
                json.dump(homographies_data, f, indent=2)
                
            logging.info(f"Homographies saved to: {self.homographies_file}")
            return True
            
        except Exception as e:
            logging.error(f"Error calculating homographies: {e}")
            return False
            
    def load_homographies(self) -> bool:
        """
        Load homographies from file
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not os.path.exists(self.homographies_file):
                logging.warning(f"Homographies file not found: {self.homographies_file}")
                return False
                
            with open(self.homographies_file, 'r') as f:
                data = json.load(f)
                
            # Load homography matrices
            self.homographies = {}
            homographies_data = data.get("homographies", {})
            
            for target_str, target_data in homographies_data.items():
                target_num = int(target_str)
                homography_matrix = np.array(target_data["homography_matrix"], dtype=np.float32)
                self.homographies[target_num] = homography_matrix
                
            logging.info(f"Loaded {len(self.homographies)} homographies from file")
            return True
            
        except Exception as e:
            logging.error(f"Error loading homographies: {e}")
            return False
            
    def transform_camera_to_target(self, camera_point: Tuple[float, float], 
                                 target_number: int) -> Tuple[float, float]:
        """
        Transform a point from camera coordinates to target (alvobase) coordinates
        
        Args:
            camera_point: (x, y) coordinates in camera reference frame
            target_number: Target number to use for transformation
            
        Returns:
            Tuple[float, float]: (x, y) coordinates in target reference frame
        """
        try:
            if target_number not in self.homographies:
                logging.error(f"No homography found for target {target_number}")
                return camera_point
                
            homography = self.homographies[target_number]
            
            # Convert point to homogeneous coordinates
            point_homo = np.array([[camera_point[0], camera_point[1]]], dtype=np.float32)
            
            # Apply homography transformation
            transformed_point = cv2.perspectiveTransform(
                point_homo.reshape(1, 1, 2), 
                homography
            )
            
            # Extract x, y coordinates
            x, y = transformed_point[0, 0]
            return (float(x), float(y))
            
        except Exception as e:
            logging.error(f"Error transforming point {camera_point} for target {target_number}: {e}")
            return camera_point
            
    def get_homography_info(self) -> Dict:
        """Get information about loaded homographies"""
        try:
            if not os.path.exists(self.homographies_file):
                return {}
                
            with open(self.homographies_file, 'r') as f:
                return json.load(f)
                
        except Exception as e:
            logging.error(f"Error reading homography info: {e}")
            return {}
            
    def has_homography_for_target(self, target_number: int) -> bool:
        """Check if homography exists for a target"""
        return target_number in self.homographies 