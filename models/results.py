import json
import os
import cv2
import numpy as np
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple


class ResultsModel:
    """Model for calculating relative positions and generating result images"""
    
    def __init__(self):
        self.results_file = os.path.join("data", "results_data.json")
        
    def calculate_relative_positions(self, detection_data: Dict, calibration_data: Dict) -> Dict:
        """
        Calculate relative positions of shots relative to target blob positions
        
        Args:
            detection_data: Detection data with absolute shot positions
            calibration_data: Calibration data with blob positions
            
        Returns:
            Dict: Results data with relative positions
        """
        try:
            blob_positions = calibration_data['fixed_target_positions']
            parameters = calibration_data['parameters']
            number_of_targets = parameters.get('number_of_targets', 1)
            
            results_data = {
                "processing_timestamp": datetime.now().isoformat(),
                "calibration_info": {
                    "blob_positions": blob_positions,
                    "number_of_targets": number_of_targets
                },
                "targets": []
            }
            
            # Process each target
            for target_data in detection_data.get('targets', []):
                target_number = target_data['target_number']
                
                # Get blob positions for this target (4 blobs per target)
                target_blob_start = (target_number - 1) * 4
                target_blob_end = target_blob_start + 4
                
                if target_blob_end <= len(blob_positions):
                    target_blobs = blob_positions[target_blob_start:target_blob_end]
                    
                    # Calculate target bounding box
                    target_bounds = self._calculate_target_bounds(target_blobs)
                    
                    # Process shots for this target
                    processed_shots = []
                    for shot in target_data.get('shots', []):
                        absolute_pos = shot['absolute_position']
                        
                        # Calculate relative position within target bounds
                        relative_pos = self._calculate_relative_position(
                            absolute_pos, target_bounds
                        )
                        
                        processed_shot = {
                            "timestamp": shot["timestamp"],
                            "shot_number": shot["shot_number"],
                            "absolute_position": absolute_pos,
                            "relative_position": relative_pos,
                            "detection_strategy": shot["detection_strategy"],
                            "frame_number": shot["frame_number"]
                        }
                        
                        # Add intensity_info if present
                        if "intensity_info" in shot:
                            processed_shot["intensity_info"] = shot["intensity_info"]
                            
                        processed_shots.append(processed_shot)
                    
                    target_result = {
                        "target_number": target_number,
                        "blob_positions": target_blobs,
                        "target_bounds": target_bounds,
                        "shots": processed_shots
                    }
                    
                    results_data["targets"].append(target_result)
                else:
                    logging.warning(f"Not enough blob positions for target {target_number}")
                    
            return results_data
            
        except Exception as e:
            logging.error(f"Error calculating relative positions: {e}")
            return {}
            
    def _calculate_target_bounds(self, blob_positions: List[List[float]]) -> Dict[str, float]:
        """
        Calculate bounding box of target from blob positions
        
        Args:
            blob_positions: List of 4 blob positions [[x,y], ...]
            
        Returns:
            Dict with min_x, max_x, min_y, max_y, width, height, center_x, center_y
        """
        try:
            positions_array = np.array(blob_positions)
            
            min_x = float(np.min(positions_array[:, 0]))
            max_x = float(np.max(positions_array[:, 0]))
            min_y = float(np.min(positions_array[:, 1]))
            max_y = float(np.max(positions_array[:, 1]))
            
            width = max_x - min_x
            height = max_y - min_y
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2
            
            return {
                "min_x": min_x,
                "max_x": max_x,
                "min_y": min_y,
                "max_y": max_y,
                "width": width,
                "height": height,
                "center_x": center_x,
                "center_y": center_y
            }
            
        except Exception as e:
            logging.error(f"Error calculating target bounds: {e}")
            return {}
            
    def _calculate_relative_position(self, absolute_pos: Dict[str, float], 
                                   target_bounds: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate relative position of shot within target bounds
        
        Args:
            absolute_pos: Absolute position in camera coordinates
            target_bounds: Target bounding box
            
        Returns:
            Dict with relative_x, relative_y (normalized 0-1), offset_x, offset_y (pixels from center)
        """
        try:
            abs_x = absolute_pos['x']
            abs_y = absolute_pos['y']
            
            # Calculate offset from target center
            offset_x = abs_x - target_bounds['center_x']
            offset_y = abs_y - target_bounds['center_y']
            
            # Calculate relative position (0-1 within target bounds)
            if target_bounds['width'] > 0:
                relative_x = (abs_x - target_bounds['min_x']) / target_bounds['width']
            else:
                relative_x = 0.5
                
            if target_bounds['height'] > 0:
                relative_y = (abs_y - target_bounds['min_y']) / target_bounds['height']
            else:
                relative_y = 0.5
            
            return {
                "relative_x": float(relative_x),
                "relative_y": float(relative_y),
                "offset_x": float(offset_x),
                "offset_y": float(offset_y)
            }
            
        except Exception as e:
            logging.error(f"Error calculating relative position: {e}")
            return {"relative_x": 0.5, "relative_y": 0.5, "offset_x": 0.0, "offset_y": 0.0}
            
    def save_results_data(self, results_data: Dict) -> bool:
        """Save results data to JSON file"""
        try:
            os.makedirs("data", exist_ok=True)
            
            with open(self.results_file, 'w') as f:
                json.dump(results_data, f, indent=2)
                
            logging.info(f"Results data saved to: {self.results_file}")
            return True
            
        except Exception as e:
            logging.error(f"Error saving results data: {e}")
            return False
            
    def load_results_data(self) -> Optional[Dict]:
        """Load results data from JSON file"""
        try:
            if not os.path.exists(self.results_file):
                return None
                
            with open(self.results_file, 'r') as f:
                return json.load(f)
                
        except Exception as e:
            logging.error(f"Error loading results data: {e}")
            return None
            
    def generate_target_images(self, results_data: Dict) -> bool:
        """
        Generate result images for each target with labeled shots
        
        Args:
            results_data: Results data with relative positions
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load alvobase image
            alvobase_path = os.path.join("resources", "drawable", "alvobase.jpg")
            if not os.path.exists(alvobase_path):
                logging.error(f"Alvobase image not found: {alvobase_path}")
                return False
                
            base_image = cv2.imread(alvobase_path)
            if base_image is None:
                logging.error("Failed to load alvobase image")
                return False
                
            success_count = 0
            
            # Generate image for each target
            for target_data in results_data.get('targets', []):
                target_number = target_data['target_number']
                shots = target_data.get('shots', [])
                
                if self._create_target_result_image(target_number, shots, base_image):
                    success_count += 1
                    
            if success_count > 0:
                logging.info(f"Generated {success_count} result images successfully")
                return True
            else:
                logging.error("Failed to generate any result images")
                return False
                
        except Exception as e:
            logging.error(f"Error generating target images: {e}")
            return False
            
    def _create_target_result_image(self, target_number: int, shots: List[Dict], 
                                  base_image: np.ndarray) -> bool:
        """
        Create result image for a specific target with labeled shots
        
        Args:
            target_number: Target number
            shots: List of shots with relative positions
            base_image: Base alvobase image
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create a copy of base image
            result_image = base_image.copy()
            img_height, img_width = result_image.shape[:2]
            
            # Draw shots on the image using relative positions
            self._draw_shots_with_smart_labels(result_image, shots, img_width, img_height)
            
            # Add target information - simplified title
            info_text = f"Alvo {target_number}"
            cv2.putText(result_image, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(result_image, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
            
            # Create output directory if it doesn't exist
            os.makedirs("output", exist_ok=True)
            
            # Save result image with new filename format
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            output_filename = f"target-{target_number}-{timestamp}.jpg"
            output_path = os.path.join("output", output_filename)
            
            success = cv2.imwrite(output_path, result_image)
            if success:
                logging.info(f"Saved result image: {output_path}")
                return True
            else:
                logging.error(f"Failed to save result image: {output_path}")
                return False
                
        except Exception as e:
            logging.error(f"Error creating target result image: {e}")
            return False
            
    def _draw_shots_with_smart_labels(self, result_image: np.ndarray, shots: List[Dict], 
                                    img_width: int, img_height: int):
        """
        Draw shots with smart label positioning to avoid overlaps
        
        Args:
            result_image: Image to draw on
            shots: List of shots with relative positions
            img_width: Image width
            img_height: Image height
        """
        try:
            # Track used areas to avoid overlaps
            used_areas = []
            valid_shots = []
            
            # First pass: Draw shots and collect their positions
            for shot in shots:
                relative_pos = shot['relative_position']
                
                # Convert relative position to image coordinates
                shot_x = int(relative_pos['relative_x'] * img_width)
                shot_y = int(relative_pos['relative_y'] * img_height)
                
                # Ensure shot is within image bounds
                if 0 <= shot_x < img_width and 0 <= shot_y < img_height:
                    # Draw shot circle
                    cv2.circle(result_image, (shot_x, shot_y), 5, (0, 0, 255), -1)  # Red filled circle (reduced size)
                    cv2.circle(result_image, (shot_x, shot_y), 7, (255, 255, 255), 2)  # White border (reduced size)
                    
                    # Add the shot position to used areas (16x16 pixels)
                    shot_area = (shot_x - 8, shot_y - 8, 16, 16)
                    used_areas.append(shot_area)
                    
                    valid_shots.append((shot, shot_x, shot_y))
                    
                    logging.info(f"Shot {shot['shot_number']}: relative({relative_pos['relative_x']:.3f}, {relative_pos['relative_y']:.3f}) -> image({shot_x}, {shot_y})")
                else:
                    logging.warning(f"Shot {shot['shot_number']} outside image bounds: ({shot_x}, {shot_y})")
            
            # Second pass: Add labels with overlap avoidance
            for shot, shot_x, shot_y in valid_shots:
                self._add_shot_label(result_image, shot, shot_x, shot_y, used_areas)
                
        except Exception as e:
            logging.error(f"Error drawing shots with labels: {e}")
            
    def _add_shot_label(self, result_image: np.ndarray, shot: Dict, shot_x: int, shot_y: int, 
                       used_areas: List[Tuple[int, int, int, int]]):
        """
        Add label for a shot with smart positioning to avoid overlaps
        
        Args:
            result_image: Image to draw on
            shot: Shot data
            shot_x, shot_y: Shot position in image coordinates
            used_areas: List of used areas to avoid
        """
        try:
            label_text = str(shot['shot_number'])
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4 
            outline_thickness = 2
            text_thickness = 1
            text_size, _ = cv2.getTextSize(label_text, font, font_scale, text_thickness)
            
            # Add padding around text for collision detection
            text_width = text_size[0] + 6
            text_height = text_size[1] + 6
            
            # Try different positions
            label_offsets = [
                (12, -text_height//2),              # right
                (12, -(text_height + 3)),           # top-right
                (12, 12),                           # bottom-right
                (-(text_width + 3), -text_height//2), # left
                (-(text_width + 3), -(text_height + 3)), # top-left
                (-(text_width + 3), 12),            # bottom-left
                (-text_width//2, -(text_height + 8)), # top-center
                (-text_width//2, 15),               # bottom-center
                (15, -5),                           # top-right diagonal
                (15, 8),                            # bottom-right diagonal
                (-15, -5),                          # top-left diagonal
                (-15, 8)                            # bottom-left diagonal
            ]
            
            label_placed = False
            for offset_x, offset_y in label_offsets:
                label_x = shot_x + offset_x
                label_y = shot_y + offset_y
                
                # Check bounds
                if (label_x >= 0 and label_y >= 0 and 
                    label_x + text_width < result_image.shape[1] and
                    label_y + text_height < result_image.shape[0]):
                    
                    # Check for overlaps
                    label_area = (label_x, label_y, text_width, text_height)
                    if not self._rectangles_overlap_any(label_area, used_areas):
                        # Place label with red font and white outline (no background)
                        text_y = label_y + text_size[1] + 3
                        text_x = label_x + 3
                        
                        # Draw white outline
                        cv2.putText(result_image, label_text, (text_x, text_y), 
                                  font, font_scale, (255, 255, 255), outline_thickness)
                        
                        # Draw red text
                        cv2.putText(result_image, label_text, (text_x, text_y), 
                                  font, font_scale, (0, 0, 255), text_thickness)
                        
                        # Add to used areas
                        used_areas.append(label_area)
                        label_placed = True
                        break
            
            if not label_placed:
                logging.warning(f"Could not place label for shot {shot['shot_number']} - trying fallback position")
                # Fallback: place label directly on the shot with smaller offset
                text_y = shot_y + text_size[1]//2
                text_x = shot_x - text_size[0]//2
                
                # Ensure fallback position is within bounds
                if text_x >= 0 and text_y >= 0 and text_x + text_size[0] < result_image.shape[1] and text_y < result_image.shape[0]:
                    # Draw white outline
                    cv2.putText(result_image, label_text, (text_x, text_y), 
                              font, font_scale, (255, 255, 255), outline_thickness)
                    
                    # Draw red text
                    cv2.putText(result_image, label_text, (text_x, text_y), 
                              font, font_scale, (0, 0, 255), text_thickness)
                
        except Exception as e:
            logging.error(f"Error adding shot label: {e}")
            
    def _rectangles_overlap_any(self, rect: Tuple[int, int, int, int], 
                              used_areas: List[Tuple[int, int, int, int]]) -> bool:
        """Check if rectangle overlaps with any used area"""
        for used_rect in used_areas:
            if self._rectangles_overlap(rect, used_rect):
                return True
        return False
        
    def _rectangles_overlap(self, rect1: Tuple[int, int, int, int], 
                          rect2: Tuple[int, int, int, int]) -> bool:
        """Check if two rectangles overlap"""
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2
        
        # Check if one rectangle is to the left of the other
        if x1 + w1 <= x2 or x2 + w2 <= x1:
            return False
        
        # Check if one rectangle is above the other
        if y1 + h1 <= y2 or y2 + h2 <= y1:
            return False
        
        # If we get here, the rectangles overlap
        return True 