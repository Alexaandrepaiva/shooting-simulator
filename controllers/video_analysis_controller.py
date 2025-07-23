import os
import logging
import threading
from datetime import datetime
from typing import Optional, Callable
from models.calibration import CalibrationModel
from models.shot_detection import ShotDetectionModel


class VideoAnalysisController:
    """Controller for analyzing recorded videos to detect laser shots"""
    
    def __init__(self):
        self.calibration_model = CalibrationModel()
        self.shot_detection_model = ShotDetectionModel()
        self.is_analyzing = False
        self.current_analysis_thread = None
        
    def analyze_video_async(self, video_path: str, progress_callback: Optional[Callable] = None,
                           completion_callback: Optional[Callable] = None):
        """
        Start video analysis in a background thread
        
        Args:
            video_path: Path to the video file to analyze
            progress_callback: Optional callback for progress updates (0-100)
            completion_callback: Optional callback when analysis is complete
        """
        if self.is_analyzing:
            logging.warning("Video analysis already in progress")
            return
            
        logging.info(f"Starting async video analysis for: {video_path}")
        
        # Start analysis in background thread
        self.current_analysis_thread = threading.Thread(
            target=self._analyze_video_worker,
            args=(video_path, progress_callback, completion_callback)
        )
        self.current_analysis_thread.daemon = True
        self.current_analysis_thread.start()
        
    def _analyze_video_worker(self, video_path: str, progress_callback: Optional[Callable],
                            completion_callback: Optional[Callable]):
        """Worker method for video analysis thread"""
        try:
            self.is_analyzing = True
            
            # Load calibration data
            if not self.calibration_model.load_calibration_data():
                logging.error("No calibration data found. Cannot analyze video.")
                if completion_callback:
                    completion_callback(False, "No calibration data found")
                return
                
            calibration_data = {
                'parameters': self.calibration_model.get_all_parameters(),
                'fixed_target_positions': self.calibration_model.get_blob_positions()
            }
            
            # Check if we have enough blob positions
            blob_positions = calibration_data['fixed_target_positions']
            number_of_targets = calibration_data['parameters'].get('number_of_targets', 1)
            required_blobs = number_of_targets * 4
            
            if len(blob_positions) < required_blobs:
                error_msg = f"Insufficient blob positions: {len(blob_positions)}/{required_blobs}"
                logging.error(error_msg)
                if completion_callback:
                    completion_callback(False, error_msg)
                return
                
            logging.info(f"Analyzing video with {number_of_targets} targets, {len(blob_positions)} blob positions")
            
            # Analyze video for shot detection
            detected_shots = self.shot_detection_model.detect_shots_in_video(
                video_path, calibration_data, progress_callback
            )
            
            # Generate result images
            if detected_shots:
                self._generate_result_images(detected_shots, calibration_data)
                
            # Call completion callback
            if completion_callback:
                completion_callback(True, f"Analysis complete. Found {len(detected_shots)} shots")
                
            logging.info(f"Video analysis completed successfully. Detected {len(detected_shots)} shots")
            
        except Exception as e:
            logging.error(f"Error during video analysis: {e}")
            if completion_callback:
                completion_callback(False, f"Analysis failed: {str(e)}")
        finally:
            self.is_analyzing = False
            
    def _generate_result_images(self, detected_shots: list, calibration_data: dict):
        """Generate result images showing shots on target"""
        try:
            # Load base target image
            base_image_path = os.path.join("resources", "drawable", "alvobase.jpg")
            if not os.path.exists(base_image_path):
                logging.warning(f"Base target image not found: {base_image_path}")
                return
                
            import cv2
            import numpy as np
            
            base_image = cv2.imread(base_image_path)
            if base_image is None:
                logging.error("Failed to load base target image")
                return
                
            # Group shots by target
            shots_by_target = {}
            for shot in detected_shots:
                target_num = shot['target_number']
                if target_num not in shots_by_target:
                    shots_by_target[target_num] = []
                shots_by_target[target_num].append(shot)
                
            # Generate image for each target
            for target_num, target_shots in shots_by_target.items():
                self._create_target_result_image(target_num, target_shots, base_image, calibration_data)
                
        except Exception as e:
            logging.error(f"Error generating result images: {e}")
            
    def _create_target_result_image(self, target_number: int, shots: list, 
                                  base_image, calibration_data: dict):
        """Create result image for a specific target"""
        try:
            import cv2
            import numpy as np
            
            # Create a copy of base image
            result_image = base_image.copy()
            
            # Get image dimensions
            img_height, img_width = result_image.shape[:2]
            
            # Get blob positions for this target (4 blobs per target)
            blob_positions = calibration_data['fixed_target_positions']
            target_blob_start = (target_number - 1) * 4
            target_blob_end = target_blob_start + 4
            
            if target_blob_end > len(blob_positions):
                logging.warning(f"Not enough blob positions for target {target_number}")
                return
                
            target_blobs = blob_positions[target_blob_start:target_blob_end]
            
            # Convert blob coordinates to image coordinates
            # The blob coordinates define the warped area, we need to map shots back to the base image
            
            # For simplicity, we'll map shots to the base image using the center as origin
            img_center_x = img_width // 2
            img_center_y = img_height // 2
            
            # Scale factor to map from warped coordinates to image coordinates
            scale_factor = min(img_width, img_height) / 400  # Assuming warped size was ~300-400
            
            # Draw shots on the image using target coordinates with sophisticated label positioning
            self._draw_shots_with_smart_labels(result_image, shots, img_width, img_height)
                              
            # Add target information - simplified title
            info_text = f"Alvo {target_number}"
            cv2.putText(result_image, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
            cv2.putText(result_image, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
                       
            # Add timestamp
            timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            text_size = cv2.getTextSize(timestamp, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            text_x = result_image.shape[1] - text_size[0] - 10
            text_y = result_image.shape[0] - 10
            
            cv2.putText(result_image, timestamp, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(result_image, timestamp, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 100), 1)
            
            # Create output directory if it doesn't exist
            os.makedirs("output", exist_ok=True)
            
            # Save result image with new filename format
            timestamp_filename = datetime.now().strftime("%Y%m%d-%H%M%S")
            result_filename = f"target-{target_number}-{timestamp_filename}.jpg"
            result_path = os.path.join("output", result_filename)
            
            cv2.imwrite(result_path, result_image)
            logging.info(f"Result image saved: {result_path}")
            
        except Exception as e:
            logging.error(f"Error creating result image for target {target_number}: {e}")
            
    def _draw_shots_with_smart_labels(self, result_image, shots, img_width, img_height):
        """Draw shots with sophisticated label positioning to avoid overlaps"""
        import cv2
        import numpy as np
        
        # Track used areas to avoid overlaps
        used_areas = []
        valid_shots = []
        
        # First pass: Draw shots and collect their positions
        for shot in shots:
            target_pos = shot['target_position']
            
            # Use target coordinates directly (homography already maps to alvobase coordinates)
            shot_x = int(target_pos['x'])
            shot_y = int(target_pos['y'])
            
            # Ensure shot is within image bounds
            if 0 <= shot_x < img_width and 0 <= shot_y < img_height:
                # Draw smaller shot circle
                cv2.circle(result_image, (shot_x, shot_y), 5, (0, 0, 255), -1)  # Red filled circle (reduced size)
                cv2.circle(result_image, (shot_x, shot_y), 7, (255, 255, 255), 2)  # White border (reduced size)
                
                # Add the shot position to used areas (smaller area)
                shot_area = (shot_x - 8, shot_y - 8, 16, 16)
                used_areas.append(shot_area)
                
                valid_shots.append((shot, shot_x, shot_y))
                
                logging.info(f"Shot {shot['shot_number']}: camera({shot['camera_position']['x']:.1f}, {shot['camera_position']['y']:.1f}) -> target({shot_x}, {shot_y})")
            else:
                logging.warning(f"Shot {shot['shot_number']} outside image bounds: ({shot_x}, {shot_y})")
        
        # Function to check if two rectangles overlap
        def rectangles_overlap_any(rect, used_areas_list):
            """Check if rectangle overlaps with any used area"""
            for used_rect in used_areas_list:
                if rectangles_overlap(rect, used_rect):
                    return True
            return False
        
        def rectangles_overlap(rect1, rect2):
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
        
        # Second pass: Add labels with overlap avoidance
        for shot, shot_x, shot_y in valid_shots:
            # Define potential positions for the label
            label_text = str(shot['shot_number'])
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4  # Reduced font size
            outline_thickness = 2
            text_thickness = 1
            text_size, _ = cv2.getTextSize(label_text, font, font_scale, text_thickness)
            
            # Add padding around text for collision detection
            text_width = text_size[0] + 6
            text_height = text_size[1] + 6
            
            # Try different positions with more options for better placement
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
                    if not rectangles_overlap_any(label_area, used_areas):
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
            
    def stop_analysis(self):
        """Stop current video analysis if running"""
        if self.is_analyzing and self.current_analysis_thread:
            logging.info("Stopping video analysis...")
            # Note: We can't forcefully stop the thread, but we can mark it as stopped
            # The analysis will complete but we won't process results
            self.is_analyzing = False
            
    def is_analysis_running(self) -> bool:
        """Check if video analysis is currently running"""
        return self.is_analyzing 