import cv2
import numpy as np
import json
import os
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any
import logging
from models.homography import HomographyModel


class LaserState:
    """State tracking for individual laser spots"""
    def __init__(self, position: Tuple[float, float], frame_number: int):
        self.position = position
        self.first_seen_frame = frame_number
        self.last_seen_frame = frame_number
        self.total_frames_visible = 1
        self.positions_history = [position]


class ShotDetectionModel:
    """Model for detecting laser shots using multiple detection strategies with state machine"""
    
    def __init__(self):
        self.debugging = True  # Enable debugging output
        self.strategies = [
            "rgb_enhanced",
            "rgb_subtraction", 
            "hsv_subtraction",
            "hsv_direct",
            "frame_difference"
        ]
        self.homography_model = HomographyModel()
        
    def detect_shots_in_video(self, video_path: str, calibration_data: Dict, 
                            progress_callback=None) -> List[Dict]:
        """
        Detect shots in video using multiple strategies as redundancy
        
        Args:
            video_path: Path to the video file
            calibration_data: Calibration parameters and blob positions
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of detected shots with their data
        """
        logging.info(f"Starting shot detection analysis for video: {video_path}")
        
        # Load calibration parameters
        params = calibration_data.get('parameters', {})
        blob_positions = calibration_data.get('fixed_target_positions', [])
        shots_per_series = params.get('shots_per_series', 3)
        number_of_targets = params.get('number_of_targets', 1)
        
        if len(blob_positions) < 4:
            raise ValueError(f"Insufficient blob positions: {len(blob_positions)}, need 4 for warping")
            
        # Load homographies for coordinate transformation
        if not self.homography_model.load_homographies():
            logging.error("Failed to load homographies. Cannot proceed with shot detection.")
            return []
            
        all_detected_shots = []
        
        # Process each target
        for target_num in range(1, number_of_targets + 1):
            # Get the 4 blob positions for this target
            target_blob_start = (target_num - 1) * 4
            target_blob_end = target_blob_start + 4
            
            if target_blob_end <= len(blob_positions):
                target_blobs = blob_positions[target_blob_start:target_blob_end]
                
                # Detect shots for this target
                target_shots = self._detect_shots_for_target(
                    video_path, target_num, target_blobs, shots_per_series, progress_callback
                )
                
                all_detected_shots.extend(target_shots)
            else:
                logging.warning(f"Not enough blob positions for target {target_num}")
                
        # Always save shots to JSON file (even if empty to clear previous data)
        self._save_shots_to_json(all_detected_shots)
        logging.info("Detection data saved")
            
        return all_detected_shots
        
    def _detect_shots_for_target(self, video_path: str, target_number: int, 
                               blob_positions: List[List[float]], shots_per_series: int,
                               progress_callback=None) -> List[Dict]:
        """Detect shots for a specific target using multiple strategies until one finds enough shots"""
        
        logging.info(f"=== Processing Target {target_number} ===")
        logging.info(f"Target {target_number} blob positions: {blob_positions}")
        
        # Validate blob positions for this target
        if len(blob_positions) != 4:
            logging.error(f"Target {target_number}: Expected 4 blob positions, got {len(blob_positions)}")
            return []
            
        # Calculate target region bounds for validation
        x_coords = [pos[0] for pos in blob_positions]
        y_coords = [pos[1] for pos in blob_positions]
        target_bounds = {
            'min_x': min(x_coords),
            'max_x': max(x_coords),
            'min_y': min(y_coords),
            'max_y': max(y_coords)
        }
        
        logging.info(f"Target {target_number} region bounds: "
                    f"X=[{target_bounds['min_x']:.1f}, {target_bounds['max_x']:.1f}], "
                    f"Y=[{target_bounds['min_y']:.1f}, {target_bounds['max_y']:.1f}]")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Try strategies until one finds enough shots or all are exhausted
        strategy_results = []
        
        for strategy_idx, strategy_name in enumerate(self.strategies):
            logging.info(f"Target {target_number} - Using strategy {strategy_idx + 1}/{len(self.strategies)}: {strategy_name}")
            
            # Reset video position for each strategy
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # Try this strategy
            detected_shots = self._detect_shots_with_state_machine(
                cap, target_number, blob_positions, strategy_name, shots_per_series, progress_callback, total_frames
            )
            
            # Validate that detected shots are within the target region
            validated_shots = self._validate_shots_in_target_region(detected_shots, target_bounds, target_number)
            
            # Store results for this strategy
            strategy_results.append({
                'strategy_name': strategy_name,
                'shots_found': len(validated_shots),
                'shots_data': validated_shots
            })
            
            logging.info(f"Target {target_number} - Strategy {strategy_name} found {len(detected_shots)} shots, {len(validated_shots)} validated")
            
            # If this strategy found at least the expected number of shots, use it and stop
            if len(validated_shots) >= shots_per_series:
                logging.info(f"Target {target_number} - Strategy {strategy_name} found sufficient shots ({len(validated_shots)}>={shots_per_series}), using all {len(validated_shots)} shots")
                cap.release()
                
                # Update shot numbers to be sequential for all detected shots
                for i, shot in enumerate(validated_shots):
                    shot['shot_number'] = i + 1
                
                return validated_shots
        
        cap.release()
        
        # If no strategy found enough shots, use the one that found the most
        if not strategy_results:
            logging.warning(f"Target {target_number} - No strategies returned any results")
            return []
        
        # Log results from all strategies tried
        logging.info(f"Target {target_number} - No strategy found enough shots. Strategy comparison:")
        for result in strategy_results:
            logging.info(f"  {result['strategy_name']}: {result['shots_found']} shots")
        
        best_strategy = max(strategy_results, key=lambda x: x['shots_found'])
        
        logging.info(f"Target {target_number} - Using best strategy '{best_strategy['strategy_name']}' with {best_strategy['shots_found']} shots (expected {shots_per_series})")
        
        # Use ALL shots from the best strategy
        best_shots = best_strategy['shots_data']
        
        # Update shot numbers to be sequential for all detected shots
        for i, shot in enumerate(best_shots):
            shot['shot_number'] = i + 1
            
        return best_shots
        
    def _detect_shots_with_state_machine(self, cap, target_number: int, blob_positions: List[List[float]], 
                                       strategy_name: str, shots_per_series: int, progress_callback, total_frames: int) -> List[Dict]:
        """
        Pure state machine for shot detection:
        - State: 'no_laser' or 'laser_present'
        - Transition no_laser -> laser_present: Start new laser
        - Transition laser_present -> no_laser: Complete shot
        - Stay in laser_present: Continue same laser (update position)
        """
        
        detected_shots = []
        frame_count = 0
        previous_frame = None
        
        # Pure state machine
        current_state = "no_laser"  # Current state: "no_laser" or "laser_present"
        current_laser = None  # Current laser being tracked (LaserState object)
        completed_shots = []  # List of completed shots
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                laser_detected = False
                laser_position = None
                
                # Detect if laser is present in current frame
                if previous_frame is not None:
                    laser_positions = self._detect_all_laser_positions(
                        frame, previous_frame, blob_positions, strategy_name
                    )
                    
                    if laser_positions:
                        laser_detected = True
                        # Use first detected laser position (simplest approach)
                        laser_position = laser_positions[0]
                
                # State machine transitions
                if current_state == "no_laser":
                    if laser_detected:
                        # Transition: no_laser -> laser_present
                        current_state = "laser_present"
                        current_laser = LaserState(laser_position, frame_count)
                        # logging.debug(f"Laser started at frame {frame_count}: {laser_position}")
                        
                elif current_state == "laser_present":
                    if laser_detected:
                        # Stay in laser_present state - update current laser
                        current_laser.position = laser_position
                        current_laser.last_seen_frame = frame_count
                        current_laser.total_frames_visible += 1
                        current_laser.positions_history.append(laser_position)
                    else:
                        # Transition: laser_present -> no_laser
                        current_state = "no_laser"
                        completed_shots.append(current_laser)
                        # logging.info(f"Shot completed: laser from frame {current_laser.first_seen_frame} to {current_laser.last_seen_frame}")
                        current_laser = None
                
                previous_frame = frame.copy()
                frame_count += 1
                
                # Progress callback - removed to reduce logging
                # if progress_callback and frame_count % 30 == 0:
                #     progress = (frame_count / total_frames) * 100
                #     progress_callback(progress)
                    
        except Exception as e:
            logging.error(f"Error in strategy {strategy_name}: {e}")
        
        # Handle case where video ends while laser is still present
        if current_state == "laser_present" and current_laser is not None:
            completed_shots.append(current_laser)
            # logging.info(f"Final shot completed: laser from frame {current_laser.first_seen_frame} to {current_laser.last_seen_frame}")
        
        # Convert completed laser states to shot data (return ALL detected shots, not limited to shots_per_series)
        for i, laser_state in enumerate(completed_shots):
            # Use the average position of the laser during its lifetime
            avg_position = self._calculate_average_position(laser_state.positions_history)
            
            shot_data = {
                "timestamp": datetime.now().isoformat(),
                "target_number": target_number,
                "shot_number": i + 1,
                "absolute_position": {
                    "x": avg_position[0],
                    "y": avg_position[1]
                },
                "detection_strategy": strategy_name,
                "frame_number": laser_state.first_seen_frame,
                "laser_info": {
                    "duration_frames": laser_state.total_frames_visible,
                    "first_seen": laser_state.first_seen_frame,
                    "last_seen": laser_state.last_seen_frame
                }
            }
            
            detected_shots.append(shot_data)
            
        # Log summary with shot positions
        if detected_shots:
            shot_positions = [f"({shot['absolute_position']['x']:.1f}, {shot['absolute_position']['y']:.1f})" for shot in detected_shots]
            logging.info(f"Strategy {strategy_name} completed: {len(detected_shots)} shots at positions {', '.join(shot_positions)}")
        else:
            logging.info(f"Strategy {strategy_name} completed: 0 shots detected")
            
        return detected_shots
        
    def _detect_all_laser_positions(self, frame: np.ndarray, previous_frame: np.ndarray, 
                                  blob_positions: List[List[float]], strategy: str) -> List[Tuple[float, float]]:
        """
        Detect ALL laser positions in the current frame (not just the first one)
        
        Args:
            frame: Current frame
            previous_frame: Previous frame  
            blob_positions: Target blob positions
            strategy: Detection strategy to use
            
        Returns:
            List of all detected laser positions in this frame
        """
        try:
            # Create target region mask
            mask = self._create_target_region_mask(frame, blob_positions)
            
            # Get detection mask using the specific strategy
            if strategy == "rgb_enhanced":
                detection_mask = self._get_rgb_enhanced_mask(frame, mask, previous_frame)
            elif strategy == "rgb_subtraction":
                detection_mask = self._get_rgb_subtraction_mask(frame, mask, previous_frame)
            elif strategy == "hsv_subtraction":
                detection_mask = self._get_hsv_subtraction_mask(frame, mask, previous_frame)
            elif strategy == "hsv_direct":
                detection_mask = self._get_hsv_direct_mask(frame, mask, previous_frame)
            elif strategy == "frame_difference":
                detection_mask = self._get_frame_difference_mask(frame, mask, previous_frame)
            else:
                return []
                
            if detection_mask is None:
                return []
                
            # Find all contours (potential laser spots)
            contours, _ = cv2.findContours(detection_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            laser_positions = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if 1 <= area <= 500:  # Valid area range for laser spots
                    M = cv2.moments(contour)
                    if M["m00"] > 0:
                        cx = M["m10"] / M["m00"]
                        cy = M["m01"] / M["m00"]
                        
                        # Validate this is actually a laser spot
                        if self._validate_laser_spot(frame, (cx, cy), strategy):
                            laser_positions.append((cx, cy))
            
            return laser_positions
            
        except Exception as e:
            logging.error(f"Error detecting laser positions: {e}")
            return []
            
    def _calculate_average_position(self, positions_history: List[Tuple[float, float]]) -> Tuple[float, float]:
        """Calculate the average position from a list of positions"""
        if not positions_history:
            return (0.0, 0.0)
            
        avg_x = sum(pos[0] for pos in positions_history) / len(positions_history)
        avg_y = sum(pos[1] for pos in positions_history) / len(positions_history)
        return (avg_x, avg_y)
        
    def _create_target_region_mask(self, frame: np.ndarray, blob_positions: List[List[float]]) -> np.ndarray:
        """
        Create a mask for the target region based on blob positions
        This creates a polygon mask using the exact blob positions rather than a bounding rectangle
        
        Args:
            frame: Camera frame
            blob_positions: 4 blob positions defining the target corners (ordered)
            
        Returns:
            np.ndarray: Binary mask for the target region
        """
        try:
            # Create mask with same dimensions as frame
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            
            if len(blob_positions) != 4:
                logging.error(f"Expected 4 blob positions, got {len(blob_positions)}")
                return mask
                
            # Convert blob positions to numpy array and ensure integer coordinates
            points = np.array(blob_positions, dtype=np.float32)
            
            # Ensure points are within frame bounds
            points[:, 0] = np.clip(points[:, 0], 0, frame.shape[1] - 1)
            points[:, 1] = np.clip(points[:, 1], 0, frame.shape[0] - 1)
            
            # Convert to integer coordinates for cv2.fillPoly
            points_int = points.astype(np.int32)
            
            # Create a polygon mask using the exact blob positions
            # This ensures we only analyze the specific target area
            cv2.fillPoly(mask, [points_int], 255)
            
            # Calculate mask area for logging
            mask_area = cv2.countNonZero(mask)
            total_area = frame.shape[0] * frame.shape[1]
            coverage_percent = (mask_area / total_area) * 100
            
            logging.debug(f"Target region mask: {mask_area} pixels ({coverage_percent:.1f}% of frame)")
            logging.debug(f"Target corners: {points_int.tolist()}")
            
            return mask
            
        except Exception as e:
            logging.error(f"Error creating target region mask: {e}")
            # Return empty mask instead of full frame to avoid false detections
            return np.zeros(frame.shape[:2], dtype=np.uint8)
            
    def _get_rgb_enhanced_mask(self, frame: np.ndarray, mask: np.ndarray, 
                             previous_frame: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Enhanced RGB detection mask for 650nm red laser - improved sensitivity"""
        try:
            if previous_frame is None:
                return None
                
            # Calculate frame difference
            frame_diff = cv2.absdiff(frame, previous_frame)
            gray_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
            red_diff = cv2.absdiff(frame[:, :, 2], previous_frame[:, :, 2])
            
            # Much more relaxed movement thresholds for better sensitivity
            mean_gray_diff = np.mean(gray_diff)
            mean_red_diff = np.mean(red_diff)
            red_ratio = mean_red_diff / (mean_gray_diff + 1e-6)
            
            # Relaxed thresholds - allow very subtle changes
            if mean_gray_diff < 0.3 or red_ratio < 0.4:
                return None
                
            # Apply much lower and more sensitive thresholds
            _, red_thresh_conservative = cv2.threshold(red_diff, 1, 255, cv2.THRESH_BINARY)
            _, red_thresh_moderate = cv2.threshold(red_diff, 3, 255, cv2.THRESH_BINARY)
            _, red_thresh_aggressive = cv2.threshold(red_diff, 8, 255, cv2.THRESH_BINARY)
            
            # Add red color validation from current frame
            current_red = frame[:, :, 2]
            current_green = frame[:, :, 1]
            current_blue = frame[:, :, 0]
            
            # Create red dominance mask with relaxed criteria
            red_dominant = (current_red > current_green * 0.8) & (current_red > current_blue * 0.8)
            red_intense = current_red > 20  # Very low intensity threshold
            red_color_mask = (red_dominant & red_intense).astype(np.uint8) * 255
            
            # Try different threshold combinations, starting with most conservative
            test_masks = [
                cv2.bitwise_and(red_thresh_conservative, mask),
                cv2.bitwise_and(red_thresh_moderate, mask),
                cv2.bitwise_and(red_thresh_aggressive, mask)
            ]
            
            final_mask = None
            for test_mask in test_masks:
                # Combine with red color validation
                combined_mask = cv2.bitwise_and(test_mask, red_color_mask)
                
                # Apply morphological operations to clean up noise
                kernel = np.ones((2, 2), np.uint8)
                cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
                cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)
                
                # Apply final target mask
                final_candidate = cv2.bitwise_and(cleaned_mask, mask)
                
                # Check if we found any pixels
                if cv2.countNonZero(final_candidate) >= 1:
                    final_mask = final_candidate
                    break
            
            # If no mask worked with color validation, try without it (fallback)
            if final_mask is None:
                for test_mask in test_masks:
                    # Apply morphological operations
                    kernel = np.ones((2, 2), np.uint8)
                    cleaned_mask = cv2.morphologyEx(test_mask, cv2.MORPH_OPEN, kernel)
                    
                    if cv2.countNonZero(cleaned_mask) >= 1:
                        final_mask = cleaned_mask
                        break
                
            return final_mask
            
        except Exception as e:
            return None
            
    def _get_rgb_subtraction_mask(self, frame: np.ndarray, mask: np.ndarray, 
                                previous_frame: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """RGB subtraction detection mask"""
        try:
            if previous_frame is None:
                return None
                
            # Extract red channels
            current_red = frame[:, :, 2]
            previous_red = previous_frame[:, :, 2]
            
            # Calculate red difference
            red_diff = cv2.absdiff(current_red, previous_red)
            red_mean = np.mean(red_diff)
            
            if red_mean < 1.0:
                return None
                
            # Calculate threshold
            red_std = np.std(red_diff)
            threshold_value = max(8, red_mean + 1.5 * red_std)
            _, red_thresh = cv2.threshold(red_diff, threshold_value, 255, cv2.THRESH_BINARY)
            
            # Check red dominance in current frame
            red_channel = frame[:, :, 2]
            green_channel = frame[:, :, 1]
            blue_channel = frame[:, :, 0]
            
            red_dominant = (red_channel > green_channel * 1.1) & (red_channel > blue_channel * 1.1)
            red_intense = red_channel > 40
            red_dominant = red_dominant & red_intense
            red_dominant = red_dominant.astype(np.uint8) * 255
            
            # Combine masks
            combined_mask = cv2.bitwise_and(red_thresh, red_dominant)
            final_mask = cv2.bitwise_and(combined_mask, mask)
            
            # Apply morphological operations
            kernel = np.ones((3, 3), np.uint8)
            final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
            final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
            
            return final_mask if cv2.countNonZero(final_mask) >= 1 else None
            
        except Exception as e:
            return None
            
    def _get_hsv_subtraction_mask(self, frame: np.ndarray, mask: np.ndarray, 
                                previous_frame: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """HSV detection with frame subtraction mask"""
        try:
            if previous_frame is None:
                return None
                
            # Calculate difference
            diff_image = cv2.absdiff(frame, previous_frame)
            diff_intensity = np.mean(diff_image)
            
            if diff_intensity < 1.0:
                return None
                
            # Apply HSV detection to difference
            red_mask = self._detect_red_laser_spot_hsv(diff_image)
            final_mask = cv2.bitwise_and(red_mask, mask)
            
            return final_mask if cv2.countNonZero(final_mask) >= 1 else None
            
        except Exception as e:
            return None
            
    def _get_hsv_direct_mask(self, frame: np.ndarray, mask: np.ndarray, 
                           previous_frame: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Direct HSV detection mask"""
        try:
            red_mask = self._detect_red_laser_spot_hsv_relaxed(frame)
            final_mask = cv2.bitwise_and(red_mask, mask)
            
            return final_mask if cv2.countNonZero(final_mask) >= 1 else None
            
        except Exception as e:
            return None
            
    def _get_frame_difference_mask(self, frame: np.ndarray, mask: np.ndarray, 
                                 previous_frame: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Frame difference detection mask"""
        try:
            if previous_frame is None:
                return None
            
            # Calculate frame difference
            frame_diff = cv2.absdiff(frame, previous_frame)
            gray_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
            
            # Apply mask
            masked_diff = cv2.bitwise_and(gray_diff, mask)
            
            # Check if there's significant change
            mean_intensity = np.mean(masked_diff)
            max_intensity = np.max(masked_diff)
            
            if mean_intensity < 8 or max_intensity < 20:
                return None
            
            # Apply threshold
            threshold = max(3, mean_intensity * 0.5)
            _, binary_diff = cv2.threshold(masked_diff, threshold, 255, cv2.THRESH_BINARY)
            
            # Apply gentle morphological operations
            kernel = np.ones((2, 2), np.uint8)
            binary_diff = cv2.morphologyEx(binary_diff, cv2.MORPH_OPEN, kernel)
            binary_diff = cv2.morphologyEx(binary_diff, cv2.MORPH_CLOSE, kernel)
            
            return binary_diff if cv2.countNonZero(binary_diff) >= 1 else None
            
        except Exception as e:
            return None
            
    def _detect_red_laser_spot_hsv(self, frame: np.ndarray) -> np.ndarray:
        """Detect red laser spot using HSV color space"""
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # Red color ranges
        lower_red1 = np.array([0, 150, 150])
        upper_red1 = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

        lower_red2 = np.array([160, 150, 150])
        upper_red2 = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

        red_mask = cv2.bitwise_or(mask1, mask2)

        # Morphological operations
        kernel = np.ones((3, 3), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        red_mask = cv2.dilate(red_mask, kernel, iterations=1)

        return red_mask
        
    def _detect_red_laser_spot_hsv_relaxed(self, frame: np.ndarray) -> np.ndarray:
        """Relaxed HSV detection with more permissive parameters"""
        blurred = cv2.GaussianBlur(frame, (3, 3), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # More relaxed red ranges
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([20, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

        lower_red2 = np.array([150, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

        red_mask = cv2.bitwise_or(mask1, mask2)

        # Gentle morphological operations
        kernel = np.ones((2, 2), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        red_mask = cv2.dilate(red_mask, kernel, iterations=1)

        return red_mask
        
    def _validate_laser_spot(self, frame: np.ndarray, position: Tuple[float, float], 
                           strategy: str) -> bool:
        """
        Validate that a detected position is actually a laser spot
        
        Args:
            frame: Current frame
            position: Detected position (x, y)
            strategy: Detection strategy used
            
        Returns:
            bool: True if position is valid laser spot
        """
        try:
            x, y = int(position[0]), int(position[1])
            
            # Check bounds
            if not (0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]):
                return False
            
            # For RGB-based strategies, validate red dominance
            if strategy in ["rgb_enhanced", "rgb_subtraction"]:
                roi_size = 3
                x1, x2 = max(0, x-roi_size), min(frame.shape[1], x+roi_size+1)
                y1, y2 = max(0, y-roi_size), min(frame.shape[0], y+roi_size+1)
                
                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    b_mean, g_mean, r_mean = np.mean(roi, axis=(0,1))
                    red_dominance = r_mean / (g_mean + b_mean + 1e-6)
                    return red_dominance > 0.7 and r_mean > 40
            
            # For other strategies, basic validation
            return True
            
        except Exception as e:
            return False
            
    def _save_shots_to_json(self, shots: List[Dict]) -> str:
        """Save detected shots to detection_data.json file organized by target"""
        try:
            # Ensure data directory exists
            os.makedirs("data", exist_ok=True)
            
            # Output path
            json_path = os.path.join("data", "detection_data.json")
            
            # Group shots by target
            targets_data = {}
            
            for shot in shots:
                target_num = shot['target_number']
                
                if target_num not in targets_data:
                    targets_data[target_num] = {
                        "target_number": target_num,
                        "shots": []
                    }
                    
                # Add shot to target data
                shot_data = {
                    "timestamp": shot["timestamp"],
                    "shot_number": shot["shot_number"],
                    "absolute_position": shot["absolute_position"],
                    "detection_strategy": shot["detection_strategy"],
                    "frame_number": shot["frame_number"]
                }
                
                # Add laser_info if present
                if "laser_info" in shot:
                    shot_data["laser_info"] = shot["laser_info"]
                    
                targets_data[target_num]["shots"].append(shot_data)
            
            # Convert to list format and sort by target number
            detection_data = {
                "analysis_timestamp": datetime.now().isoformat(),
                "total_shots": len(shots),
                "targets": [targets_data[target_num] for target_num in sorted(targets_data.keys())]
            }
            
            # Save to JSON (overwrites existing file)
            with open(json_path, 'w') as f:
                json.dump(detection_data, f, indent=2)
            
            return json_path
            
        except Exception as e:
            logging.error(f"Error saving detection data to JSON: {e}")
            return None 

    def _validate_shots_in_target_region(self, shots: List[Dict], target_bounds: Dict, target_number: int) -> List[Dict]:
        """
        Validate that detected shots are within the target region bounds
        
        Args:
            shots: List of detected shots
            target_bounds: Target region bounds {min_x, max_x, min_y, max_y}
            target_number: Target number for logging
            
        Returns:
            List of validated shots within the target region
        """
        validated_shots = []
        
        for shot in shots:
            shot_x = shot['absolute_position']['x']
            shot_y = shot['absolute_position']['y']
            
            # Check if shot is within target bounds (with small tolerance)
            tolerance = 20  # pixels
            in_x_bounds = (target_bounds['min_x'] - tolerance <= shot_x <= target_bounds['max_x'] + tolerance)
            in_y_bounds = (target_bounds['min_y'] - tolerance <= shot_y <= target_bounds['max_y'] + tolerance)
            
            if in_x_bounds and in_y_bounds:
                validated_shots.append(shot)
                logging.debug(f"Target {target_number} - Shot at ({shot_x:.1f}, {shot_y:.1f}) validated")
            else:
                logging.warning(f"Target {target_number} - Shot at ({shot_x:.1f}, {shot_y:.1f}) outside target region, discarded")
                
        return validated_shots 