import cv2
import numpy as np
import json
import os
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any
import logging
from models.homography import HomographyModel


class ShotDetectionModel:
    """Model for detecting laser shots using multiple detection strategies"""
    
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
        """Detect shots for a specific target using multiple strategies in fallback order"""
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        detected_shots = []
        
        # Try each strategy until we find enough shots
        for strategy_idx, strategy_name in enumerate(self.strategies):
            logging.info(f"Analysing target {target_number} strategy {strategy_idx + 1}/{len(self.strategies)}: {strategy_name}")
            
            # Special handling for frame_difference strategy
            if strategy_name == "frame_difference":
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                detected_shots = self._detect_shots_frame_difference(
                    cap, target_number, blob_positions, shots_per_series, progress_callback, total_frames
                )
            else:
                # Standard frame-by-frame detection for other strategies
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                detected_shots = self._detect_shots_standard(
                    cap, target_number, blob_positions, strategy_name, shots_per_series, progress_callback, total_frames
                )
            
            logging.info(f"Analysing target {target_number} strategy {strategy_idx + 1}/{len(self.strategies)}: {len(detected_shots)} shots found")
            
            # If we found enough shots, stop trying other strategies
            if len(detected_shots) >= shots_per_series:
                break
                
        cap.release()
        return detected_shots
        
    def _detect_shot_in_camera_frame(self, frame: np.ndarray, previous_frame: np.ndarray,
                                   blob_positions: List[List[float]], strategy: str) -> Optional[Tuple[float, float]]:
        """
        Detect shot in the original camera frame, focused on the target region
        
        Args:
            frame: Current frame in camera coordinates
            previous_frame: Previous frame in camera coordinates  
            blob_positions: 4 blob positions defining the target region
            strategy: Detection strategy to use
            
        Returns:
            Tuple[float, float]: Shot position in camera coordinates, or None if no shot detected
        """
        try:
            # Create a mask for the target region defined by blob positions
            mask = self._create_target_region_mask(frame, blob_positions)
            
            # Apply detection strategy in camera frame
            shot_position = self._detect_shot_with_strategy(frame, mask, previous_frame, strategy)
            
            return shot_position
            
        except Exception as e:
            logging.error(f"Error detecting shot in camera frame: {e}")
            return None
            
    def _create_target_region_mask(self, frame: np.ndarray, blob_positions: List[List[float]]) -> np.ndarray:
        """
        Create a mask for the target region defined by blob positions
        
        Args:
            frame: Camera frame
            blob_positions: 4 blob positions defining the target corners
            
        Returns:
            np.ndarray: Binary mask for the target region
        """
        try:
            # Create mask with same dimensions as frame
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            
            # Convert blob positions to integer points
            points = np.array(blob_positions, dtype=np.int32)
            
            # Create polygon mask from blob positions
            cv2.fillPoly(mask, [points], 255)
            
            return mask
            
        except Exception as e:
            logging.error(f"Error creating target region mask: {e}")
            # Return full frame mask as fallback
            return np.ones(frame.shape[:2], dtype=np.uint8) * 255
        
    def _warp_to_square(self, frame: np.ndarray, points: List[List[float]], 
                       size: int) -> Tuple[np.ndarray, np.ndarray, int, Tuple[int, int], np.ndarray]:
        """Apply warping transformation to frame using blob coordinates"""
        try:
            # Convert points to proper format
            points_array = np.array(points, dtype="float32")
            
            # Define square points for warping (corners of the target area)
            square_points = np.float32([[0, 0], [size, 0], [size, size], [0, size]])
            
            # Calculate homography matrix
            matrix = cv2.getPerspectiveTransform(points_array, square_points)
            
            # Apply warping
            warped_frame = cv2.warpPerspective(frame, matrix, (size, size))
            
            # Create mask for the target area
            increased_radius = size // 2 - 18
            mask_center = (size // 2, size // 2)
            mask = np.zeros((size, size), dtype="uint8")
            cv2.circle(mask, mask_center, increased_radius, 255, -1)
            
            return warped_frame, mask, increased_radius, mask_center, matrix
            
        except Exception as e:
            logging.error(f"Error in warping: {e}")
            # Return original frame with basic mask if warping fails
            mask = np.ones((frame.shape[0], frame.shape[1]), dtype="uint8") * 255
            return frame, mask, frame.shape[0]//2, (frame.shape[1]//2, frame.shape[0]//2), np.eye(3)
            
    def _detect_shot_with_strategy(self, frame: np.ndarray, mask: np.ndarray, 
                                 previous_frame: np.ndarray, strategy: str) -> Optional[Tuple[float, float]]:
        """Detect shot using specific strategy"""
        try:
            if strategy == "rgb_enhanced":
                return self._detect_red_spot_enhanced_rgb(frame, mask, previous_frame)
            elif strategy == "rgb_subtraction":
                return self._detect_red_spot_rgb_subtraction(frame, mask, previous_frame)
            elif strategy == "hsv_subtraction":
                return self._detect_red_spot_with_hsv_subtraction(frame, mask, previous_frame)
            elif strategy == "hsv_direct":
                return self._detect_red_spot_hsv_direct(frame, mask, previous_frame)
            elif strategy == "frame_difference":
                return self._detect_shot_frame_difference_single(frame, mask, previous_frame)
            else:
                logging.warning(f"Unknown strategy: {strategy}")
                return None
                
        except Exception as e:
            logging.error(f"Error in strategy {strategy}: {e}")
            return None
            
    def _detect_red_spot_enhanced_rgb(self, frame: np.ndarray, mask: np.ndarray, 
                                    previous_frame: Optional[np.ndarray]) -> Optional[Tuple[float, float]]:
        """Enhanced RGB detection for 650nm red laser"""
        try:
            if previous_frame is None:
                return None
                
            # Calculate frame difference
            frame_diff = cv2.absdiff(frame, previous_frame)
            gray_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
            red_diff = cv2.absdiff(frame[:, :, 2], previous_frame[:, :, 2])
            
            # Check for sufficient movement
            mean_gray_diff = np.mean(gray_diff)
            mean_red_diff = np.mean(red_diff)
            red_ratio = mean_red_diff / (mean_gray_diff + 1e-6)
            
            if mean_gray_diff < 1.0 or red_ratio < 0.9:
                return None
                
            # Apply thresholds
            _, red_thresh_conservative = cv2.threshold(red_diff, 5, 255, cv2.THRESH_BINARY)
            _, red_thresh_aggressive = cv2.threshold(red_diff, 15, 255, cv2.THRESH_BINARY)
            
            # Choose appropriate threshold
            test_mask = cv2.bitwise_and(red_thresh_aggressive, mask)
            if cv2.countNonZero(test_mask) > 1:
                final_mask = test_mask
            else:
                final_mask = cv2.bitwise_and(red_thresh_conservative, mask)
                
            return self._validate_and_find_red_spot(frame, final_mask)
            
        except Exception as e:
            return None
            
    def _detect_red_spot_rgb_subtraction(self, frame: np.ndarray, mask: np.ndarray, 
                                       previous_frame: Optional[np.ndarray]) -> Optional[Tuple[float, float]]:
        """RGB subtraction method"""
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
            
            # Check red dominance
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
            
            if cv2.countNonZero(final_mask) < 1:
                return None
                
            return self._find_laser_spot_by_contours_strict(final_mask, frame)
            
        except Exception as e:
            return None
            
    def _detect_red_spot_with_hsv_subtraction(self, frame: np.ndarray, mask: np.ndarray, 
                                            previous_frame: Optional[np.ndarray]) -> Optional[Tuple[float, float]]:
        """HSV detection with frame subtraction"""
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
            
            return self._find_laser_spot_by_contours(final_mask)
            
        except Exception as e:
            return None
            
    def _detect_red_spot_hsv_direct(self, frame: np.ndarray, mask: np.ndarray, 
                                  previous_frame: Optional[np.ndarray]) -> Optional[Tuple[float, float]]:
        """Direct HSV detection on current frame"""
        try:
            red_mask = self._detect_red_laser_spot_hsv_relaxed(frame)
            final_mask = cv2.bitwise_and(red_mask, mask)
            
            return self._find_laser_spot_by_contours(final_mask)
            
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
        
    def _validate_and_find_red_spot(self, frame: np.ndarray, candidate_mask: np.ndarray) -> Optional[Tuple[float, float]]:
        """Validate and find the best red spot candidate"""
        contours, _ = cv2.findContours(candidate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        best_spot = None
        best_score = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if not (1 <= area <= 500):
                continue
                
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
                
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            
            # Validate red dominance
            x, y = int(cx), int(cy)
            if (0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]):
                roi_size = 3
                x1, x2 = max(0, x-roi_size), min(frame.shape[1], x+roi_size+1)
                y1, y2 = max(0, y-roi_size), min(frame.shape[0], y+roi_size+1)
                
                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    b_mean, g_mean, r_mean = np.mean(roi, axis=(0,1))
                    red_dominance = r_mean / (g_mean + b_mean + 1e-6)
                    intensity_score = r_mean / 255.0
                    area_score = min(area / 50.0, 1.0)
                    
                    total_score = red_dominance * intensity_score * area_score
                    
                    if total_score > best_score and red_dominance > 0.8:
                        best_score = total_score
                        best_spot = (cx, cy)
                        
        return best_spot
        
    def _find_laser_spot_by_contours_strict(self, mask: np.ndarray, frame: np.ndarray) -> Optional[Tuple[float, float]]:
        """Find laser spot with strict validation"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 1 <= area <= 300:
                valid_contours.append((contour, area))
                
        if not valid_contours:
            return None
            
        best_spot = None
        best_score = 0
        
        for contour, area in valid_contours:
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
                
            center_x = M["m10"] / M["m00"]
            center_y = M["m01"] / M["m00"]
            
            # Validate RGB
            x, y = int(center_x), int(center_y)
            if not (0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]):
                continue
                
            roi_size = 2
            x1, x2 = max(0, x-roi_size), min(frame.shape[1], x+roi_size+1)
            y1, y2 = max(0, y-roi_size), min(frame.shape[0], y+roi_size+1)
            
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue
                
            b_mean, g_mean, r_mean = np.mean(roi, axis=(0,1))
            red_dominance_avg = r_mean / (g_mean + b_mean + 1e-6)
            red_intensity = r_mean / 255.0
            red_contrast = (r_mean - g_mean) + (r_mean - b_mean)
            
            if (red_dominance_avg > 0.7 and red_intensity > 0.1 and 
                red_contrast > 5 and r_mean > 50):
                
                score = red_dominance_avg * red_intensity * (area / 50.0)
                
                if score > best_score:
                    best_score = score
                    best_spot = (center_x, center_y)
                    
        return best_spot
        
    def _find_laser_spot_by_contours(self, mask: np.ndarray) -> Optional[Tuple[float, float]]:
        """Find laser spot using contours"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 1 <= area <= 1000:
                valid_contours.append((contour, area))
                
        if not valid_contours:
            # Fallback to moments
            M = cv2.moments(mask)
            if M["m00"] > 0:
                center_x = M["m10"] / M["m00"]
                center_y = M["m01"] / M["m00"]
                return (center_x, center_y)
            return None
            
        # Choose largest contour
        largest_contour = max(valid_contours, key=lambda x: x[1])[0]
        
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            return None
            
        center_x = M["m10"] / M["m00"]
        center_y = M["m01"] / M["m00"]
        
        return (center_x, center_y)
        
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
                targets_data[target_num]["shots"].append({
                    "timestamp": shot["timestamp"],
                    "shot_number": shot["shot_number"],
                    "camera_position": shot["camera_position"],
                    "target_position": shot["target_position"],
                    "detection_strategy": shot["detection_strategy"],
                    "frame_number": shot["frame_number"]
                })
            
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

    def _detect_shots_standard(self, cap, target_number: int, blob_positions: List[List[float]], 
                              strategy_name: str, shots_per_series: int, progress_callback, total_frames: int) -> List[Dict]:
        """Standard frame-by-frame detection for rgb_enhanced, rgb_subtraction, hsv_subtraction, hsv_direct"""
        
        detected_shots = []
        frame_count = 0
        previous_frame = None
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Process frame for shot detection
                if previous_frame is not None:
                    # Detect shot in the original frame (camera coordinates)
                    shot_position_camera = self._detect_shot_in_camera_frame(
                        frame, previous_frame, blob_positions, strategy_name
                    )
                    
                    if shot_position_camera is not None:
                        # Transform camera coordinates to target coordinates using homography
                        target_position = self.homography_model.transform_camera_to_target(
                            shot_position_camera, target_number
                        )
                        
                        # Create shot data
                        shot_data = {
                            "timestamp": datetime.now().isoformat(),
                            "target_number": target_number,
                            "shot_number": len(detected_shots) + 1,
                            "camera_position": {
                                "x": shot_position_camera[0],
                                "y": shot_position_camera[1]
                            },
                            "target_position": {
                                "x": target_position[0],
                                "y": target_position[1]
                            },
                            "detection_strategy": strategy_name,
                            "frame_number": frame_count
                        }
                        
                        detected_shots.append(shot_data)
                        
                        # Check if we have enough shots
                        if len(detected_shots) >= shots_per_series:
                            break
                
                previous_frame = frame.copy()
                frame_count += 1
                
                # Progress callback
                if progress_callback and frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    progress_callback(progress)
                    
        except Exception as e:
            logging.error(f"Error in strategy {strategy_name}: {e}")
            
        return detected_shots
        
    def _detect_shots_frame_difference(self, cap, target_number: int, blob_positions: List[List[float]], 
                                     shots_per_series: int, progress_callback, total_frames: int) -> List[Dict]:
        """
        Detect shots using frame difference strategy with rising edge detection
        This method processes the entire video to detect shots by tracking intensity changes
        """
        
        detected_shots = []
        frame_count = 0
        previous_frame = None
        
        # Fine-tuned parameters for shot detection to get 5 shots
        min_intensity_threshold = 6   # Lower threshold to catch more shots
        shot_duration_threshold = 3   # Keep duration requirement
        cooldown_frames = 8          # Reduce cooldown to allow more shots
        
        # State tracking
        current_intensity_state = "low"  # "low" or "high"
        high_intensity_start_frame = None
        frames_since_last_shot = 0
        
        # Statistics for adaptive thresholding
        intensity_history = []
        max_intensity_seen = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if previous_frame is not None:
                    # Create target region mask
                    mask = self._create_target_region_mask(frame, blob_positions)
                    
                    # Calculate frame difference within target region
                    frame_diff = cv2.absdiff(frame, previous_frame)
                    
                    # Convert to grayscale and apply mask
                    gray_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
                    masked_diff = cv2.bitwise_and(gray_diff, mask)
                    
                    # Calculate intensity metrics
                    mean_intensity = np.mean(masked_diff)
                    max_intensity = np.max(masked_diff)
                    max_intensity_seen = max(max_intensity_seen, max_intensity)
                    
                    # Keep track of intensity history for adaptive thresholding
                    intensity_history.append(mean_intensity)
                    if len(intensity_history) > 30:  # Keep last 30 frames
                        intensity_history.pop(0)
                    
                    # Calculate adaptive threshold - more sensitive approach
                    if len(intensity_history) >= 10:
                        baseline_intensity = np.median(intensity_history)
                        intensity_std = np.std(intensity_history)
                        # Use a lower multiplier for higher sensitivity
                        adaptive_threshold = max(min_intensity_threshold, baseline_intensity + 1.2 * intensity_std)
                    else:
                        adaptive_threshold = min_intensity_threshold
                    
                    # State machine for shot detection
                    frames_since_last_shot += 1
                    
                    if current_intensity_state == "low":
                        # Looking for rising edge (shot appearing) - balanced sensitivity
                        # Require both significant mean and peak intensity
                        if ((mean_intensity > adaptive_threshold and max_intensity > adaptive_threshold * 1.8) or 
                            max_intensity > 35):  # Slightly lower absolute threshold
                            current_intensity_state = "high"
                            high_intensity_start_frame = frame_count
                    
                    elif current_intensity_state == "high":
                        # In high intensity state, check for falling edge or shot confirmation
                        frames_in_high_state = frame_count - high_intensity_start_frame
                        
                        # Check if intensity dropped (falling edge) - more flexible approach
                        falling_edge_detected = False
                        
                        # Primary falling edge criteria
                        if mean_intensity < adaptive_threshold * 0.8 and max_intensity < adaptive_threshold * 1.5:
                            falling_edge_detected = True
                        
                        # Alternative falling edge: significant drop from recent high
                        elif len(intensity_history) >= 5:
                            recent_max = max(intensity_history[-5:])
                            if max_intensity < recent_max * 0.6:
                                falling_edge_detected = True
                        
                        # Force shot detection if we've been in high state for reasonable duration
                        # This handles cases where the falling edge is gradual - balanced criteria
                        force_shot_detection = (frames_in_high_state >= 8 and 
                                              frames_in_high_state <= 18 and
                                              frames_since_last_shot >= cooldown_frames and
                                              max_intensity_seen > 25)  # Lower intensity requirement
                        
                        if falling_edge_detected or force_shot_detection:
                            # Falling edge detected - this completes a shot if it lasted long enough
                            if (frames_in_high_state >= shot_duration_threshold and 
                                frames_since_last_shot >= cooldown_frames):
                                
                                # Find the shot position using the high intensity region
                                # Use a lower threshold for better detection
                                search_threshold = min(adaptive_threshold * 0.3, 3.0)
                                shot_position = self._find_shot_position_in_difference(
                                    frame, previous_frame, mask, search_threshold
                                )
                                
                                if shot_position is not None:
                                    # Transform to target coordinates
                                    target_position = self.homography_model.transform_camera_to_target(
                                        shot_position, target_number
                                    )
                                    
                                    # Create shot data
                                    shot_data = {
                                        "timestamp": datetime.now().isoformat(),
                                        "target_number": target_number,
                                        "shot_number": len(detected_shots) + 1,
                                        "camera_position": {
                                            "x": shot_position[0],
                                            "y": shot_position[1]
                                        },
                                        "target_position": {
                                            "x": target_position[0],
                                            "y": target_position[1]
                                        },
                                        "detection_strategy": "frame_difference",
                                        "frame_number": frame_count,
                                        "intensity_info": {
                                            "mean_intensity": mean_intensity,
                                            "max_intensity": max_intensity,
                                            "threshold": adaptive_threshold,
                                            "duration_frames": frames_in_high_state,
                                            "detection_type": "force" if force_shot_detection else "edge"
                                        }
                                    }
                                    
                                    detected_shots.append(shot_data)
                                    frames_since_last_shot = 0
                                    
                                    # Check if we have enough shots
                                    if len(detected_shots) >= shots_per_series:
                                        break
                            
                            current_intensity_state = "low"
                        
                        # If in high state too long without falling edge, reset (noise rejection)
                        elif frames_in_high_state > 25:  # Allow slightly longer high states
                            current_intensity_state = "low"
                
                previous_frame = frame.copy()
                frame_count += 1
                
                # Progress callback
                if progress_callback and frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    progress_callback(progress)
                    
        except Exception as e:
            logging.error(f"Error in frame difference detection: {e}")
            
        return detected_shots
    
    def _find_shot_position_in_difference(self, frame: np.ndarray, previous_frame: np.ndarray, 
                                        mask: np.ndarray, threshold: float) -> Optional[Tuple[float, float]]:
        """
        Find the position of the shot in the difference image
        """
        try:
            # Calculate frame difference
            frame_diff = cv2.absdiff(frame, previous_frame)
            gray_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
            
            # Apply mask and threshold - use a lower threshold for better sensitivity
            masked_diff = cv2.bitwise_and(gray_diff, mask)
            adaptive_threshold = max(3, threshold)  # Ensure minimum threshold of 3
            _, binary_diff = cv2.threshold(masked_diff, adaptive_threshold, 255, cv2.THRESH_BINARY)
            
            # Apply gentle morphological operations to clean up noise
            kernel = np.ones((2, 2), np.uint8)  # Smaller kernel for better preservation
            binary_diff = cv2.morphologyEx(binary_diff, cv2.MORPH_OPEN, kernel)
            binary_diff = cv2.morphologyEx(binary_diff, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(binary_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                # Fallback to center of mass with lower threshold
                fallback_threshold = max(1, threshold * 0.3)
                _, binary_fallback = cv2.threshold(masked_diff, fallback_threshold, 255, cv2.THRESH_BINARY)
                moments = cv2.moments(binary_fallback)
                if moments["m00"] > 0:
                    cx = moments["m10"] / moments["m00"]
                    cy = moments["m01"] / moments["m00"]
                    return (cx, cy)
                return None
            
            # Find the largest significant contour with more permissive area
            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area >= 1:  # Very low minimum area for high sensitivity
                    valid_contours.append((contour, area))
            
            if not valid_contours:
                return None
            
            # Use the largest contour or the one with best properties
            best_contour = None
            best_score = 0
            
            for contour, area in valid_contours:
                # Calculate a score based on area and compactness
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    compactness = 4 * np.pi * area / (perimeter * perimeter)
                    score = area * compactness  # Favor larger, more compact regions
                    if score > best_score:
                        best_score = score
                        best_contour = contour
            
            if best_contour is None:
                best_contour = max(valid_contours, key=lambda x: x[1])[0]
            
            # Calculate centroid
            moments = cv2.moments(best_contour)
            if moments["m00"] > 0:
                cx = moments["m10"] / moments["m00"]
                cy = moments["m01"] / moments["m00"]
                return (cx, cy)
            
            return None
            
        except Exception as e:
            return None
    
    def _detect_shot_frame_difference_single(self, frame: np.ndarray, mask: np.ndarray, 
                                           previous_frame: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        Single frame detection for frame difference strategy (for compatibility with existing interface)
        This is a simplified version for use in the standard detection loop
        """
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
            
            # Find shot position
            return self._find_shot_position_in_difference(frame, previous_frame, mask, mean_intensity * 0.5)
            
        except Exception as e:
            return None 