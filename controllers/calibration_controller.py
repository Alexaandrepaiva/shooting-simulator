import cv2
import numpy as np
import logging
import json
import os
from PIL import Image
import customtkinter as ctk
from models.calibration import CalibrationModel
from models.homography import HomographyModel


class CalibrationController:
    """Controller for calibration view and blob detection"""
    
    def __init__(self, app_controller):
        self.app_controller = app_controller
        self.view = None
        self.calibration_model = CalibrationModel()
        self.homography_model = HomographyModel()
        
        # Camera access through app controller - no longer managed here
        self.is_running = False
        self.after_id = None
        
        # Blob detection
        self.detector = None
        self.detector_params = cv2.SimpleBlobDetector_Params()
        self.recent_blob_detections = []
        self.fixed_target_positions = []
        
        # Setup blob detector parameters
        self._configure_detector_params()
        
    def set_view(self, view):
        """Set the view that this controller manages"""
        self.view = view
        
    def start_calibration(self):
        """Start the calibration process"""
        try:
            # Load saved parameters if available
            self.calibration_model.load_calibration_data()
            
            # Update view with loaded parameters
            if self.view:
                self.view.set_parameter_values(self.calibration_model.get_all_parameters())
            
            # Check if shared camera is available
            if not self.app_controller.is_camera_available():
                raise Exception("Shared camera not available")
            
            # Start video capture loop
            self.is_running = True
            self._update_frame()
            
            logging.info("Calibration started successfully")
            
        except Exception as e:
            logging.error(f"Error starting calibration: {e}")
            
    def stop_calibration(self):
        """Stop the calibration process"""
        self.is_running = False
        
        # Cancel scheduled updates
        if self.after_id:
            try:
                self.view.parent.after_cancel(self.after_id)
            except:
                pass
            self.after_id = None
            
        # Note: Camera is not released here - it's managed by AppController
        logging.info("Calibration stopped")
        
    def _configure_detector_params(self):
        """Configure blob detector parameters"""
        # Filter by Area
        self.detector_params.filterByArea = True
        self.detector_params.minArea = 30
        self.detector_params.maxArea = 400
        
        # Filter by Circularity
        self.detector_params.filterByCircularity = True
        self.detector_params.minCircularity = 0.5
        
        # Disable other filters to match original code
        self.detector_params.filterByConvexity = False
        self.detector_params.filterByInertia = False
        
        # Set distance between blobs
        self.detector_params.minDistBetweenBlobs = 5
        
        # Create detector
        self.detector = cv2.SimpleBlobDetector_create(self.detector_params)
        
    def _update_frame(self):
        """Update video frame and detect blobs"""
        if not self.is_running:
            return
            
        # Get shared camera from app controller
        camera = self.app_controller.get_camera()
        if not camera:
            # Schedule next update
            if self.is_running:
                self.after_id = self.view.parent.after(30, self._update_frame)
            return
            
        try:
            ret, frame = camera.read()
            if not ret:
                self.after_id = self.view.parent.after(30, self._update_frame)
                return
                
            # Update detector parameters with current values
            self._update_detector_params()
            
            # Detect blobs
            processed_frame = self._detect_blobs(frame)
            
            # Convert frame for display
            display_image = self._convert_frame_for_display(processed_frame)
            
            # Update view
            if self.view:
                self.view.update_video_frame(display_image)
                
        except Exception as e:
            logging.error(f"Error updating frame: {e}")
            
        # Schedule next update
        if self.is_running:
            self.after_id = self.view.parent.after(30, self._update_frame)
            
    def _update_detector_params(self):
        """Update detector parameters with current values from view"""
        params = self.calibration_model.get_all_parameters()
        
        self.detector_params.minArea = max(1, params['min_area'])
        self.detector_params.maxArea = max(self.detector_params.minArea, params['max_area'])
        self.detector_params.minCircularity = params['min_circularity']
        
        # Recreate detector with updated parameters
        self.detector = cv2.SimpleBlobDetector_create(self.detector_params)
        
    def _detect_blobs(self, frame):
        """Detect blobs in the frame and draw green circles"""
        # Optimize frame for blob detection
        optimized_frame = self._optimize_frame(frame)
        
        # Detect keypoints
        keypoints = self.detector.detect(optimized_frame)
        
        # Store keypoints for recurrent blob detection
        self._store_keypoints(keypoints)
        
        # Find recurrent blobs
        all_points = self._find_recurrent_blobs()
        clustered_points = self._cluster_blobs(all_points, distance_threshold=10)
        
        # Select recurrent blobs based on number of targets (4 blobs per target)
        number_of_targets = self.calibration_model.get_parameter('number_of_targets')
        expected_blobs = number_of_targets * 4  # Exactly 4 blobs per target
        selected_clusters = self._select_recurrent_blobs(clustered_points, expected_blobs)
        
        # Store positions for saving
        self.fixed_target_positions = [np.mean(cluster[1], axis=0).tolist() for cluster in selected_clusters]
        
        # Log blob detection info
        if len(self.fixed_target_positions) > 0:
            logging.debug(f"Detecting {len(self.fixed_target_positions)} blobs for {number_of_targets} targets (expected: {expected_blobs})")
        
        # Draw green circles on detected blobs
        result_frame = frame.copy()
        self._paint_blobs_green(result_frame)
        
        return result_frame
            
    def _optimize_frame(self, frame):
        """Optimize frame for blob detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        return blurred
        
    def _store_keypoints(self, keypoints):
        """Store keypoints for recurrent blob detection"""
        current_points = [(kp.pt[0], kp.pt[1]) for kp in keypoints]
        
        # Add to recent detections
        self.recent_blob_detections.append(current_points)
        
        # Keep only last 30 frames
        if len(self.recent_blob_detections) > 30:
            self.recent_blob_detections.pop(0)
            
    def _find_recurrent_blobs(self):
        """Find blobs that appear consistently across multiple frames"""
        all_points = []
        
        for frame_points in self.recent_blob_detections:
            all_points.extend(frame_points)
            
        return all_points
        
    def _cluster_blobs(self, points, distance_threshold=10):
        """Cluster nearby blob detections"""
        if not points:
            return []
            
        clusters = []
        points = list(points)
        
        while points:
            # Start new cluster with first point
            current_point = points.pop(0)
            cluster = [current_point]
            
            # Find all points within threshold distance
            i = 0
            while i < len(points):
                point = points[i]
                
                # Check if point is close to any point in current cluster
                close_to_cluster = False
                for cluster_point in cluster:
                    distance = np.sqrt((point[0] - cluster_point[0])**2 + (point[1] - cluster_point[1])**2)
                    if distance <= distance_threshold:
                        close_to_cluster = True
                        break
                        
                if close_to_cluster:
                    cluster.append(points.pop(i))
                else:
                    i += 1
                    
            clusters.append((len(cluster), cluster))
            
        # Sort by cluster size (number of detections)
        clusters.sort(key=lambda x: x[0], reverse=True)
        
        return clusters
        
    def _select_recurrent_blobs(self, clustered_points, max_blobs):
        """Select the most recurrent blobs up to max_blobs"""
        # Take clusters with most detections
        selected_clusters = clustered_points[:max_blobs]
        
        # Filter clusters that have at least 5 detections
        min_detections = 5
        selected_clusters = [cluster for cluster in selected_clusters if cluster[0] >= min_detections]
        
        return selected_clusters
        
    def _paint_blobs_green(self, frame):
        """Paint green circles on detected blob positions"""
        for position in self.fixed_target_positions:
            center = (int(position[0]), int(position[1]))
            cv2.circle(frame, center, 8, (0, 255, 0), 2)  # Green circle
            
    def _convert_frame_for_display(self, frame):
        """Convert OpenCV frame to CTkImage for display"""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(rgb_frame)
            
            # Resize to fit display area
            pil_image = pil_image.resize((640, 480), Image.Resampling.LANCZOS)
            
            # Create CTkImage
            ctk_image = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=(640, 480))
            
            return ctk_image
            
        except Exception as e:
            logging.error(f"Error converting frame for display: {e}")
            return None
            
    def on_parameter_change(self, param_name, value):
        """Handle parameter changes from the view"""
        self.calibration_model.update_parameter(param_name, value)
        
    def save_parameters(self):
        """Save calibration parameters and blob positions to JSON"""
        try:
            # Update model with current blob positions
            self.calibration_model.set_blob_positions(self.fixed_target_positions)
            
            # Save to JSON file
            filepath = self.calibration_model.save_calibration_data()
            
            num_targets = self.calibration_model.get_parameter('number_of_targets')
            expected_blobs = num_targets * 4
            
            logging.info(f"Calibration parameters saved successfully to {filepath}")
            logging.info(f"Targets: {num_targets}, Expected blobs: {expected_blobs}, Detected blobs: {len(self.fixed_target_positions)}")
            
            if len(self.fixed_target_positions) < expected_blobs:
                logging.warning(f"Warning: Only detected {len(self.fixed_target_positions)} blobs, expected {expected_blobs} for {num_targets} targets")
            
            # Calculate and save homographies
            calibration_data = {
                'parameters': self.calibration_model.get_all_parameters(),
                'fixed_target_positions': self.fixed_target_positions
            }
            
            homography_success = self.homography_model.calculate_and_save_homographies(calibration_data)
            if homography_success:
                logging.info("Homographies calculated and saved successfully")
            else:
                logging.error("Failed to calculate homographies")
            
            # Stop loading state before navigation
            if self.view:
                self.view.stop_save_loading()
            
            # Navigate to simulation view after calibration
            self.stop_calibration()
            self.app_controller.navigate_to_simulation()
            
        except Exception as e:
            logging.error(f"Error saving calibration parameters: {e}")
            # Stop loading state on error
            if self.view:
                self.view.stop_save_loading()
            
    def cleanup(self):
        """Clean up resources when closing"""
        self.stop_calibration() 