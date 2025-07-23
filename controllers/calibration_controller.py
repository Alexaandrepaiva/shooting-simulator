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
        
        # Camera and detection variables
        self.camera = None
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
            
            # Initialize camera
            self._initialize_camera()
            
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
            
        # Release camera
        if self.camera:
            self.camera.release()
            self.camera = None
            
        logging.info("Calibration stopped")
        
    def _initialize_camera(self):
        """Initialize the camera"""
        try:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                raise Exception("Could not open camera")
                
            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            logging.info("Camera initialized successfully")
            
        except Exception as e:
            logging.error(f"Error initializing camera: {e}")
            raise
            
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
        if not self.is_running or not self.camera:
            return
            
        try:
            ret, frame = self.camera.read()
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
        try:
            # Convert to HSV for better color separation
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Apply CLAHE to V channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            h, s, v = cv2.split(hsv)
            v = clahe.apply(v)
            hsv = cv2.merge([h, s, v])
            
            # Convert back to BGR then to grayscale
            enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            return blurred
            
        except Exception as e:
            logging.error(f"Error optimizing frame: {e}")
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
    def _store_keypoints(self, keypoints):
        """Store keypoints for recurrent detection"""
        self.recent_blob_detections.append(keypoints)
        if len(self.recent_blob_detections) > 10:
            self.recent_blob_detections.pop(0)
            
    def _find_recurrent_blobs(self):
        """Find points that appear recurrently"""
        all_points = []
        for kps in self.recent_blob_detections:
            for kp in kps:
                all_points.append((int(kp.pt[0]), int(kp.pt[1])))
        return all_points
        
    def _cluster_blobs(self, points, distance_threshold=10):
        """Cluster nearby points"""
        clusters = []
        for point in points:
            found_cluster = False
            for cluster in clusters:
                for member_point in cluster[1]:
                    if np.linalg.norm(np.array(point) - np.array(member_point)) < distance_threshold:
                        cluster[1].append(point)
                        found_cluster = True
                        break
                if found_cluster:
                    break
            if not found_cluster:
                clusters.append((point, [point]))
        return clusters
        
    def _select_recurrent_blobs(self, clustered_points, expected_blobs):
        """Select the most recurrent blobs - exactly expected_blobs count"""
        sorted_clusters = sorted(clustered_points, key=lambda x: len(x[1]), reverse=True)
        selected_clusters = sorted_clusters[:expected_blobs]
        return selected_clusters
        
    def _paint_blobs_green(self, frame):
        """Paint green circles on detected blob positions"""
        for pos in self.fixed_target_positions:
            pos_tuple = tuple(int(x) for x in pos)
            cv2.circle(frame, pos_tuple, 12, (0, 200, 0), 2)
            
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
            
            # Navigate to simulation view after calibration
            self.stop_calibration()
            self.app_controller.navigate_to_simulation()
            
        except Exception as e:
            logging.error(f"Error saving calibration parameters: {e}")
            
    def cleanup(self):
        """Clean up resources when closing"""
        self.stop_calibration() 