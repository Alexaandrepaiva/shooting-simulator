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
            
            # Step 1: Analyze video for shot detection (saves absolute positions to detection_data.json)
            detected_shots = self.shot_detection_model.detect_shots_in_video(
                video_path, calibration_data, progress_callback
            )
            
            # Step 2: Process detection data and apply homography transformations
            if detected_shots:
                logging.info("Processing detection data and applying homography transformations...")
                
                # Import required controllers
                from controllers.results_controller import ResultsController
                from models.results import ResultsModel
                
                # Initialize results controller
                results_controller = ResultsController()
                
                # Process detection data to apply homography and get target positions
                success = results_controller.process_detection_data()
                if not success:
                    logging.error("Failed to process detection data")
                    if completion_callback:
                        completion_callback(False, "Failed to process detection data")
                    return
                
                # Step 3: Generate result images using the processed results data
                results_model = ResultsModel()
                results_data = results_model.load_results_data()
                if results_data:
                    image_success = results_model.generate_target_images(results_data)
                    if image_success:
                        logging.info("Result images generated successfully")
                    else:
                        logging.warning("Failed to generate result images")
                else:
                    logging.error("No results data found for image generation")
                
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