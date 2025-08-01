import cv2
import os
import logging
import threading
import json
import numpy as np
from datetime import datetime
from PIL import Image
import customtkinter as ctk
from typing import Optional, Dict, Any, List, Tuple
from controllers.analysis_controller import AnalysisController
from models.calibration import CalibrationModel
from models.results import ResultsModel
from utils.enhanced_video_recorder import EnhancedVideoRecorder


class SimulationController:
    """Controller for simulation view and video recording pipeline"""
    
    def __init__(self, app_controller):
        self.app_controller = app_controller
        self.view = None
        
        # Camera access through app controller - no longer managed here
        self.is_camera_running = False
        self.camera_thread = None
        self.after_id = None
        
        # Enhanced video recording
        self.enhanced_recorder = EnhancedVideoRecorder()
        self.is_recording = False
        self.video_filepath: Optional[str] = None
        self.frames_written = 0
        self.recording_start_time = None
        
        # Video settings - optimized for shot detection
        self.fps = 30
        self.bitrate = 15000000  # 15 Mbps for high quality
        self.format_type = 'MJPEG'  # Can be 'MJPEG' or 'RAW'
        self.video_save_directory = "Video"
        
        # Create video directory if it doesn't exist
        os.makedirs(self.video_save_directory, exist_ok=True)
        
        # Analysis controller
        self.analysis_controller = AnalysisController()
        
        # Results processing models
        self.calibration_model = CalibrationModel()
        self.results_model = ResultsModel()
        
    def set_view(self, view):
        """Set the view that this controller manages"""
        self.view = view
        
    def start_simulation(self):
        """Start the simulation (camera feed)"""
        try:
            # Check if shared camera is available
            if not self.app_controller.is_camera_available():
                raise Exception("Shared camera not available")
                
            self._start_camera_feed()
            logging.info("Simulation started successfully")
            
        except Exception as e:
            logging.error(f"Error starting simulation: {e}")
            if self.view:
                self.view.show_no_camera_message()
                
    def stop_simulation(self):
        """Stop the simulation and clean up resources"""
        # Stop recording if active
        if self.is_recording:
            self.stop_recording()
            
        # Stop camera feed
        self._stop_camera_feed()
        
        # Note: Camera is not released here - it's managed by AppController
        logging.info("Simulation stopped")
        
    def _start_camera_feed(self):
        """Start the camera feed loop"""
        self.is_camera_running = True
        self._update_camera_frame()
        
    def _stop_camera_feed(self):
        """Stop the camera feed loop"""
        self.is_camera_running = False
        
        # Cancel scheduled updates
        if self.after_id:
            try:
                if self.view and self.view.parent:
                    self.view.parent.after_cancel(self.after_id)
            except:
                pass
            self.after_id = None
            
    def _update_camera_frame(self):
        """Update camera frame and handle video recording"""
        if not self.is_camera_running:
            return
            
        # Get shared camera from app controller
        camera = self.app_controller.get_camera()
        if not camera:
            # Schedule next update
            if self.is_camera_running and self.view and self.view.parent:
                self.after_id = self.view.parent.after(30, self._update_camera_frame)
            return
            
        try:
            ret, frame = camera.read()
            if not ret:
                # Schedule next update
                if self.is_camera_running and self.view and self.view.parent:
                    self.after_id = self.view.parent.after(30, self._update_camera_frame)
                return
                
            # Record frame if recording is active (for OpenCV fallback)
            if self.is_recording:
                self._record_frame(frame)
                
            # Convert frame for display
            display_image = self._convert_frame_for_display(frame)
            
            # Update view
            if self.view:
                self.view.update_video_frame(display_image)
                
        except Exception as e:
            logging.error(f"Error updating camera frame: {e}")
            
        # Schedule next update
        if self.is_camera_running and self.view and self.view.parent:
            self.after_id = self.view.parent.after(30, self._update_camera_frame)
            
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
            
    def start_recording(self):
        """Start enhanced video recording"""
        if self.is_recording:
            logging.warning("Recording already in progress")
            if self.view:
                self.view.stop_recording_loading()
            return
            
        try:
            # Generate unique filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"shooting_{timestamp}.avi"
            self.video_filepath = os.path.join(self.video_save_directory, filename)
            
            # Get shared camera
            camera = self.app_controller.get_camera()
            if not camera:
                raise RuntimeError("Camera not available")
                
            # Configure enhanced recorder for high-quality capture
            frame_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            self.enhanced_recorder.configure(
                width=frame_width,
                height=frame_height,
                fps=self.fps,
                bitrate=self.bitrate,
                format_type=self.format_type,
                use_hardware_acceleration=True
            )
            
            # Start enhanced recording
            if not self.enhanced_recorder.start_recording(self.video_filepath, camera):
                raise RuntimeError("Failed to start enhanced video recording")
                
            self.is_recording = True
            self.frames_written = 0
            self.recording_start_time = datetime.now()
            
            # Update view state
            if self.view:
                self.view.update_recording_state(True)
                self.view.stop_recording_loading()
                
            logging.info(f"🎬 Started enhanced recording to: {self.video_filepath}")
            
        except Exception as e:
            logging.error(f"Error starting recording: {e}")
            self.is_recording = False
            # Stop loading state on error
            if self.view:
                self.view.stop_recording_loading()
                
    def stop_recording(self):
        """Stop enhanced video recording"""
        if not self.is_recording:
            logging.warning("No recording in progress")
            if self.view:
                self.view.stop_recording_loading()
            return
            
        try:
            self.is_recording = False
            recording_end_time = datetime.now()
            
            # Stop enhanced recorder
            success = self.enhanced_recorder.stop_recording()
            if not success:
                logging.warning("Enhanced recorder reported stop issues")
                
            # Get final recording stats
            stats = self.enhanced_recorder.get_recording_stats()
            self.frames_written = stats.get('frames_written', 0)
            frames_dropped = stats.get('frames_dropped', 0)
            
            # Calculate recording duration
            duration = 0
            if self.recording_start_time:
                duration = (recording_end_time - self.recording_start_time).total_seconds()
                
            # Update view state
            if self.view:
                self.view.update_recording_state(False)
                self.view.stop_recording_loading()
                
            # Log recording results
            avg_fps = self.frames_written / duration if duration > 0 else 0
            logging.info(f"🎬 Recording stopped: {self.frames_written} frames, {duration:.2f}s, {avg_fps:.1f} FPS")
            if frames_dropped > 0:
                logging.warning(f"⚠️ {frames_dropped} frames were dropped during recording")
            else:
                logging.info("✅ No frames dropped - high quality recording achieved!")
                
            # Start video analysis automatically
            if self.video_filepath and os.path.exists(self.video_filepath):
                logging.info("Starting automatic video analysis...")
                self.analysis_controller.analyze_video_async(
                    self.video_filepath,
                    progress_callback=self._analysis_progress_callback,
                    completion_callback=self._analysis_completion_callback
                )
            
            return self.video_filepath
            
        except Exception as e:
            logging.error(f"Error stopping recording: {e}")
            # Stop loading state on error
            if self.view:
                self.view.stop_recording_loading()
            return None
            
    def _record_frame(self, frame):
        """Record a single frame to video file (for OpenCV fallback backend)"""
        try:
            if self.is_recording:
                # The enhanced recorder handles frame recording
                # For OpenCV fallback, it will use this method
                if self.enhanced_recorder:
                    success = self.enhanced_recorder.record_frame(frame)
                    if success:
                        # Update frame count from recorder stats
                        stats = self.enhanced_recorder.get_recording_stats()
                        self.frames_written = stats.get('frames_written', 0)
                        
                        # Log progress every 100 frames
                        if self.frames_written % 100 == 0:
                            logging.debug(f"Frames recorded: {self.frames_written}")
                    
        except Exception as e:
            logging.error(f"Error recording frame: {e}")
            
    def handle_recalibrate(self):
        """Handle recalibration request"""
        try:
            # Stop current simulation
            self.stop_simulation()
            
            # Navigate back to calibration
            self.app_controller.navigate_to_calibration()
            
            # Stop loading state after successful navigation
            if self.view:
                self.view.stop_recalibrate_loading()
                
        except Exception as e:
            logging.error(f"Error during recalibration: {e}")
            # Stop loading state on error
            if self.view:
                self.view.stop_recalibrate_loading()
        
    def cleanup(self):
        """Clean up resources when closing"""
        self.stop_simulation()
        
        # Clean up enhanced recorder
        if self.enhanced_recorder:
            self.enhanced_recorder.cleanup()
        
    def get_recording_status(self):
        """Get current recording status information"""
        base_status = {
            'is_recording': self.is_recording,
            'frames_written': self.frames_written,
            'video_filepath': self.video_filepath,
            'recording_start_time': self.recording_start_time
        }
        
        # Add enhanced recorder stats if available
        if self.enhanced_recorder:
            enhanced_stats = self.enhanced_recorder.get_recording_stats()
            base_status.update(enhanced_stats)
            
        return base_status
        
    def _analysis_progress_callback(self, progress: float):
        """Callback for video analysis progress updates"""
        logging.info(f"Video analysis progress: {progress:.1f}%")
        
    def _analysis_completion_callback(self, success: bool, message: str):
        """Callback for video analysis completion"""
        if success:
            logging.info(f"Video analysis completed successfully: {message}")
            # Automatically continue with data processing and result generation
            self._continue_pipeline_after_analysis()
        else:
            logging.error(f"Video analysis failed: {message}")
            
    def _continue_pipeline_after_analysis(self):
        """Continue the pipeline after video analysis completes"""
        try:
            logging.info("Starting automatic data processing...")
            
            # Step 3: Process detection data to generate relative positions
            processing_success = self.process_detection_data()
            if not processing_success:
                logging.error("Data processing failed. Pipeline stopped.")
                return
                
            logging.info("Data processing completed successfully. Starting result image generation...")
            
            # Step 4: Generate result images with shots and blob positions
            result_generation_success = self.generate_result_images()
            if result_generation_success:
                logging.info("Complete pipeline finished successfully!")
                logging.info("Results available in:")
                logging.info("  - Detection data: data/detection_data.json")
                logging.info("  - Results data: data/results_data.json")
                
                # List generated result images
                status = self.get_processing_status()
                if status.get("result_images"):
                    logging.info("  - Result images:")
                    for image_file in status["result_images"]:
                        logging.info(f"    * data/{image_file}")
            else:
                logging.error("Result image generation failed.")
                
        except Exception as e:
            logging.error(f"Error in pipeline continuation: {e}")
            
    def is_analysis_running(self) -> bool:
        """Check if video analysis is currently running"""
        return self.analysis_controller.is_analysis_running()
        
    # Results processing methods (moved from ResultsController)
    def process_detection_data(self) -> bool:
        """
        Process detection data and calibration data to generate relative positions
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load calibration data
            if not self.calibration_model.load_calibration_data():
                logging.error("No calibration data found. Cannot process detection data.")
                return False
                
            calibration_data = {
                'parameters': self.calibration_model.get_all_parameters(),
                'fixed_target_positions': self.calibration_model.get_blob_positions()
            }
            
            # Load detection data
            detection_data = self._load_detection_data()
            if not detection_data:
                logging.error("No detection data found. Cannot process.")
                return False
                
            # Process data to generate relative positions
            results_data = self.results_model.calculate_relative_positions(
                detection_data, calibration_data
            )
            
            # Save results data
            success = self.results_model.save_results_data(results_data)
            if success:
                logging.info("Detection data processed successfully and results saved.")
                return True
            else:
                logging.error("Failed to save results data.")
                return False
                
        except Exception as e:
            logging.error(f"Error processing detection data: {e}")
            return False
            
    def generate_result_images(self) -> bool:
        """
        Generate result images with labeled shots using alvobase image and results data
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load results data
            results_data = self.results_model.load_results_data()
            if not results_data:
                logging.error("No results data found. Run process-data first.")
                return False
                
            # Generate images for each target
            success = self.results_model.generate_target_images(results_data)
            if success:
                logging.info("Result images generated successfully.")
                return True
            else:
                logging.error("Failed to generate result images.")
                return False
                
        except Exception as e:
            logging.error(f"Error generating result images: {e}")
            return False
            
    def _load_detection_data(self) -> Optional[Dict]:
        """Load detection data from JSON file"""
        try:
            detection_file = os.path.join("data", "detection_data.json")
            if not os.path.exists(detection_file):
                return None
                
            with open(detection_file, 'r') as f:
                return json.load(f)
                
        except Exception as e:
            logging.error(f"Error loading detection data: {e}")
            return None
            
    def get_processing_status(self) -> Dict[str, Any]:
        """Get status of data processing pipeline"""
        status = {
            "calibration_data_exists": False,
            "detection_data_exists": False,
            "results_data_exists": False,
            "result_images_exist": False
        }
        
        try:
            # Check calibration data
            status["calibration_data_exists"] = os.path.exists(
                os.path.join("data", "calibration_data.json")
            )
            
            # Check detection data
            status["detection_data_exists"] = os.path.exists(
                os.path.join("data", "detection_data.json")
            )
            
            # Check results data
            status["results_data_exists"] = os.path.exists(
                os.path.join("data", "results_data.json")
            )
            
            # Check result images (look for any target result images)
            result_images = []
            if os.path.exists("data"):
                for file in os.listdir("data"):
                    if file.startswith("target_") and file.endswith("_result.jpg"):
                        result_images.append(file)
            status["result_images_exist"] = len(result_images) > 0
            status["result_images"] = result_images
            
        except Exception as e:
            logging.error(f"Error checking processing status: {e}")
            
        return status 