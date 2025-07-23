import cv2
import os
import logging
import threading
from datetime import datetime
from PIL import Image
import customtkinter as ctk
from typing import Optional
from controllers.video_analysis_controller import VideoAnalysisController
from controllers.results_controller import ResultsController


class SimulationController:
    """Controller for simulation view and video recording pipeline"""
    
    def __init__(self, app_controller):
        self.app_controller = app_controller
        self.view = None
        
        # Camera and video feed
        self.camera = None
        self.is_camera_running = False
        self.camera_thread = None
        self.after_id = None
        
        # Video recording
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.is_recording = False
        self.video_filepath: Optional[str] = None
        self.frames_written = 0
        self.recording_start_time = None
        
        # Video settings
        self.fps = 30
        self.codec = 'mp4v'
        self.video_save_directory = "Video"
        
        # Create video directory if it doesn't exist
        os.makedirs(self.video_save_directory, exist_ok=True)
        
        # Video analysis controller
        self.video_analysis_controller = VideoAnalysisController()
        
        # Results controller for data processing and result generation
        self.results_controller = ResultsController()
        
    def set_view(self, view):
        """Set the view that this controller manages"""
        self.view = view
        
    def start_simulation(self):
        """Start the simulation (camera feed)"""
        try:
            self._initialize_camera()
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
        
        # Release camera
        if self.camera:
            self.camera.release()
            self.camera = None
            
        logging.info("Simulation stopped")
        
    def _initialize_camera(self):
        """Initialize the camera"""
        try:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                raise Exception("Could not open camera")
                
            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, self.fps)
            
            logging.info("Camera initialized successfully")
            
        except Exception as e:
            logging.error(f"Error initializing camera: {e}")
            raise
            
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
        if not self.is_camera_running or not self.camera:
            return
            
        try:
            ret, frame = self.camera.read()
            if not ret:
                # Schedule next update
                if self.is_camera_running and self.view and self.view.parent:
                    self.after_id = self.view.parent.after(30, self._update_camera_frame)
                return
                
            # Record frame if recording is active
            if self.is_recording and self.video_writer:
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
            
            # Resize to fit display area (adjust as needed)
            pil_image = pil_image.resize((640, 480), Image.Resampling.LANCZOS)
            
            # Create CTkImage
            ctk_image = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=(640, 480))
            
            return ctk_image
            
        except Exception as e:
            logging.error(f"Error converting frame for display: {e}")
            return None
            
    def start_recording(self):
        """Start video recording"""
        if self.is_recording:
            logging.warning("Recording already in progress")
            if self.view:
                self.view.stop_recording_loading()
            return
            
        try:
            # Create video filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"shooting_{timestamp}.mp4"
            self.video_filepath = os.path.join(self.video_save_directory, filename)
            
            # Get frame dimensions from camera
            if self.camera:
                frame_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            else:
                frame_width, frame_height = 640, 480
                
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            self.video_writer = cv2.VideoWriter(
                self.video_filepath, fourcc, self.fps, (frame_width, frame_height)
            )
            
            if not self.video_writer.isOpened():
                raise RuntimeError("Failed to initialize video writer")
                
            self.is_recording = True
            self.frames_written = 0
            self.recording_start_time = datetime.now()
            
            # Update view state
            if self.view:
                self.view.update_recording_state(True)
                self.view.stop_recording_loading()
                
            logging.info(f"Started recording to: {self.video_filepath}")
            
        except Exception as e:
            logging.error(f"Error starting recording: {e}")
            self.is_recording = False
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            # Stop loading state on error
            if self.view:
                self.view.stop_recording_loading()
                
    def stop_recording(self):
        """Stop video recording"""
        if not self.is_recording:
            logging.warning("No recording in progress")
            if self.view:
                self.view.stop_recording_loading()
            return
            
        try:
            self.is_recording = False
            recording_end_time = datetime.now()
            
            # Release video writer
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
                
            # Calculate recording duration
            if self.recording_start_time:
                duration = (recording_end_time - self.recording_start_time).total_seconds()
                
            # Update view state
            if self.view:
                self.view.update_recording_state(False)
                self.view.stop_recording_loading()
                
            logging.info(f"Recording stopped. Saved {self.frames_written} frames to: {self.video_filepath}")
            logging.info(f"Recording duration: {duration:.2f} seconds")
            
            # Start video analysis automatically
            if self.video_filepath and os.path.exists(self.video_filepath):
                logging.info("Starting automatic video analysis...")
                self.video_analysis_controller.analyze_video_async(
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
        """Record a single frame to video file"""
        try:
            if self.video_writer and self.is_recording:
                self.video_writer.write(frame)
                self.frames_written += 1
                
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
        
    def get_recording_status(self):
        """Get current recording status information"""
        return {
            'is_recording': self.is_recording,
            'frames_written': self.frames_written,
            'video_filepath': self.video_filepath,
            'recording_start_time': self.recording_start_time
        }
        
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
            processing_success = self.results_controller.process_detection_data()
            if not processing_success:
                logging.error("Data processing failed. Pipeline stopped.")
                return
                
            logging.info("Data processing completed successfully. Starting result image generation...")
            
            # Step 4: Generate result images with shots and blob positions
            result_generation_success = self.results_controller.generate_result_images()
            if result_generation_success:
                logging.info("Complete pipeline finished successfully!")
                logging.info("Results available in:")
                logging.info("  - Detection data: data/detection_data.json")
                logging.info("  - Results data: data/results_data.json")
                
                # List generated result images
                status = self.results_controller.get_processing_status()
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
        return self.video_analysis_controller.is_analysis_running() 