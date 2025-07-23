#!/usr/bin/env python3
"""
Pipeline CLI for Shooting Simulator
Allows testing video recording functionality independently.
"""

import argparse
import sys
import os
import cv2
import logging
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from controllers.video_analysis_controller import VideoAnalysisController
from controllers.results_controller import ResultsController


class PipelineCLI:
    """Command-line interface for testing pipeline components"""
    
    def __init__(self):
        self.setup_logging()
        self.video_save_directory = "Video"
        os.makedirs(self.video_save_directory, exist_ok=True)
        
    def setup_logging(self):
        """Setup logging for CLI operations"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler()
            ]
        )
        
    def record_video(self):
        """
        Record video from camera until 'q' key is pressed
        Uses the same filename pattern as the app
        """
        logging.info("Starting video recording... Press 'q' to stop recording")
        
        try:
            # Initialize camera
            camera = cv2.VideoCapture(0)
            if not camera.isOpened():
                raise Exception("Could not open camera")
                
            # Set camera properties (same as app)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camera.set(cv2.CAP_PROP_FPS, 30)
            
            # Get frame dimensions
            frame_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = 30
            
            # Create output filename (same pattern as app)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"shooting_{timestamp}.mp4"
            output_path = os.path.join(self.video_save_directory, filename)
            
            # Initialize video writer (same codec as app)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
            
            if not video_writer.isOpened():
                raise RuntimeError("Failed to initialize video writer")
                
            logging.info(f"Recording to: {output_path}")
            logging.info("Press 'q' to stop recording...")
            
            # Record video
            start_time = datetime.now()
            frames_recorded = 0
            
            while True:
                ret, frame = camera.read()
                if not ret:
                    logging.warning("Failed to capture frame")
                    continue
                    
                # Write frame
                video_writer.write(frame)
                frames_recorded += 1
                
                # Display frame for monitoring (optional)
                cv2.imshow('Recording - Press Q to stop', frame)
                
                # Check for 'q' key press to stop recording
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    logging.info("Stop recording requested by user")
                    break
                    
                # Show progress every 5 seconds
                elapsed = (datetime.now() - start_time).total_seconds()
                if frames_recorded % (fps * 5) == 0 and frames_recorded > 0:
                    logging.info(f"Recording... {elapsed:.1f} seconds elapsed")
                    
            # Calculate final duration
            final_duration = (datetime.now() - start_time).total_seconds()
            
            # Cleanup
            video_writer.release()
            camera.release()
            cv2.destroyAllWindows()
            
            logging.info(f"Recording completed!")
            logging.info(f"- File: {output_path}")
            logging.info(f"- Frames: {frames_recorded}")
            logging.info(f"- Duration: {final_duration:.2f} seconds")
            logging.info(f"- Average FPS: {frames_recorded / final_duration:.2f}")
            
            return output_path
            
        except Exception as e:
            logging.error(f"Error during video recording: {e}")
            # Cleanup on error
            try:
                if 'video_writer' in locals():
                    video_writer.release()
                if 'camera' in locals():
                    camera.release()
                cv2.destroyAllWindows()
            except:
                pass
            return None
            
    def analyze_video(self, video_path: str):
        """
        Analyze a video file to detect laser shots and update detection_data.json
        
        This performs the same analysis as the main system:
        - Uses calibration data from data/calibration_data.json
        - Applies warped frame analysis using blob coordinates
        - Uses 5 detection strategies: RGB Enhanced, RGB Subtraction, HSV Subtraction, HSV Direct, and Frame Difference (fallback)
        - Frame Difference strategy uses consecutive frame comparison without color filtering as last resort
        - Saves results to data/detection_data.json (overwrites existing file)
        - Generates result images in Video/ directory
        
        Args:
            video_path: Path to the video file to analyze
        """
        logging.info(f"Starting video analysis for: {video_path}")
        logging.info("This will update data/detection_data.json with shot detection results")
        
        if not os.path.exists(video_path):
            logging.error(f"Video file not found: {video_path}")
            return
            
        try:
            # Check if homographies exist, generate if missing
            homographies_file = os.path.join("data", "homographies.json")
            if not os.path.exists(homographies_file):
                logging.info("Homographies file missing. Generating from calibration data...")
                if not self.generate_homographies():
                    logging.error("Cannot proceed without homographies. Please check calibration data.")
                    return
            
            # Initialize video analysis controller
            analysis_controller = VideoAnalysisController()
            
            # Analyze video synchronously for CLI (same as system analysis)
            analysis_controller._analyze_video_worker(
                video_path,
                progress_callback=self._analysis_progress,
                completion_callback=self._analysis_complete
            )
            
        except Exception as e:
            logging.error(f"Error during video analysis: {e}")
            
    def generate_homographies(self):
        """
        Generate homographies.json from existing calibration data
        This is useful when the homographies file is missing
        """
        logging.info("Generating homographies from calibration data...")
        
        try:
            from models.calibration import CalibrationModel
            from models.homography import HomographyModel
            
            # Load calibration data
            calibration_model = CalibrationModel()
            if not calibration_model.load_calibration_data():
                logging.error("No calibration data found. Please run calibration first.")
                return False
                
            # Prepare calibration data
            calibration_data = {
                'parameters': calibration_model.get_all_parameters(),
                'fixed_target_positions': calibration_model.get_blob_positions()
            }
            
            # Generate homographies
            homography_model = HomographyModel()
            success = homography_model.calculate_and_save_homographies(calibration_data)
            
            if success:
                logging.info("Homographies generated successfully!")
                return True
            else:
                logging.error("Failed to generate homographies")
                return False
                
        except Exception as e:
            logging.error(f"Error generating homographies: {e}")
            return False
            
    def process_data(self):
        """
        Process detection data and calibration data to generate relative positions
        
        This step:
        - Loads calibration data (blob positions) from data/calibration_data.json
        - Loads detection data (absolute shot positions) from data/detection_data.json
        - Calculates relative positions of shots relative to target blob positions
        - Saves results to data/results_data.json
        """
        logging.info("Processing detection data to generate relative positions...")
        
        try:
            # Initialize results controller
            results_controller = ResultsController()
            
            # Check status
            status = results_controller.get_processing_status()
            if not status["calibration_data_exists"]:
                logging.error("Calibration data not found. Please run calibration first.")
                return False
                
            if not status["detection_data_exists"]:
                logging.error("Detection data not found. Please run analize-video first.")
                return False
                
            # Process the data
            success = results_controller.process_detection_data()
            if success:
                logging.info("Data processing completed successfully!")
                logging.info("Results saved to: data/results_data.json")
                return True
            else:
                logging.error("Data processing failed.")
                return False
                
        except Exception as e:
            logging.error(f"Error during data processing: {e}")
            return False
            
    def generate_result(self):
        """
        Generate result images with labeled shots using alvobase image and results data
        
        This step:
        - Loads results data from data/results_data.json
        - Uses alvobase image as base
        - Plots each shot using relative positions
        - Adds smart label positioning for shot numbers
        - Saves result images to data/ directory as target_X_result.jpg
        """
        logging.info("Generating result images with labeled shots...")
        
        try:
            # Initialize results controller
            results_controller = ResultsController()
            
            # Check status
            status = results_controller.get_processing_status()
            if not status["results_data_exists"]:
                logging.error("Results data not found. Please run process-data first.")
                return False
                
            # Generate result images
            success = results_controller.generate_result_images()
            if success:
                logging.info("Result image generation completed successfully!")
                
                # List generated images
                updated_status = results_controller.get_processing_status()
                if updated_status["result_images"]:
                    logging.info("Generated images:")
                    for image_file in updated_status["result_images"]:
                        logging.info(f"  - data/{image_file}")
                        
                return True
            else:
                logging.error("Result image generation failed.")
                return False
                
        except Exception as e:
            logging.error(f"Error during result image generation: {e}")
            return False
            
    def _analysis_progress(self, progress: float):
        """Progress callback for video analysis"""
        if int(progress) % 10 == 0:  # Log every 10%
            logging.info(f"Analysis progress: {progress:.1f}%")
            
    def _analysis_complete(self, success: bool, message: str):
        """Completion callback for video analysis"""
        if success:
            logging.info(f"Analysis completed: {message}")
            logging.info(f"Detection data saved to: data/detection_data.json")
        else:
            logging.error(f"Analysis failed: {message}")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description='Pipeline CLI for Shooting Simulator')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Record video command
    record_parser = subparsers.add_parser('record', help='Record video from camera until Q key is pressed')
    
    # Analyze video command
    analyze_parser = subparsers.add_parser('analize-video', help='Analize video file to detect laser shots')
    analyze_parser.add_argument('video_path', help='Path to the video file to analize')
    
    # Process data command
    subparsers.add_parser('process-data', help='Process detection and calibration data to generate relative positions')
    
    # Generate result command
    subparsers.add_parser('generate-result', help='Generate result images with labeled shots using alvobase image')
    
    # Generate homographies command
    subparsers.add_parser('generate-homographies', help='Generate homographies from calibration data')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
        
    # Initialize CLI
    cli = PipelineCLI()
    
    # Execute command
    if args.command == 'record':
        cli.record_video()
    elif args.command == 'analize-video':
        cli.analyze_video(args.video_path)
    elif args.command == 'process-data':
        cli.process_data()
    elif args.command == 'generate-result':
        cli.generate_result()
    elif args.command == 'generate-homographies':
        cli.generate_homographies()
    else:
        logging.error(f"Unknown command: {args.command}")


if __name__ == '__main__':
    main() 