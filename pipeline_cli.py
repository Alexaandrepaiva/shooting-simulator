#!/usr/bin/env python3
"""
Pipeline CLI for Shooting Simulator
Allows testing each step of the workflow independently through terminal commands.
"""

import argparse
import sys
import os
import cv2
import json
import logging
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from controllers.analysis_controller import AnalysisController
from utils.enhanced_video_recorder import EnhancedVideoRecorder


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
        Uses enhanced C++ recorder with fallback to OpenCV [[memory:4135206]]
        
        Returns:
            str: Path to recorded video file, or None if failed
        """
        logging.info("🎬 Starting enhanced video recording... Press 'q' to stop recording")
        
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
            filename = f"shooting_{timestamp}.avi"
            output_path = os.path.join(self.video_save_directory, filename)
            
            # Initialize enhanced recorder
            with EnhancedVideoRecorder() as recorder:
                # Configure for high-quality recording to avoid frame loss
                recorder.configure(
                    width=frame_width,
                    height=frame_height,
                    fps=fps,
                    bitrate=15000000,  # 15 Mbps for high quality
                    format_type='MJPEG',
                    use_hardware_acceleration=True
                )
                
                # Start enhanced recording
                if not recorder.start_recording(output_path, camera):
                    raise RuntimeError("Failed to start enhanced video recording")
                    
                logging.info(f"🎬 Recording to: {output_path}")
                logging.info("Press 'q' to stop recording...")
                
                # Record video with real-time monitoring
                start_time = datetime.now()
                last_stats_time = start_time
                
                while True:
                    # Display frame for monitoring
                    ret, frame = camera.read()
                    if ret:
                        cv2.imshow('Enhanced Recording - Press Q to stop', frame)
                        
                        # For OpenCV fallback, manually record frame
                        recorder.record_frame(frame)
                    
                    # Check for 'q' key press to stop recording
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == ord('Q'):
                        logging.info("Stop recording requested by user")
                        break
                        
                    # Show progress every 5 seconds
                    current_time = datetime.now()
                    if (current_time - last_stats_time).total_seconds() >= 5.0:
                        stats = recorder.get_recording_stats()
                        elapsed = (current_time - start_time).total_seconds()
                        
                        frames_written = stats.get('frames_written', 0)
                        frames_dropped = stats.get('frames_dropped', 0)
                        avg_fps = stats.get('average_fps', 0)
                        backend = stats.get('backend', 'unknown')
                        
                        logging.info(f"📊 Recording stats: {frames_written} frames, "
                                   f"{frames_dropped} dropped, {avg_fps:.1f} FPS, "
                                   f"{elapsed:.1f}s elapsed, backend: {backend}")
                        last_stats_time = current_time
                        
                # Stop recording
                if not recorder.stop_recording():
                    logging.warning("Enhanced recorder reported stop issues")
                    
                # Get final stats
                final_stats = recorder.get_recording_stats()
                final_duration = (datetime.now() - start_time).total_seconds()
                
                # Cleanup
                camera.release()
                cv2.destroyAllWindows()
                
                # Log final results
                frames_written = final_stats.get('frames_written', 0)
                frames_dropped = final_stats.get('frames_dropped', 0)
                backend = final_stats.get('backend', 'unknown')
                avg_fps = frames_written / final_duration if final_duration > 0 else 0
                
                logging.info(f"🎬 Enhanced recording completed!")
                logging.info(f"- File: {output_path}")
                logging.info(f"- Frames written: {frames_written}")
                logging.info(f"- Frames dropped: {frames_dropped}")
                logging.info(f"- Duration: {final_duration:.2f} seconds")
                logging.info(f"- Average FPS: {avg_fps:.2f}")
                logging.info(f"- Backend used: {backend}")
                
                if frames_dropped == 0:
                    logging.info("✅ Perfect recording - no frames lost!")
                elif frames_dropped > 0:
                    logging.warning(f"⚠️ {frames_dropped} frames were dropped")
                    
                return output_path
            
        except Exception as e:
            logging.error(f"Error during enhanced video recording: {e}")
            # Cleanup on error
            try:
                if 'camera' in locals():
                    camera.release()
                cv2.destroyAllWindows()
            except:
                pass
            return None

    def check_calibration(self):
        """
        Check calibration data status and display information
        
        Returns:
            dict: Calibration status information
        """
        logging.info("Checking calibration data...")
        
        try:
            from models.calibration import CalibrationModel
            
            calibration_model = CalibrationModel()
            
            calibration_file = os.path.join("data", "calibration_data.json")
            if not os.path.exists(calibration_file):
                logging.error("❌ Calibration data not found!")
                logging.info("Run calibration in the app first to generate calibration_data.json")
                return {"exists": False}
            
            # Load calibration data
            success = calibration_model.load_calibration_data()
            if not success:
                logging.error("❌ Failed to load calibration data")
                return {"exists": True, "valid": False}
            
            # Get calibration info
            params = calibration_model.get_all_parameters()
            blob_positions = calibration_model.get_blob_positions()
            
            logging.info("✅ Calibration data found and valid!")
            logging.info(f"  📋 Parameters:")
            for key, value in params.items():
                logging.info(f"    - {key}: {value}")
            logging.info(f"  🎯 Targets: {params.get('number_of_targets', 1)}")
            logging.info(f"  📍 Blob positions: {len(blob_positions)} detected")
            
            expected_blobs = params.get('number_of_targets', 1) * 4
            if len(blob_positions) < expected_blobs:
                logging.warning(f"⚠️  Expected {expected_blobs} blobs, found {len(blob_positions)}")
            else:
                logging.info(f"✅ Blob count correct: {len(blob_positions)}/{expected_blobs}")
            
            return {
                "exists": True,
                "valid": True,
                "parameters": params,
                "blob_positions": blob_positions,
                "blob_count": len(blob_positions),
                "expected_blobs": expected_blobs
            }
            
        except Exception as e:
            logging.error(f"Error checking calibration: {e}")
            return {"exists": False, "error": str(e)}

    def check_homographies(self):
        """
        Check homography data status and display information
        
        Returns:
            dict: Homography status information
        """
        logging.info("Checking homography data...")
        
        try:
            from models.homography import HomographyModel
            
            homography_file = os.path.join("data", "homographies.json")
            if not os.path.exists(homography_file):
                logging.error("❌ Homography data not found!")
                logging.info("Run 'generate-homographies' to create homographies.json")
                return {"exists": False}
            
            # Load homography data
            homography_model = HomographyModel()
            success = homography_model.load_homographies()
            if not success:
                logging.error("❌ Failed to load homography data")
                return {"exists": True, "valid": False}
            
            # Get homography info
            info = homography_model.get_homography_info()
            
            logging.info("✅ Homography data found and valid!")
            logging.info(f"  📐 Alvobase dimensions: {info.get('alvobase_dimensions', {})}")
            logging.info(f"  🎯 Number of targets: {info.get('number_of_targets', 0)}")
            
            homographies = info.get('homographies', {})
            logging.info(f"  🔄 Homographies calculated: {len(homographies)}")
            
            for target_id, target_data in homographies.items():
                camera_points = target_data.get('camera_points', [])
                logging.info(f"    Target {target_id}: {len(camera_points)} camera points")
            
            return {
                "exists": True,
                "valid": True,
                "info": info,
                "target_count": len(homographies)
            }
            
        except Exception as e:
            logging.error(f"Error checking homographies: {e}")
            return {"exists": False, "error": str(e)}

    def analyze_video(self, video_path: str):
        """
        Analyze a video file to detect laser shots (Step 1 of workflow)
        
        This step:
        - Detects shots in camera coordinates using 5 strategies
        - Saves absolute positions to data/detection_data.json
        - Does NOT apply homography transformations (that's done in process-data)
        
        Args:
            video_path: Path to the video file to analyze
            
        Returns:
            bool: True if successful, False otherwise
        """
        logging.info(f"🎯 STEP 1: Analyzing video for shot detection")
        logging.info(f"Video: {video_path}")
        
        if not os.path.exists(video_path):
            logging.error(f"❌ Video file not found: {video_path}")
            return False
            
        try:
            # Check prerequisites
            cal_status = self.check_calibration()
            if not cal_status.get("valid", False):
                logging.error("❌ Valid calibration data required for video analysis")
                return False
            
            hom_status = self.check_homographies()
            if not hom_status.get("valid", False):
                logging.info("⚠️  Homographies missing, generating...")
                if not self.generate_homographies():
                    logging.error("❌ Cannot proceed without homographies")
                    return False
            
            # Initialize video analysis controller
            analysis_controller = AnalysisController()
            
            # Analyze video (this will save to detection_data.json and then process with homography)
            success = False
            message = ""
            
            def completion_callback(success_result, completion_message):
                nonlocal success, message
                success = success_result
                message = completion_message
            
            analysis_controller._analyze_video_worker(
                video_path,
                progress_callback=self._analysis_progress,
                completion_callback=completion_callback
            )
            
            if success:
                logging.info(f"✅ Video analysis completed: {message}")
                return True
            else:
                logging.error(f"❌ Video analysis failed: {message}")
                return False
            
        except Exception as e:
            logging.error(f"❌ Error during video analysis: {e}")
            return False

    def process_data(self):
        """
        Process detection data and apply homography transformations (Step 2 of workflow)
        
        This step:
        - Loads absolute shot positions from data/detection_data.json
        - Applies homography transformations to convert to alvobase coordinates  
        - Calculates grouping circles (center and radius) for each target
        - Saves target positions and grouping data to data/results_data.json
        
        Returns:
            bool: True if successful, False otherwise
        """
        logging.info("🔄 STEP 2: Processing detection data with homography transformations")
        
        try:
            # Check prerequisites
            detection_file = os.path.join("data", "detection_data.json")
            if not os.path.exists(detection_file):
                logging.error("❌ Detection data not found. Run 'analize-video' first.")
                return False
            
            cal_status = self.check_calibration()
            if not cal_status.get("valid", False):
                logging.error("❌ Valid calibration data required")
                return False
            
            hom_status = self.check_homographies()
            if not hom_status.get("valid", False):
                logging.error("❌ Valid homography data required")
                return False
            
            # Import required models for data processing
            from models.calibration import CalibrationModel
            from models.results import ResultsModel
            
            # Initialize models
            calibration_model = CalibrationModel()
            results_model = ResultsModel()
            
            # Load calibration data
            if not calibration_model.load_calibration_data():
                logging.error("❌ No calibration data found. Cannot process detection data.")
                return False
                
            calibration_data = {
                'parameters': calibration_model.get_all_parameters(),
                'fixed_target_positions': calibration_model.get_blob_positions()
            }
            
            # Load detection data
            detection_data = self._load_detection_data()
            if not detection_data:
                logging.error("❌ No detection data found. Cannot process.")
                return False
                
            # Process data to generate relative positions
            results_data = results_model.calculate_relative_positions(
                detection_data, calibration_data
            )
            
            # Save results data
            success = results_model.save_results_data(results_data)
            if success:
                logging.info("✅ Data processing completed successfully!")
                logging.info("📄 Results saved to: data/results_data.json")
                return True
            else:
                logging.error("❌ Data processing failed")
                return False
                
        except Exception as e:
            logging.error(f"❌ Error during data processing: {e}")
            return False

    def generate_result_images(self):
        """
        Generate result images with labeled shots and grouping circles (Step 3 of workflow)
        
        This step:
        - Loads target positions from data/results_data.json
        - Uses alvobase image as base
        - Draws shots using target coordinates (alvobase coordinate system)
        - Draws grouping circles with center point (blue) and radius line (green)
        - Saves labeled result images to output/ directory
        
        Returns:
            bool: True if successful, False otherwise
        """
        logging.info("🖼️  STEP 3: Generating result images with labeled shots and grouping circles")
        
        try:
            from models.results import ResultsModel
            
            # Check prerequisites
            results_file = os.path.join("data", "results_data.json")
            if not os.path.exists(results_file):
                logging.error("❌ Results data not found. Run 'process-data' first.")
                return False
            
            # Check alvobase image
            alvobase_path = os.path.join("resources", "drawable", "alvobase.jpg")
            if not os.path.exists(alvobase_path):
                logging.error(f"❌ Alvobase image not found: {alvobase_path}")
                return False
            
            # Initialize results model
            results_model = ResultsModel()
            
            # Load results data
            results_data = results_model.load_results_data()
            if not results_data:
                logging.error("❌ Failed to load results data")
                return False
            
            # Generate images
            success = results_model.generate_target_images(results_data)
            if success:
                logging.info("✅ Result image generation completed successfully!")
                
                # List generated images
                output_dir = "output"
                if os.path.exists(output_dir):
                    images = [f for f in os.listdir(output_dir) if f.endswith('.jpg')]
                    if images:
                        logging.info("📸 Generated images:")
                        for image_file in sorted(images):
                            logging.info(f"  - output/{image_file}")
                
                return True
            else:
                logging.error("❌ Result image generation failed")
                return False
                
        except Exception as e:
            logging.error(f"❌ Error during result image generation: {e}")
            return False

    def view_detection_data(self):
        """
        Display detection data summary
        
        Returns:
            dict: Detection data summary
        """
        logging.info("📊 Viewing detection data...")
        
        try:
            detection_file = os.path.join("data", "detection_data.json")
            if not os.path.exists(detection_file):
                logging.error("❌ Detection data not found")
                return {"exists": False}
            
            with open(detection_file, 'r') as f:
                data = json.load(f)
            
            logging.info("✅ Detection data found!")
            logging.info(f"  📅 Analysis timestamp: {data.get('analysis_timestamp', 'Unknown')}")
            logging.info(f"  🎯 Total shots: {data.get('total_shots', 0)}")
            
            targets = data.get('targets', [])
            logging.info(f"  🎪 Targets: {len(targets)}")
            
            for target in targets:
                target_num = target.get('target_number', 'Unknown')
                shots = target.get('shots', [])
                logging.info(f"    Target {target_num}: {len(shots)} shots")
                
                for shot in shots:
                    abs_pos = shot.get('absolute_position', {})
                    strategy = shot.get('detection_strategy', 'unknown')
                    logging.info(f"      Shot {shot.get('shot_number', '?')}: ({abs_pos.get('x', 0):.1f}, {abs_pos.get('y', 0):.1f}) [{strategy}]")
            
            return {"exists": True, "data": data}
            
        except Exception as e:
            logging.error(f"❌ Error viewing detection data: {e}")
            return {"exists": False, "error": str(e)}

    def view_results_data(self):
        """
        Display results data summary
        
        Returns:
            dict: Results data summary
        """
        logging.info("📊 Viewing results data...")
        
        try:
            results_file = os.path.join("data", "results_data.json")
            if not os.path.exists(results_file):
                logging.error("❌ Results data not found")
                return {"exists": False}
            
            with open(results_file, 'r') as f:
                data = json.load(f)
            
            logging.info("✅ Results data found!")
            logging.info(f"  📅 Processing timestamp: {data.get('processing_timestamp', 'Unknown')}")
            
            targets = data.get('targets', [])
            logging.info(f"  🎪 Targets: {len(targets)}")
            
            for target in targets:
                target_num = target.get('target_number', 'Unknown')
                shots = target.get('shots', [])
                logging.info(f"    Target {target_num}: {len(shots)} shots")
                
                # Display grouping circle information
                grouping_circle = target.get('grouping_circle', {})
                if grouping_circle.get('shot_count', 0) > 0:
                    center_x = grouping_circle.get('center_x', 0)
                    center_y = grouping_circle.get('center_y', 0)
                    radius = grouping_circle.get('radius', 0)
                    shot_count = grouping_circle.get('shot_count', 0)
                    logging.info(f"      🎯 Grouping: center({center_x:.1f}, {center_y:.1f}), radius={radius:.1f}, shots={shot_count}")
                
                for shot in shots:
                    abs_pos = shot.get('absolute_position', {})
                    target_pos = shot.get('target_position', {})
                    strategy = shot.get('detection_strategy', 'unknown')
                    logging.info(f"      Shot {shot.get('shot_number', '?')}:")
                    logging.info(f"        Camera: ({abs_pos.get('x', 0):.1f}, {abs_pos.get('y', 0):.1f})")
                    logging.info(f"        Target: ({target_pos.get('x', 0):.1f}, {target_pos.get('y', 0):.1f})")
                    logging.info(f"        Strategy: {strategy}")
            
            return {"exists": True, "data": data}
            
        except Exception as e:
            logging.error(f"❌ Error viewing results data: {e}")
            return {"exists": False, "error": str(e)}

    def view_grouping_circles(self):
        """
        Display grouping circle information for all targets
        
        Returns:
            dict: Grouping circles summary
        """
        logging.info("🎯 Viewing grouping circles...")
        
        try:
            results_file = os.path.join("data", "results_data.json")
            if not os.path.exists(results_file):
                logging.error("❌ Results data not found")
                return {"exists": False}
            
            with open(results_file, 'r') as f:
                data = json.load(f)
            
            logging.info("✅ Grouping circles found!")
            logging.info(f"  📅 Processing timestamp: {data.get('processing_timestamp', 'Unknown')}")
            
            targets = data.get('targets', [])
            logging.info(f"  🎪 Targets: {len(targets)}")
            
            grouping_data = []
            for target in targets:
                target_num = target.get('target_number', 'Unknown')
                grouping_circle = target.get('grouping_circle', {})
                
                if grouping_circle.get('shot_count', 0) > 0:
                    center_x = grouping_circle.get('center_x', 0)
                    center_y = grouping_circle.get('center_y', 0)
                    radius = grouping_circle.get('radius', 0)
                    shot_count = grouping_circle.get('shot_count', 0)
                    
                    logging.info(f"    🎯 Target {target_num}:")
                    logging.info(f"      Center: ({center_x:.1f}, {center_y:.1f})")
                    logging.info(f"      Radius: {radius:.1f}")
                    logging.info(f"      Shots: {shot_count}")
                    logging.info(f"      Precision: {'High' if radius < 50 else 'Medium' if radius < 100 else 'Low'}")
                    
                    grouping_data.append({
                        "target": target_num,
                        "center": [center_x, center_y],
                        "radius": radius,
                        "shots": shot_count
                    })
                else:
                    logging.info(f"    ⚫ Target {target_num}: No grouping data")
            
            return {"exists": True, "groupings": grouping_data}
            
        except Exception as e:
            logging.error(f"❌ Error viewing grouping circles: {e}")
            return {"exists": False, "error": str(e)}

    def status(self):
        """
        Display overall pipeline status
        
        Returns:
            dict: Complete pipeline status
        """
        logging.info("📋 PIPELINE STATUS")
        logging.info("=" * 50)
        
        # Check each step
        cal_status = self.check_calibration()
        hom_status = self.check_homographies()
        
        detection_exists = os.path.exists(os.path.join("data", "detection_data.json"))
        results_exists = os.path.exists(os.path.join("data", "results_data.json"))
        
        output_dir = "output"
        result_images = []
        if os.path.exists(output_dir):
            result_images = [f for f in os.listdir(output_dir) if f.endswith('.jpg')]
        
        # Summary
        logging.info("🔍 WORKFLOW STEPS:")
        logging.info(f"  1. Calibration:     {'✅' if cal_status.get('valid') else '❌'}")
        logging.info(f"  2. Homographies:    {'✅' if hom_status.get('valid') else '❌'}")
        logging.info(f"  3. Detection Data:  {'✅' if detection_exists else '❌'}")
        logging.info(f"  4. Results Data:    {'✅' if results_exists else '❌'}")
        logging.info(f"  5. Result Images:   {'✅' if result_images else '❌'} ({len(result_images)} images)")
        
        logging.info("\n📁 DATA FILES:")
        data_files = [
            "calibration_data.json",
            "homographies.json", 
            "detection_data.json",
            "results_data.json"
        ]
        
        for filename in data_files:
            filepath = os.path.join("data", filename)
            if os.path.exists(filepath):
                size = os.path.getsize(filepath)
                logging.info(f"  ✅ {filename} ({size} bytes)")
            else:
                logging.info(f"  ❌ {filename} (missing)")
        
        return {
            "calibration": cal_status,
            "homographies": hom_status,
            "detection_exists": detection_exists,
            "results_exists": results_exists,
            "result_images": len(result_images)
        }

    def generate_homographies(self):
        """
        Generate homographies.json from existing calibration data
        
        Returns:
            bool: True if successful, False otherwise
        """
        logging.info("🔧 Generating homographies from calibration data...")
        
        try:
            from models.calibration import CalibrationModel
            from models.homography import HomographyModel
            
            # Check calibration first
            cal_status = self.check_calibration()
            if not cal_status.get("valid", False):
                logging.error("❌ Valid calibration data required")
                return False
            
            # Load calibration data
            calibration_model = CalibrationModel()
            if not calibration_model.load_calibration_data():
                logging.error("❌ Failed to load calibration data")
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
                logging.info("✅ Homographies generated successfully!")
                logging.info("📄 Saved to: data/homographies.json")
                return True
            else:
                logging.error("❌ Failed to generate homographies")
                return False
                
        except Exception as e:
            logging.error(f"❌ Error generating homographies: {e}")
            return False

    def clean_data(self):
        """
        Clean all generated data files (keeps calibration data)
        
        Returns:
            bool: True if successful, False otherwise  
        """
        logging.info("🧹 Cleaning generated data files...")
        
        files_to_clean = [
            os.path.join("data", "detection_data.json"),
            os.path.join("data", "results_data.json"),
        ]
        
        # Clean output images
        output_dir = "output"
        if os.path.exists(output_dir):
            for filename in os.listdir(output_dir):
                if filename.endswith('.jpg'):
                    files_to_clean.append(os.path.join(output_dir, filename))
        
        cleaned_count = 0
        for filepath in files_to_clean:
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                    logging.info(f"🗑️  Removed: {filepath}")
                    cleaned_count += 1
                except Exception as e:
                    logging.error(f"❌ Failed to remove {filepath}: {e}")
        
        logging.info(f"✅ Cleaned {cleaned_count} files")
        logging.info("ℹ️  Calibration data and homographies preserved")
        return True

    def _analysis_progress(self, progress: float):
        """Progress callback for video analysis"""
        if int(progress) % 10 == 0:  # Log every 10%
            logging.info(f"🔄 Analysis progress: {progress:.1f}%")
            
    def _analysis_complete(self, success: bool, message: str):
        """Completion callback for video analysis"""
        if success:
            logging.info(f"✅ Analysis completed: {message}")
        else:
            logging.error(f"❌ Analysis failed: {message}")
            
    def _load_detection_data(self):
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


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Pipeline CLI for Shooting Simulator - Test each workflow step independently',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
WORKFLOW STEPS:
  1. Calibration (done in app) → saves calibration_data.json
  2. generate-homographies → calculates coordinate transformations  
  3. record → captures video footage
  4. analize-video → detects shots in camera coordinates
  5. process-data → applies homography transformations + calculates grouping circles
  6. generate-result-images → creates labeled result images with grouping circles

EXAMPLES:
  python pipeline_cli.py status
  python pipeline_cli.py record
  python pipeline_cli.py analize-video Video/shooting_20241124_143022.avi
  python pipeline_cli.py process-data
  python pipeline_cli.py generate-result-images
  python pipeline_cli.py view-grouping-circles
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Status and inspection commands
    subparsers.add_parser('status', help='Show overall pipeline status')
    subparsers.add_parser('check-calibration', help='Check calibration data status')
    subparsers.add_parser('check-homographies', help='Check homography data status')
    subparsers.add_parser('view-detection-data', help='Display detection data summary')
    subparsers.add_parser('view-results-data', help='Display results data summary')
    subparsers.add_parser('view-grouping-circles', help='Display grouping circle information for all targets')
    
    # Workflow commands
    subparsers.add_parser('record', help='Record video from camera until Q key is pressed')
    
    analyze_parser = subparsers.add_parser('analize-video', help='Analyze video file to detect laser shots (Step 1)')
    analyze_parser.add_argument('video_path', help='Path to the video file to analyze')
    
    subparsers.add_parser('process-data', help='Process detection data with homography transformations + grouping circles (Step 2)')
    subparsers.add_parser('generate-result-images', help='Generate result images with labeled shots and grouping circles (Step 3)')
    
    # Utility commands
    subparsers.add_parser('generate-homographies', help='Generate/regenerate homographies from calibration data')
    subparsers.add_parser('clean-data', help='Clean all generated data files (preserves calibration)')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
        
    # Initialize CLI
    cli = PipelineCLI()
    
    # Execute command
    if args.command == 'status':
        cli.status()
    elif args.command == 'check-calibration':
        cli.check_calibration()
    elif args.command == 'check-homographies':
        cli.check_homographies()
    elif args.command == 'view-detection-data':
        cli.view_detection_data()
    elif args.command == 'view-results-data':
        cli.view_results_data()
    elif args.command == 'view-grouping-circles':
        cli.view_grouping_circles()
    elif args.command == 'record':
        cli.record_video()
    elif args.command == 'analize-video':
        cli.analyze_video(args.video_path)
    elif args.command == 'process-data':
        cli.process_data()
    elif args.command == 'generate-result-images':
        cli.generate_result_images()
    elif args.command == 'generate-homographies':
        cli.generate_homographies()
    elif args.command == 'clean-data':
        cli.clean_data()
    else:
        logging.error(f"Unknown command: {args.command}")


if __name__ == '__main__':
    main() 