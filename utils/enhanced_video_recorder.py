"""
Enhanced Video Recorder for Shooting Simulator

This module provides a high-performance video recorder that uses:
1. C++ Windows Media Foundation backend for maximum performance (primary)
2. OpenCV fallback for compatibility

The recorder is designed to prevent frame loss during short-duration shots
by using less compression and operating system-level capture methods.
"""

import os
import sys
import cv2
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, Dict, Any
import time


class EnhancedVideoRecorder:
    """
    Enhanced video recorder with C++ backend and OpenCV fallback
    
    Features:
    - High-performance C++ recording using Windows Media Foundation
    - Automatic fallback to OpenCV if C++ backend fails
    - Frame loss detection and reporting
    - High bitrate recording for better shot detection
    - Thread-safe operation
    """
    
    def __init__(self):
        self.cpp_recorder = None
        self.opencv_recorder = None
        self.is_recording = False
        self.recording_thread = None
        self.stop_recording_event = threading.Event()
        
        # Recording configuration
        self.config = {
            'width': 640,
            'height': 480,
            'fps': 30,
            'bitrate': 12000000,  # 12 Mbps for high quality
            'format': 'MJPEG',     # MJPEG for AVI files
            'use_hardware_acceleration': True,
            'use_cpp_backend': True
        }
        
        # Recording state
        self.current_backend = None
        self.video_filepath = None
        self.frames_written = 0
        self.recording_start_time = None
        
        # Try to load C++ backend
        self._try_load_cpp_backend()
        
    def _try_load_cpp_backend(self):
        """Attempt to load the C++ high-performance backend"""
        try:
            # Add cpp_video_recorder to path
            cpp_dir = Path(__file__).parent.parent / "cpp_video_recorder"
            if cpp_dir.exists():
                sys.path.insert(0, str(cpp_dir))
                
            from cpp_video_recorder.python_wrapper import HighPerformanceVideoRecorder
            self.cpp_recorder = HighPerformanceVideoRecorder()
            logging.info("✅ C++ high-performance video recorder loaded successfully")
            return True
            
        except Exception as e:
            logging.warning(f"⚠️ C++ video recorder not available: {e}")
            logging.info("Will use OpenCV fallback for video recording")
            self.cpp_recorder = None
            return False
            
    def configure(self, width: int = 640, height: int = 480, fps: int = 30,
                 bitrate: int = 12000000, format_type: str = 'MJPEG',
                 use_hardware_acceleration: bool = True,
                 force_opencv: bool = False) -> bool:
        """
        Configure the video recorder
        
        Args:
            width: Video width in pixels
            height: Video height in pixels
            fps: Frames per second
            bitrate: Video bitrate in bits per second
            format_type: Video format ('MJPEG' or 'RAW')
            use_hardware_acceleration: Enable hardware acceleration
            force_opencv: Force use of OpenCV backend
            
        Returns:
            bool: True if configuration successful
        """
        self.config.update({
            'width': width,
            'height': height,
            'fps': fps,
            'bitrate': bitrate,
            'format': format_type,
            'use_hardware_acceleration': use_hardware_acceleration,
            'use_cpp_backend': not force_opencv and self.cpp_recorder is not None
        })
        
        logging.info(f"📹 Video recorder configured: {width}x{height}@{fps}fps, "
                    f"{format_type}, {bitrate/1000000:.1f}Mbps")
        logging.info(f"Backend: {'C++ (High Performance)' if self.config['use_cpp_backend'] else 'OpenCV (Fallback)'}")
        
        return True
        
    def start_recording(self, output_path: str, camera) -> bool:
        """
        Start video recording
        
        Args:
            output_path: Path where video will be saved
            camera: OpenCV camera object for frame capture
            
        Returns:
            bool: True if recording started successfully
        """
        if self.is_recording:
            logging.warning("Recording already in progress")
            return False
            
        self.video_filepath = output_path
        self.frames_written = 0
        self.recording_start_time = datetime.now()
        
        # Try C++ backend first if available
        if self.config['use_cpp_backend'] and self.cpp_recorder:
            if self._start_cpp_recording(camera):
                return True
            else:
                logging.warning("C++ backend failed, falling back to OpenCV")
                
        # Fall back to OpenCV
        return self._start_opencv_recording(camera)
        
    def _start_cpp_recording(self, camera) -> bool:
        """Start recording with C++ backend"""
        try:
            # Initialize C++ recorder
            success = self.cpp_recorder.initialize(
                width=self.config['width'],
                height=self.config['height'],
                fps=self.config['fps'],
                bitrate=self.config['bitrate'],
                format_type=self.config['format'],
                output_path=self.video_filepath,
                use_hardware_acceleration=self.config['use_hardware_acceleration']
            )
            
            if not success:
                error = self.cpp_recorder.get_last_error()
                logging.error(f"Failed to initialize C++ recorder: {error}")
                return False
                
            # Start recording
            if not self.cpp_recorder.start_recording():
                error = self.cpp_recorder.get_last_error()
                logging.error(f"Failed to start C++ recording: {error}")
                return False
                
            self.current_backend = 'cpp'
            self.is_recording = True
            
            # Start recording thread
            self.stop_recording_event.clear()
            self.recording_thread = threading.Thread(
                target=self._cpp_recording_loop,
                args=(camera,),
                daemon=True
            )
            self.recording_thread.start()
            
            logging.info(f"🎬 Started C++ recording to: {self.video_filepath}")
            return True
            
        except Exception as e:
            logging.error(f"C++ recording setup failed: {e}")
            return False
            
    def _start_opencv_recording(self, camera) -> bool:
        """Start recording with OpenCV backend"""
        try:
            # Get camera properties
            if camera:
                frame_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            else:
                frame_width = self.config['width']
                frame_height = self.config['height']
                
            # Choose codec based on format
            if self.config['format'] == 'RAW':
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Motion JPEG for less compression
            elif self.config['format'] == 'MJPEG':
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Motion JPEG for AVI files
            else:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Default H.264
                
            # Create video writer
            self.opencv_recorder = cv2.VideoWriter(
                self.video_filepath,
                fourcc,
                self.config['fps'],
                (frame_width, frame_height)
            )
            
            if not self.opencv_recorder.isOpened():
                logging.error("Failed to initialize OpenCV video writer")
                return False
                
            self.current_backend = 'opencv'
            self.is_recording = True
            
            logging.info(f"🎬 Started OpenCV recording to: {self.video_filepath}")
            return True
            
        except Exception as e:
            logging.error(f"OpenCV recording setup failed: {e}")
            return False
            
    def _cpp_recording_loop(self, camera):
        """Recording loop for C++ backend"""
        try:
            frame_interval = 1.0 / self.config['fps']
            next_frame_time = time.time()
            
            while not self.stop_recording_event.is_set():
                current_time = time.time()
                
                # Maintain frame rate
                if current_time >= next_frame_time:
                    # Capture frame using C++ backend
                    if self.cpp_recorder.capture_frame():
                        self.frames_written += 1
                        
                        # Log progress every 100 frames
                        if self.frames_written % 100 == 0:
                            stats = self.cpp_recorder.get_stats()
                            if stats:
                                logging.debug(f"Recording: {stats.frames_written} frames, "
                                            f"{stats.frames_dropped} dropped, "
                                            f"{stats.average_fps:.1f} FPS")
                    else:
                        error = self.cpp_recorder.get_last_error()
                        if error != "No error":
                            logging.warning(f"Frame capture issue: {error}")
                    
                    next_frame_time += frame_interval
                else:
                    # Small sleep to prevent busy waiting
                    time.sleep(0.001)
                    
        except Exception as e:
            logging.error(f"C++ recording loop error: {e}")
            
    def record_frame(self, frame) -> bool:
        """
        Record a single frame (for OpenCV backend)
        
        Args:
            frame: OpenCV frame to record
            
        Returns:
            bool: True if frame recorded successfully
        """
        if not self.is_recording:
            return False
            
        if self.current_backend == 'opencv' and self.opencv_recorder:
            try:
                self.opencv_recorder.write(frame)
                self.frames_written += 1
                
                # Log progress every 100 frames
                if self.frames_written % 100 == 0:
                    logging.debug(f"Frames recorded: {self.frames_written}")
                    
                return True
            except Exception as e:
                logging.error(f"Error recording frame: {e}")
                return False
                
        # For C++ backend, frames are captured automatically in the loop
        return self.current_backend == 'cpp'
        
    def stop_recording(self) -> bool:
        """
        Stop video recording
        
        Returns:
            bool: True if recording stopped successfully
        """
        if not self.is_recording:
            return True
            
        self.is_recording = False
        success = True
        
        try:
            if self.current_backend == 'cpp':
                success = self._stop_cpp_recording()
            elif self.current_backend == 'opencv':
                success = self._stop_opencv_recording()
                
            # Calculate final stats
            if self.recording_start_time:
                duration = (datetime.now() - self.recording_start_time).total_seconds()
                avg_fps = self.frames_written / duration if duration > 0 else 0
                
                logging.info(f"🎬 Recording stopped: {self.frames_written} frames, "
                           f"{duration:.2f}s, {avg_fps:.1f} avg FPS")
                           
            return success
            
        except Exception as e:
            logging.error(f"Error stopping recording: {e}")
            return False
            
    def _stop_cpp_recording(self) -> bool:
        """Stop C++ recording"""
        success = True
        
        # Signal recording thread to stop
        self.stop_recording_event.set()
        
        # Wait for recording thread to finish
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=5.0)
            if self.recording_thread.is_alive():
                logging.warning("Recording thread did not stop in time")
                
        # Stop C++ recorder
        if self.cpp_recorder:
            if not self.cpp_recorder.stop_recording():
                error = self.cpp_recorder.get_last_error()
                logging.error(f"Failed to stop C++ recording: {error}")
                success = False
                
            # Get final stats
            stats = self.cpp_recorder.get_stats()
            if stats and stats.frames_dropped > 0:
                logging.warning(f"⚠️ {stats.frames_dropped} frames were dropped during recording")
                
        return success
        
    def _stop_opencv_recording(self) -> bool:
        """Stop OpenCV recording"""
        if self.opencv_recorder:
            self.opencv_recorder.release()
            self.opencv_recorder = None
            
        return True
        
    def get_recording_stats(self) -> Dict[str, Any]:
        """
        Get current recording statistics
        
        Returns:
            dict: Recording statistics
        """
        stats = {
            'is_recording': self.is_recording,
            'frames_written': self.frames_written,
            'video_filepath': self.video_filepath,
            'recording_start_time': self.recording_start_time,
            'backend': self.current_backend,
            'frames_dropped': 0,
            'average_fps': 0.0
        }
        
        # Get additional stats from C++ backend
        if self.current_backend == 'cpp' and self.cpp_recorder:
            try:
                cpp_stats = self.cpp_recorder.get_stats()
                if cpp_stats:
                    stats.update({
                        'frames_written': cpp_stats.frames_written,
                        'frames_dropped': cpp_stats.frames_dropped,
                        'average_fps': cpp_stats.average_fps
                    })
            except Exception as e:
                logging.debug(f"Could not get C++ stats: {e}")
                
        return stats
        
    def cleanup(self):
        """Clean up resources"""
        if self.is_recording:
            self.stop_recording()
            
        if self.cpp_recorder:
            try:
                self.cpp_recorder.cleanup()
            except Exception as e:
                logging.debug(f"C++ cleanup error: {e}")
                
        if self.opencv_recorder:
            self.opencv_recorder.release()
            self.opencv_recorder = None
            
    def __enter__(self):
        """Context manager entry"""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()
        
    def __del__(self):
        """Destructor"""
        self.cleanup()


def test_enhanced_recorder():
    """Test function for enhanced video recorder"""
    import time
    
    logging.basicConfig(level=logging.INFO)
    
    with EnhancedVideoRecorder() as recorder:
        # Configure for high-quality recording
        recorder.configure(
            width=640,
            height=480,
            fps=30,
            bitrate=15000000,  # 15 Mbps
            format_type='MJPEG'
        )
        
        # Mock camera for testing
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            print("No camera available for testing")
            return False
            
        # Test recording
        output_path = f"test_enhanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
        
        print("Starting enhanced recording test...")
        if recorder.start_recording(output_path, camera):
            print("Recording for 3 seconds...")
            time.sleep(3.0)
            
            # Get stats during recording
            stats = recorder.get_recording_stats()
            print(f"Recording stats: {stats}")
            
            if recorder.stop_recording():
                print("Test completed successfully!")
                print(f"Video saved to: {output_path}")
                return True
            else:
                print("Failed to stop recording")
                return False
        else:
            print("Failed to start recording")
            return False


if __name__ == "__main__":
    test_enhanced_recorder() 