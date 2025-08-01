"""
Python wrapper for high-performance C++ video recorder
Uses ctypes to interface with the compiled C++ DLL
"""

import os
import sys
import ctypes
from ctypes import Structure, POINTER, c_char_p, c_int, c_bool, c_void_p, c_size_t, c_double
import logging
from typing import Optional, Tuple
from pathlib import Path


class RecordingStats(Structure):
    """Structure to hold recording statistics"""
    _fields_ = [
        ("frames_written", c_size_t),
        ("frames_dropped", c_size_t),
        ("recording_duration", c_double),
        ("average_fps", c_double)
    ]


class HighPerformanceVideoRecorder:
    """Python wrapper for the C++ high-performance video recorder"""
    
    def __init__(self):
        self._dll = None
        self._handle = None
        self._load_dll()
        
    def _load_dll(self):
        """Load the C++ DLL"""
        try:
            # Try to find the DLL in the same directory as this file
            dll_dir = Path(__file__).parent
            dll_paths = [
                dll_dir / "video_recorder.dll",
                dll_dir / "Release" / "video_recorder.dll",
                dll_dir / "Debug" / "video_recorder.dll",
                Path.cwd() / "video_recorder.dll"
            ]
            
            dll_path = None
            for path in dll_paths:
                if path.exists():
                    dll_path = path
                    break
                    
            if not dll_path:
                raise FileNotFoundError(f"Could not find video_recorder.dll in any of these locations: {dll_paths}")
                
            self._dll = ctypes.CDLL(str(dll_path))
            self._setup_function_signatures()
            logging.info(f"Successfully loaded video recorder DLL from: {dll_path}")
            
        except Exception as e:
            logging.error(f"Failed to load video recorder DLL: {e}")
            raise
            
    def _setup_function_signatures(self):
        """Set up function signatures for the DLL functions"""
        # CreateVideoRecorder
        self._dll.CreateVideoRecorder.restype = c_void_p
        self._dll.CreateVideoRecorder.argtypes = []
        
        # DestroyVideoRecorder
        self._dll.DestroyVideoRecorder.restype = None
        self._dll.DestroyVideoRecorder.argtypes = [c_void_p]
        
        # SetRecordingConfig
        self._dll.SetRecordingConfig.restype = c_bool
        self._dll.SetRecordingConfig.argtypes = [
            c_void_p,      # handle
            c_int,         # width
            c_int,         # height
            c_int,         # fps
            c_int,         # bitrate
            c_char_p,      # format
            c_char_p,      # output_path
            c_bool         # use_hw_accel
        ]
        
        # InitializeRecorder
        self._dll.InitializeRecorder.restype = c_bool
        self._dll.InitializeRecorder.argtypes = [c_void_p]
        
        # StartVideoRecording
        self._dll.StartVideoRecording.restype = c_bool
        self._dll.StartVideoRecording.argtypes = [c_void_p]
        
        # StopVideoRecording
        self._dll.StopVideoRecording.restype = c_bool
        self._dll.StopVideoRecording.argtypes = [c_void_p]
        
        # IsVideoRecording
        self._dll.IsVideoRecording.restype = c_bool
        self._dll.IsVideoRecording.argtypes = [c_void_p]
        
        # CaptureVideoFrame
        self._dll.CaptureVideoFrame.restype = c_bool
        self._dll.CaptureVideoFrame.argtypes = [c_void_p]
        
        # GetRecordingStats
        self._dll.GetRecordingStats.restype = c_bool
        self._dll.GetRecordingStats.argtypes = [
            c_void_p,                    # handle
            POINTER(c_size_t),          # frames_written
            POINTER(c_size_t),          # frames_dropped
            POINTER(c_double),          # duration
            POINTER(c_double)           # avg_fps
        ]
        
        # GetRecorderError
        self._dll.GetRecorderError.restype = c_char_p
        self._dll.GetRecorderError.argtypes = [c_void_p]
        
        # CleanupRecorder
        self._dll.CleanupRecorder.restype = None
        self._dll.CleanupRecorder.argtypes = [c_void_p]
        
    def initialize(self, width: int = 640, height: int = 480, fps: int = 30, 
                  bitrate: int = 8000000, format_type: str = "MJPEG", 
                  output_path: str = "", use_hardware_acceleration: bool = True) -> bool:
        """
        Initialize the video recorder with configuration
        
        Args:
            width: Video width in pixels
            height: Video height in pixels
            fps: Frames per second
            bitrate: Video bitrate in bits per second
            format_type: Video format ("MJPEG" or "RAW")
            output_path: Output file path
            use_hardware_acceleration: Enable hardware acceleration
            
        Returns:
            bool: True if initialization successful
        """
        if not self._dll:
            logging.error("DLL not loaded")
            return False
            
        # Create recorder handle
        self._handle = self._dll.CreateVideoRecorder()
        if not self._handle:
            logging.error("Failed to create video recorder handle")
            return False
            
        # Configure the recorder
        format_bytes = format_type.encode('utf-8')
        output_bytes = output_path.encode('utf-8')
        
        success = self._dll.SetRecordingConfig(
            self._handle,
            width,
            height, 
            fps,
            bitrate,
            format_bytes,
            output_bytes,
            use_hardware_acceleration
        )
        
        if not success:
            error = self.get_last_error()
            logging.error(f"Failed to configure video recorder: {error}")
            self._cleanup_handle()
            return False
            
        # Verify initialization
        if not self._dll.InitializeRecorder(self._handle):
            error = self.get_last_error()
            logging.error(f"Recorder initialization failed: {error}")
            self._cleanup_handle()
            return False
            
        logging.info(f"Video recorder initialized: {width}x{height}@{fps}fps, {format_type}, bitrate={bitrate}")
        return True
        
    def configure(self, width: int = 640, height: int = 480, fps: int = 30,
                 bitrate: int = 8000000, format_type: str = "MJPEG",
                 use_hardware_acceleration: bool = True) -> bool:
        """
        Configure the video recorder
        
        Args:
            width: Video width in pixels
            height: Video height in pixels
            fps: Frames per second
            bitrate: Video bitrate in bits per second
            format_type: Video format ("MJPEG" or "RAW")
            use_hardware_acceleration: Enable hardware acceleration
            
        Returns:
            bool: True if configuration successful
        """
        if not self._dll:
            logging.error("DLL not loaded")
            return False
            
        # Create recorder handle
        self._handle = self._dll.CreateVideoRecorder()
        if not self._handle:
            logging.error("Failed to create video recorder handle")
            return False
            
        # Configure the recorder (output path will be set later)
        format_bytes = format_type.encode('utf-8')
        output_bytes = "".encode('utf-8')  # Empty output path for now
        
        success = self._dll.SetRecordingConfig(
            self._handle,
            width,
            height, 
            fps,
            bitrate,
            format_bytes,
            output_bytes,
            use_hardware_acceleration
        )
        
        if not success:
            error = self.get_last_error()
            logging.error(f"Failed to configure video recorder: {error}")
            self._cleanup_handle()
            return False
            
        # Verify initialization
        if not self._dll.InitializeRecorder(self._handle):
            error = self.get_last_error()
            logging.error(f"Recorder initialization failed: {error}")
            self._cleanup_handle()
            return False
            
        logging.info(f"Video recorder initialized: {width}x{height}@{fps}fps, {format_type}, bitrate={bitrate}")
        return True
        
    def start_recording(self) -> bool:
        """
        Start video recording
        
        Returns:
            bool: True if recording started successfully
        """
        if not self._handle:
            logging.error("Recorder not initialized")
            return False
            
        success = self._dll.StartVideoRecording(self._handle)
        if not success:
            error = self.get_last_error()
            logging.error(f"Failed to start recording: {error}")
            
        return success
        
    def stop_recording(self) -> bool:
        """
        Stop video recording
        
        Returns:
            bool: True if recording stopped successfully
        """
        if not self._handle:
            return True
            
        success = self._dll.StopVideoRecording(self._handle)
        if not success:
            error = self.get_last_error()
            logging.error(f"Failed to stop recording: {error}")
            
        return success
        
    def capture_frame(self) -> bool:
        """
        Capture a single video frame
        
        Returns:
            bool: True if frame captured successfully
        """
        if not self._handle:
            return False
            
        return self._dll.CaptureVideoFrame(self._handle)
        
    def is_recording(self) -> bool:
        """
        Check if currently recording
        
        Returns:
            bool: True if recording is active
        """
        if not self._handle:
            return False
            
        return self._dll.IsVideoRecording(self._handle)
        
    def get_stats(self) -> Optional[RecordingStats]:
        """
        Get recording statistics
        
        Returns:
            RecordingStats or None if failed
        """
        if not self._handle:
            return None
            
        frames_written = c_size_t()
        frames_dropped = c_size_t()
        duration = c_double()
        avg_fps = c_double()
        
        success = self._dll.GetRecordingStats(
            self._handle,
            ctypes.byref(frames_written),
            ctypes.byref(frames_dropped),
            ctypes.byref(duration),
            ctypes.byref(avg_fps)
        )
        
        if not success:
            return None
            
        stats = RecordingStats()
        stats.frames_written = frames_written.value
        stats.frames_dropped = frames_dropped.value
        stats.recording_duration = duration.value
        stats.average_fps = avg_fps.value
        
        return stats
        
    def get_last_error(self) -> str:
        """
        Get the last error message
        
        Returns:
            str: Error message
        """
        if not self._handle or not self._dll:
            return "Recorder not initialized"
            
        error_ptr = self._dll.GetRecorderError(self._handle)
        if error_ptr:
            return error_ptr.decode('utf-8')
        return "No error"
        
    def cleanup(self):
        """Clean up resources"""
        if self._handle:
            self._dll.CleanupRecorder(self._handle)
            self._cleanup_handle()
            
    def _cleanup_handle(self):
        """Clean up the recorder handle"""
        if self._handle and self._dll:
            self._dll.DestroyVideoRecorder(self._handle)
            self._handle = None
            
    def __enter__(self):
        """Context manager entry"""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()
        
    def __del__(self):
        """Destructor"""
        self.cleanup()


class VideoRecorderError(Exception):
    """Custom exception for video recorder errors"""
    pass


def test_video_recorder():
    """Test function for the video recorder"""
    import time
    
    logging.basicConfig(level=logging.INFO)
    
    try:
        with HighPerformanceVideoRecorder() as recorder:
            # Test with MJPEG format
            success = recorder.initialize(
                width=640,
                height=480,
                fps=30,
                bitrate=8000000,
                format_type="MJPEG",
                output_path="test_recording.avi",
                use_hardware_acceleration=True
            )
            
            if not success:
                print(f"Failed to initialize: {recorder.get_last_error()}")
                return False
                
            print("Starting recording...")
            if not recorder.start_recording():
                print(f"Failed to start recording: {recorder.get_last_error()}")
                return False
                
            # Record for 5 seconds
            start_time = time.time()
            while time.time() - start_time < 5.0:
                if not recorder.capture_frame():
                    print(f"Frame capture failed: {recorder.get_last_error()}")
                    
                time.sleep(1.0 / 30)  # 30 FPS
                
                # Print stats every second
                if int(time.time() - start_time) % 1 == 0:
                    stats = recorder.get_stats()
                    if stats:
                        print(f"Stats: {stats.frames_written} frames, "
                              f"{stats.frames_dropped} dropped, "
                              f"{stats.average_fps:.1f} FPS")
                              
            print("Stopping recording...")
            if not recorder.stop_recording():
                print(f"Failed to stop recording: {recorder.get_last_error()}")
                return False
                
            # Final stats
            stats = recorder.get_stats()
            if stats:
                print(f"Final stats: {stats.frames_written} frames written, "
                      f"{stats.frames_dropped} frames dropped, "
                      f"{stats.recording_duration:.2f}s duration, "
                      f"{stats.average_fps:.1f} average FPS")
                      
        print("Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        return False


if __name__ == "__main__":
    test_video_recorder() 