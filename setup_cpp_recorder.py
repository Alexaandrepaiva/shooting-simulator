#!/usr/bin/env python3
"""
Setup script for C++ High-Performance Video Recorder

This script will:
1. Check system requirements
2. Build the C++ video recorder
3. Test the installation
4. Update the existing requirements if needed

Usage:
    python setup_cpp_recorder.py
    python setup_cpp_recorder.py --test-only
    python setup_cpp_recorder.py --force-rebuild
"""

import os
import sys
import subprocess
import shutil
import logging
from pathlib import Path
import argparse
import platform


def setup_logging(verbose=False):
    """Set up logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )


def check_system_requirements():
    """Check if the system meets requirements for building"""
    logging.info("🔍 Checking system requirements...")
    
    # Check OS
    if platform.system() != 'Windows':
        logging.error("❌ This C++ recorder is designed for Windows only")
        return False
        
    logging.info(f"✅ Operating System: {platform.system()} {platform.release()}")
    
    # Check Windows version (Windows 10 or later)
    version = platform.version()
    logging.info(f"✅ Windows Version: {version}")
    
    # Check Python version
    python_version = sys.version_info
    if python_version < (3, 7):
        logging.error(f"❌ Python 3.7+ required, found {python_version.major}.{python_version.minor}")
        return False
        
    logging.info(f"✅ Python Version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    return True


def check_build_tools():
    """Check if build tools are available"""
    logging.info("🔧 Checking build tools...")
    
    # Check for Visual Studio or Build Tools
    vs_paths = [
        "C:/Program Files/Microsoft Visual Studio/2022",
        "C:/Program Files/Microsoft Visual Studio/2019",
        "C:/Program Files (x86)/Microsoft Visual Studio/2019",
        "C:/Program Files (x86)/Microsoft Build Tools"
    ]
    
    vs_found = False
    for vs_path in vs_paths:
        if Path(vs_path).exists():
            logging.info(f"✅ Found Visual Studio/Build Tools at: {vs_path}")
            vs_found = True
            break
    
    if not vs_found:
        logging.warning("⚠️ Visual Studio not found in common locations")
        logging.info("💡 You can install Visual Studio Community 2019+ or Build Tools for Visual Studio")
        logging.info("   Download from: https://visualstudio.microsoft.com/downloads/")
    
    # Check for CMake
    cmake_found = False
    try:
        result = subprocess.run(["cmake", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            version = result.stdout.split('\n')[0]
            logging.info(f"✅ CMake found: {version}")
            cmake_found = True
    except FileNotFoundError:
        logging.warning("⚠️ CMake not found in PATH")
        logging.info("💡 You can install CMake from: https://cmake.org/download/")
    
    return vs_found or cmake_found


def build_cpp_recorder(force_rebuild=False):
    """Build the C++ video recorder"""
    logging.info("🔨 Building C++ video recorder...")
    
    cpp_dir = Path("cpp_video_recorder")
    if not cpp_dir.exists():
        logging.error(f"❌ C++ source directory not found: {cpp_dir}")
        return False
    
    # Check if already built
    dll_path = cpp_dir / "video_recorder.dll"
    if dll_path.exists() and not force_rebuild:
        logging.info("✅ C++ recorder already built (use --force-rebuild to rebuild)")
        return True
    
    # Run build script
    build_script = cpp_dir / "build.py"
    if not build_script.exists():
        logging.error(f"❌ Build script not found: {build_script}")
        return False
    
    try:
        logging.info("Running build script...")
        result = subprocess.run([
            sys.executable, str(build_script),
            "--build-type", "Release",
            "--method", "auto"
        ], cwd=cpp_dir, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            logging.info("✅ C++ recorder built successfully!")
            logging.debug(f"Build output:\n{result.stdout}")
            return True
        else:
            logging.error(f"❌ Build failed with return code {result.returncode}")
            logging.error(f"Build stderr:\n{result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logging.error("❌ Build timed out")
        return False
    except Exception as e:
        logging.error(f"❌ Build failed: {e}")
        return False


def test_cpp_recorder():
    """Test the C++ video recorder"""
    logging.info("🧪 Testing C++ video recorder...")
    
    try:
        # Add cpp_video_recorder to path
        cpp_dir = Path("cpp_video_recorder")
        sys.path.insert(0, str(cpp_dir))
        
        # Test import
        from python_wrapper import HighPerformanceVideoRecorder
        logging.info("✅ Python wrapper imported successfully")
        
        # Test initialization
        with HighPerformanceVideoRecorder() as recorder:
            success = recorder.initialize(
                width=640, height=480, fps=30,
                output_path="test_init.avi"
            )
            
            if success:
                logging.info("✅ C++ recorder initialization test passed")
                return True
            else:
                error = recorder.get_last_error()
                logging.error(f"❌ C++ recorder initialization failed: {error}")
                return False
                
    except ImportError as e:
        logging.error(f"❌ Failed to import C++ recorder: {e}")
        logging.info("💡 This usually means the DLL was not built or is missing dependencies")
        return False
    except Exception as e:
        logging.error(f"❌ C++ recorder test failed: {e}")
        return False


def test_fallback_recorder():
    """Test the fallback OpenCV recorder"""
    logging.info("🧪 Testing fallback OpenCV recorder...")
    
    try:
        from utils.enhanced_video_recorder import EnhancedVideoRecorder
        
        with EnhancedVideoRecorder() as recorder:
            # Force OpenCV backend
            success = recorder.configure(force_opencv=True)
            
            if success:
                logging.info("✅ OpenCV fallback recorder test passed")
                return True
            else:
                logging.error("❌ OpenCV fallback recorder test failed")
                return False
                
    except Exception as e:
        logging.error(f"❌ Enhanced recorder test failed: {e}")
        return False


def update_requirements():
    """Update requirements.txt if needed"""
    logging.info("📋 Checking requirements...")
    
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        logging.warning("⚠️ requirements.txt not found")
        return
        
    # Read current requirements
    current_requirements = requirements_file.read_text().strip().split('\n')
    
    # Check for numpy (needed for enhanced recorder)
    has_numpy = any(line.startswith('numpy') for line in current_requirements if line.strip())
    
    if not has_numpy:
        logging.info("➕ Adding numpy to requirements.txt")
        with open(requirements_file, 'a') as f:
            f.write('\nnumpy>=1.19.0  # For enhanced video recorder\n')
            
    logging.info("✅ Requirements checked")


def create_build_instructions():
    """Create build instructions file"""
    instructions = """
# C++ High-Performance Video Recorder - Build Instructions

## Automatic Setup (Recommended)

Run the setup script:
```bash
python setup_cpp_recorder.py
```

## Manual Build

### Prerequisites
1. Windows 10 or later
2. Visual Studio 2019+ or Build Tools for Visual Studio
3. CMake (optional, but recommended)

### Build Steps

1. Navigate to cpp_video_recorder directory:
```bash
cd cpp_video_recorder
```

2. Build with Python script:
```bash
python build.py --build-type Release
```

3. Or build with CMake manually:
```bash
mkdir build
cd build
cmake .. -A x64
cmake --build . --config Release
```

### Testing

Test the built recorder:
```bash
python setup_cpp_recorder.py --test-only
```

## Troubleshooting

### Common Issues

1. **"Visual Studio not found"**
   - Install Visual Studio Community 2019+ or Build Tools for Visual Studio
   - Download from: https://visualstudio.microsoft.com/downloads/

2. **"CMake not found"**
   - Install CMake from: https://cmake.org/download/
   - Add CMake to your PATH

3. **"DLL not found"**
   - Make sure the build completed successfully
   - Check that video_recorder.dll exists in cpp_video_recorder/

4. **"Media Foundation errors"**
   - Ensure you're on Windows 10 or later
   - Update Windows to the latest version

### Fallback Mode

If the C++ recorder fails to build or initialize, the system will automatically
fall back to OpenCV recording, which provides basic functionality but may
be more susceptible to frame loss during high-speed capture.

## Performance Benefits

The C++ recorder provides:
- Higher recording quality (up to 15 Mbps vs 8 Mbps)
- Lower CPU usage through hardware acceleration
- Better frame timing and reduced frame loss
- Direct OS-level camera access via Windows Media Foundation

## File Locations

- C++ source: `cpp_video_recorder/video_recorder.cpp`
- Python wrapper: `cpp_video_recorder/python_wrapper.py`
- Enhanced recorder: `utils/enhanced_video_recorder.py`
- Build script: `cpp_video_recorder/build.py`
- This setup script: `setup_cpp_recorder.py`
"""
    
    instructions_file = Path("CPP_RECORDER_SETUP.md")
    instructions_file.write_text(instructions.strip())
    logging.info(f"📝 Build instructions written to: {instructions_file}")


def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description="Setup C++ High-Performance Video Recorder")
    parser.add_argument("--test-only", action="store_true", 
                       help="Only test existing installation")
    parser.add_argument("--force-rebuild", action="store_true", 
                       help="Force rebuild even if DLL exists")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Verbose output")
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    logging.info("🚀 Setting up C++ High-Performance Video Recorder")
    logging.info("=" * 60)
    
    # Check system requirements
    if not check_system_requirements():
        logging.error("❌ System requirements not met")
        return False
    
    # Test-only mode
    if args.test_only:
        logging.info("🧪 Testing existing installation...")
        
        cpp_success = test_cpp_recorder()
        fallback_success = test_fallback_recorder()
        
        if cpp_success:
            logging.info("✅ C++ high-performance recorder is working!")
        elif fallback_success:
            logging.info("⚠️ C++ recorder not available, but OpenCV fallback is working")
        else:
            logging.error("❌ Neither C++ nor OpenCV recorder is working")
            return False
            
        return True
    
    # Check build tools
    if not check_build_tools():
        logging.warning("⚠️ Build tools not fully available")
        logging.info("You can still try to build, or install the missing tools")
        
        response = input("Continue anyway? (y/N): ").strip().lower()
        if response != 'y':
            logging.info("Setup cancelled")
            return False
    
    # Build C++ recorder
    if not build_cpp_recorder(args.force_rebuild):
        logging.error("❌ Failed to build C++ recorder")
        logging.info("💡 You can still use the application with OpenCV fallback")
        
        # Test fallback
        if test_fallback_recorder():
            logging.info("✅ OpenCV fallback is working")
        else:
            logging.error("❌ Even OpenCV fallback failed")
            return False
    else:
        # Test C++ recorder
        if not test_cpp_recorder():
            logging.warning("⚠️ C++ recorder built but test failed")
            if test_fallback_recorder():
                logging.info("✅ OpenCV fallback is available")
            else:
                logging.error("❌ No working recorder available")
                return False
        else:
            logging.info("✅ C++ high-performance recorder is ready!")
    
    # Update requirements
    update_requirements()
    
    # Create build instructions
    create_build_instructions()
    
    logging.info("=" * 60)
    logging.info("🎉 Setup completed successfully!")
    logging.info("")
    logging.info("The enhanced video recorder is now ready to use.")
    logging.info("It will automatically use the best available backend:")
    logging.info("  1. C++ High-Performance (if available)")
    logging.info("  2. OpenCV Fallback (always available)")
    logging.info("")
    logging.info("You can now run your shooting simulator with improved")
    logging.info("video recording that should capture fast shots without")
    logging.info("dropping frames!")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 