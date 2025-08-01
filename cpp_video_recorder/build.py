#!/usr/bin/env python3
"""
Build script for the C++ high-performance video recorder
Handles compilation and setup for Windows
"""

import os
import sys
import subprocess
import shutil
import logging
from pathlib import Path
import argparse


def setup_logging(verbose=False):
    """Set up logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )


def find_visual_studio():
    """Find Visual Studio installation"""
    # Common VS installation paths
    vs_paths = [
        "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Auxiliary/Build",
        "C:/Program Files/Microsoft Visual Studio/2022/Professional/VC/Auxiliary/Build", 
        "C:/Program Files/Microsoft Visual Studio/2022/Enterprise/VC/Auxiliary/Build",
        "C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/VC/Auxiliary/Build",
        "C:/Program Files (x86)/Microsoft Visual Studio/2019/Professional/VC/Auxiliary/Build",
        "C:/Program Files (x86)/Microsoft Visual Studio/2019/Enterprise/VC/Auxiliary/Build",
    ]
    
    for vs_path in vs_paths:
        vcvars_path = Path(vs_path) / "vcvars64.bat"
        if vcvars_path.exists():
            logging.info(f"Found Visual Studio at: {vs_path}")
            return str(vcvars_path)
    
    return None


def find_cmake():
    """Find CMake installation"""
    cmake_paths = [
        "cmake",  # In PATH
        "C:/Program Files/CMake/bin/cmake.exe",
        "C:/Program Files (x86)/CMake/bin/cmake.exe",
    ]
    
    for cmake_path in cmake_paths:
        try:
            result = subprocess.run([cmake_path, "--version"], 
                                 capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                logging.info(f"Found CMake: {cmake_path}")
                logging.info(f"CMake version: {result.stdout.split()[2]}")
                return cmake_path
        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue
    
    return None


def run_command(cmd, cwd=None, env=None):
    """Run a command and log output"""
    logging.debug(f"Running command: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    logging.debug(f"Working directory: {cwd or os.getcwd()}")
    
    try:
        result = subprocess.run(
            cmd, 
            cwd=cwd, 
            env=env,
            capture_output=True, 
            text=True, 
            timeout=300,  # 5 minutes
            shell=True if isinstance(cmd, str) else False
        )
        
        if result.stdout:
            logging.debug(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            logging.debug(f"STDERR:\n{result.stderr}")
            
        if result.returncode != 0:
            logging.error(f"Command failed with return code {result.returncode}")
            logging.error(f"STDERR: {result.stderr}")
            return False
            
        return True
        
    except subprocess.TimeoutExpired:
        logging.error("Command timed out")
        return False
    except Exception as e:
        logging.error(f"Command execution failed: {e}")
        return False


def build_with_cmake(source_dir, build_type="Release", clean=False):
    """Build using CMake"""
    logging.info("Building with CMake...")
    
    # Find CMake
    cmake = find_cmake()
    if not cmake:
        logging.error("CMake not found. Please install CMake and add it to PATH")
        return False
    
    build_dir = source_dir / "build"
    
    # Clean build directory if requested
    if clean and build_dir.exists():
        logging.info("Cleaning build directory...")
        shutil.rmtree(build_dir)
    
    # Create build directory
    build_dir.mkdir(exist_ok=True)
    
    # Configure
    logging.info("Configuring with CMake...")
    configure_cmd = [
        cmake, 
        "-S", str(source_dir),
        "-B", str(build_dir),
        f"-DCMAKE_BUILD_TYPE={build_type}",
        "-A", "x64"  # 64-bit build
    ]
    
    if not run_command(configure_cmd):
        logging.error("CMake configuration failed")
        return False
    
    # Build
    logging.info("Building with CMake...")
    build_cmd = [
        cmake,
        "--build", str(build_dir),
        "--config", build_type,
        "--parallel"
    ]
    
    if not run_command(build_cmd):
        logging.error("CMake build failed")
        return False
    
    # Copy DLL to source directory
    dll_src = build_dir / build_type / "video_recorder.dll"
    dll_dst = source_dir / "video_recorder.dll"
    
    if dll_src.exists():
        shutil.copy2(dll_src, dll_dst)
        logging.info(f"DLL copied to: {dll_dst}")
    else:
        # Try alternative paths
        alt_paths = [
            build_dir / "bin" / build_type / "video_recorder.dll",
            build_dir / "bin" / "video_recorder.dll",
            build_dir / "video_recorder.dll"
        ]
        
        for alt_path in alt_paths:
            if alt_path.exists():
                shutil.copy2(alt_path, dll_dst)
                logging.info(f"DLL copied from alternative path: {alt_path}")
                break
        else:
            logging.warning("Could not find built DLL to copy")
    
    return True


def build_with_msvc(source_dir, build_type="Release"):
    """Build using MSVC directly"""
    logging.info("Building with MSVC...")
    
    # Find Visual Studio
    vcvars = find_visual_studio()
    if not vcvars:
        logging.error("Visual Studio not found. Please install Visual Studio 2019 or later")
        return False
    
    # Set up environment
    env = os.environ.copy()
    
    # Run vcvars64.bat to set up MSVC environment
    vcvars_cmd = f'"{vcvars}" && set'
    try:
        result = subprocess.run(vcvars_cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            # Parse environment variables
            for line in result.stdout.split('\n'):
                if '=' in line:
                    key, value = line.split('=', 1)
                    env[key] = value
        else:
            logging.error("Failed to set up MSVC environment")
            return False
    except Exception as e:
        logging.error(f"Failed to run vcvars64.bat: {e}")
        return False
    
    # Compile command
    sources = ["video_recorder.cpp"]
    includes = ["/I."]
    libs = ["mf.lib", "mfplat.lib", "mfreadwrite.lib", "mfuuid.lib"]
    
    compile_cmd = [
        "cl.exe",
        "/std:c++20",
        "/EHsc",
        "/MD" if build_type == "Release" else "/MDd",
        "/O2" if build_type == "Release" else "/Od",
        "/DWIN32", "/D_WINDOWS", "/D_USRDLL",
        "/LD",  # Create DLL
        *includes,
        *sources,
        "/link",
        *libs,
        "/OUT:video_recorder.dll"
    ]
    
    if not run_command(compile_cmd, cwd=source_dir, env=env):
        logging.error("MSVC compilation failed")
        return False
    
    return True


def test_build(source_dir):
    """Test the built DLL"""
    logging.info("Testing the built DLL...")
    
    dll_path = source_dir / "video_recorder.dll"
    if not dll_path.exists():
        logging.error("DLL not found for testing")
        return False
    
    # Test with Python wrapper
    try:
        sys.path.insert(0, str(source_dir))
        from python_wrapper import HighPerformanceVideoRecorder
        
        with HighPerformanceVideoRecorder() as recorder:
            success = recorder.initialize(
                width=640, height=480, fps=30,
                output_path="test.avi"
            )
            
            if success:
                logging.info("DLL test successful!")
                return True
            else:
                logging.error(f"DLL test failed: {recorder.get_last_error()}")
                return False
                
    except Exception as e:
        logging.error(f"DLL test failed with exception: {e}")
        return False


def main():
    """Main build function"""
    parser = argparse.ArgumentParser(description="Build C++ video recorder")
    parser.add_argument("--build-type", choices=["Debug", "Release"], 
                       default="Release", help="Build type")
    parser.add_argument("--method", choices=["cmake", "msvc", "auto"], 
                       default="auto", help="Build method")
    parser.add_argument("--clean", action="store_true", 
                       help="Clean build directory first")
    parser.add_argument("--test", action="store_true", 
                       help="Test the built DLL")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Verbose output")
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    # Get source directory
    source_dir = Path(__file__).parent
    logging.info(f"Source directory: {source_dir}")
    logging.info(f"Build type: {args.build_type}")
    
    # Choose build method
    build_method = args.method
    if build_method == "auto":
        # Try CMake first, fall back to MSVC
        if find_cmake():
            build_method = "cmake"
        elif find_visual_studio():
            build_method = "msvc"
        else:
            logging.error("Neither CMake nor Visual Studio found")
            return False
    
    logging.info(f"Using build method: {build_method}")
    
    # Build
    success = False
    if build_method == "cmake":
        success = build_with_cmake(source_dir, args.build_type, args.clean)
    elif build_method == "msvc":
        success = build_with_msvc(source_dir, args.build_type)
    
    if not success:
        logging.error("Build failed")
        return False
    
    logging.info("Build completed successfully!")
    
    # Test if requested
    if args.test:
        if not test_build(source_dir):
            logging.error("Build test failed")
            return False
    
    logging.info("All operations completed successfully!")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 