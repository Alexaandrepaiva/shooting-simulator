cmake_minimum_required(VERSION 3.16)
project(VideoRecorder)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler-specific options
if(MSVC)
    # Use static runtime for better distribution
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    
    # MSVC specific flags
    add_compile_options(
        /W4          # Warning level 4
        /WX          # Treat warnings as errors
        /permissive- # Disable non-conforming code
        /Zc:__cplusplus # Enable correct __cplusplus macro
    )
    
    # Release optimizations
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(/O2 /Ob2 /DNDEBUG)
    endif()
    
    # Debug options
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(/Od /Zi)
    endif()
else()
    # GCC/Clang flags
    add_compile_options(
        -Wall -Wextra -Wpedantic
        -Werror
    )
    
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(-O3 -DNDEBUG)
    endif()
    
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(-g -O0)
    endif()
endif()

# Find required packages
find_package(PkgConfig QUIET)

# Windows-specific setup
if(WIN32)
    # Enable Windows 10 features
    add_definitions(-DWINVER=0x0A00 -D_WIN32_WINNT=0x0A00)
    
    # Required Windows libraries
    set(WINDOWS_LIBS
        mf
        mfplat
        mfreadwrite
        mfuuid
        ole32
        oleaut32
        kernel32
        user32
        gdi32
        winspool
        shell32
        comctl32
        advapi32
        uuid
    )
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Source files
set(SOURCES
    video_recorder.cpp
    video_recorder.h
)

# Create shared library (DLL on Windows)
add_library(video_recorder SHARED ${SOURCES})

# Link libraries
if(WIN32)
    target_link_libraries(video_recorder ${WINDOWS_LIBS})
endif()

# Set output properties
set_target_properties(video_recorder PROPERTIES
    # Output name
    OUTPUT_NAME "video_recorder"
    
    # Windows-specific properties
    $<$<PLATFORM_ID:Windows>:
        WINDOWS_EXPORT_ALL_SYMBOLS ON
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
        ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    >
    
    # Set position-independent code for shared library
    POSITION_INDEPENDENT_CODE ON
)

# Platform-specific installation
if(WIN32)
    # Install DLL to Python package directory
    install(TARGETS video_recorder
        RUNTIME DESTINATION .
        LIBRARY DESTINATION .
        ARCHIVE DESTINATION .
    )
    
    # Copy to common locations for development
    add_custom_command(TARGET video_recorder POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:video_recorder>
        ${CMAKE_CURRENT_SOURCE_DIR}/video_recorder.dll
        COMMENT "Copying DLL to source directory"
    )
else()
    install(TARGETS video_recorder
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
    )
endif()

# Optional: Create a test executable
option(BUILD_TESTS "Build test executable" OFF)
if(BUILD_TESTS)
    add_executable(test_recorder test_main.cpp)
    target_link_libraries(test_recorder video_recorder)
    
    if(WIN32)
        target_link_libraries(test_recorder ${WINDOWS_LIBS})
    endif()
endif()

# Print configuration summary
message(STATUS "")
message(STATUS "=== Video Recorder Build Configuration ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Target platform: ${CMAKE_SYSTEM_NAME}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID}")
if(WIN32)
    message(STATUS "Windows libraries: ${WINDOWS_LIBS}")
endif()
message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "Build tests: ${BUILD_TESTS}")
message(STATUS "==========================================")
message(STATUS "") 