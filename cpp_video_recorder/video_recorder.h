#pragma once

#include <windows.h>
#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <mferror.h>
#include <mfcaptureengine.h>
#include <winrt/base.h>
#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <mutex>
#include <atomic>
#include <chrono>

// Link required libraries
#pragma comment(lib, "mf.lib")
#pragma comment(lib, "mfplat.lib")
#pragma comment(lib, "mfreadwrite.lib")
#pragma comment(lib, "mfuuid.lib")

namespace VideoRecorder {

class HighPerformanceVideoRecorder {
public:
    struct RecordingConfig {
        int width = 640;
        int height = 480;
        int fps = 30;
        int bitrate = 8000000;  // 8 Mbps for high quality
        std::string format = "MJPEG";  // MJPEG or RAW
        std::string outputPath;
        bool useHardwareAcceleration = true;
    };

    struct RecordingStats {
        size_t framesWritten = 0;
        size_t framesDropped = 0;
        double recordingDuration = 0.0;
        double averageFps = 0.0;
        std::chrono::high_resolution_clock::time_point startTime;
    };

private:
    // Media Foundation interfaces
    winrt::com_ptr<IMFMediaSource> m_pVideoSource;
    winrt::com_ptr<IMFSourceReader> m_pSourceReader;
    winrt::com_ptr<IMFSinkWriter> m_pSinkWriter;
    winrt::com_ptr<IMFMediaType> m_pInputType;
    winrt::com_ptr<IMFMediaType> m_pOutputType;
    
    // Recording state
    std::atomic<bool> m_isRecording;
    std::atomic<bool> m_isInitialized;
    std::mutex m_statsMutex;
    RecordingConfig m_config;
    RecordingStats m_stats;
    
    // Stream indices
    DWORD m_streamIndex;
    LONGLONG m_rtStart;
    LONGLONG m_frameTimestamp;
    
    // Buffer management
    std::vector<BYTE> m_frameBuffer;
    size_t m_frameSize;

public:
    HighPerformanceVideoRecorder();
    ~HighPerformanceVideoRecorder();

    // Main interface
    bool Initialize(const RecordingConfig& config);
    bool StartRecording();
    bool StopRecording();
    bool IsRecording() const;
    bool IsInitialized() const;
    
    // Frame capture
    bool CaptureFrame();
    bool WriteFrameBuffer(const BYTE* frameData, size_t dataSize);
    
    // Configuration
    bool SetVideoSource(int deviceIndex = 0);
    bool ConfigureVideoFormat();
    bool ConfigureSinkWriter();
    
    // Statistics
    RecordingStats GetStats() const;
    void ResetStats();
    
    // Cleanup
    void Cleanup();
    
    // Error handling
    std::string GetLastError() const;

private:
    // Helper methods
    HRESULT CreateVideoDeviceSource(IMFMediaSource** ppSource, int deviceIndex);
    HRESULT ConfigureSourceReader();
    HRESULT SetupOutputType();
    HRESULT SetupInputType();
    bool IsFormatSupported(const GUID& format);
    void UpdateStats();
    
    // Error tracking
    mutable std::string m_lastError;
    void SetError(const std::string& error);
    void SetError(HRESULT hr, const std::string& context);
};

// C interface for Python integration
extern "C" {
    // Handle type for Python
    typedef void* VideoRecorderHandle;
    
    // Core functions
    VideoRecorderHandle* CreateVideoRecorder();
    void DestroyVideoRecorder(VideoRecorderHandle* handle);
    
    // Configuration
    bool SetRecordingConfig(VideoRecorderHandle* handle, 
                           int width, int height, int fps, 
                           int bitrate, const char* format, 
                           const char* outputPath, bool useHwAccel);
    
    // Recording control
    bool InitializeRecorder(VideoRecorderHandle* handle);
    bool StartVideoRecording(VideoRecorderHandle* handle);
    bool StopVideoRecording(VideoRecorderHandle* handle);
    bool IsVideoRecording(VideoRecorderHandle* handle);
    
    // Frame capture
    bool CaptureVideoFrame(VideoRecorderHandle* handle);
    
    // Statistics
    bool GetRecordingStats(VideoRecorderHandle* handle,
                          size_t* framesWritten, size_t* framesDropped,
                          double* duration, double* avgFps);
    
    // Error handling
    const char* GetRecorderError(VideoRecorderHandle* handle);
    
    // Cleanup
    void CleanupRecorder(VideoRecorderHandle* handle);
}

} // namespace VideoRecorder 