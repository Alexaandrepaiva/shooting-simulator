#include "video_recorder.h"
#include <comdef.h>
#include <sstream>
#include <algorithm>

namespace VideoRecorder {

HighPerformanceVideoRecorder::HighPerformanceVideoRecorder()
    : m_isRecording(false)
    , m_isInitialized(false)
    , m_streamIndex(0)
    , m_rtStart(0)
    , m_frameTimestamp(0)
    , m_frameSize(0)
{
    // Initialize Media Foundation
    HRESULT hr = MFStartup(MF_VERSION);
    if (FAILED(hr)) {
        SetError(hr, "Failed to initialize Media Foundation");
    }
}

HighPerformanceVideoRecorder::~HighPerformanceVideoRecorder() {
    Cleanup();
    MFShutdown();
}

bool HighPerformanceVideoRecorder::Initialize(const RecordingConfig& config) {
    if (m_isRecording) {
        SetError("Cannot initialize while recording is in progress");
        return false;
    }

    Cleanup();
    m_config = config;
    
    // Create video source
    if (!SetVideoSource(0)) {
        return false;
    }
    
    // Configure video format
    if (!ConfigureVideoFormat()) {
        return false;
    }
    
    // Configure sink writer for output
    if (!ConfigureSinkWriter()) {
        return false;
    }
    
    // Calculate frame size for buffer management
    m_frameSize = m_config.width * m_config.height * 3; // RGB24
    m_frameBuffer.resize(m_frameSize);
    
    m_isInitialized = true;
    ResetStats();
    
    return true;
}

bool HighPerformanceVideoRecorder::StartRecording() {
    if (!m_isInitialized) {
        SetError("Recorder not initialized");
        return false;
    }
    
    if (m_isRecording) {
        SetError("Recording already in progress");
        return false;
    }
    
    // Begin writing
    HRESULT hr = m_pSinkWriter->BeginWriting();
    if (FAILED(hr)) {
        SetError(hr, "Failed to begin writing");
        return false;
    }
    
    // Reset timing
    m_rtStart = 0;
    m_frameTimestamp = 0;
    m_stats.startTime = std::chrono::high_resolution_clock::now();
    
    m_isRecording = true;
    return true;
}

bool HighPerformanceVideoRecorder::StopRecording() {
    if (!m_isRecording) {
        return true;
    }
    
    m_isRecording = false;
    
    // Finalize the recording
    if (m_pSinkWriter) {
        HRESULT hr = m_pSinkWriter->Finalize();
        if (FAILED(hr)) {
            SetError(hr, "Failed to finalize recording");
            return false;
        }
    }
    
    UpdateStats();
    return true;
}

bool HighPerformanceVideoRecorder::CaptureFrame() {
    if (!m_isRecording || !m_pSourceReader) {
        return false;
    }
    
    DWORD streamIndex;
    DWORD flags;
    LONGLONG timestamp;
    winrt::com_ptr<IMFSample> pSample;
    
    // Read sample from source
    HRESULT hr = m_pSourceReader->ReadSample(
        MF_SOURCE_READER_FIRST_VIDEO_STREAM,
        0,
        &streamIndex,
        &flags,
        &timestamp,
        pSample.put()
    );
    
    if (FAILED(hr)) {
        SetError(hr, "Failed to read video sample");
        return false;
    }
    
    if (flags & MF_SOURCE_READER_STREAMTICK) {
        // Stream tick without data, try again
        return true;
    }
    
    if (!pSample) {
        return true; // No sample available
    }
    
    // Set timestamp for output
    LONGLONG outputTimestamp = m_frameTimestamp;
    LONGLONG frameDuration = 10000000LL / m_config.fps; // 100ns units
    
    hr = pSample->SetSampleTime(outputTimestamp);
    if (SUCCEEDED(hr)) {
        hr = pSample->SetSampleDuration(frameDuration);
    }
    
    if (FAILED(hr)) {
        SetError(hr, "Failed to set sample timing");
        return false;
    }
    
    // Write sample to sink
    hr = m_pSinkWriter->WriteSample(m_streamIndex, pSample.get());
    if (FAILED(hr)) {
        SetError(hr, "Failed to write sample");
        std::lock_guard<std::mutex> lock(m_statsMutex);
        m_stats.framesDropped++;
        return false;
    }
    
    // Update stats and timing
    {
        std::lock_guard<std::mutex> lock(m_statsMutex);
        m_stats.framesWritten++;
    }
    
    m_frameTimestamp += frameDuration;
    return true;
}

bool HighPerformanceVideoRecorder::SetVideoSource(int deviceIndex) {
    // Create video capture device
    HRESULT hr = CreateVideoDeviceSource(m_pVideoSource.put(), deviceIndex);
    if (FAILED(hr)) {
        SetError(hr, "Failed to create video device source");
        return false;
    }
    
    // Create source reader
    hr = ConfigureSourceReader();
    if (FAILED(hr)) {
        return false;
    }
    
    return true;
}

bool HighPerformanceVideoRecorder::ConfigureVideoFormat() {
    if (!m_pSourceReader) {
        SetError("Source reader not initialized");
        return false;
    }
    
    // Set up input type (from camera)
    HRESULT hr = SetupInputType();
    if (FAILED(hr)) {
        return false;
    }
    
    // Set media type on source reader
    hr = m_pSourceReader->SetCurrentMediaType(
        MF_SOURCE_READER_FIRST_VIDEO_STREAM,
        nullptr,
        m_pInputType.get()
    );
    
    if (FAILED(hr)) {
        SetError(hr, "Failed to set source media type");
        return false;
    }
    
    return true;
}

bool HighPerformanceVideoRecorder::ConfigureSinkWriter() {
    // Create sink writer
    winrt::com_ptr<IMFAttributes> pAttributes;
    HRESULT hr = MFCreateAttributes(pAttributes.put(), 1);
    if (FAILED(hr)) {
        SetError(hr, "Failed to create attributes");
        return false;
    }
    
    // Enable hardware acceleration if requested
    if (m_config.useHardwareAcceleration) {
        hr = pAttributes->SetUINT32(MF_SINK_WRITER_DISABLE_THROTTLING, TRUE);
        if (FAILED(hr)) {
            SetError(hr, "Failed to disable throttling");
            return false;
        }
    }
    
    // Convert output path to wide string
    std::wstring wOutputPath(m_config.outputPath.begin(), m_config.outputPath.end());
    
    hr = MFCreateSinkWriterFromURL(
        wOutputPath.c_str(),
        nullptr,
        pAttributes.get(),
        m_pSinkWriter.put()
    );
    
    if (FAILED(hr)) {
        SetError(hr, "Failed to create sink writer");
        return false;
    }
    
    // Set up output type
    hr = SetupOutputType();
    if (FAILED(hr)) {
        return false;
    }
    
    // Add stream to sink writer
    hr = m_pSinkWriter->AddStream(m_pOutputType.get(), &m_streamIndex);
    if (FAILED(hr)) {
        SetError(hr, "Failed to add stream to sink writer");
        return false;
    }
    
    // Set input type for sink writer (same as source)
    hr = m_pSinkWriter->SetInputMediaType(
        m_streamIndex,
        m_pInputType.get(),
        nullptr
    );
    
    if (FAILED(hr)) {
        SetError(hr, "Failed to set input media type on sink writer");
        return false;
    }
    
    return true;
}

HRESULT HighPerformanceVideoRecorder::CreateVideoDeviceSource(IMFMediaSource** ppSource, int deviceIndex) {
    *ppSource = nullptr;
    
    winrt::com_ptr<IMFAttributes> pAttributes;
    HRESULT hr = MFCreateAttributes(pAttributes.put(), 1);
    if (FAILED(hr)) return hr;
    
    hr = pAttributes->SetGUID(
        MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE,
        MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID
    );
    if (FAILED(hr)) return hr;
    
    IMFActivate** ppDevices = nullptr;
    UINT32 count = 0;
    
    hr = MFEnumDeviceSources(pAttributes.get(), &ppDevices, &count);
    if (FAILED(hr)) return hr;
    
    if (count == 0 || deviceIndex >= static_cast<int>(count)) {
        if (ppDevices) {
            for (UINT32 i = 0; i < count; i++) {
                if (ppDevices[i]) ppDevices[i]->Release();
            }
            CoTaskMemFree(ppDevices);
        }
        return E_INVALIDARG;
    }
    
    hr = ppDevices[deviceIndex]->ActivateObject(IID_PPV_ARGS(ppSource));
    
    // Cleanup
    for (UINT32 i = 0; i < count; i++) {
        if (ppDevices[i]) ppDevices[i]->Release();
    }
    CoTaskMemFree(ppDevices);
    
    return hr;
}

HRESULT HighPerformanceVideoRecorder::ConfigureSourceReader() {
    winrt::com_ptr<IMFAttributes> pAttributes;
    HRESULT hr = MFCreateAttributes(pAttributes.put(), 2);
    if (FAILED(hr)) return hr;
    
    // Enable video processing for better performance
    hr = pAttributes->SetUINT32(MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING, TRUE);
    if (FAILED(hr)) return hr;
    
    // Disable throttling for real-time capture
    hr = pAttributes->SetUINT32(MF_SOURCE_READER_DISABLE_DXVA, FALSE);
    if (FAILED(hr)) return hr;
    
    hr = MFCreateSourceReaderFromMediaSource(
        m_pVideoSource.get(),
        pAttributes.get(),
        m_pSourceReader.put()
    );
    
    return hr;
}

HRESULT HighPerformanceVideoRecorder::SetupOutputType() {
    HRESULT hr = MFCreateMediaType(m_pOutputType.put());
    if (FAILED(hr)) return hr;
    
    hr = m_pOutputType->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);
    if (FAILED(hr)) return hr;
    
    // Choose codec based on configuration
    GUID codecGuid = MFVideoFormat_H264;
    if (m_config.format == "RAW") {
        codecGuid = MFVideoFormat_RGB24;
    } else if (m_config.format == "MJPEG") {
        codecGuid = MFVideoFormat_MJPG;
    }
    
    hr = m_pOutputType->SetGUID(MF_MT_SUBTYPE, codecGuid);
    if (FAILED(hr)) return hr;
    
    hr = MFSetAttributeSize(m_pOutputType.get(), MF_MT_FRAME_SIZE, m_config.width, m_config.height);
    if (FAILED(hr)) return hr;
    
    hr = MFSetAttributeRatio(m_pOutputType.get(), MF_MT_FRAME_RATE, m_config.fps, 1);
    if (FAILED(hr)) return hr;
    
    hr = m_pOutputType->SetUINT32(MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive);
    if (FAILED(hr)) return hr;
    
    // Set bitrate for compressed formats
    if (m_config.format != "RAW") {
        hr = m_pOutputType->SetUINT32(MF_MT_AVG_BITRATE, m_config.bitrate);
        if (FAILED(hr)) return hr;
    }
    
    return S_OK;
}

HRESULT HighPerformanceVideoRecorder::SetupInputType() {
    HRESULT hr = MFCreateMediaType(m_pInputType.put());
    if (FAILED(hr)) return hr;
    
    hr = m_pInputType->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);
    if (FAILED(hr)) return hr;
    
    hr = m_pInputType->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_RGB24);
    if (FAILED(hr)) return hr;
    
    hr = MFSetAttributeSize(m_pInputType.get(), MF_MT_FRAME_SIZE, m_config.width, m_config.height);
    if (FAILED(hr)) return hr;
    
    hr = MFSetAttributeRatio(m_pInputType.get(), MF_MT_FRAME_RATE, m_config.fps, 1);
    if (FAILED(hr)) return hr;
    
    hr = m_pInputType->SetUINT32(MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive);
    if (FAILED(hr)) return hr;
    
    return S_OK;
}

RecordingStats HighPerformanceVideoRecorder::GetStats() const {
    std::lock_guard<std::mutex> lock(m_statsMutex);
    RecordingStats stats = m_stats;
    
    // Calculate current duration and FPS
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - stats.startTime);
    stats.recordingDuration = duration.count() / 1000.0;
    
    if (stats.recordingDuration > 0) {
        stats.averageFps = stats.framesWritten / stats.recordingDuration;
    }
    
    return stats;
}

void HighPerformanceVideoRecorder::ResetStats() {
    std::lock_guard<std::mutex> lock(m_statsMutex);
    m_stats = RecordingStats();
    m_stats.startTime = std::chrono::high_resolution_clock::now();
}

void HighPerformanceVideoRecorder::UpdateStats() {
    // Stats are updated in real-time during frame capture
}

void HighPerformanceVideoRecorder::Cleanup() {
    if (m_isRecording) {
        StopRecording();
    }
    
    m_pSinkWriter.reset();
    m_pSourceReader.reset();
    m_pVideoSource.reset();
    m_pInputType.reset();
    m_pOutputType.reset();
    
    m_isInitialized = false;
}

void HighPerformanceVideoRecorder::SetError(const std::string& error) {
    m_lastError = error;
}

void HighPerformanceVideoRecorder::SetError(HRESULT hr, const std::string& context) {
    _com_error err(hr);
    std::wstring wErrMsg = err.ErrorMessage();
    std::string errMsg(wErrMsg.begin(), wErrMsg.end());
    
    std::ostringstream oss;
    oss << context << ": " << errMsg << " (HRESULT: 0x" << std::hex << hr << ")";
    m_lastError = oss.str();
}

std::string HighPerformanceVideoRecorder::GetLastError() const {
    return m_lastError;
}

bool HighPerformanceVideoRecorder::IsRecording() const {
    return m_isRecording;
}

bool HighPerformanceVideoRecorder::IsInitialized() const {
    return m_isInitialized;
}

// C interface implementation
extern "C" {
    VideoRecorderHandle* CreateVideoRecorder() {
        try {
            return reinterpret_cast<VideoRecorderHandle*>(new HighPerformanceVideoRecorder());
        } catch (...) {
            return nullptr;
        }
    }
    
    void DestroyVideoRecorder(VideoRecorderHandle* handle) {
        if (handle) {
            delete reinterpret_cast<HighPerformanceVideoRecorder*>(handle);
        }
    }
    
    bool SetRecordingConfig(VideoRecorderHandle* handle, 
                           int width, int height, int fps, 
                           int bitrate, const char* format, 
                           const char* outputPath, bool useHwAccel) {
        if (!handle) return false;
        
        auto* recorder = reinterpret_cast<HighPerformanceVideoRecorder*>(handle);
        HighPerformanceVideoRecorder::RecordingConfig config;
        config.width = width;
        config.height = height;
        config.fps = fps;
        config.bitrate = bitrate;
        config.format = format ? format : "MJPEG";
        config.outputPath = outputPath ? outputPath : "";
        config.useHardwareAcceleration = useHwAccel;
        
        return recorder->Initialize(config);
    }
    
    bool InitializeRecorder(VideoRecorderHandle* handle) {
        if (!handle) return false;
        auto* recorder = reinterpret_cast<HighPerformanceVideoRecorder*>(handle);
        return recorder->IsInitialized();
    }
    
    bool StartVideoRecording(VideoRecorderHandle* handle) {
        if (!handle) return false;
        auto* recorder = reinterpret_cast<HighPerformanceVideoRecorder*>(handle);
        return recorder->StartRecording();
    }
    
    bool StopVideoRecording(VideoRecorderHandle* handle) {
        if (!handle) return false;
        auto* recorder = reinterpret_cast<HighPerformanceVideoRecorder*>(handle);
        return recorder->StopRecording();
    }
    
    bool IsVideoRecording(VideoRecorderHandle* handle) {
        if (!handle) return false;
        auto* recorder = reinterpret_cast<HighPerformanceVideoRecorder*>(handle);
        return recorder->IsRecording();
    }
    
    bool CaptureVideoFrame(VideoRecorderHandle* handle) {
        if (!handle) return false;
        auto* recorder = reinterpret_cast<HighPerformanceVideoRecorder*>(handle);
        return recorder->CaptureFrame();
    }
    
    bool GetRecordingStats(VideoRecorderHandle* handle,
                          size_t* framesWritten, size_t* framesDropped,
                          double* duration, double* avgFps) {
        if (!handle) return false;
        
        auto* recorder = reinterpret_cast<HighPerformanceVideoRecorder*>(handle);
        auto stats = recorder->GetStats();
        
        if (framesWritten) *framesWritten = stats.framesWritten;
        if (framesDropped) *framesDropped = stats.framesDropped;
        if (duration) *duration = stats.recordingDuration;
        if (avgFps) *avgFps = stats.averageFps;
        
        return true;
    }
    
    const char* GetRecorderError(VideoRecorderHandle* handle) {
        if (!handle) return "Invalid handle";
        auto* recorder = reinterpret_cast<HighPerformanceVideoRecorder*>(handle);
        static std::string errorStr = recorder->GetLastError();
        return errorStr.c_str();
    }
    
    void CleanupRecorder(VideoRecorderHandle* handle) {
        if (!handle) return;
        auto* recorder = reinterpret_cast<HighPerformanceVideoRecorder*>(handle);
        recorder->Cleanup();
    }
}

} // namespace VideoRecorder 