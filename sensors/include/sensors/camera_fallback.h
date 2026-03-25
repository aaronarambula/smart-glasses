#pragma once

#include "lidar_base.h"
#include "camera_classifier.h"

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace sensors {

// Front-camera fallback that converts a monocular OpenCV stream into a narrow
// synthetic scan aimed at the most prominent obstacle in front of the user.
//
// This is intentionally pragmatic:
//   - no pipeline rewrite
//   - emits ScanFrame so clustering/tracking/TTC/audio continue to work
//   - uses simple contour-based obstacle detection with pinhole distance
//     estimation from the apparent obstacle height in pixels
class CameraFallback : public LidarBase {
public:
    explicit CameraFallback(std::string port_uri);
    CameraFallback(int camera_index,
                   int frame_width,
                   int frame_height,
                   float scan_hz,
                   float hfov_deg,
                   float focal_px,
                   float obstacle_height_mm);
    ~CameraFallback() override;

    bool open() override;
    bool start() override;
    void stop() override;
    void close() override;

    ScanFrame get_latest_frame() const override;
    void set_frame_callback(FrameCallback cb) override;

    bool        is_open() const override;
    bool        is_running() const override;
    std::string error_message() const override;
    std::string model_name() const override { return "CameraFallback"; }
    CameraClassification last_classification() const;

    void configure(int camera_index,
                   int frame_width,
                   int frame_height,
                   float scan_hz,
                   float hfov_deg,
                   float focal_px,
                   float obstacle_height_mm);

private:
    struct Impl;

    bool parse_uri();
    void read_loop();
    void publish_frame(ScanFrame frame);
    void set_error(const std::string& msg);

    ScanFrame make_empty_frame() const;
    ScanFrame make_detection_frame(float distance_mm,
                                   float center_angle_deg,
                                   float angular_width_deg) const;
    bool detect_obstacle(ScanFrame& frame_out);

    int   camera_index_        = 0;
    int   frame_width_         = 640;
    int   frame_height_        = 480;
    float scan_hz_             = 10.0f;
    float hfov_deg_            = 62.0f;
    float focal_px_            = 700.0f;
    float obstacle_height_mm_  = 1700.0f;
    bool  configured_          = false;

    std::atomic<bool> open_{ false };
    std::atomic<bool> running_{ false };
    std::thread       read_thread_;

    mutable std::mutex frame_mutex_;
    ScanFrame          latest_frame_;
    uint64_t           frame_counter_ = 0;

    mutable std::mutex frame_cb_mutex_;
    FrameCallback      frame_callback_;

    mutable std::mutex error_mutex_;
    std::string        error_message_;

    mutable std::mutex class_mutex_;
    CameraClassification last_classification_;
    std::unique_ptr<CameraObjectClassifier> classifier_;

    std::unique_ptr<Impl> impl_;
};

} // namespace sensors
