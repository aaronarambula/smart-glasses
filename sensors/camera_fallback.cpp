#include "sensors/camera_fallback.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <sstream>
#include <thread>
#include <vector>

#ifdef HAVE_OPENCV
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#endif

namespace sensors {

namespace {
constexpr std::array<float, 5> kClusterOffsetsDeg{ -2.0f, -1.0f, 0.0f, 1.0f, 2.0f };
constexpr float kMinContourAreaPx = 600.0f;
constexpr int kMinBBoxHeightPx = 18;
constexpr float kMinDistanceMm = 250.0f;
constexpr float kMaxDistanceMm = 6000.0f;
constexpr float kMaxAngularWidthDeg = 28.0f;
constexpr int kCameraWarmupReads = 10;
constexpr int kMaxConsecutiveReadFailures = 5;
constexpr auto kCameraWarmupDelay = std::chrono::milliseconds(40);
constexpr auto kCameraRetryDelay = std::chrono::milliseconds(150);
} // namespace

struct CameraFallback::Impl {
#ifdef HAVE_OPENCV
    cv::VideoCapture cap;
    cv::Mat frame;
    cv::Mat gray;
    cv::Mat blurred;
    cv::Mat edges;
    cv::Mat morph;
#endif
};

#ifdef HAVE_OPENCV
bool CameraFallback::open_capture_device()
{
    impl_->cap.release();

    const int backends[] = { cv::CAP_V4L2, cv::CAP_ANY };
    for (const int backend : backends) {
        bool opened = false;
        if (backend == cv::CAP_ANY) {
            opened = impl_->cap.open(camera_index_);
        } else {
            opened = impl_->cap.open(camera_index_, backend);
        }
        if (!opened) continue;

        if (frame_width_ > 0)  impl_->cap.set(cv::CAP_PROP_FRAME_WIDTH, frame_width_);
        if (frame_height_ > 0) impl_->cap.set(cv::CAP_PROP_FRAME_HEIGHT, frame_height_);
        if (scan_hz_ > 0.0f)   impl_->cap.set(cv::CAP_PROP_FPS, scan_hz_);

        cv::Mat probe;
        for (int i = 0; i < kCameraWarmupReads; ++i) {
            if (impl_->cap.read(probe) && !probe.empty()) {
                impl_->frame = probe;
                return true;
            }
            std::this_thread::sleep_for(kCameraWarmupDelay);
        }

        impl_->cap.release();
    }

    return false;
}
#endif

CameraFallback::CameraFallback(std::string port_uri)
    : LidarBase(std::move(port_uri))
    , impl_(std::make_unique<Impl>())
{}

CameraFallback::CameraFallback(int camera_index,
                               int frame_width,
                               int frame_height,
                               float scan_hz,
                               float hfov_deg,
                               float focal_px,
                               float obstacle_height_mm)
    : LidarBase("camera://configured")
    , impl_(std::make_unique<Impl>())
{
    configure(camera_index, frame_width, frame_height, scan_hz, hfov_deg,
              focal_px, obstacle_height_mm);
}

CameraFallback::~CameraFallback()
{
    stop();
    close();
}

void CameraFallback::configure(int camera_index,
                               int frame_width,
                               int frame_height,
                               float scan_hz,
                               float hfov_deg,
                               float focal_px,
                               float obstacle_height_mm)
{
    camera_index_ = std::max(0, camera_index);
    frame_width_ = std::max(160, frame_width);
    frame_height_ = std::max(120, frame_height);
    scan_hz_ = std::max(1.0f, scan_hz);
    hfov_deg_ = std::clamp(hfov_deg, 20.0f, 140.0f);
    focal_px_ = std::max(50.0f, focal_px);
    obstacle_height_mm_ = std::max(200.0f, obstacle_height_mm);
    configured_ = true;
}

bool CameraFallback::parse_uri()
{
    static const std::string prefix = "camera://";
    if (port_.compare(0, prefix.size(), prefix) != 0) {
        set_error("CameraFallback expects port like camera://0?width=640&height=480&hz=10");
        return false;
    }

    std::string spec = port_.substr(prefix.size());
    std::string params;
    const size_t qpos = spec.find('?');
    if (qpos != std::string::npos) {
        params = spec.substr(qpos + 1);
        spec = spec.substr(0, qpos);
    }

    if (!spec.empty()) {
        camera_index_ = std::max(0, std::stoi(spec));
    }

    std::stringstream qs(params);
    std::string kv;
    while (std::getline(qs, kv, '&')) {
        if (kv.empty()) continue;
        const size_t eq = kv.find('=');
        const std::string key = kv.substr(0, eq);
        const std::string value = (eq == std::string::npos) ? "" : kv.substr(eq + 1);

        if (key == "width" && !value.empty()) {
            frame_width_ = std::max(160, std::stoi(value));
        } else if (key == "height" && !value.empty()) {
            frame_height_ = std::max(120, std::stoi(value));
        } else if (key == "hz" && !value.empty()) {
            scan_hz_ = std::max(1.0f, std::stof(value));
        } else if ((key == "hfov" || key == "hfov-deg") && !value.empty()) {
            hfov_deg_ = std::clamp(std::stof(value), 20.0f, 140.0f);
        } else if ((key == "focal-px" || key == "focal_px") && !value.empty()) {
            focal_px_ = std::max(50.0f, std::stof(value));
        } else if ((key == "obstacle-height-mm" || key == "obstacle_height_mm") && !value.empty()) {
            obstacle_height_mm_ = std::max(200.0f, std::stof(value));
        }
    }

    return true;
}

bool CameraFallback::open()
{
    if (open_.load()) return true;

#ifndef HAVE_OPENCV
    set_error("CameraFallback requires OpenCV. Rebuild after installing libopencv-dev");
    return false;
#else
    if (!configured_ && !parse_uri()) return false;

    if (!open_capture_device()) {
        set_error("Cannot read frames from camera index " + std::to_string(camera_index_));
        return false;
    }

    classifier_ = std::make_unique<CameraObjectClassifier>();
    try {
        classifier_->load_weights("aaronnet_camera_cls.bin");
    } catch (const std::exception&) {
        classifier_.reset();
    }

    open_.store(true);
    return true;
#endif
}

bool CameraFallback::start()
{
    if (!open_.load()) {
        set_error("start() called before open()");
        return false;
    }
    if (running_.load()) return true;

    running_.store(true);
    read_thread_ = std::thread(&CameraFallback::read_loop, this);
    return true;
}

void CameraFallback::stop()
{
    running_.store(false);
    if (read_thread_.joinable()) {
        read_thread_.join();
    }
}

void CameraFallback::close()
{
    stop();
#ifdef HAVE_OPENCV
    if (impl_ && impl_->cap.isOpened()) {
        impl_->cap.release();
    }
#endif
    open_.store(false);
}

ScanFrame CameraFallback::get_latest_frame() const
{
    std::lock_guard<std::mutex> lk(frame_mutex_);
    return latest_frame_;
}

void CameraFallback::set_frame_callback(FrameCallback cb)
{
    std::lock_guard<std::mutex> lk(frame_cb_mutex_);
    frame_callback_ = std::move(cb);
}

bool CameraFallback::is_open() const
{
    return open_.load();
}

bool CameraFallback::is_running() const
{
    return running_.load();
}

std::string CameraFallback::error_message() const
{
    std::lock_guard<std::mutex> lk(error_mutex_);
    return error_message_;
}

CameraClassification CameraFallback::last_classification() const
{
    std::lock_guard<std::mutex> lk(class_mutex_);
    return last_classification_;
}

void CameraFallback::set_error(const std::string& msg)
{
    std::lock_guard<std::mutex> lk(error_mutex_);
    error_message_ = msg;
}

void CameraFallback::publish_frame(ScanFrame frame)
{
    {
        std::lock_guard<std::mutex> lk(frame_mutex_);
        latest_frame_ = frame;
    }

    FrameCallback cb;
    {
        std::lock_guard<std::mutex> lk(frame_cb_mutex_);
        cb = frame_callback_;
    }
    if (cb) cb(frame);
}

ScanFrame CameraFallback::make_empty_frame() const
{
    ScanFrame frame;
    frame.timestamp = std::chrono::steady_clock::now();
    frame.frame_id = frame_counter_;
    frame.sensor_rpm = 0.0f;
    return frame;
}

ScanFrame CameraFallback::make_detection_frame(float distance_mm,
                                               float center_angle_deg,
                                               float angular_width_deg) const
{
    ScanFrame frame;
    frame.timestamp = std::chrono::steady_clock::now();
    frame.frame_id = frame_counter_;
    frame.sensor_rpm = 0.0f;

    const float clamped_width = std::clamp(angular_width_deg, 3.0f, kMaxAngularWidthDeg);
    for (size_t i = 0; i < kClusterOffsetsDeg.size(); ++i) {
        ScanPoint p;
        p.angle_deg = center_angle_deg + (kClusterOffsetsDeg[i] * (clamped_width / 6.0f));
        while (p.angle_deg < 0.0f) p.angle_deg += 360.0f;
        while (p.angle_deg >= 360.0f) p.angle_deg -= 360.0f;
        p.distance_mm = distance_mm;
        p.quality = 180;
        p.is_new_scan = (i == 0);
        frame.points.push_back(p);
    }
    return frame;
}

bool CameraFallback::detect_obstacle(ScanFrame& frame_out)
{
#ifndef HAVE_OPENCV
    (void)frame_out;
    return false;
#else
    if (!impl_->cap.read(impl_->frame) || impl_->frame.empty()) {
        set_error("Camera read failed");
        return false;
    }

    cv::cvtColor(impl_->frame, impl_->gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(impl_->gray, impl_->blurred, cv::Size(5, 5), 0.0);
    cv::Canny(impl_->blurred, impl_->edges, 20.0, 80.0);

    const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::dilate(impl_->edges, impl_->morph, kernel, cv::Point(-1, -1), 2);
    cv::morphologyEx(impl_->morph, impl_->morph, cv::MORPH_CLOSE, kernel,
                     cv::Point(-1, -1), 1);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(impl_->morph, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    const int width = impl_->frame.cols;
    const int height = impl_->frame.rows;
    const int center_left = width / 8;
    const int center_right = width - center_left;
    const int horizon_cut = height / 10;

    double best_score = 0.0;
    cv::Rect best_box;

    for (const auto& contour : contours) {
        const double area = cv::contourArea(contour);
        if (area < kMinContourAreaPx) continue;

        const cv::Rect box = cv::boundingRect(contour);
        if (box.height < kMinBBoxHeightPx) continue;
        if (box.y < horizon_cut) continue;

        const int box_center_x = box.x + box.width / 2;
        if (box_center_x < center_left || box_center_x > center_right) continue;

        const double solidity_score = area / static_cast<double>(std::max(1, box.area()));
        const double center_bias =
            1.0 - (std::abs(box_center_x - width / 2) / static_cast<double>(width / 2));
        const double height_bias =
            std::min(2.0, static_cast<double>(box.height) / static_cast<double>(kMinBBoxHeightPx));
        const double score =
            area * (0.35 + solidity_score) * (0.35 + center_bias) * height_bias;

        if (score > best_score) {
            best_score = score;
            best_box = box;
        }
    }

    if (best_score <= 0.0) {
        {
            std::lock_guard<std::mutex> lk(class_mutex_);
            last_classification_ = CameraClassification{};
        }
        frame_out = make_empty_frame();
        return true;
    }

    const float bbox_height_px = static_cast<float>(best_box.height);
    const float distance_mm =
        std::clamp((obstacle_height_mm_ * focal_px_) / bbox_height_px,
                   kMinDistanceMm, kMaxDistanceMm);

    const float bbox_center_x = static_cast<float>(best_box.x + best_box.width * 0.5f);
    const float pixel_offset = bbox_center_x - (static_cast<float>(width) * 0.5f);
    const float angle_per_pixel = hfov_deg_ / static_cast<float>(width);
    float center_angle = -pixel_offset * angle_per_pixel;
    if (center_angle < 0.0f) center_angle += 360.0f;

    const float angular_width =
        std::max(4.0f, static_cast<float>(best_box.width) * angle_per_pixel);

    {
        std::lock_guard<std::mutex> lk(class_mutex_);
        if (classifier_) {
            last_classification_ = classifier_->classify(impl_->frame(best_box).clone());
        } else {
            last_classification_ = CameraClassification{};
        }
    }

    frame_out = make_detection_frame(distance_mm, center_angle, angular_width);
    return true;
#endif
}

void CameraFallback::read_loop()
{
    const auto sleep_for = std::chrono::duration<float>(1.0f / std::max(1.0f, scan_hz_));
    int consecutive_failures = 0;

    while (running_.load()) {
        ScanFrame frame;
        if (!detect_obstacle(frame)) {
            ++consecutive_failures;
            if (consecutive_failures >= kMaxConsecutiveReadFailures) {
#ifdef HAVE_OPENCV
                if (open_.load()) {
                    open_capture_device();
                }
#endif
                consecutive_failures = 0;
                std::this_thread::sleep_for(kCameraRetryDelay);
            }
            frame = make_empty_frame();
        } else {
            consecutive_failures = 0;
        }

        frame.frame_id = frame_counter_++;
        publish_frame(std::move(frame));
        std::this_thread::sleep_for(sleep_for);
    }
}

} // namespace sensors
