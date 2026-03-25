#pragma once

#include "lidar_base.h"

#include <atomic>
#include <deque>
#include <mutex>
#include <string>
#include <thread>

struct gpiod_chip;
struct gpiod_line;
struct gpiod_line_request;

namespace sensors {

// Single-beam HC-SR04-style ultrasonic sensor wrapped in the existing
// LidarBase interface by synthesising a narrow forward cluster.
class UltrasonicFallback : public LidarBase {
public:
    explicit UltrasonicFallback(std::string port_uri);
    UltrasonicFallback(int trigger_pin, int echo_pin, float scan_hz, float mock_distance_mm);
    ~UltrasonicFallback() override;

    bool open() override;
    bool start() override;
    void stop() override;
    void close() override;

    ScanFrame get_latest_frame() const override;
    void set_frame_callback(FrameCallback cb) override;

    bool        is_open() const override;
    bool        is_running() const override;
    std::string error_message() const override;
    std::string model_name() const override { return "UltrasonicFallback"; }

    int   trigger_pin() const { return trigger_pin_; }
    int   echo_pin()    const { return echo_pin_; }
    float scan_hz()     const { return scan_hz_; }
    float mock_distance_mm() const { return mock_distance_mm_; }
    void configure(int trigger_pin, int echo_pin, float scan_hz, float mock_distance_mm);

private:
    bool parse_uri();
    void read_loop();
    void publish_frame(ScanFrame frame);
    void set_error(const std::string& msg);

    ScanFrame make_frame(float distance_mm) const;
    float read_distance_mm();

#ifdef __linux__
#if defined(HAVE_LIBGPIOD_V2)
    bool open_gpiod();
    bool wait_for_gpio_value(bool target,
                             std::chrono::microseconds timeout) const;
#elif defined(HAVE_LIBGPIOD_V1)
    bool open_gpiod();
    bool wait_for_gpio_value(bool target,
                             std::chrono::microseconds timeout) const;
#else
    bool ensure_gpio_pin(int pin, const char* direction, bool* exported_flag);
    bool write_gpio_value(int pin, bool high);
    bool read_gpio_value(int pin, bool& high) const;
    bool wait_for_gpio_value(int pin, bool target,
                             std::chrono::microseconds timeout) const;
#endif
#endif

    int   trigger_pin_      = 23;
    int   echo_pin_         = 24;
    float scan_hz_          = 10.0f;
    float max_distance_mm_  = 4000.0f;
    float mock_distance_mm_ = 0.0f;
    bool  mock_mode_        = false;
    bool  configured_       = false;

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

    mutable std::mutex filter_mutex_;
    std::deque<float>  recent_distances_mm_;
    float              last_good_distance_mm_ = 0.0f;
    std::chrono::steady_clock::time_point last_good_time_{};

#ifdef __linux__
#if defined(HAVE_LIBGPIOD_V2)
    gpiod_chip* chip_ = nullptr;
    gpiod_line_request* trigger_request_ = nullptr;
    gpiod_line_request* echo_request_ = nullptr;
#elif defined(HAVE_LIBGPIOD_V1)
    gpiod_chip* chip_ = nullptr;
    gpiod_line* trigger_line_ = nullptr;
    gpiod_line* echo_line_ = nullptr;
#else
    bool trigger_exported_ = false;
    bool echo_exported_    = false;
#endif
#endif
};

} // namespace sensors
