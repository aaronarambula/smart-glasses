#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>

struct gpiod_chip;
struct gpiod_line;
struct gpiod_line_request;

namespace audio {

struct HapticsConfig {
    bool enabled = false;
    int gpio_pin = -1;          // BCM numbering
    bool active_low = true;     // LOW = on for common transistor/buzzer boards
    int pulse_count = 2;
    int pulse_on_ms = 180;
    int pulse_off_ms = 120;
    bool verbose = false;
};

class HapticsEngine {
public:
    explicit HapticsEngine(HapticsConfig config = HapticsConfig{});
    ~HapticsEngine();

    HapticsEngine(const HapticsEngine&) = delete;
    HapticsEngine& operator=(const HapticsEngine&) = delete;
    HapticsEngine(HapticsEngine&&) = delete;
    HapticsEngine& operator=(HapticsEngine&&) = delete;

    bool start();
    void stop();

    bool is_running() const { return running_.load(); }
    bool is_enabled() const { return config_.enabled && config_.gpio_pin >= 0; }
    bool is_available() const { return available_.load(); }
    std::string error_message() const;

    void pulse_caution();

    HapticsConfig& config() { return config_; }
    const HapticsConfig& config() const { return config_; }

private:
    void worker_loop();
    bool open_gpio();
    void close_gpio();
    bool write_active(bool active);
    void set_error(const std::string& msg);

    HapticsConfig config_;

    std::atomic<bool> running_{ false };
    std::atomic<bool> stop_flag_{ false };
    std::atomic<bool> available_{ false };
    std::thread worker_;

    std::mutex mutex_;
    std::condition_variable cv_;
    bool pending_pulse_ = false;

    mutable std::mutex error_mutex_;
    std::string error_message_;

#ifdef __linux__
#if defined(HAVE_LIBGPIOD_V2)
    gpiod_chip* chip_ = nullptr;
    gpiod_line_request* request_ = nullptr;
#elif defined(HAVE_LIBGPIOD_V1)
    gpiod_chip* chip_ = nullptr;
    gpiod_line* line_ = nullptr;
#endif
#endif
};

} // namespace audio
