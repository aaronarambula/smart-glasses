#include "audio/haptics_engine.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <thread>

#ifdef __linux__
#if defined(HAVE_LIBGPIOD_V2) || defined(HAVE_LIBGPIOD_V1)
#include <gpiod.h>
#endif
#endif

namespace audio {

HapticsEngine::HapticsEngine(HapticsConfig config)
    : config_(std::move(config))
{}

HapticsEngine::~HapticsEngine()
{
    stop();
}

bool HapticsEngine::start()
{
    if (running_.load()) return true;
    if (!is_enabled()) return true;
    if (!open_gpio()) return false;

    stop_flag_.store(false);
    running_.store(true);
    worker_ = std::thread(&HapticsEngine::worker_loop, this);
    return true;
}

void HapticsEngine::stop()
{
    stop_flag_.store(true);
    cv_.notify_all();
    if (worker_.joinable()) {
        worker_.join();
    }
    running_.store(false);
    close_gpio();
}

std::string HapticsEngine::error_message() const
{
    std::lock_guard<std::mutex> lock(error_mutex_);
    return error_message_;
}

void HapticsEngine::pulse_caution()
{
    if (!is_enabled() || !available_.load()) return;

    {
        std::lock_guard<std::mutex> lock(mutex_);
        pending_pulse_ = true;
    }
    cv_.notify_one();
}

void HapticsEngine::worker_loop()
{
    while (!stop_flag_.load()) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&] { return stop_flag_.load() || pending_pulse_; });
        if (stop_flag_.load()) break;
        pending_pulse_ = false;
        lock.unlock();

        for (int i = 0; i < std::max(1, config_.pulse_count); ++i) {
            if (stop_flag_.load()) break;
            if (!write_active(true)) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(
                std::max(1, config_.pulse_on_ms)));
            if (!write_active(false)) break;
            if (i + 1 < std::max(1, config_.pulse_count)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(
                    std::max(1, config_.pulse_off_ms)));
            }
        }
    }

    write_active(false);
}

bool HapticsEngine::open_gpio()
{
    available_.store(false);
    set_error("");

#if !defined(__linux__)
    set_error("Haptics GPIO currently requires Linux/libgpiod");
    return false;
#elif defined(HAVE_LIBGPIOD_V2)
    static constexpr const char* kChipPaths[] = {
        "/dev/gpiochip0", "/dev/gpiochip1", "/dev/gpiochip2",
        "/dev/gpiochip3", "/dev/gpiochip4", "/dev/gpiochip5"
    };

    for (const char* chip_path : kChipPaths) {
        gpiod_chip* chip = gpiod_chip_open(chip_path);
        if (!chip) continue;

        gpiod_line_settings* settings = gpiod_line_settings_new();
        gpiod_line_config* line_cfg = gpiod_line_config_new();
        gpiod_request_config* req_cfg = gpiod_request_config_new();
        if (!settings || !line_cfg || !req_cfg) {
            if (settings) gpiod_line_settings_free(settings);
            if (line_cfg) gpiod_line_config_free(line_cfg);
            if (req_cfg) gpiod_request_config_free(req_cfg);
            gpiod_chip_close(chip);
            continue;
        }

        gpiod_line_settings_set_direction(settings, GPIOD_LINE_DIRECTION_OUTPUT);
        gpiod_line_settings_set_output_value(settings, GPIOD_LINE_VALUE_INACTIVE);

        const unsigned int offset = static_cast<unsigned int>(config_.gpio_pin);
        if (gpiod_line_config_add_line_settings(line_cfg, &offset, 1, settings) != 0) {
            gpiod_line_settings_free(settings);
            gpiod_line_config_free(line_cfg);
            gpiod_request_config_free(req_cfg);
            gpiod_chip_close(chip);
            continue;
        }

        gpiod_request_config_set_consumer(req_cfg, "smart_glasses_haptics");
        gpiod_line_request* req = gpiod_chip_request_lines(chip, req_cfg, line_cfg);

        gpiod_line_settings_free(settings);
        gpiod_line_config_free(line_cfg);
        gpiod_request_config_free(req_cfg);

        if (!req) {
            gpiod_chip_close(chip);
            continue;
        }

        chip_ = chip;
        request_ = req;
        if (!write_active(false)) {
            close_gpio();
            return false;
        }
        available_.store(true);
        if (config_.verbose) {
            std::cout << "  ✓ Haptics   : GPIO" << config_.gpio_pin
                      << " (" << (config_.active_low ? "active-low" : "active-high")
                      << ")\n";
        }
        return true;
    }
#elif defined(HAVE_LIBGPIOD_V1)
    chip_ = gpiod_chip_open_by_name("gpiochip0");
    if (!chip_) {
        set_error("Failed to open gpiochip0 for haptics");
        return false;
    }

    line_ = gpiod_chip_get_line(chip_, config_.gpio_pin);
    if (!line_) {
        set_error("Failed to get haptics GPIO line");
        close_gpio();
        return false;
    }

    if (gpiod_line_request_output(line_, "smart_glasses_haptics", 0) != 0) {
        set_error("Failed to request haptics GPIO output line");
        close_gpio();
        return false;
    }

    if (!write_active(false)) {
        close_gpio();
        return false;
    }
#else
    set_error("Haptics GPIO requires libgpiod on this platform");
    return false;
#endif

    set_error("Failed to request haptics GPIO line");
    return false;
}

void HapticsEngine::close_gpio()
{
#ifdef __linux__
#if defined(HAVE_LIBGPIOD_V2)
    if (request_) {
        gpiod_line_request_release(request_);
        request_ = nullptr;
    }
    if (chip_) {
        gpiod_chip_close(chip_);
        chip_ = nullptr;
    }
#elif defined(HAVE_LIBGPIOD_V1)
    if (line_) {
        gpiod_line_release(line_);
        line_ = nullptr;
    }
    if (chip_) {
        gpiod_chip_close(chip_);
        chip_ = nullptr;
    }
#endif
#endif
    available_.store(false);
}

bool HapticsEngine::write_active(bool active)
{
    if (!is_enabled()) return true;

    const int value = active
        ? (config_.active_low ? 0 : 1)
        : (config_.active_low ? 1 : 0);

#if !defined(__linux__)
    (void)value;
    return false;
#elif defined(HAVE_LIBGPIOD_V2)
    if (!request_) return false;
    const unsigned int offset = static_cast<unsigned int>(config_.gpio_pin);
    gpiod_line_value line_value =
        (value != 0) ? GPIOD_LINE_VALUE_ACTIVE : GPIOD_LINE_VALUE_INACTIVE;
    if (gpiod_line_request_set_value(request_, offset, line_value) != 0) {
        set_error("Failed to write haptics GPIO value");
        return false;
    }
    return true;
#elif defined(HAVE_LIBGPIOD_V1)
    if (!line_) return false;
    if (gpiod_line_set_value(line_, value) != 0) {
        set_error("Failed to write haptics GPIO value");
        return false;
    }
    return true;
#else
    (void)value;
    return false;
#endif
}

void HapticsEngine::set_error(const std::string& msg)
{
    std::lock_guard<std::mutex> lock(error_mutex_);
    error_message_ = msg;
}

} // namespace audio
