#include "sensors/ultrasonic_fallback.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <thread>
#include <vector>

#ifdef __linux__
#ifdef HAVE_LIBGPIOD
#include <gpiod.h>
#else
#include <unistd.h>
#endif
#endif

namespace sensors {

namespace {
constexpr std::array<float, 5> kSyntheticAnglesDeg{ 356.0f, 358.0f, 0.0f, 2.0f, 4.0f };
constexpr size_t kMedianWindow = 5;
constexpr auto kHoldLastGoodFor = std::chrono::milliseconds(500);
#ifdef __linux__
constexpr float kSpeedOfSoundMmPerUs = 0.343f;

std::string gpio_path(int pin, const char* leaf)
{
    std::ostringstream ss;
    ss << "/sys/class/gpio/gpio" << pin << "/" << leaf;
    return ss.str();
}

std::string trim_copy(std::string s)
{
    s.erase(std::remove_if(s.begin(), s.end(), [](unsigned char c) {
        return c == '\n' || c == '\r' || c == ' ' || c == '\t';
    }), s.end());
    return s;
}
#endif
} // namespace

UltrasonicFallback::UltrasonicFallback(std::string port_uri)
    : LidarBase(std::move(port_uri))
{}

UltrasonicFallback::~UltrasonicFallback()
{
    stop();
    close();
}

bool UltrasonicFallback::parse_uri()
{
    if (port_.rfind("ultrasonic://", 0) != 0) {
        set_error("UltrasonicFallback expects port like ultrasonic://23,24?hz=10 or ultrasonic://mock?mm=1200");
        return false;
    }

    std::string spec = port_.substr(std::string("ultrasonic://").size());
    std::string params;
    const size_t qpos = spec.find('?');
    if (qpos != std::string::npos) {
        params = spec.substr(qpos + 1);
        spec = spec.substr(0, qpos);
    }

    if (spec == "mock") {
        if (mock_distance_mm_ <= 0.0f) {
            mock_distance_mm_ = 1200.0f;
        }
    } else if (!spec.empty()) {
        const size_t comma = spec.find(',');
        if (comma == std::string::npos) {
            set_error("UltrasonicFallback URI must include trigger and echo pins");
            return false;
        }
        trigger_pin_ = std::stoi(spec.substr(0, comma));
        echo_pin_ = std::stoi(spec.substr(comma + 1));
    }

    std::stringstream qs(params);
    std::string kv;
    while (std::getline(qs, kv, '&')) {
        if (kv.empty()) continue;
        const size_t eq = kv.find('=');
        const std::string key = kv.substr(0, eq);
        const std::string value = (eq == std::string::npos) ? "" : kv.substr(eq + 1);

        if (key == "hz" && !value.empty()) {
            scan_hz_ = std::max(1.0f, std::stof(value));
        } else if ((key == "max-mm" || key == "max_mm") && !value.empty()) {
            max_distance_mm_ = std::clamp(std::stof(value), 200.0f, 6000.0f);
        } else if (key == "mm" && !value.empty()) {
            mock_distance_mm_ = std::max(0.0f, std::stof(value));
        }
    }

    return true;
}

bool UltrasonicFallback::open()
{
    if (open_.load()) return true;
    if (!parse_uri()) return false;

#ifndef __linux__
    if (mock_distance_mm_ <= 0.0f) {
        set_error("UltrasonicFallback GPIO mode currently requires Linux; use ultrasonic://mock?mm=1200 for desktop testing");
        return false;
    }
    open_.store(true);
    return true;
#else
    if (mock_distance_mm_ > 0.0f) {
        open_.store(true);
        return true;
    }
#ifdef HAVE_LIBGPIOD
    if (!open_gpiod()) return false;
#else
    if (!ensure_gpio_pin(trigger_pin_, "out", &trigger_exported_)) return false;
    if (!ensure_gpio_pin(echo_pin_, "in", &echo_exported_)) return false;
    if (!write_gpio_value(trigger_pin_, false)) return false;
#endif

    open_.store(true);
    return true;
#endif
}

bool UltrasonicFallback::start()
{
    if (!open_.load()) {
        set_error("start() called before open()");
        return false;
    }
    if (running_.load()) return true;

    running_.store(true);
    read_thread_ = std::thread(&UltrasonicFallback::read_loop, this);
    return true;
}

void UltrasonicFallback::stop()
{
    running_.store(false);
    if (read_thread_.joinable()) {
        read_thread_.join();
    }
}

void UltrasonicFallback::close()
{
    stop();
#ifdef __linux__
#ifdef HAVE_LIBGPIOD
    if (echo_line_) {
        gpiod_line_release(echo_line_);
        echo_line_ = nullptr;
    }
    if (trigger_line_) {
        gpiod_line_release(trigger_line_);
        trigger_line_ = nullptr;
    }
    if (chip_) {
        gpiod_chip_close(chip_);
        chip_ = nullptr;
    }
#else
    if (trigger_exported_) {
        std::ofstream unexport_file("/sys/class/gpio/unexport");
        if (unexport_file) unexport_file << trigger_pin_;
        trigger_exported_ = false;
    }
    if (echo_exported_) {
        std::ofstream unexport_file("/sys/class/gpio/unexport");
        if (unexport_file) unexport_file << echo_pin_;
        echo_exported_ = false;
    }
#endif
#endif
    open_.store(false);
}

ScanFrame UltrasonicFallback::get_latest_frame() const
{
    std::lock_guard<std::mutex> lock(frame_mutex_);
    return latest_frame_;
}

void UltrasonicFallback::set_frame_callback(FrameCallback cb)
{
    std::lock_guard<std::mutex> lock(frame_cb_mutex_);
    frame_callback_ = std::move(cb);
}

bool UltrasonicFallback::is_open() const
{
    return open_.load();
}

bool UltrasonicFallback::is_running() const
{
    return running_.load();
}

std::string UltrasonicFallback::error_message() const
{
    std::lock_guard<std::mutex> lock(error_mutex_);
    return error_message_;
}

void UltrasonicFallback::set_error(const std::string& msg)
{
    std::lock_guard<std::mutex> lock(error_mutex_);
    error_message_ = msg;
}

ScanFrame UltrasonicFallback::make_frame(float distance_mm) const
{
    ScanFrame frame;
    frame.timestamp = std::chrono::steady_clock::now();
    frame.sensor_rpm = scan_hz_ * 60.0f;

    if (!(distance_mm > 0.0f) || distance_mm > max_distance_mm_) {
        return frame;
    }

    frame.points.reserve(kSyntheticAnglesDeg.size());
    for (size_t i = 0; i < kSyntheticAnglesDeg.size(); ++i) {
        ScanPoint p;
        p.angle_deg = kSyntheticAnglesDeg[i];
        p.distance_mm = distance_mm;
        p.quality = 255;
        p.is_new_scan = (i == 0);
        frame.points.push_back(p);
    }

    return frame;
}

float UltrasonicFallback::read_distance_mm()
{
    float raw_distance_mm = 0.0f;
    if (mock_distance_mm_ > 0.0f) {
        raw_distance_mm = mock_distance_mm_;
    } else {

#ifndef __linux__
        raw_distance_mm = 0.0f;
#else
#ifdef HAVE_LIBGPIOD
        if (gpiod_line_set_value(trigger_line_, 0) != 0) return 0.0f;
        std::this_thread::sleep_for(std::chrono::microseconds(2));
        if (gpiod_line_set_value(trigger_line_, 1) != 0) return 0.0f;
        std::this_thread::sleep_for(std::chrono::microseconds(10));
        if (gpiod_line_set_value(trigger_line_, 0) != 0) return 0.0f;

        if (!wait_for_gpio_value(true, std::chrono::milliseconds(30))) {
            raw_distance_mm = 0.0f;
        } else {
            const auto pulse_start = std::chrono::steady_clock::now();
            if (!wait_for_gpio_value(false, std::chrono::milliseconds(30))) {
                raw_distance_mm = 0.0f;
            } else {
                const auto pulse_end = std::chrono::steady_clock::now();
                const auto pulse_us =
                    std::chrono::duration_cast<std::chrono::microseconds>(pulse_end - pulse_start).count();
                raw_distance_mm = static_cast<float>(pulse_us) * (kSpeedOfSoundMmPerUs * 0.5f);
                raw_distance_mm = std::clamp(raw_distance_mm, 0.0f, max_distance_mm_);
            }
        }
#else
        if (!write_gpio_value(trigger_pin_, false)) return 0.0f;
        std::this_thread::sleep_for(std::chrono::microseconds(2));
        if (!write_gpio_value(trigger_pin_, true)) return 0.0f;
        std::this_thread::sleep_for(std::chrono::microseconds(10));
        if (!write_gpio_value(trigger_pin_, false)) return 0.0f;

        if (!wait_for_gpio_value(echo_pin_, true, std::chrono::milliseconds(30))) {
            raw_distance_mm = 0.0f;
        } else {
            const auto pulse_start = std::chrono::steady_clock::now();

            if (!wait_for_gpio_value(echo_pin_, false, std::chrono::milliseconds(30))) {
                raw_distance_mm = 0.0f;
            } else {
                const auto pulse_end = std::chrono::steady_clock::now();
                const auto pulse_us =
                    std::chrono::duration_cast<std::chrono::microseconds>(pulse_end - pulse_start).count();
                raw_distance_mm = static_cast<float>(pulse_us) * (kSpeedOfSoundMmPerUs * 0.5f);
                raw_distance_mm = std::clamp(raw_distance_mm, 0.0f, max_distance_mm_);
            }
        }
#endif
#endif
    }

    std::lock_guard<std::mutex> lock(filter_mutex_);
    const auto now = std::chrono::steady_clock::now();

    if (raw_distance_mm > 0.0f && raw_distance_mm <= max_distance_mm_) {
        recent_distances_mm_.push_back(raw_distance_mm);
        while (recent_distances_mm_.size() > kMedianWindow) {
            recent_distances_mm_.pop_front();
        }

        std::vector<float> sorted(recent_distances_mm_.begin(), recent_distances_mm_.end());
        std::sort(sorted.begin(), sorted.end());
        const float median = sorted[sorted.size() / 2];

        last_good_distance_mm_ = median;
        last_good_time_ = now;
        return median;
    }

    if (last_good_distance_mm_ > 0.0f &&
        (now - last_good_time_) <= kHoldLastGoodFor) {
        return last_good_distance_mm_;
    }

    recent_distances_mm_.clear();
    last_good_distance_mm_ = 0.0f;
    return 0.0f;
}

void UltrasonicFallback::publish_frame(ScanFrame frame)
{
    frame.frame_id = frame_counter_++;

    FrameCallback cb;
    {
        std::lock_guard<std::mutex> lock(frame_mutex_);
        latest_frame_ = frame;
    }
    {
        std::lock_guard<std::mutex> lock(frame_cb_mutex_);
        cb = frame_callback_;
    }
    if (cb) cb(frame);
}

void UltrasonicFallback::read_loop()
{
    const auto period = std::chrono::duration<double>(1.0 / std::max(1.0f, scan_hz_));

    while (running_.load()) {
        const auto loop_start = std::chrono::steady_clock::now();

        ScanFrame frame = make_frame(read_distance_mm());
        publish_frame(std::move(frame));

        const auto elapsed = std::chrono::steady_clock::now() - loop_start;
        const auto sleep_for = period - elapsed;
        if (sleep_for > std::chrono::duration<double>::zero()) {
            std::this_thread::sleep_for(sleep_for);
        }
    }
}

#ifdef __linux__
#ifdef HAVE_LIBGPIOD
bool UltrasonicFallback::open_gpiod()
{
    static constexpr const char* kChipNames[] = {
        "gpiochip0", "gpiochip1", "gpiochip2", "gpiochip3", "gpiochip4", "gpiochip5"
    };

    for (const char* chip_name : kChipNames) {
        gpiod_chip* chip = gpiod_chip_open_by_name(chip_name);
        if (!chip) continue;

        gpiod_line* trig = gpiod_chip_get_line(chip, trigger_pin_);
        gpiod_line* echo = gpiod_chip_get_line(chip, echo_pin_);
        if (!trig || !echo) {
            if (chip) gpiod_chip_close(chip);
            continue;
        }

        if (gpiod_line_request_output(trig, "smart_glasses_ultrasonic", 0) != 0) {
            gpiod_chip_close(chip);
            continue;
        }
        if (gpiod_line_request_input(echo, "smart_glasses_ultrasonic") != 0) {
            gpiod_line_release(trig);
            gpiod_chip_close(chip);
            continue;
        }

        chip_ = chip;
        trigger_line_ = trig;
        echo_line_ = echo;
        return true;
    }

    set_error("Cannot request GPIO lines via libgpiod; verify GPIO23/GPIO24 and install libgpiod-dev");
    return false;
}

bool UltrasonicFallback::wait_for_gpio_value(
    bool target,
    std::chrono::microseconds timeout) const
{
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
        const int value = gpiod_line_get_value(echo_line_);
        if (value < 0) return false;
        if ((value != 0) == target) return true;
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    return false;
}
#else
bool UltrasonicFallback::ensure_gpio_pin(int pin, const char* direction, bool* exported_flag)
{
    const std::string dir_path = gpio_path(pin, "direction");
    {
        std::ifstream existing(dir_path);
        if (!existing.good()) {
            std::ofstream export_file("/sys/class/gpio/export");
            if (!export_file) {
                set_error("Cannot open /sys/class/gpio/export; check GPIO permissions");
                return false;
            }
            export_file << pin;
            if (!export_file) {
                set_error("Failed to export GPIO pin " + std::to_string(pin));
                return false;
            }
            *exported_flag = true;
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    }

    std::ofstream direction_file(dir_path);
    if (!direction_file) {
        set_error("Cannot configure GPIO" + std::to_string(pin) + " direction");
        return false;
    }
    direction_file << direction;
    if (!direction_file) {
        set_error("Failed to set GPIO" + std::to_string(pin) + " direction to " + direction);
        return false;
    }
    return true;
}

bool UltrasonicFallback::write_gpio_value(int pin, bool high)
{
    std::ofstream value_file(gpio_path(pin, "value"));
    if (!value_file) {
        set_error("Cannot open GPIO" + std::to_string(pin) + " value for write");
        return false;
    }
    value_file << (high ? "1" : "0");
    return static_cast<bool>(value_file);
}

bool UltrasonicFallback::read_gpio_value(int pin, bool& high) const
{
    std::ifstream value_file(gpio_path(pin, "value"));
    if (!value_file) return false;

    std::string s;
    value_file >> s;
    s = trim_copy(s);
    high = (s == "1");
    return true;
}

bool UltrasonicFallback::wait_for_gpio_value(
    int pin,
    bool target,
    std::chrono::microseconds timeout) const
{
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    bool state = false;
    while (std::chrono::steady_clock::now() < deadline) {
        if (!read_gpio_value(pin, state)) return false;
        if (state == target) return true;
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    return false;
}
#endif
#endif

} // namespace sensors
