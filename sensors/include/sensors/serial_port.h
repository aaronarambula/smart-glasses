#pragma once

// ─── serial_port.h ───────────────────────────────────────────────────────────
// POSIX RAII wrapper around a serial (UART/USB-serial) file descriptor.
// Designed for Raspberry Pi Linux (works on any POSIX system).
//
// Handles:
//   - Opening /dev/ttyUSB0, /dev/ttyAMA0, /dev/serial0, etc.
//   - Baud rate configuration (B9600 → B4000000 via termios)
//   - 8N1 framing, raw (non-canonical) mode
//   - Blocking reads with configurable timeout
//   - Non-blocking reads
//   - Flushing TX/RX buffers
//   - RAII: destructor closes the fd automatically
//
// Usage:
//   SerialPort port("/dev/ttyUSB0", 115200);
//   if (!port.open()) { /* check port.error_message() */ }
//   port.write(cmd_bytes);
//   uint8_t buf[64];
//   int n = port.read(buf, sizeof(buf));

#include <string>
#include <vector>
#include <cstdint>
#include <cstring>
#include <cerrno>

// POSIX / Linux headers — only compiled on the Pi
#include <fcntl.h>
#include <unistd.h>
#include <termios.h>
#include <sys/ioctl.h>

namespace sensors {

// ─── SerialPort ──────────────────────────────────────────────────────────────

class SerialPort {
public:
    // ── Construction ──────────────────────────────────────────────────────────

    // port_path : e.g. "/dev/ttyUSB0" or "/dev/ttyAMA0"
    // baud_rate : numeric baud rate, e.g. 115200 or 230400
    SerialPort(std::string port_path, uint32_t baud_rate)
        : port_path_(std::move(port_path))
        , baud_rate_(baud_rate)
        , fd_(-1)
    {}

    // Destructor closes the port if still open.
    ~SerialPort() { close(); }

    // Non-copyable (owns an fd).
    SerialPort(const SerialPort&)            = delete;
    SerialPort& operator=(const SerialPort&) = delete;

    // Movable.
    SerialPort(SerialPort&& o) noexcept
        : port_path_(std::move(o.port_path_))
        , baud_rate_(o.baud_rate_)
        , fd_(o.fd_)
        , error_(std::move(o.error_))
    {
        o.fd_ = -1;
    }

    SerialPort& operator=(SerialPort&& o) noexcept {
        if (this != &o) {
            close();
            port_path_ = std::move(o.port_path_);
            baud_rate_ = o.baud_rate_;
            fd_        = o.fd_;
            error_     = std::move(o.error_);
            o.fd_      = -1;
        }
        return *this;
    }

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    // Opens the port and configures it.
    // Returns true on success; false on failure (check error_message()).
    bool open() {
        // O_RDWR  : read + write (some sensors need a command to start)
        // O_NOCTTY: don't let the port become the controlling terminal
        // O_NDELAY: don't block waiting for DCD signal on open
        fd_ = ::open(port_path_.c_str(), O_RDWR | O_NOCTTY | O_NDELAY);
        if (fd_ < 0) {
            set_errno_error("open()");
            return false;
        }

        // Restore blocking mode after open (O_NDELAY only affected open itself).
        if (::fcntl(fd_, F_SETFL, 0) < 0) {
            set_errno_error("fcntl(F_SETFL, 0)");
            ::close(fd_);
            fd_ = -1;
            return false;
        }

        if (!configure()) {
            ::close(fd_);
            fd_ = -1;
            return false;
        }

        return true;
    }

    // Closes the port.  Safe to call even if not open.
    void close() {
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
    }

    // ── Status ────────────────────────────────────────────────────────────────

    bool        is_open()       const { return fd_ >= 0; }
    int         fd()            const { return fd_; }
    std::string error_message() const { return error_; }
    const std::string& path()   const { return port_path_; }
    uint32_t    baud_rate()     const { return baud_rate_; }

    // ── I/O ───────────────────────────────────────────────────────────────────

    // Writes `len` bytes from `data`.
    // Returns the number of bytes written, or -1 on error.
    int write(const uint8_t* data, size_t len) {
        if (fd_ < 0) { error_ = "port not open"; return -1; }
        ssize_t n = ::write(fd_, data, len);
        if (n < 0) { set_errno_error("write()"); }
        return static_cast<int>(n);
    }

    // Convenience overload for a byte vector.
    int write(const std::vector<uint8_t>& data) {
        return write(data.data(), data.size());
    }

    // Blocking read: waits up to timeout_ms milliseconds for data.
    // Returns the number of bytes actually read (may be less than len),
    // 0 on timeout, or -1 on error.
    int read(uint8_t* buf, size_t len, int timeout_ms = 1000) {
        if (fd_ < 0) { error_ = "port not open"; return -1; }

        fd_set read_fds;
        FD_ZERO(&read_fds);
        FD_SET(fd_, &read_fds);

        struct timeval tv;
        tv.tv_sec  = timeout_ms / 1000;
        tv.tv_usec = (timeout_ms % 1000) * 1000;

        int ret = ::select(fd_ + 1, &read_fds, nullptr, nullptr, &tv);
        if (ret < 0) {
            set_errno_error("select()");
            return -1;
        }
        if (ret == 0) {
            // Timeout — not an error, just no data yet.
            return 0;
        }

        ssize_t n = ::read(fd_, buf, len);
        if (n < 0) {
            set_errno_error("read()");
            return -1;
        }
        return static_cast<int>(n);
    }

    // Reads exactly `len` bytes, blocking until all arrive or timeout expires.
    // Returns true if all bytes were read; false on timeout or error.
    bool read_exact(uint8_t* buf, size_t len, int timeout_ms = 2000) {
        size_t total = 0;
        const auto deadline = std::chrono::steady_clock::now()
                            + std::chrono::milliseconds(timeout_ms);
        while (total < len) {
            int remaining_ms = static_cast<int>(
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    deadline - std::chrono::steady_clock::now()).count());
            if (remaining_ms <= 0) {
                error_ = "read_exact() timeout";
                return false;
            }
            int n = read(buf + total, len - total, remaining_ms);
            if (n < 0) return false;   // error already set
            total += static_cast<size_t>(n);
        }
        return true;
    }

    // Returns the number of bytes waiting in the kernel receive buffer.
    int bytes_available() const {
        if (fd_ < 0) return 0;
        int bytes = 0;
        ::ioctl(fd_, FIONREAD, &bytes);
        return bytes;
    }

    // ── Buffer control ────────────────────────────────────────────────────────

    // Discard all data in the OS receive buffer (flush RX).
    void flush_input() {
        if (fd_ >= 0) ::tcflush(fd_, TCIFLUSH);
    }

    // Discard all data in the OS transmit buffer (flush TX).
    void flush_output() {
        if (fd_ >= 0) ::tcflush(fd_, TCOFLUSH);
    }

    // Flush both directions.
    void flush() {
        if (fd_ >= 0) ::tcflush(fd_, TCIOFLUSH);
    }

    // ── Baud rate change after open ───────────────────────────────────────────

    // Allows changing baud rate without closing and reopening.
    bool set_baud_rate(uint32_t baud) {
        baud_rate_ = baud;
        if (fd_ >= 0) return configure();
        return true;  // will be applied on next open()
    }

private:
    // ── Internals ─────────────────────────────────────────────────────────────

    std::string port_path_;
    uint32_t    baud_rate_;
    int         fd_;
    std::string error_;

    // Converts a numeric baud rate to the POSIX termios speed_t constant.
    // Returns B0 if the rate is not supported.
    static speed_t to_speed_t(uint32_t baud) {
        switch (baud) {
            case 9600:    return B9600;
            case 19200:   return B19200;
            case 38400:   return B38400;
            case 57600:   return B57600;
            case 115200:  return B115200;
            case 230400:  return B230400;
#ifdef B460800
            case 460800:  return B460800;
#endif
#ifdef B500000
            case 500000:  return B500000;
#endif
#ifdef B576000
            case 576000:  return B576000;
#endif
#ifdef B921600
            case 921600:  return B921600;
#endif
#ifdef B1000000
            case 1000000: return B1000000;
#endif
#ifdef B1152000
            case 1152000: return B1152000;
#endif
#ifdef B1500000
            case 1500000: return B1500000;
#endif
#ifdef B2000000
            case 2000000: return B2000000;
#endif
#ifdef B2500000
            case 2500000: return B2500000;
#endif
#ifdef B3000000
            case 3000000: return B3000000;
#endif
#ifdef B3500000
            case 3500000: return B3500000;
#endif
#ifdef B4000000
            case 4000000: return B4000000;
#endif
            default:      return B0;
        }
    }

    // Applies termios settings for 8N1 raw mode at baud_rate_.
    bool configure() {
        speed_t speed = to_speed_t(baud_rate_);
        if (speed == B0) {
            error_ = "Unsupported baud rate: " + std::to_string(baud_rate_);
            return false;
        }

        struct termios tty;
        std::memset(&tty, 0, sizeof(tty));

        if (::tcgetattr(fd_, &tty) != 0) {
            set_errno_error("tcgetattr()");
            return false;
        }

        // ── Input baud rate ───────────────────────────────────────────────────
        ::cfsetispeed(&tty, speed);

        // ── Output baud rate ──────────────────────────────────────────────────
        ::cfsetospeed(&tty, speed);

        // ── Raw mode (cfmakeraw equivalent, explicit for portability) ─────────
        // Disable input processing: no break signal, no CR→NL translation,
        // no parity checking, no 8th-bit stripping, no XON/XOFF.
        tty.c_iflag &= ~(IGNBRK | BRKINT | PARMRK | ISTRIP
                       | INLCR  | IGNCR  | ICRNL  | IXON);

        // Disable output processing (no NL→CR/NL, etc.).
        tty.c_oflag &= ~OPOST;

        // Disable canonical mode, echo, extended processing, signals.
        tty.c_lflag &= ~(ECHO | ECHONL | ICANON | ISIG | IEXTEN);

        // ── 8N1 character framing ─────────────────────────────────────────────
        tty.c_cflag &= ~(CSIZE | PARENB | CSTOPB);   // clear size, parity, 2-stop
        tty.c_cflag |=  CS8;                          // 8 data bits
        tty.c_cflag |=  CREAD | CLOCAL;               // enable receiver, ignore modem

        // ── Timing: return as soon as ≥1 byte arrives, no extra wait ──────────
        // VMIN=0, VTIME=1 → return after 0.1 s if no byte arrives (avoids
        // blocking the read thread indefinitely when the sensor is silent).
        tty.c_cc[VMIN]  = 0;
        tty.c_cc[VTIME] = 1;   // tenths of a second

        if (::tcsetattr(fd_, TCSANOW, &tty) != 0) {
            set_errno_error("tcsetattr()");
            return false;
        }

        return true;
    }

    // Formats errno into the error_ string.
    void set_errno_error(const char* context) {
        error_ = std::string(context) + ": " + std::strerror(errno);
    }
};

} // namespace sensors

// ─── chrono include (needed by read_exact) ────────────────────────────────────
#include <chrono>