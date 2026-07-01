#pragma once

#include <cstdarg>
#include <cstdio>
#include <ctime>

namespace radixforge {

enum class LogLevel : int { DEBUG = 0, INFO = 1, WARN = 2, ERROR = 3 };

// Global log level — change at runtime before starting the server.
inline LogLevel g_log_level = LogLevel::INFO;

inline void set_log_level(LogLevel level) { g_log_level = level; }

inline void log_impl(LogLevel level, const char* tag, const char* fmt, va_list args) {
    if (level < g_log_level) return;

    const char* lvl_str = "INFO ";
    switch (level) {
        case LogLevel::DEBUG: lvl_str = "DEBUG"; break;
        case LogLevel::WARN:  lvl_str = "WARN "; break;
        case LogLevel::ERROR: lvl_str = "ERROR"; break;
        default: break;
    }

    time_t now = time(nullptr);
    struct tm tm_buf;
    localtime_r(&now, &tm_buf);
    fprintf(stderr, "%02d:%02d:%02d [%s] [%s] ",
            tm_buf.tm_hour, tm_buf.tm_min, tm_buf.tm_sec, lvl_str, tag);
    vfprintf(stderr, fmt, args);
    fputc('\n', stderr);
}

inline void log_info(const char* tag, const char* fmt, ...) {
    va_list a; va_start(a, fmt); log_impl(LogLevel::INFO,  tag, fmt, a); va_end(a);
}
inline void log_warn(const char* tag, const char* fmt, ...) {
    va_list a; va_start(a, fmt); log_impl(LogLevel::WARN,  tag, fmt, a); va_end(a);
}
inline void log_error(const char* tag, const char* fmt, ...) {
    va_list a; va_start(a, fmt); log_impl(LogLevel::ERROR, tag, fmt, a); va_end(a);
}
inline void log_debug(const char* tag, const char* fmt, ...) {
    va_list a; va_start(a, fmt); log_impl(LogLevel::DEBUG, tag, fmt, a); va_end(a);
}

} // namespace radixforge

#define RF_INFO(tag, ...)  ::radixforge::log_info(tag,  __VA_ARGS__)
#define RF_WARN(tag, ...)  ::radixforge::log_warn(tag,  __VA_ARGS__)
#define RF_ERROR(tag, ...) ::radixforge::log_error(tag, __VA_ARGS__)
#define RF_DEBUG(tag, ...) ::radixforge::log_debug(tag, __VA_ARGS__)
