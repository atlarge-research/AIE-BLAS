#pragma once

#include <source_location>

#include "aieblas/detail/util/cxx_compat.hpp"

// Use macro since we cannot combine source_location::current with variable
// arguments
#define log(...) aieblas::log_impl(std::source_location::current(), __VA_ARGS__)

namespace aieblas {
    enum class log_level : unsigned {
        unknown,
        debug,
        verbose,
        status,
        notice,
        warning,
        error
    };

    void set_log_level(log_level level);
    log_level get_log_level();
    std::string log_header(log_level level, const std::source_location& loc);

    template <typename... Args>
    inline void log_impl(const std::source_location& loc, log_level level,
                         std::format_string<Args...> fmt, Args &&...args) {
        if (level >= get_log_level()) {
            std::string message = std::format(std::runtime_format(fmt),
                                            std::forward<Args>(args)...);
            std::println("{}{}", log_header(level, loc), message);
        }
    }

    template <typename... Args>
    inline void log_impl(const std::source_location& loc,
                         std::format_string<Args...> fmt, Args &&...args) {
         log_impl(loc, log_level::status, std::runtime_format(fmt),
                  std::forward<Args>(args)...);
    }

    inline log_level log_level_from_str(std::string_view str) {
        if (str == "debug") {
            return log_level::debug;
        } else if (str == "verbose") {
            return log_level::verbose;
        } else if (str == "status") {
            return log_level::status;
        } else if (str == "notice") {
            return log_level::notice;
        } else if (str == "warning") {
            return log_level::warning;
        } else if (str == "error") {
            return log_level::error;
        } else {
            return log_level::unknown;
        }
    }

    constexpr inline const char *log_level_to_str(log_level level) {
        switch (level) {
        case log_level::debug:
            return "debug";
        case log_level::verbose:
            return "verbose";
        case log_level::status:
            return "status";
        case log_level::notice:
            return "notice";
        case log_level::warning:
            return "warning";
        case log_level::error:
            return "error";
        default:
            return "unknown";
        }
    }

} // namespace aieblas
