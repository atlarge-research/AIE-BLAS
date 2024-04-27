#include <chrono>

#include "aieblas/detail/util.hpp"
#include "aieblas/detail/util/logging.hpp"

namespace aieblas {

static log_level current_log_level = log_level::status;

void set_log_level(log_level level) {
    current_log_level = level;
}

log_level get_log_level() {
    return current_log_level;
}

std::string log_header(log_level level, const std::source_location& loc) {
    const auto now = std::chrono::system_clock::now();
    const std::time_t t_c = std::chrono::system_clock::to_time_t(now);
    const std::tm &tm = *std::localtime(&t_c);

    std::string message = std::format("[{:02}:{:02}] ", tm.tm_hour, tm.tm_min);
    if (get_log_level() <= log_level::debug) {
        std::error_code ec;
        std::string_view filename = loc.file_name();
        fs::path proximate = fs::proximate(filename, ec);
        if (!ec) {
            filename = proximate.c_str();
        }
        message.append(std::format("{}:{} in {}: ",
                                   filename, loc.line(), loc.function_name()));
    }

    return message;
}

} // namespace aieblas
