#pragma once

#include "aieblas/detail/util/cxx_compat.hpp"
#include "aieblas/detail/util/logging.hpp"
#include <filesystem>
namespace fs = std::filesystem;

namespace util {
    inline void create_dir(const fs::path &directory) {
        log(aieblas::log_level::verbose, "Creating directory '{}'",
            directory.native());
        fs::create_directory(directory);
    }
}
