#pragma once

#include <stdexcept>
#include <string>

namespace aieblas {
namespace codegen {

struct kernel_parameter {
    std::string kernel;
    std::string parameter;

    inline bool
    operator==(const aieblas::codegen::kernel_parameter &rhs) const noexcept {
        return this->kernel == rhs.kernel && this->parameter == rhs.parameter;
    }

    inline bool
    operator==(const aieblas::codegen::kernel_parameter &&rhs) const noexcept {
        return *this == rhs;
    }
};

class parse_error : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

} // namespace codegen
} // namespace aieblas

template <> struct std::hash<aieblas::codegen::kernel_parameter> {
    inline std::size_t
    operator()(const aieblas::codegen::kernel_parameter &s) const noexcept {
        std::size_t h1 = std::hash<std::string>{}(s.kernel);
        std::size_t h2 = std::hash<std::string>{}(s.parameter);
        return h1 ^ (h2 << 1);
    }
};
