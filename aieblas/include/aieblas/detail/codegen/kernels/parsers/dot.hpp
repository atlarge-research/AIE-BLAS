#pragma once
#include <nlohmann/json.hpp>

#include "aieblas/detail/codegen/kernels.hpp"
#include "aieblas/detail/codegen/kernels/dot.hpp"
#include "aieblas/detail/util.hpp"

namespace aieblas {
namespace codegen {
namespace generators {
inline std::unique_ptr<dot_options> get_dot_options(nlohmann::json json) {
    return std::make_unique<dot_options>();
}
} // namespace generators
} // namespace codegen
} // namespace aieblas
