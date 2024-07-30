#pragma once
#include <nlohmann/json.hpp>

#include "aieblas/detail/codegen/kernels.hpp"
#include "aieblas/detail/codegen/kernels/asum.hpp"
#include "aieblas/detail/util.hpp"

namespace aieblas {
namespace codegen {
namespace generators {
inline std::unique_ptr<asum_options> get_asum_options(nlohmann::json json) {
    return std::make_unique<asum_options>();
}
} // namespace generators
} // namespace codegen
} // namespace aieblas
