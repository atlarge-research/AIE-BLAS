#pragma once
#include <nlohmann/json.hpp>

#include "aieblas/detail/codegen/kernels.hpp"
#include "aieblas/detail/codegen/kernels/iamax.hpp"
#include "aieblas/detail/util.hpp"

namespace aieblas {
namespace codegen {
namespace generators {
inline std::unique_ptr<iamax_options> get_iamax_options(nlohmann::json json) {
    return std::make_unique<iamax_options>();
}
} // namespace generators
} // namespace codegen
} // namespace aieblas
