#pragma once
#include <nlohmann/json.hpp>

#include "aieblas/detail/codegen/kernels.hpp"
#include "aieblas/detail/codegen/kernels/nrm2.hpp"
#include "aieblas/detail/util.hpp"

namespace aieblas {
namespace codegen {
namespace generators {
inline std::unique_ptr<nrm2_options> get_nrm2_options(nlohmann::json json) {
    return std::make_unique<nrm2_options>();
}
} // namespace generators
} // namespace codegen
} // namespace aieblas
