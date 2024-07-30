#pragma once

#include <memory>
#include <nlohmann/json.hpp>

#include "aieblas/detail/codegen/kernels.hpp"

namespace aieblas {
namespace codegen {

std::unique_ptr<kernel_options> get_kernel_options(blas_op kernel,
                                                   nlohmann::json json);

} // namespace codegen
} // namespace aieblas
