#include "aieblas/detail/codegen/json_parser/get_kernel_options.hpp"
#include "aieblas/detail/util.hpp"

#include "aieblas/detail/codegen/kernels/parsers/asum.hpp"
#include "aieblas/detail/codegen/kernels/parsers/axpy.hpp"
#include "aieblas/detail/codegen/kernels/parsers/dot.hpp"
#include "aieblas/detail/codegen/kernels/parsers/gemv.hpp"
#include "aieblas/detail/codegen/kernels/parsers/iamax.hpp"
#include "aieblas/detail/codegen/kernels/parsers/nrm2.hpp"
#include "aieblas/detail/codegen/kernels/parsers/rot.hpp"
#include "aieblas/detail/codegen/kernels/parsers/scal.hpp"

namespace aieblas {
namespace codegen {

std::unique_ptr<kernel_options> get_kernel_options(blas_op kernel,
                                                   nlohmann::json json) {
    switch (kernel) {
    case blas_op::asum:
        return generators::get_asum_options(json);
    case blas_op::axpy:
        return generators::get_axpy_options(json);
    case blas_op::dot:
        return generators::get_dot_options(json);
    case blas_op::gemv:
        return generators::get_gemv_options(json);
    case blas_op::iamax:
        return generators::get_iamax_options(json);
    case blas_op::nrm2:
        return generators::get_nrm2_options(json);
    case blas_op::rot:
        return generators::get_rot_options(json);
    case blas_op::scal:
        return generators::get_scal_options(json);

    default:
        throw std::runtime_error(
            std::format("Unknown blas operation '{}'", blas_op_to_str(kernel)));
    }
}

} // namespace codegen
} // namespace aieblas
