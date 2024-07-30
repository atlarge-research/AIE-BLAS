#include "aieblas/detail/codegen/generator.hpp"
#include "aieblas/detail/codegen/kernels.hpp"
#include "aieblas/detail/util.hpp"

#include "aieblas/detail/codegen/kernels/asum.hpp"
#include "aieblas/detail/codegen/kernels/axpy.hpp"
#include "aieblas/detail/codegen/kernels/dot.hpp"
#include "aieblas/detail/codegen/kernels/gemv.hpp"
#include "aieblas/detail/codegen/kernels/iamax.hpp"
#include "aieblas/detail/codegen/kernels/nrm2.hpp"
#include "aieblas/detail/codegen/kernels/rot.hpp"
#include "aieblas/detail/codegen/kernels/scal.hpp"

namespace aieblas {
namespace codegen {

std::unique_ptr<kernel_generator> get_kernel_generator(const kernel &kernel) {
    switch (kernel.operation) {
    case blas_op::asum:
        return std::make_unique<generators::asum_generator>(kernel);
    case blas_op::axpy:
        return std::make_unique<generators::axpy_generator>(kernel);
    case blas_op::dot:
        return std::make_unique<generators::dot_generator>(kernel);
    case blas_op::gemv:
        return std::make_unique<generators::gemv_generator>(kernel);
    case blas_op::iamax:
        return std::make_unique<generators::iamax_generator>(kernel);
    case blas_op::nrm2:
        return std::make_unique<generators::nrm2_generator>(kernel);
    case blas_op::rot:
        return std::make_unique<generators::rot_generator>(kernel);
    case blas_op::scal:
        return std::make_unique<generators::scal_generator>(kernel);
    default:
        throw std::runtime_error(
            std::format("Unsupported kernel operation '{}'",
                        blas_op_to_str(kernel.operation)));
    }
}

std::vector<kernel_arg> get_kernel_args(blas_op operation) {
    switch (operation) {
    case blas_op::asum:
        return generators::get_asum_args();
    case blas_op::axpy:
        return generators::get_axpy_args();
    case blas_op::dot:
        return generators::get_dot_args();
    case blas_op::gemv:
        return generators::get_gemv_args();
    case blas_op::iamax:
        return generators::get_iamax_args();
    case blas_op::nrm2:
        return generators::get_nrm2_args();
    case blas_op::rot:
        return generators::get_rot_args();
    case blas_op::scal:
        return generators::get_scal_args();
    default:
        throw std::runtime_error(
            std::format("Unsupported kernel operation '{}'",
                        blas_op_to_str(operation)));
    }
}

} // namespace codegen
} // namespace aieblas
