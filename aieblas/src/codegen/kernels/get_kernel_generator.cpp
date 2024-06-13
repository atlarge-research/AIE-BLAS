#include "aieblas/detail/codegen/generator.hpp"
#include "aieblas/detail/codegen/kernels.hpp"
#include "aieblas/detail/util.hpp"

#include "aieblas/detail/codegen/kernels/dot.hpp"
#include "aieblas/detail/codegen/kernels/scale.hpp"

namespace aieblas {
namespace codegen {

std::unique_ptr<kernel_generator> get_kernel_generator(const kernel &kernel) {
    switch (kernel.operation) {
    case blas_op::dot:
        return std::make_unique<generators::dot_generator>(kernel);
    case blas_op::scale:
        return std::make_unique<generators::scale_generator>(kernel);
    default:
        throw std::runtime_error(
            std::format("Unsupported kernel operation '{}'",
                        blas_op_to_str(kernel.operation)));
    }
}

std::vector<kernel_arg> get_kernel_args(blas_op operation) {
    switch (operation) {
    case blas_op::dot:
        return generators::get_dot_args();
    case blas_op::scale:
        return generators::get_scale_args();
    default:
        throw std::runtime_error(
            std::format("Unsupported kernel operation '{}'",
                        blas_op_to_str(operation)));
    }
}

} // namespace codegen
} // namespace aieblas
