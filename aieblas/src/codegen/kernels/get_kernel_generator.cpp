#include "aieblas/detail/codegen/generator.hpp"
#include "aieblas/detail/codegen/kernels.hpp"
#include "aieblas/detail/util.hpp"

#include "aieblas/detail/codegen/kernels/scale.hpp"

namespace aieblas {
namespace codegen {

std::unique_ptr<kernel_generator> get_kernel_generator(const kernel &kernel) {
    if (kernel.operation == blas_op::scale) {
        return std::make_unique<generators::scale_generator>(kernel);
    } else {
        throw std::runtime_error(
            std::format("Unsupported kernel operation '{}'",
                        blas_op_to_str(kernel.operation)));
    }
}

} // namespace codegen
} // namespace aieblas
