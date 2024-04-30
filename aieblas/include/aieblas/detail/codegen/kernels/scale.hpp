#pragma once

#include "aieblas/detail/util.hpp"
#include "aieblas/detail/codegen/generator.hpp"
#include "aieblas/detail/codegen/kernels.hpp"

namespace aieblas {
namespace codegen {
namespace generators {
class scale_generator : public kernel_generator {
public:
    scale_generator(const kernel &kernel)
    : kernel_generator(kernel), dtype(datatype_to_str(kernel.type)) {}

    virtual ~scale_generator() {}

    void gen_kernel_args(generator &gen) override;
    void gen_kernel_body(generator &gen) override;

private:
    const char *const dtype;
};

} // generators
} // codegen
} // aieblas
