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
    : kernel_generator(kernel), dtype(aie_dtype(kernel.type, kernel.vsize)),
      in(kernel.connections.at("in")), ctrl(kernel.connections.at("ctrl")),
      out((kernel.connections.at("out"))) {}

    virtual ~scale_generator() {}

    void gen_kernel_args(generator &gen) override;
    void gen_kernel_body(generator &gen) override;

    void gen_mm2s(generator &gen) override;
    void gen_s2mm(generator &gen) override;

    void gen_link(generator &gen) override;

private:
    const std::string dtype;

    const connection in;
    const connection ctrl;
    const connection out;
};

std::vector<kernel_arg> get_scale_args();

} // generators
} // codegen
} // aieblas
