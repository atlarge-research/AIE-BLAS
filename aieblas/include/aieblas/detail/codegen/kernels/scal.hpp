#pragma once

#include "aieblas/detail/util.hpp"
#include "aieblas/detail/codegen/generator.hpp"
#include "aieblas/detail/codegen/kernels.hpp"

namespace aieblas {
namespace codegen {
namespace generators {
class scal_generator : public kernel_generator {
public:
    scal_generator(const kernel &kernel)
    : kernel_generator(kernel), dtype(aie_dtype(kernel.type, kernel.vsize)),
      x(kernel.connections.at("x")), alpha(kernel.connections.at("alpha")),
      out((kernel.connections.at("out"))) {}

    virtual ~scal_generator() {}

    void gen_kernel_glob(generator &gen) override;
    void gen_kernel_args(generator &gen) override;
    void gen_kernel_body(generator &gen) override;

    bool need_mm2s() const override;
    void gen_mm2s(generator &gen) override;
    bool need_s2mm() const override;
    void gen_s2mm(generator &gen) override;

    void gen_link(generator &gen) override;

private:
    const std::string dtype;

    const connection x;
    const connection alpha;
    const connection out;
};

std::vector<kernel_arg> get_scal_args();

} // generators
} // codegen
} // aieblas
