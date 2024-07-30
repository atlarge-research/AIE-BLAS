#pragma once

#include "aieblas/detail/util.hpp"
#include "aieblas/detail/codegen/generator.hpp"
#include "aieblas/detail/codegen/kernels.hpp"

namespace aieblas {
namespace codegen {
namespace generators {

class dot_options : public kernel_options {
public:
    dot_options() : kernel_options() {}

    virtual ~dot_options() {}

    bool disabled_arg(const std::string &arg) const override {
        return false;
    };
};

class dot_generator : public kernel_generator {
public:
    dot_generator(const kernel &kernel)
    : kernel_generator(kernel), dtype(aie_dtype(kernel.type, kernel.vsize)),
      x(kernel.connections.at("x")), y(kernel.connections.at("y")),
      out((kernel.connections.at("out"))),
      options(dynamic_cast<dot_options &>(*kernel.extra_options)) {}

    virtual ~dot_generator() {}

    void gen_kernel_glob(generator &gen) override;
    void gen_kernel_args(generator &gen) override;
    void gen_kernel_body(generator &gen) override;

    std::vector<pl_kernel_generator> get_pl_generators() override;
    bool need_mm2s() const;
    void gen_mm2s(generator &gen);
    bool need_s2mm() const;
    void gen_s2mm(generator &gen);

    void gen_link(generator &gen) override;

private:
    const std::string dtype;

    const connection x;
    const connection y;
    const connection out;

    const dot_options &options;
};

std::vector<kernel_arg> get_dot_args();

} // generators
} // codegen
} // aieblas
