#pragma once

#include "aieblas/detail/util.hpp"
#include "aieblas/detail/codegen/generator.hpp"
#include "aieblas/detail/codegen/kernels.hpp"

namespace aieblas {
namespace codegen {
namespace generators {

class axpy_options : public kernel_options {
public:
    axpy_options(const value &alpha_) : kernel_options(), alpha(alpha_) {}

    virtual ~axpy_options() {}

    bool disabled_arg(const std::string &arg) const override {
        if (arg == "alpha") {
            return alpha.set;
        }

        return false;
    };

    const value alpha;
};

class axpy_generator : public kernel_generator {
public:
    axpy_generator(const kernel &kernel)
    : kernel_generator(kernel), dtype(aie_dtype(kernel.type, kernel.vsize)),
      x(kernel.connections.at("x")), y(kernel.connections.at("y")),
      alpha(kernel.connections.at("alpha")),
      out((kernel.connections.at("out"))),
      options(dynamic_cast<axpy_options &>(*kernel.extra_options)) {}

    virtual ~axpy_generator() {}

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
    const connection alpha;
    const connection out;

    const axpy_options &options;
};

std::vector<kernel_arg> get_axpy_args();

} // generators
} // codegen
} // aieblas
