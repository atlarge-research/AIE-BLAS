#pragma once

#include "aieblas/detail/util.hpp"
#include "aieblas/detail/codegen/generator.hpp"
#include "aieblas/detail/codegen/kernels.hpp"

namespace aieblas {
namespace codegen {
namespace generators {

class gemv_options : public kernel_options {
public:
    gemv_options(const value &alpha_, const value &beta_) : kernel_options(),
    alpha(alpha_), beta(beta_) {}

    virtual ~gemv_options() {}

    bool disabled_arg(const std::string &arg) const override {
        if (arg == "alpha") {
            return alpha.set;
        } else if (arg == "beta") {
            return beta.set;
        }

        return false;
    };

    const value alpha;
    const value beta;
};

class gemv_generator : public kernel_generator {
public:
    gemv_generator(const kernel &kernel)
    : kernel_generator(kernel), dtype(aie_dtype(kernel.type, kernel.vsize)),
      A(kernel.connections.at("A")), x(kernel.connections.at("x")),
      y(kernel.connections.at("y")), alpha(kernel.connections.at("alpha")),
      beta(kernel.connections.at("beta")), out((kernel.connections.at("out"))),
      options(dynamic_cast<gemv_options &>(*kernel.extra_options))
    {}

    virtual ~gemv_generator() {}

    void gen_kernel_glob(generator &gen) override;
    void gen_kernel_args(generator &gen) override;
    void gen_kernel_body(generator &gen) override;

    std::vector<pl_kernel_generator> get_pl_generators() override;
    void gen_mm2s_A(generator &gen);
    void gen_mm2s_scalar(generator &gen);
    void gen_mm2s_x(generator &gen);
    void gen_mm2s_y(generator &gen);
    void gen_s2mm(generator &gen);

    void gen_link(generator &gen) override;

private:
    const std::string dtype;

    const connection A;
    const connection x;
    const connection y;
    const connection alpha;
    const connection beta;
    const connection out;

    const gemv_options &options;
};

std::vector<kernel_arg> get_gemv_args();

} // generators
} // codegen
} // aieblas
