#pragma once

#include "aieblas/detail/util.hpp"
#include "aieblas/detail/codegen/generator.hpp"
#include "aieblas/detail/codegen/kernels.hpp"

namespace aieblas {
namespace codegen {
namespace generators {

class iamax_options : public kernel_options {
public:
    iamax_options() : kernel_options() {}

    virtual ~iamax_options() {}

    bool disabled_arg(const std::string &arg) const override {
        return false;
    };
};

class iamax_generator : public kernel_generator {
public:
    iamax_generator(const kernel &kernel)
    : kernel_generator(kernel), dtype(datatype_to_str(kernel.type)),
      x(kernel.connections.at("x")), out((kernel.connections.at("out"))),
      options(dynamic_cast<iamax_options &>(*kernel.extra_options))  {}

    virtual ~iamax_generator() {}

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
    const connection out;

    const iamax_options &options;
};

std::vector<kernel_arg> get_iamax_args();

} // generators
} // codegen
} // aieblas
