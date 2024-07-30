#pragma once

#include "aieblas/detail/util.hpp"
#include "aieblas/detail/codegen/generator.hpp"
#include "aieblas/detail/codegen/kernels.hpp"

namespace aieblas {
namespace codegen {
namespace generators {

class rot_options : public kernel_options {
public:
    rot_options(const value &c_, const value &s_) : kernel_options() {}

    virtual ~rot_options() {}

    bool disabled_arg(const std::string &arg) const override {
        if (arg == "c") {
            return c.set;
        } else if (arg == "s") {
            return s.set;
        }

        return false;
    };

    const value c;
    const value s;
};

class rot_generator : public kernel_generator {
public:
    rot_generator(const kernel &kernel)
    : kernel_generator(kernel), dtype(aie_dtype(kernel.type, kernel.vsize)),
      x(kernel.connections.at("x")), y(kernel.connections.at("y")),
      c(kernel.connections.at("c")), s(kernel.connections.at("s")),
      out_x((kernel.connections.at("out_x"))),
      out_y((kernel.connections.at("out_y"))),
      options(dynamic_cast<rot_options &>(*kernel.extra_options)) {}

    virtual ~rot_generator() {}

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
    const connection c;
    const connection s;
    const connection out_x;
    const connection out_y;

    const rot_options &options;
};

std::vector<kernel_arg> get_rot_args();

} // generators
} // codegen
} // aieblas
