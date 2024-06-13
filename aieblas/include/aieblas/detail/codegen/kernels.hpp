#pragma once

#include <fstream>

#include "aieblas/detail/util.hpp"
#include "aieblas/detail/codegen/generator.hpp"

namespace aieblas {
namespace codegen {

std::vector<kernel_arg> get_kernel_args(blas_op operation);

class kernel_generator {
public:
    kernel_generator(const kernel &kernel)
        : k(kernel) {}

    virtual ~kernel_generator() {}

    virtual void gen_kernel_args(generator &gen) = 0;
    virtual void gen_kernel_body(generator &gen) = 0;

    inline std::vector<kernel_arg> get_kernel_args() const {
        return ::aieblas::codegen::get_kernel_args(k.operation);
    }

    virtual void gen_mm2s(generator &gen) = 0;
    virtual void gen_s2mm(generator &gen) = 0;

    virtual void gen_link(generator &gen) = 0;

protected:
    const kernel &k;
};

std::unique_ptr<kernel_generator> get_kernel_generator(const kernel &kernel);

void generate_kernel_src(generator &gen, kernel_generator &kernel_gen,
                         const kernel &kernel);

void generate_kernel_hdr(generator &gen, kernel_generator &kernel_gen,
                         const kernel &kernel);

} // codegen
} // aieblas
