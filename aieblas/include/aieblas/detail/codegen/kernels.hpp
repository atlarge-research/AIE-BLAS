#pragma once

#include <fstream>

#include "aieblas/detail/util.hpp"
#include "aieblas/detail/codegen/generator.hpp"

namespace aieblas {
namespace codegen {

struct pl_kernel_generator {
    std::string name;
    std::function<void(generator &)> generator;
};

std::vector<kernel_arg> get_kernel_args(blas_op operation);

struct value {
    bool set;
    dtype type;
    union {
        int64_t signed_v;
        uint64_t unsigned_v;
        float float_v;
    };
    value() : set(false), type(dtype::unknown) {}
    value(int64_t val): set(true), type(dtype::int64), signed_v(val) {}
    value(uint64_t val): set(true), type(dtype::uint64), unsigned_v(val) {}
    value(float val): set(true), type(dtype::float32), float_v(val) {}

    std::string to_string() const {
        if (!set) {
            return std::string();
        }
        switch (type) {
            case dtype::int64:
                return std::to_string(signed_v);
            case dtype::uint64:
                return std::to_string(unsigned_v);
            case dtype::float32:
                return std::to_string(float_v);
            default:
                throw std::runtime_error("Unknown value datatype");
        }
    }
};

class kernel_generator {
public:
    kernel_generator(const kernel &kernel)
        : k(kernel) {}

    virtual ~kernel_generator() {}


    virtual void gen_kernel_glob(generator &gen) = 0;
    virtual void gen_kernel_args(generator &gen) = 0;
    virtual void gen_kernel_body(generator &gen) = 0;

    inline std::vector<kernel_arg> get_kernel_args() const {
        return ::aieblas::codegen::get_kernel_args(k.operation);
    }

    virtual std::vector<pl_kernel_generator> get_pl_generators() = 0;

    virtual void gen_link(generator &gen) = 0;

protected:
    const kernel &k;
    std::unique_ptr<kernel_options> o;
};

std::unique_ptr<kernel_generator> get_kernel_generator(const kernel &kernel);

void generate_kernel_src(generator &gen, kernel_generator &kernel_gen,
                         const kernel &kernel);

void generate_kernel_hdr(generator &gen, kernel_generator &kernel_gen,
                         const kernel &kernel);

} // codegen
} // aieblas
