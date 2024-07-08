#include <fstream>
#include "aieblas/detail/util.hpp"
#include "aieblas/detail/codegen/generator.hpp"
#include "aieblas/detail/codegen/kernels.hpp"

namespace aieblas {
namespace codegen {
static inline void generate_pl_kernel_mm2s(generator &gen,
                                           kernel_generator &kernel_gen,
                                           const kernel &kernel) {
    gen.println("#include <ap_int.h>");
    gen.println("#include <hls_stream.h>");
    gen.println("#include <ap_axi_sdata.h>");
    gen.println();
    gen.println("extern \"C\" {{");
    kernel_gen.gen_mm2s(gen);
    gen.println("}}");
}

static inline void generate_pl_kernel_s2mm(generator &gen,
                                           kernel_generator &kernel_gen,
                                           const kernel &kernel) {
    gen.println("#include <ap_int.h>");
    gen.println("#include <hls_stream.h>");
    gen.println("#include <ap_axi_sdata.h>");
    gen.println();
    gen.println("extern \"C\" {{");
    kernel_gen.gen_s2mm(gen);
    gen.println("}}");
}

void generator::generate_pl_kernel(const kernel &kernel,
                                   const fs::path &pl_dir) {
    std::unique_ptr<kernel_generator> kernel_gen = get_kernel_generator(kernel);
    if (kernel_gen->need_mm2s()) {
        fs::path mm2s = pl_dir / std::format("{}_mm2s.cpp", kernel.user_name);
        this->open(mm2s);
        generate_pl_kernel_mm2s(*this, *kernel_gen, kernel);
        this->close();
        this->pl_kernels.push_back(mm2s);
    }

    if (kernel_gen->need_s2mm()) {
        fs::path s2mm = pl_dir / std::format("{}_s2mm.cpp", kernel.user_name);
        this->open(s2mm);
        generate_pl_kernel_s2mm(*this, *kernel_gen, kernel);
        this->close();
        this->pl_kernels.push_back(s2mm);
    }
}

void generator::generate_pl_kernels() {
    fs::path pl_dir = out_dir / "pl_kernels";
    if (fs::exists(pl_dir)) {
        log(log_level::verbose, "Removing existing pl directory");
        fs::remove_all(pl_dir);
    }
    util::create_dir(pl_dir);

    for (const kernel &kernel : d.kernels)  {
        generate_pl_kernel(kernel, pl_dir);
    }
}
} // codegen
} // aieblas
