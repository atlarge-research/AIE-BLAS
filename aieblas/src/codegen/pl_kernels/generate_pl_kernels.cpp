#include <fstream>
#include "aieblas/detail/util.hpp"
#include "aieblas/detail/codegen/generator.hpp"
#include "aieblas/detail/codegen/kernels.hpp"

namespace aieblas {
namespace codegen {
static inline void generate_pl_kernel(generator &gen,
        std::function<void(generator &)> pl_gen) {
    gen.println("#include <ap_int.h>");
    gen.println("#include <hls_stream.h>");
    gen.println("#include <ap_axi_sdata.h>");
    gen.println();
    gen.println("extern \"C\" {{");
    pl_gen(gen);
    gen.println("}}");
}


void generator::generate_pl_kernels() {
    fs::path pl_dir = out_dir / "pl_kernels";
    if (fs::exists(pl_dir)) {
        log(log_level::verbose, "Removing existing pl directory");
        fs::remove_all(pl_dir);
    }
    util::create_dir(pl_dir);

    for (const kernel &kernel : d.kernels)  {
        std::unique_ptr<kernel_generator> kernel_gen = get_kernel_generator(kernel);
        for (pl_kernel_generator &pl_gen : kernel_gen->get_pl_generators()) {
            fs::path pl = pl_dir / std::format("{}_{}.cpp", kernel.user_name, pl_gen.name);
            this->open(pl);
            generate_pl_kernel(*this, pl_gen.generator);
            this->close();
            this->pl_kernels.push_back(pl);
        }
    }
}
} // codegen
} // aieblas
