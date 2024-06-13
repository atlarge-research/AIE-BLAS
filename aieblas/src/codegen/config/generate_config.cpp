#include <fstream>
#include "aieblas/detail/util.hpp"
#include "aieblas/detail/codegen/generator.hpp"
#include "aieblas/detail/codegen/kernels.hpp"

namespace aieblas {
namespace codegen {

static inline void generate_config_kernel(generator &gen,
                                          const kernel &kernel) {
    std::unique_ptr<kernel_generator> kernel_gen = get_kernel_generator(kernel);
    kernel_gen->gen_link(gen);
}

void generator::generate_config() {
    fs::path cfg_file = out_dir / "link.cfg";

    this->open(cfg_file, comment_type::HASHTAG);
    this->println("platform={}", this->d.platform);
    this->println("");

    if (this->d.profile) {
        this->println("[profile]");
        this->println("data=all:all:all # Monitor data on all kernels and CUs");
        this->println("memory=all       # Monitor transfers for all memories");
        this->println("stall=all:all    # Monitor stalls for all CUs of all "
                      "kernels");
        this->println("exec=all:all     # Monitor execution times for all CUs");
        this->println("aie=all          # Monitor all AIE streams");
        this->println("");
    }

    this->println("[connectivity]");
    for (const kernel &kernel : this->d.kernels) {
        generate_config_kernel(*this, kernel);
        this->println("");
    }
    this->close();
}

} // codegen
} // aieblas
