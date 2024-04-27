#include <fstream>
#include "aieblas/detail/util.hpp"
#include "aieblas/detail/codegen/generator.hpp"
#include "aieblas/detail/codegen/kernels.hpp"

namespace {
inline void create_dir(const fs::path &directory) {
    log(aieblas::log_level::verbose, "Creating directory '{}'",
        directory.c_str());
    fs::create_directory(directory);
}
} // namespace

namespace aieblas {
namespace codegen {
void generator::generate_kernel(const kernel &kernel, fs::path kernel_dir) {
    fs::path kernel_src = kernel_dir / std::format("{}.cpp", kernel.user_name);
    fs::path kernel_hdr = kernel_dir / std::format("{}.hpp", kernel.user_name);

    this->open(kernel_src);
    generate_kernel_src(*this, kernel);
    this->close();
}

void generator::generate_kernels() {
    fs::path aie_dir = out_dir / "aie";
    if (fs::exists(aie_dir)) {
        log(log_level::verbose, "Removing existing aie directory");
        fs::remove_all(aie_dir);
    }
    create_dir(aie_dir);

    fs::path kernel_dir = aie_dir / "kernels";
    create_dir(kernel_dir);

    this->kernel_srcs.clear();
    this->kernel_srcs.reserve(this->d.kernels.size());
    this->kernel_hdrs.clear();
    this->kernel_hdrs.reserve(this->d.kernels.size());

    for (const kernel &kernel : this->d.kernels) {
        log("Generating kernel {}", kernel.user_name);

        this->generate_kernel(kernel, kernel_dir);
    }
}

void generate_kernel_src(generator &gen, const kernel &kernel) {
    gen.println("#include \"{}.hpp\"", kernel.user_name);
    gen.println("");
    /* TODO: ideal number of samples? */
    gen.println("#define NUM_SAMPLES 32");
    gen.println("");

    gen.print<generator::INCREASE_AFTER>("void {}(", kernel.user_name);
    gen.println(") {{");
    gen.println<generator::DECREASE_BEFORE>("}}");
}

} // codegen
} // aieblas
