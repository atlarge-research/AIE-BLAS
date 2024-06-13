#include "aieblas/codegen.hpp"
#include "aieblas/detail/util.hpp"
#include "aieblas/detail/codegen/generator.hpp"


namespace aieblas {
namespace codegen {

void codegen(fs::path json_file, fs::path output) {
    generator gen{json_file, output};

    gen.generate_kernels();
    gen.generate_graph();
    gen.generate_pl_kernels();
    gen.generate_config();
    gen.generate_cmake();
}

} // codegen
} // aieblas
