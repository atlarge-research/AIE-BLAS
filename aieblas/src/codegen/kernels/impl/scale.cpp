#include "aieblas/detail/util.hpp"
#include "aieblas/detail/codegen/kernels/scale.hpp"


namespace aieblas {
namespace codegen {
namespace generators {

void scale_generator::gen_kernel_args(generator &gen) {
    gen.print("input_window<{0}> *in, output_window<{0}> *out, "
              "input_stream<{0}> *ctrl", dtype);
}

void scale_generator::gen_kernel_body(generator &gen) {
    gen.println("{0} c_in, c_out;", dtype);
    gen.println("{0} scalar = readincr(ctrl);", dtype);
    gen.println<generator::NO_INDENT>("");
    gen.println<generator::INCREASE_AFTER>("for (unsigned i = 0; "
                                           "i < NUM_SAMPLES; i++) {{");
    gen.println("window_readincr(in, c_in);");
    gen.println("c_out = c_in * scalar;");
    gen.println("window_writeincr(out, c_out);");
    gen.println<generator::DECREASE_BEFORE>("}}");
}

} // generators
} // codegen
} // aieblas
