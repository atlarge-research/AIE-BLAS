#include "scal.hpp"

#define NUM_SAMPLES 32
// 32 / 8
#define NUM_LOOPS 4

void scal(input_window<float> *x, output_window<float> *out, input_stream<float> *alpha) {
    aie::vector<float, 8> c_x, c_out;
    aie::vector<float, 8> scalar = aie::broadcast<float, 8>(readincr(alpha));

    for (unsigned i = 0; i < NUM_LOOPS; i++) {
        c_x = window_readincr_v<8>(x);
        c_out = aie::mul(c_x, scalar).to_vector<float>();
        window_writeincr(out, c_out);
    }
}
