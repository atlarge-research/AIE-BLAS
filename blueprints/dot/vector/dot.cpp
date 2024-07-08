#include "dot.hpp"

#define NUM_SAMPLES 32
// 32 / 8
#define NUM_LOOPS 4

void dot(input_window<float> *x, output_window<float> *y, output_stream<float> *out) {
    aie::vector<float, 8> c_x, c_y;
    aie::vector<float, 8> result = aie::broadcast<float, 8>(0);

    for (unsigned i = 0; i < NUM_LOOPS; i++) {
        c_x = window_readincr_v<8>(x);
        c_y = window_readincr_v<8>(y);
        result = aie::add(aie::mul(c_x, c_y), result);
    }

    writeincr(out, aie::reduce_add(result));
}
