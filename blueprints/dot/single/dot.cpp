#include "dot.hpp"

#define NUM_SAMPLES 32
// 32 / 8
#define NUM_LOOPS 4

void dot(input_window<float> *x, output_window<float> *y, output_stream<float> *out) {
    float c_x, c_y;
    float result = 0;

    for (unsigned i = 0; i < NUM_SAMPLES; i++) {
        c_x = window_readincr(x);
        c_y = window_readincr(y);
        result += c_x * c_y;
    }

    writeincr(out, result);
}
