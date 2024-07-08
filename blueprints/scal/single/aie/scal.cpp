#include "scal.hpp"

#define NUM_SAMPLES 32

void scal(input_stream<int64> in_size_n, input_window<float> *x, output_window<float> *out, input_stream<float> *alpha) {
    float c_x, c_out;
    float scalar = readincr(alpha);
    int64 NUM_SAMPLES = readincr(in_size_n);

    for (unsigned i = 0; i < NUM_SAMPLES; i++) {
        c_x = window_readincr(x);
        c_out = c_x * scalar;
        window_writeincr(out, c_out);
    }
}
