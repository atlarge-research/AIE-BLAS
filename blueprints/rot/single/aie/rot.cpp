#include "rot.hpp"

#define NUM_SAMPLES 32

int32 chess_storage(%chess_alignof(v4int32)) rot_storage[4] = {0, 0, 0, 0};

void rot(input_window<int32> *x, input_window<int32> *y,
         output_window<int32> *out_x, output_window<int32> *out_y,
         input_stream<int32> *c, input_stream<int32> *s) {
    int32 *c_store = &rot_storage[0];
    int32 *s_store = &rot_storage[1];
    int32 *rot_set = &rot_storage[2];
    if (*rot_set == 0) {
        *rot_set = 1;
        *c_store = readincr(c);
        *s_store = readincr(s);
    }

    int32 vx, vy, vout_x, vout_y;
    int32 scalar_c = *c_store;
    int32 scalar_s = *s_store;
    for (unsigned i = 0; i < NUM_SAMPLES; i++) {
        vx = window_readincr(x);
        vy = window_readincr(y);
        vout_x = scalar_c * vx + scalar_s * vy;
        vout_y = scalar_c * vy - scalar_s * vx;
        window_writeincr(out_x, vout_x);
        window_writeincr(out_y, vout_y);
    }
}
