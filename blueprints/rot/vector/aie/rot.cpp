#include "rot.hpp"

#define NUM_SAMPLES 32
// 32 / 8
#define NUM_LOOPS 4

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

    aie::vector<int32, 8> vx, vy, vout_x, vout_y;
    aie::vector<int32, 8> scalar_c = aie::broadcast<int32, 8>(*c_store);
    aie::vector<int32, 8> scalar_s = aie::broadcast<int32, 8>(*s_store);
    for (unsigned i = 0; i < NUM_LOOPS; i++) {
        vx = window_readincr_v<8>(x);
        vy = window_readincr_v<8>(y);
        // out_x = c * x + s * y
        vout_x = aie::mac(aie::mul(scalar_c, vx), scalar_s, vy).to_vector<float>();
        // out_y = c * y - s * x
        vout_y = aie::msc(aie::mul(scalar_c, vy), scalar_s, vx).to_vector<float>();
        window_writeincr(out_x, vout_x);
        window_writeincr(out_y, vout_y);
    }
}


void rot(input_window<float> *x, input_window<float> *y,
         output_window<float> *out_x, output_window<float> *out_y,
         input_stream<float> *c, input_stream<float> *s) {
    aie::vector<float, 8> c_x, c_y, c_out_x, c_out_y;
    float scalar_c = aie::broadcast<float, 8>(readincr(c));
    float scalar_s = aie::broadcast<float, 8>(readincr(s));


}
