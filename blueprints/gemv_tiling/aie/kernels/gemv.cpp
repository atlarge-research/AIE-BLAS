#include "gemv.hpp"

#define NUM_SAMPLES 64

uint64 chess_storage(%chess_alignof(v4int64)) counter[4] = {0, 0, 0, 0};
int32 chess_storage(%chess_alignof(v8int32)) scalar_storage[8] = {0, 0, 0, 0, 0, 0, 0, 0};
int32 chess_storage(%chess_alignof(v8int32)) out_storage[NUM_SAMPLES];

void gemv(input_stream<int32> *__restrict scalar, input_window<int32> *__restrict A, input_window<int32> *__restrict x, input_window<int32> *__restrict y, output_window<int32> *__restrict out) {
    uint64 *cycle = &counter[0];
    int32 *alpha_store = &scalar_storage[0];
    int32 *beta_store = &scalar_storage[1];
    int32 *n = &scalar_storage[2];
    int32 *scalar_set = &scalar_storage[3];

    if (*scalar_set == 0) {
        *scalar_set = 1;
        *n = readincr(scalar);
        *alpha_store = readincr(scalar);
        *beta_store = readincr(scalar);
    }

    if (*cycle % NUM_SAMPLES == 0) {
        window_acquire(x);
    }

    if (*cycle % *n == 0) {
        window_acquire(y);
        window_acquire(out);
    }

    constexpr unsigned NUM_LOOPS = NUM_SAMPLES / 16;

    aie::vector<int32, 16> vx, vA;
    int32 &out_local = out_storage[*cycle % NUM_SAMPLES];
    if ((*cycle % *n) / NUM_SAMPLES == 0) {
        out_local = *beta_store * window_readincr(y);
    }
    aie::vector<int32, 16> vout = aie::zeros<int32, 16>();

    for (unsigned i = 0; i < NUM_LOOPS; i++) {
        vA = window_readincr_v<16>(A);
        vx = window_readincr_v<16>(x);
        vout = aie::add(vout, aie::mul(vA, vx).to_vector<int32>());
    }
    window_decr_v16(x, NUM_LOOPS);

    out_local += *alpha_store * aie::reduce_add(vout);

    if ((*cycle % *n) / NUM_SAMPLES == (*n / NUM_SAMPLES) - 1) {
        window_writeincr(out, out_local);
    }

    *cycle += 1;

    if (*cycle % NUM_SAMPLES == 0) {
        window_release(x);
    }

    if (*cycle % *n == 0) {
        window_release(y);
        window_release(out);
    }
}
