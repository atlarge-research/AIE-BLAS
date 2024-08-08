#pragma once
#include "aie_api/aie.hpp"
#include "aie_api/aie_adf.hpp"

void rot(input_window<float> *x, input_window<float> *y,
         output_window<float> *out_x, output_window<float> *out_y,
         input_stream<float> *c, input_stream<float> *s);
