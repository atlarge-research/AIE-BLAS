#pragma once
#include "aie_api/aie.hpp"
#include "aie_api/aie_adf.hpp"

void scal(input_window<float> *x, output_window<float> *out, input_stream<float> *alpha);
