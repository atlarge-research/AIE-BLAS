#ifndef FUNCTION_KERNELS_H
#define FUNCTION_KERNELS_H
#include "aie_api/aie.hpp"
#include "aie_api/aie_adf.hpp"
void scale(input_window<int32> *in, output_window<int32> *out, input_stream<int32> *ctrl);

#endif
