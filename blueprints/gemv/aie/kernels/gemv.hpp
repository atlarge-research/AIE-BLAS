// This file was auto-generated by aieblas on 2024-7-19 at 13:38

#pragma once
#include "aie_api/aie.hpp"
#include "aie_api/aie_adf.hpp"

void gemv(input_stream<int32> *__restrict alpha, input_window<int32> *__restrict A, input_window<int32> *__restrict x, input_stream<int32> *__restrict beta, input_window<int32> *__restrict y, output_window<int32> *__restrict out);
