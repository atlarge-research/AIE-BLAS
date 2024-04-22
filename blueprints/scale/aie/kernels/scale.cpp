#include "scale.h"

#define NUM_SAMPLES 32

void scale(input_window<int32> *in, output_window<int32> *out, input_stream<int32> *ctrl)
{
  int32 c_in, c_out;
  int32 scalar = readincr(ctrl);

  for (unsigned i = 0; i < NUM_SAMPLES; i++)
  {
    window_readincr(in, c_in);
#if defined(__AIESIM__) || defined(__X86SIM__)
    printf("[0] Received %d\n", c_in);
#endif
    c_out = c_in * scalar;
    window_writeincr(out, c_out);
#if defined(__AIESIM__) || defined(__X86SIM__)
    printf("[0] Computed %d\n", c_out);
#endif
  }
}
