#include <adf.h>
#include "kernels/scale.h"

using namespace adf;

class simpleGraph : public graph
{
private:
  kernel scalek;

public:
  input_plio in;
  input_plio ctrl;
  output_plio out;
  simpleGraph()
  {

    // Data is read written from/to file
    in = input_plio::create("DataIn", plio_32_bits, "data/input.txt");
    ctrl = input_plio::create("CtrlIn", plio_32_bits, "data/ctrl.txt");
    out = output_plio::create("DataOut", plio_32_bits, "data/output.txt");

    scalek = kernel::create(scale);

    // By default we send 32 32-bits integers
    connect<window<128>> net0(in.out[0], scalek.in[0]);
    connect<stream>(ctrl.out[0], scalek.in[1]);
    connect<window<128>> net1(scalek.out[0], out.in[0]);
    source(scalek) = "kernels/scale.cpp";

    runtime<ratio>(scalek) = 0.9;
  }
};
