
#include <ap_int.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include <stdio.h>
#include <iostream>
extern "C"
{

// Reads the data from memory and put into stream s
void mm2s(ap_int<32> *mem, int size, int scalar, hls::stream<qdma_axis<32, 0, 0, 0>> &stream_in, hls::stream<qdma_axis<32, 0, 0, 0>> &stream_ctrl) {
#pragma HLS INTERFACE m_axi port = mem offset = slave
#pragma HLS interface axis port = stream_in
#pragma HLS interface axis port = stream_ctrl

#pragma HLS INTERFACE s_axilite port = mem bundle = control
#pragma HLS INTERFACE s_axilite port = size bundle = control
#pragma HLS INTERFACE s_axilite port = scalar bundle = control
#pragma HLS interface s_axilite port = return bundle = control

    // Send scalar over ctrl stream
    qdma_axis<32,0,0,0> qdma_scalar;
    qdma_scalar.data=(ap_int<32>) scalar;
    qdma_scalar.keep_all();
    stream_ctrl.write(qdma_scalar);

    // Send data over in stream
    for (int i = 0; i < size; i++) {
#pragma HLS PIPELINE II = 1
        qdma_axis<32, 0, 0, 0> x;
        x.data = mem[i];
        x.keep_all();
        stream_in.write(x);
    }
}
}
