
#include <ap_int.h>
#include <hls_stream.h>
#include <ap_axi_sdata.h>

extern "C"
{
// Receives data from stream and store it in memory
void s2mm(ap_int<32> *mem, int size, hls::stream<qdma_axis<32, 0, 0, 0>> &stream_out) {
#pragma HLS INTERFACE m_axi port = mem offset = slave

#pragma HLS interface axis port = stream_out

#pragma HLS INTERFACE s_axilite port = mem bundle = control
#pragma HLS INTERFACE s_axilite port = size bundle = control
#pragma HLS interface s_axilite port = return bundle = control

    for (int i = 0; i < size; i++) {
#pragma HLS PIPELINE II = 1
        qdma_axis<32, 0, 0, 0> x = stream_out.read();
        mem[i] = x.data;
    }
}
}
