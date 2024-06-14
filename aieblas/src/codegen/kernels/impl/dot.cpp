#include "aieblas/detail/util.hpp"
#include "aieblas/detail/codegen/kernels/dot.hpp"


namespace aieblas {
namespace codegen {
namespace generators {

void dot_generator::gen_kernel_args(generator &gen) {
    gen.print("input_window<{0}> *x, output_window<{0}> *y, "
              "output_stream<{0}> *out", datatype_to_str(k.type));
}

void dot_generator::gen_kernel_body(generator &gen) {
    const unsigned num_samples = k.wsize * 8 / datatype_to_bits(k.type);
    gen.println("constexpr unsigned NUM_SAMPLES = {};", num_samples);
    gen.println("{0} c_x, c_y;", dtype);
    const char *loop_cond;
    if (k.vsize == 0) {
        gen.println("{0} result = 0;", dtype);
        loop_cond = "NUM_SAMPLES";
    } else {
        gen.println("{0} result = aie::broadcast<{1}, {2}>(0);",
                    dtype, datatype_to_str(k.type), k.vsize);
        gen.println("constexpr unsigned NUM_LOOPS = {};",
                    num_samples / k.vsize);
        loop_cond = "NUM_LOOPS";
    }
    gen.println<generator::NO_INDENT>("");
    gen.println<generator::INCREASE_AFTER>("for (unsigned i = 0; "
                                           "i < {}; i++) {{", loop_cond);
    if (k.vsize == 0) {
        gen.println("window_readincr(x, c_x);");
        gen.println("window_readincr(y, c_y);");
        gen.println("result += c_x * c_y;");
    } else {
        gen.println("c_x = window_read_v<{}>(x);", k.vsize);
        gen.println("c_y = window_read_v<{}>(y);", k.vsize);
        gen.println("result = aie::add(aie::mul(c_x, c_y), result);");
    }
    gen.println<generator::DECREASE_BEFORE>("}}");
    gen.println("");
    if (k.vsize == 0) {
        gen.println("writeincr(out, result);");
    } else {
        gen.println("writeincr(out, aie::reduce_add_v(result));");
    }
}

std::vector<kernel_arg> get_dot_args() {
    return std::vector<kernel_arg>{{karg_type::input_plio, "x", 1},
                                   {karg_type::input_plio, "y", 1},
                                   {karg_type::output_plio, "out", 0}};
}

void dot_generator::gen_mm2s(generator &gen) {
    unsigned bits = datatype_to_bits(k.type);

    if (x.type != connection_type::host &&
        y.type != connection_type::host) {
        return;
    }

    gen.print<generator::INCREASE_AFTER>("void {}_mm2s(", k.user_name);
    if (x.type == connection_type::host) {
        gen.print("ap_int<{0}> *mem_x, ", bits);
    }
    if (y.type == connection_type::host) {
        gen.print("ap_int<{0}> *mem_y, ", bits);
    }
    gen.print("int size, ");
    if (x.type == connection_type::host) {
        gen.print("hls::stream<qdma_axis<{0}, 0, 0, 0>> &stream_x", bits);
    }
    if (y.type == connection_type::host) {
        if (x.type == connection_type::host) {
            gen.print(", ");
        }
        gen.print("hls::stream<qdma_axis<{0}, 0, 0, 0>> &stream_y", bits);
    }
    gen.println(") {{");

    if (x.type == connection_type::host) {
        gen.println<generator::NO_INDENT>("#pragma HLS interface m_axi "
                                          "port = mem_x offset = slave");
        gen.println<generator::NO_INDENT>("#pragma HLS interface axis "
                                          "port = stream_x");
        gen.println<generator::NO_INDENT>("#pragma HLS INTERFACE s_axilite "
                                          "port = mem_x bundle = control");
    }
    if (y.type == connection_type::host) {
        gen.println<generator::NO_INDENT>("#pragma HLS interface m_axi "
                                          "port = mem_y offset = slave");
        gen.println<generator::NO_INDENT>("#pragma HLS interface axis "
                                          "port = stream_y");
        gen.println<generator::NO_INDENT>("#pragma HLS INTERFACE s_axilite "
                                          "port = mem_y bundle = control");
    }

    gen.println<generator::NO_INDENT>("#pragma HLS INTERFACE s_axilite "
                                          "port = size bundle = control");
    gen.println<generator::NO_INDENT>("#pragma HLS interface s_axilite "
                                      "port = return bundle = control");
    gen.println("");

    gen.println("// Send data over stream");
    gen.println<generator::INCREASE_AFTER>("for (int i = 0; i < size; i++) "
                                            "{{");
    gen.println<generator::NO_INDENT>("#pragma HLS pipeline II = 1");

    if (x.type == connection_type::host) {
        gen.println("qdma_axis<{},0,0,0> x;", bits);
        gen.println("x.data = mem_x[i];");
        gen.println("x.keep_all();");
        gen.println("stream_x.write(x);");
    }

    if (y.type == connection_type::host) {
        gen.println("qdma_axis<{},0,0,0> y;", bits);
        gen.println("y.data = mem_y[i];");
        gen.println("y.keep_all();");
        gen.println("stream_y.write(y);");
    }

    gen.println<generator::DECREASE_BEFORE>("}}");
    gen.println<generator::DECREASE_BEFORE>("}}");
}

void dot_generator::gen_s2mm(generator &gen) {
    unsigned bits = datatype_to_bits(k.type);

    if (out.type != connection_type::host) {
        return;
    }

    gen.println<generator::INCREASE_AFTER>(
        "void {0}_s2mm(ap_int<{1}> *mem,"
        "hls::stream<qdma_axis<{1}, 0, 0, 0>> &stream_out) {{",
        k.user_name, bits);

    gen.println<generator::NO_INDENT>("#pragma HLS interface m_axi "
                                      "port = mem offset = slave");
    gen.println<generator::NO_INDENT>("#pragma HLS interface axis "
                                      "port = stream_out");
    gen.println("");
    gen.println<generator::NO_INDENT>("#pragma HLS INTERFACE s_axilite "
                                      "port = mem bundle = control");
    gen.println<generator::NO_INDENT>("#pragma HLS interface s_axilite "
                                      "port = return bundle = control");
    gen.println("");

    gen.println("// Retrieve data from out stream");
    gen.println("*mem = stream_out.read();", bits);
    gen.println<generator::DECREASE_BEFORE>("}}");
}

void dot_generator::gen_link(generator &gen) {
    if (x.type == connection_type::host ||
        y.type == connection_type::host) {
        gen.println("nk={0}_mm2s:1:{0}_mm2s", k.user_name);
    }
    if (out.type == connection_type::host) {
        gen.println("nk={0}_s2mm:1:{0}_s2mm", k.user_name);
    }
    gen.println("");
    if (x.type == connection_type::host) {
        gen.println("sc={0}_mm2s.stream_x:ai_engine_0.{0}_x", k.user_name);
    }
    if (y.type == connection_type::host) {
        gen.println("sc={0}_mm2s.stream_y:ai_engine_0.{0}_y",
                    k.user_name);
    }
    if (out.type == connection_type::host) {
        gen.println("sc=ai_engine_0.{0}_out:{0}_s2mm.stream_out", k.user_name);
    } else if (out.type == connection_type::kernel) {
        gen.println("sc=ai_engine_0.{}_out:ai_engine_0.{}_{}", k.user_name,
                    out.kernel, out.parameter);
    }
    gen.println("");
    if (x.type == connection_type::host ||
        y.type == connection_type::host) {
        gen.println("slr={0}_mm2s:SLR0", k.user_name);
    }
    if (out.type == connection_type::host) {
        gen.println("slr={0}_s2mm:SLR0", k.user_name);
    }
    gen.println("");
    if (x.type == connection_type::host ||
        y.type == connection_type::host) {
        gen.println("sp={0}_mm2s.m_axi_gmem:MC_NOC0", k.user_name);
    }
    if (out.type == connection_type::host) {
        gen.println("sp={0}_s2mm.m_axi_gmem:MC_NOC0", k.user_name);
    }
}

} // generators
} // codegen
} // aieblas
