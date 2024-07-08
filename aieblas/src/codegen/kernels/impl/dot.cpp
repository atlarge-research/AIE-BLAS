#include "aieblas/detail/util.hpp"
#include "aieblas/detail/codegen/kernels/dot.hpp"


namespace aieblas {
namespace codegen {
namespace generators {

void dot_generator::gen_kernel_glob(generator &gen) {
    const unsigned num_samples = k.wsize * 8 / datatype_to_bits(k.type);
    gen.println("#define NUM_SAMPLES {}", num_samples);
    gen.println();
    gen.print("uint64 chess_storage(%chess_alignof(v4int64)) counter[4] = "
              "{{0, 0, 0, 0}}");
    const auto native_type = datatype_to_native_type(k.type, k.vsize);
    gen.print("{0} chess_storage(%chess_alignof({1})) result_storage[{2}] = {{",
              datatype_to_str(k.type), std::get<1>(native_type),
              std::get<0>(native_type));
    for (std::size_t i = 0; i < std::get<0>(native_type); ++i) {
        if (i != 0) {
            gen.print(", 0");
        } else {
            gen.print("0");
        }
    }
    gen.println("}};");
}

void dot_generator::gen_kernel_args(generator &gen) {
    gen.print("input_stream<uint64> *__restrict in_size_n, "
              "input_window<{0}> *__restrict x, "
              "input_window<{0}> *__restrict y, "
              "output_stream<{0}> *__restrict out", datatype_to_str(k.type));
}

void dot_generator::gen_kernel_body(generator &gen) {
    gen.println("uint64 *num_cycles = &counter[0];");
    gen.println("uint64 *cycle = &counter[1];");
    gen.println<generator::INCREASE_AFTER>("if (*num_cycles == 0) {{");
    gen.println("*num_cycles = readincr(in_size_n) / NUM_SAMPLES;");
    gen.println<generator::DECREASE_BEFORE>("}}");
    gen.println("{0} &result = *({0} *)result_storage;", dtype);
    const char *loop_cond;
    if (k.vsize == 0) {
        loop_cond = "NUM_SAMPLES";
    } else {
        gen.println("constexpr unsigned NUM_LOOPS = NUM_SAMPLES / {};",
                    k.vsize);
        loop_cond = "NUM_LOOPS";
    }
    gen.println();
    gen.println("{0} vx, vy;", dtype);
    if (k.vsize > 0) {
        gen.println("aie::accum<{0}, {1}> acc;", datatype_to_accum(k.type),
                    k.vsize);
        gen.println("acc.from_vector(result);");
    }
    gen.println<generator::INCREASE_AFTER>("for (unsigned i = 0; "
                                           "i < {}; i++) {{", loop_cond);
    if (k.vsize == 0) {
        gen.println("vx = window_readincr(x);");
        gen.println("vy = window_readincr(y);");
        gen.println("result += vx * vy;");
    } else {
        gen.println("vx = window_readincr_v<{}>(x);", k.vsize);
        gen.println("vy = window_readincr_v<{}>(y);", k.vsize);
        gen.println("acc = aie::mac(acc, vx, vy);");
    }
    gen.println<generator::DECREASE_BEFORE>("}}");
    gen.println();

    if (k.vsize > 0) {
        gen.println("result = acc.to_vector<{}>();", datatype_to_str(k.type));
    }
    gen.println("*cycle += 1;");
    gen.println();
    gen.println<generator::INCREASE_AFTER>("if (*cycle == *num_cycles) {{");
    if (k.vsize == 0) {
        gen.println("writeincr(out, result);");
    } else {
        gen.println("writeincr(out, aie::reduce_add_v(result));");
    }
    gen.println<generator::DECREASE_BEFORE>("}}");
}

std::vector<kernel_arg> get_dot_args() {
    return std::vector<kernel_arg>{{karg_type::input, "x", 1},
                                   {karg_type::input, "y", 1},
                                   {karg_type::output, "out", 0}};
}

bool dot_generator::need_mm2s() const {
    return x.type == connection_type::host
        || y.type == connection_type::host;
}

void dot_generator::gen_mm2s(generator &gen) {
    unsigned bits = datatype_to_bits(k.type);

    if (x.type != connection_type::host &&
        y.type != connection_type::host) {
        throw std::runtime_error("Internal error: mm2s not needed");
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
    gen.println();

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

bool dot_generator::need_s2mm() const {
    return out.type == connection_type::host;
}

void dot_generator::gen_s2mm(generator &gen) {
    unsigned bits = datatype_to_bits(k.type);

    if (out.type != connection_type::host) {
        throw std::runtime_error("Internal error: mm2s not needed");
    }

    gen.println<generator::INCREASE_AFTER>(
        "void {0}_s2mm(ap_int<{1}> *mem,"
        "hls::stream<qdma_axis<{1}, 0, 0, 0>> &stream_out) {{",
        k.user_name, bits);

    gen.println<generator::NO_INDENT>("#pragma HLS interface m_axi "
                                      "port = mem offset = slave");
    gen.println<generator::NO_INDENT>("#pragma HLS interface axis "
                                      "port = stream_out");
    gen.println();
    gen.println<generator::NO_INDENT>("#pragma HLS INTERFACE s_axilite "
                                      "port = mem bundle = control");
    gen.println<generator::NO_INDENT>("#pragma HLS interface s_axilite "
                                      "port = return bundle = control");
    gen.println();

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
    gen.println();
    if (x.type == connection_type::host) {
        gen.println("sc={0}_mm2s.stream_x:ai_engine_0.{0}_x", k.user_name);
    }
    if (y.type == connection_type::host) {
        gen.println("sc={0}_mm2s.stream_y:ai_engine_0.{0}_y",
                    k.user_name);
    }
    if (out.type == connection_type::host) {
        gen.println("sc=ai_engine_0.{0}_out:{0}_s2mm.stream_out", k.user_name);
    }
    gen.println();
    if (x.type == connection_type::host ||
        y.type == connection_type::host) {
        gen.println("slr={0}_mm2s:SLR0", k.user_name);
    }
    if (out.type == connection_type::host) {
        gen.println("slr={0}_s2mm:SLR0", k.user_name);
    }
    gen.println();
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
