#include "aieblas/detail/util.hpp"
#include "aieblas/detail/codegen/kernels/scal.hpp"


namespace aieblas {
namespace codegen {
namespace generators {

void scal_generator::gen_kernel_glob(generator &gen) {
    const unsigned num_samples = k.wsize * 8 / datatype_to_bits(k.type);
    gen.println("#define NUM_SAMPLES {}", num_samples);
    gen.println();
    const auto native_type = datatype_to_native_type(k.type, 2);
    gen.print("{0} chess_storage(%chess_alignof({1})) alpha_storage[{2}] = {{",
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

void scal_generator::gen_kernel_args(generator &gen) {
    gen.print("input_window<{0}> *__restrict x, "
              "output_window<{0}> *__restrict out, "
              "input_stream<{0}> *__restrict alpha", datatype_to_str(k.type));
}

void scal_generator::gen_kernel_body(generator &gen) {
    gen.println("{0} *alpha_store = &alpha_storage[0];", datatype_to_str(k.type));
    gen.println("{0} *alpha_set = &alpha_storage[1];", datatype_to_str(k.type));
    gen.println<generator::INCREASE_AFTER>("if (*alpha_set == 0) {{");
    gen.println("*alpha_set = 1;");
    gen.println("*alpha_store = readincr(alpha);");
    gen.println<generator::DECREASE_BEFORE>("}}");
    gen.println();

    gen.println("{0} c_x, c_out;", dtype);
    const char *loop_cond;
    if (k.vsize == 0) {
        gen.println("{0} scalar = *alpha_store;", dtype);
        loop_cond = "NUM_SAMPLES";
    } else {
        gen.println("{0} scalar = aie::broadcast<{1}, {2}>(*alpha_store);",
                    dtype, datatype_to_str(k.type), k.vsize);
        gen.println("constexpr unsigned NUM_LOOPS = NUM_SAMPLES / {};",
                    k.vsize);
        loop_cond = "NUM_LOOPS";
    }
    gen.println();
    gen.println<generator::INCREASE_AFTER>("for (unsigned i = 0; "
                                           "i < {}; i++) {{", loop_cond);
    if (k.vsize == 0) {
        gen.println("c_x = window_readincr(x);");
        gen.println("c_out = c_x * scalar;");
    } else {
        gen.println("c_x = window_readincr_v<{}>(x);", k.vsize);
        gen.println("c_out = aie::mul(c_x, scalar).to_vector<{}>();",
                    datatype_to_str(k.type));
    }
    gen.println("window_writeincr(out, c_out);");
    gen.println<generator::DECREASE_BEFORE>("}}");
}

std::vector<kernel_arg> get_scal_args() {
    return std::vector<kernel_arg>{{karg_type::input, "x", 1},
                                   {karg_type::input, "alpha", 0},
                                   {karg_type::output, "out", 1}};
}

bool scal_generator::need_mm2s() const {
    return x.type == connection_type::host
        || alpha.type == connection_type::host;
}

void scal_generator::gen_mm2s(generator &gen) {
    unsigned bits = datatype_to_bits(k.type);

    if (x.type != connection_type::host &&
        alpha.type != connection_type::host) {
        throw std::runtime_error("Internal error: mm2s not needed");
    }

    gen.print<generator::INCREASE_AFTER>("void {}_mm2s(", k.user_name);
    if (x.type == connection_type::host) {
        gen.print("ap_int<{0}> *mem, int size, "
                  "hls::stream<qdma_axis<{0}, 0, 0, 0>> &stream_x", bits);
    }
    if (alpha.type == connection_type::host) {
        if (x.type == connection_type::host) {
            gen.print(", ");
        }

        gen.print("ap_int<{0}> scalar, "
                  "hls::stream<qdma_axis<{0}, 0, 0, 0>> &stream_alpha", bits);
    }
    gen.println(") {{");

    if (x.type == connection_type::host) {
        gen.println<generator::NO_INDENT>("#pragma HLS interface m_axi "
                                          "port = mem offset = slave");
        gen.println<generator::NO_INDENT>("#pragma HLS interface axis "
                                          "port = stream_x");
        gen.println<generator::NO_INDENT>("#pragma HLS INTERFACE s_axilite "
                                          "port = mem bundle = control");
        gen.println<generator::NO_INDENT>("#pragma HLS INTERFACE s_axilite "
                                          "port = size bundle = control");
    }
    if (alpha.type == connection_type::host) {
        gen.println<generator::NO_INDENT>("#pragma HLS interface axis "
                                          "port = stream_alpha");
        gen.println<generator::NO_INDENT>("#pragma HLS INTERFACE s_axilite "
                                          "port = scalar bundle = control");
    }
    gen.println<generator::NO_INDENT>("#pragma HLS interface s_axilite "
                                      "port = return bundle = control");
    gen.println();

    if (alpha.type == connection_type::host) {
        gen.println("// Send scalar over alpha stream");
        gen.println("qdma_axis<{},0,0,0> qdma_scalar;", bits);
        gen.println("qdma_scalar.data = (ap_int<{}>) scalar;", bits);
        gen.println("qdma_scalar.keep_all();");
        gen.println("qdma_scalar.set_last(1);");
        gen.println("stream_alpha.write(qdma_scalar);");
        gen.println();
    }

    if (x.type == connection_type::host) {
        gen.println("// Send data over x stream");
        gen.println<generator::INCREASE_AFTER>("for (int i = 0; i < size; i++) "
                                               "{{");
        gen.println<generator::NO_INDENT>("#pragma HLS pipeline II = 1");
        gen.println("qdma_axis<{},0,0,0> x;", bits);
        gen.println("x.data = mem[i];");
        gen.println("x.keep_all();");
        gen.println("stream_x.write(x);");
        gen.println<generator::DECREASE_BEFORE>("}}");
    }
    gen.println<generator::DECREASE_BEFORE>("}}");
}

bool scal_generator::need_s2mm() const {
    return out.type == connection_type::host;
}

void scal_generator::gen_s2mm(generator &gen) {
    unsigned bits = datatype_to_bits(k.type);

    if (out.type != connection_type::host) {
        throw std::runtime_error("Internal error: mm2s not needed");
    }

    gen.println<generator::INCREASE_AFTER>(
        "void {0}_s2mm(ap_int<{1}> *mem, int size, "
        "hls::stream<qdma_axis<{1}, 0, 0, 0>> &stream_out) {{",
        k.user_name, bits);

    gen.println<generator::NO_INDENT>("#pragma HLS interface m_axi "
                                      "port = mem offset = slave");
    gen.println<generator::NO_INDENT>("#pragma HLS interface axis "
                                      "port = stream_out");
    gen.println();
    gen.println<generator::NO_INDENT>("#pragma HLS INTERFACE s_axilite "
                                      "port = mem bundle = control");
    gen.println<generator::NO_INDENT>("#pragma HLS INTERFACE s_axilite "
                                      "port = size bundle = control");
    gen.println<generator::NO_INDENT>("#pragma HLS interface s_axilite "
                                      "port = return bundle = control");
    gen.println();

    gen.println("// Retrieve data from out stream");
    gen.println<generator::INCREASE_AFTER>("for (int i = 0; i < size; i++) {{");
    gen.println<generator::NO_INDENT>("#pragma HLS pipeline II = 1");
    gen.println("qdma_axis<{},0,0,0> x = stream_out.read();", bits);
    gen.println("mem[i] = x.data;");
    gen.println<generator::DECREASE_BEFORE>("}}");
    gen.println<generator::DECREASE_BEFORE>("}}");
}

void scal_generator::gen_link(generator &gen) {
    if (x.type == connection_type::host ||
        alpha.type == connection_type::host) {
        gen.println("nk={0}_mm2s:1:{0}_mm2s", k.user_name);
    }
    if (out.type == connection_type::host) {
        gen.println("nk={0}_s2mm:1:{0}_s2mm", k.user_name);
    }
    gen.println();
    if (x.type == connection_type::host) {
        gen.println("sc={0}_mm2s.stream_x:ai_engine_0.{0}_x", k.user_name);
    }
    if (alpha.type == connection_type::host) {
        gen.println("sc={0}_mm2s.stream_alpha:ai_engine_0.{0}_alpha",
                    k.user_name);
    }
    if (out.type == connection_type::host) {
        gen.println("sc=ai_engine_0.{0}_out:{0}_s2mm.stream_out", k.user_name);
    }
    gen.println();
    if (x.type == connection_type::host ||
        alpha.type == connection_type::host) {
        gen.println("slr={0}_mm2s:SLR0", k.user_name);
    }
    if (out.type == connection_type::host) {
        gen.println("slr={0}_s2mm:SLR0", k.user_name);
    }
    gen.println();
    if (x.type == connection_type::host) {
        gen.println("sp={0}_mm2s.m_axi_gmem:MC_NOC0", k.user_name);
    }
    if (out.type == connection_type::host) {
        gen.println("sp={0}_s2mm.m_axi_gmem:MC_NOC0", k.user_name);
    }
}

} // generators
} // codegen
} // aieblas
