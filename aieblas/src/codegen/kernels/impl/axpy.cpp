#include "aieblas/detail/util.hpp"
#include "aieblas/detail/codegen/kernels/axpy.hpp"


namespace aieblas {
namespace codegen {
namespace generators {

void axpy_generator::gen_kernel_glob(generator &gen) {
    const unsigned num_samples = k.wsize * 8 / datatype_to_bits(k.type);
    gen.println("#define NUM_SAMPLES {}", num_samples);
    if (!options.alpha.set) {
        gen.println();
        const auto native_type = datatype_to_native_type(k.type, 2);
        gen.print("{0} chess_storage(%chess_alignof({1})) alpha_storage[{2}] = "
                  "{{", datatype_to_str(k.type), std::get<1>(native_type),
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
}

void axpy_generator::gen_kernel_args(generator &gen) {
    if (!options.alpha.set) {
        gen.print("input_stream<{0}> *__restrict alpha, ",
                  datatype_to_str(k.type));
    }
    gen.print("input_window<{0}> *__restrict x, "
              "input_window<{0}> *__restrict y, "
              "output_window<{0}> *__restrict out", datatype_to_str(k.type));
}

void axpy_generator::gen_kernel_body(generator &gen) {
    if (!options.alpha.set) {
        gen.println("{0} *alpha_store = &alpha_storage[0];",
                    datatype_to_str(k.type));
        gen.println("{0} *alpha_set = &alpha_storage[1];",
                    datatype_to_str(k.type));
        gen.println<generator::INCREASE_AFTER>("if (*alpha_set == 0) {{");
        gen.println("*alpha_set = 1;");
        gen.println("*alpha_store = readincr(alpha);");
        gen.println<generator::DECREASE_BEFORE>("}}");
        gen.println();
    }

    gen.println("{0} vx, vy, vout;", dtype);
    const char *loop_cond;
    if (k.vsize == 0) {
        if (options.alpha.set) {
            gen.println("{0} scalar = {1}", dtype, options.alpha.to_string());
        } else {
            gen.println("{0} scalar = *alpha_store;", dtype);
        }
        loop_cond = "NUM_SAMPLES";
    } else {
        if (options.alpha.set) {
            gen.println("{0} scalar = aie::broadcast<{1}, {2}>({3});",
                        dtype, datatype_to_str(k.type), k.vsize,
                        options.alpha.to_string());
        } else {
            gen.println("{0} scalar = aie::broadcast<{1}, {2}>(*alpha_store);",
                        dtype, datatype_to_str(k.type), k.vsize);
        }
        gen.println("constexpr unsigned NUM_LOOPS = NUM_SAMPLES / {};",
                    k.vsize);
        loop_cond = "NUM_LOOPS";
    }
    gen.println();
    gen.println<generator::INCREASE_AFTER>("for (unsigned i = 0; "
                                           "i < {}; i++) {{", loop_cond);
    if (k.vsize == 0) {
        gen.println("vx = window_readincr(x);");
        gen.println("vy = window_readincr(y);");
        gen.println("vout = scalar * vx + vy;");
    } else {
        gen.println("vx = window_readincr_v<{}>(x);", k.vsize);
        gen.println("vy = window_readincr_v<{}>(y);", k.vsize);
        gen.println("vout = aie::add(aie::mul(scalar, vx).to_vector<{}>(), "
                    "vy);", datatype_to_str(k.type));
    }
    gen.println("window_writeincr(out, vout);");
    gen.println<generator::DECREASE_BEFORE>("}}");
}

std::vector<kernel_arg> get_axpy_args() {
    return std::vector<kernel_arg>{{karg_type::input, "alpha", 0},
                                   {karg_type::input, "x", 1},
                                   {karg_type::input, "y", 1},
                                   {karg_type::output, "out", 1}};
}

bool axpy_generator::need_mm2s() const {
    return x.type == connection_type::host
        || y.type == connection_type::host
        || alpha.type == connection_type::host;
}

void axpy_generator::gen_mm2s(generator &gen) {
    unsigned bits = datatype_to_bits(k.type);

    if (x.type != connection_type::host &&
        y.type != connection_type::host &&
        alpha.type != connection_type::host) {
        throw std::runtime_error("Internal error: mm2s not needed");
    }

    gen.print<generator::INCREASE_AFTER>("void {}_mm2s(", k.user_name);
    if (x.type == connection_type::host || y.type == connection_type::host) {
        gen.print("int size, ");
    }
    if (x.type == connection_type::host) {
        gen.print("ap_int<{0}> *mem_x, ", bits);
    }
    if (y.type == connection_type::host) {
        gen.print("ap_int<{0}> *mem_y, ", bits);
    }
    if (alpha.type == connection_type::host) {
        gen.print("ap_int<{0}> scalar, ", bits);
    }
    if (x.type == connection_type::host) {
        gen.print("hls::stream<qdma_axis<{0}, 0, 0, 0>> &stream_x", bits);
        if (y.type == connection_type::host
            || alpha.type == connection_type::host) {
                gen.print(", ");
        }
    }
    if (y.type == connection_type::host) {
        gen.print("hls::stream<qdma_axis<{0}, 0, 0, 0>> &stream_y", bits);
        if (alpha.type == connection_type::host) {
                gen.print(", ");
        }
    }
    if (alpha.type == connection_type::host) {
        gen.print("hls::stream<qdma_axis<{0}, 0, 0, 0>> &stream_alpha", bits);
    }
    gen.println(") {{");

    if (x.type == connection_type::host || y.type == connection_type::host) {
        gen.println<generator::NO_INDENT>("#pragma HLS INTERFACE s_axilite "
                                          "port = size bundle = control");
    }
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

    if (x.type == connection_type::host || y.type == connection_type::host) {
        gen.println<generator::INCREASE_AFTER>("for (int i = 0; i < size; i++) "
                                               "{{");
        gen.println<generator::NO_INDENT>("#pragma HLS pipeline II = 1");
        if (x.type == connection_type::host) {
            gen.println("// Send data over x stream");
            gen.println("qdma_axis<{},0,0,0> x;", bits);
            gen.println("x.data = mem_x[i];");
            gen.println("x.keep_all();");
            gen.println("stream_x.write(x);");
        }
        if (y.type == connection_type::host) {
            gen.println("// Send data over y stream");
            gen.println("qdma_axis<{},0,0,0> y;", bits);
            gen.println("y.data = mem_y[i];");
            gen.println("y.keep_all();");
            gen.println("stream_y.write(y);");
        }
        gen.println<generator::DECREASE_BEFORE>("}}");
    }
    gen.println<generator::DECREASE_BEFORE>("}}");
}

bool axpy_generator::need_s2mm() const {
    return out.type == connection_type::host;
}

void axpy_generator::gen_s2mm(generator &gen) {
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

std::vector<pl_kernel_generator> axpy_generator::get_pl_generators() {
    std::vector<pl_kernel_generator> generators;
    if (need_mm2s()) {
        generators.emplace_back("mm2s", [this](generator &gen) {
                this->gen_mm2s(gen);
            });
    }

    if (need_s2mm()) {
        generators.emplace_back("s2mm", [this](generator &gen) {
                this->gen_s2mm(gen);
            });
    }

    return generators;
}

void axpy_generator::gen_link(generator &gen) {
    if (x.type == connection_type::host ||
        y.type == connection_type::host ||
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
    if (y.type == connection_type::host) {
        gen.println("sc={0}_mm2s.stream_y:ai_engine_0.{0}_y", k.user_name);
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
        y.type == connection_type::host ||
        alpha.type == connection_type::host) {
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
