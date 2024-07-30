#include "aieblas/detail/util.hpp"
#include "aieblas/detail/codegen/kernels/gemv.hpp"


namespace aieblas {
namespace codegen {
namespace generators {

void gemv_generator::gen_kernel_glob(generator &gen) {
    const unsigned num_samples = k.wsize * 8 / datatype_to_bits(k.type);
    gen.println("#define NUM_SAMPLES {}", num_samples);
    gen.println();
    gen.println("uint64 chess_storage(%chess_alignof(v4int64)) counter[4] = "
                "{{0, 0, 0, 0}};");
    if (!options.alpha.set || !options.beta.set) {
        const auto native_type = datatype_to_native_type(k.type, 2);
        gen.print("{0} chess_storage(%chess_alignof({1})) scalar[{2}] = {{",
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
}

void gemv_generator::gen_kernel_args(generator &gen) {
    if (!options.alpha.set) {
        gen.print("input_stream<{0}> *__restrict alpha, ",
                  datatype_to_str(k.type));
    }
    gen.print("input_window<{0}> *__restrict A, "
              "input_window<{0}> *__restrict x, ", datatype_to_str(k.type));
    if (!options.beta.set) {
        gen.print("input_stream<{0}> *__restrict beta, ",
                  datatype_to_str(k.type));
    }
    gen.print("input_window<{0}> *__restrict y, "
              "output_window<{0}> *__restrict out", datatype_to_str(k.type));
}

void gemv_generator::gen_kernel_body(generator &gen) {
    gen.println("uint64 *cycle = &counter[0];");
    if (!options.alpha.set) {
        gen.println("{0} *alpha_store = &scalar[0];", datatype_to_str(k.type));
    }
    if (!options.beta.set) {
        gen.println("{0} *beta_store = &scalar[1];", datatype_to_str(k.type));
    }
    if (!options.alpha.set || !options.beta.set) {
        gen.println("{0} *scalar_set = &scalar[2];", datatype_to_str(k.type));
        gen.println<generator::INCREASE_AFTER>("if (*scalar_set == 0) {{");
        gen.println("*scalar_set = 1;");
        if (!options.alpha.set) {
            gen.println("*alpha_store = readincr(alpha);");
        }
        if (!options.beta.set) {
            gen.println("*beta_store = readincr(beta);");
        }
        gen.println<generator::DECREASE_BEFORE>("}}");
        gen.println();
    }

    gen.println<generator::INCREASE_AFTER>("if (*cycle == 0) {{");
    gen.println("window_acquire(x);");
    gen.println<generator::DECREASE_BEFORE>("}}");

    gen.println<generator::INCREASE_AFTER>("if (*cycle % NUM_SAMPLES == 0) {{");
    gen.println("window_acquire(y);");
    gen.println("window_acquire(out);");
    gen.println<generator::DECREASE_BEFORE>("}}");

    gen.println("{0} vx, vA;", dtype);
    const char *loop_cond;
    if (k.vsize == 0) {
        gen.println("{0} vout = 0;", dtype);
        loop_cond = "NUM_SAMPLES";
    } else {
        gen.println("{0} vout = aie::zeros<{1}, {2}>();",
                    dtype, datatype_to_str(k.type), k.vsize);
        gen.println("constexpr unsigned NUM_LOOPS = NUM_SAMPLES / {};",
                    k.vsize);
        loop_cond = "NUM_LOOPS";
    }
    gen.println();
    gen.println<generator::INCREASE_AFTER>("for (unsigned i = 0; "
                                           "i < {}; i++) {{", loop_cond);
    if (k.vsize == 0) {
        gen.println("vA = window_readincr(A);");
        gen.println("vx = window_readincr(x);");
        gen.println("vout += vA * vx;");
    } else {
        gen.println("vA = window_readincr_v<{}>(A);", k.vsize);
        gen.println("vx = window_readincr_v<{}>(x);", k.vsize);
        gen.println("vout = aie::add(vout, aie::mul(vA, vx)"
                    ".to_vector<{}>());", datatype_to_str(k.type));
    }
    gen.println<generator::DECREASE_BEFORE>("}}");
    if (k.vsize == 0) {
        gen.println("window_decr(x, NUM_SAMPLES);");
    } else {
        gen.println("window_decr_v{}(x, NUM_LOOPS);", k.vsize);
    }
    gen.println();
    gen.println("{0} vy = window_readincr(y);", datatype_to_str(k.type));

    std::string alpha_val;
    if (options.alpha.set) {
        alpha_val = options.alpha.to_string();
    } else {
        alpha_val = "*alpha_store";
    }
    std::string beta_val;
    if (options.beta.set) {
        beta_val = options.beta.to_string();
    } else {
        beta_val = "*beta_store";
    }
    if (k.vsize == 0) {
        gen.println("window_writeincr(out, {} * vout + {} * vy);", alpha_val,
                    beta_val);
    } else {
        gen.println("window_writeincr(out, {} * aie::reduce_add(vout) + "
                    "{} * vy);", alpha_val, beta_val);
    }

    gen.println();
    gen.println("*cycle += 1;");
    gen.println();
    gen.println<generator::INCREASE_AFTER>("if (*cycle % NUM_SAMPLES == 0) {{");
    gen.println("window_release(y);");
    gen.println("window_release(out);");
    gen.println<generator::DECREASE_BEFORE>("}}");
}

std::vector<kernel_arg> get_gemv_args() {
    return std::vector<kernel_arg>{{karg_type::input, "alpha", 0},
                                   {karg_type::input, "A", 2},
                                   {karg_type::input, "x", 1},
                                   {karg_type::input, "beta", 0},
                                   {karg_type::input, "y", 1},
                                   {karg_type::output, "out", 1}};
}

void gemv_generator::gen_mm2s_A(generator &gen) {
    unsigned bits = datatype_to_bits(k.type);

    if (A.type != connection_type::host) {
        throw std::runtime_error("Internal error: mm2s_A not needed");
    }

    gen.println<generator::NO_INDENT>("#define n 64");
    gen.println();

    gen.print<generator::INCREASE_AFTER>("void {}_mm2s_A(", k.user_name);
    gen.println("ap_int<{0}> *mem_A, int m, "
                "hls::stream<qdma_axis<{0}, 0, 0, 0>> &stream_A) {{", bits);

    gen.println<generator::NO_INDENT>("#pragma HLS interface m_axi "
                                      "port = mem_A offset = slave");
    gen.println<generator::NO_INDENT>("#pragma HLS interface axis "
                                      "port = stream_A");
    gen.println<generator::NO_INDENT>("#pragma HLS INTERFACE s_axilite "
                                      "port = mem_A bundle = control");
    gen.println<generator::NO_INDENT>("#pragma HLS INTERFACE s_axilite "
                                      "port = m bundle = control");
    gen.println<generator::NO_INDENT>("#pragma HLS interface s_axilite "
                                      "port = return bundle = control");
    gen.println();

    if (x.type == connection_type::host) {
        gen.println("// Send data over stream");
        gen.println("int mn = m * n;");
        gen.println<generator::INCREASE_AFTER>("for (int i = 0; i < mn; i++) "
                                               "{{");
        gen.println<generator::NO_INDENT>("#pragma HLS pipeline II = 1");
        gen.println("qdma_axis<{},0,0,0> A;", bits);
        gen.println("A.data = mem_A[i];");
        gen.println("A.keep_all();");
        gen.println("stream_A.write(A);");
        gen.println<generator::DECREASE_BEFORE>("}}");
    }
    gen.println<generator::DECREASE_BEFORE>("}}");
}

void gemv_generator::gen_mm2s_x(generator &gen) {
    unsigned bits = datatype_to_bits(k.type);

    if (x.type != connection_type::host) {
        throw std::runtime_error("Internal error: mm2s_x not needed");
    }

    gen.println<generator::NO_INDENT>("#define n 64");
    gen.println();

    gen.print<generator::INCREASE_AFTER>("void {}_mm2s_x(", k.user_name);
    gen.println("ap_int<{0}> *mem_x, "
                "hls::stream<qdma_axis<{0}, 0, 0, 0>> &stream_x) {{", bits);

    gen.println<generator::NO_INDENT>("#pragma HLS interface m_axi "
                                      "port = mem_x offset = slave");
    gen.println<generator::NO_INDENT>("#pragma HLS interface axis "
                                      "port = stream_x");
    gen.println<generator::NO_INDENT>("#pragma HLS INTERFACE s_axilite "
                                      "port = mem_x bundle = control");
    gen.println<generator::NO_INDENT>("#pragma HLS interface s_axilite "
                                      "port = return bundle = control");
    gen.println();

    if (x.type == connection_type::host) {
        gen.println("// Send data over stream");
        gen.println<generator::INCREASE_AFTER>("for (int i = 0; i < n; i++) "
                                               "{{");
        gen.println<generator::NO_INDENT>("#pragma HLS unroll");
        gen.println("qdma_axis<{},0,0,0> x;", bits);
        gen.println("x.data = mem_x[i];");
        gen.println("x.keep_all();");
        gen.println("stream_x.write(x);");
        gen.println<generator::DECREASE_BEFORE>("}}");
    }
    gen.println<generator::DECREASE_BEFORE>("}}");
}

void gemv_generator::gen_mm2s_y(generator &gen) {
    unsigned bits = datatype_to_bits(k.type);

    if (y.type != connection_type::host) {
        throw std::runtime_error("Internal error: mm2s_y not needed");
    }

    gen.print<generator::INCREASE_AFTER>("void {}_mm2s_y(", k.user_name);
    gen.println("ap_int<{0}> *mem_y, int m, "
                "hls::stream<qdma_axis<{0}, 0, 0, 0>> &stream_y) {{", bits);

    gen.println<generator::NO_INDENT>("#pragma HLS interface m_axi "
                                      "port = mem_y offset = slave");
    gen.println<generator::NO_INDENT>("#pragma HLS interface axis "
                                      "port = stream_y");
    gen.println<generator::NO_INDENT>("#pragma HLS INTERFACE s_axilite "
                                      "port = mem_y bundle = control");
    gen.println<generator::NO_INDENT>("#pragma HLS INTERFACE s_axilite "
                                      "port = m bundle = control");
    gen.println<generator::NO_INDENT>("#pragma HLS interface s_axilite "
                                      "port = return bundle = control");
    gen.println();

    if (x.type == connection_type::host) {
        gen.println("// Send data over stream");
        gen.println<generator::INCREASE_AFTER>("for (int i = 0; i < m; i++) "
                                               "{{");
        gen.println<generator::NO_INDENT>("#pragma HLS pipeline II = 1");
        gen.println("qdma_axis<{},0,0,0> y;", bits);
        gen.println("y.data = mem_y[i];");
        gen.println("y.keep_all();");
        gen.println("stream_y.write(y);");
        gen.println<generator::DECREASE_BEFORE>("}}");
    }
    gen.println<generator::DECREASE_BEFORE>("}}");
}

void gemv_generator::gen_mm2s_scalar(generator &gen) {
    unsigned bits = datatype_to_bits(k.type);

    if (alpha.type != connection_type::host
        && beta.type != connection_type::host) {
        throw std::runtime_error("Internal error: mm2s_scalar not needed");
    }

    gen.print<generator::INCREASE_AFTER>("void {}_mm2s_scalar(", k.user_name);
    if (alpha.type == connection_type::host) {
        gen.print("ap_int<{0}> alpha, ", bits);
    }
    if (beta.type == connection_type::host) {
        gen.print("ap_int<{0}> beta, ", bits);
    }
    if (alpha.type == connection_type::host) {
        gen.print("hls::stream<qdma_axis<{0}, 0, 0, 0>> &stream_alpha", bits);
        if (beta.type == connection_type::host) {
            gen.print(", ");
        }
    }
    if (beta.type == connection_type::host) {
        gen.print("hls::stream<qdma_axis<{0}, 0, 0, 0>> &stream_beta", bits);
    }

    gen.println(") {{");

    if (alpha.type == connection_type::host) {
        gen.println<generator::NO_INDENT>("#pragma HLS interface axis "
                                          "port = stream_alpha");
        gen.println<generator::NO_INDENT>("#pragma HLS INTERFACE s_axilite "
                                          "port = alpha bundle = control");
    }
    if (beta.type == connection_type::host) {
        gen.println<generator::NO_INDENT>("#pragma HLS interface axis "
                                          "port = stream_beta");
        gen.println<generator::NO_INDENT>("#pragma HLS INTERFACE s_axilite "
                                          "port = beta bundle = control");
    }
    gen.println<generator::NO_INDENT>("#pragma HLS interface s_axilite "
                                      "port = return bundle = control");
    gen.println();

    if (alpha.type == connection_type::host) {
        gen.println("qdma_axis<{},0,0,0> scalar_alpha;", bits);
        gen.println("scalar_alpha.data = alpha;");
        gen.println("scalar_alpha.keep_all();");
        gen.println("scalar_alpha.set_last(1);");
        gen.println("stream_alpha.write(scalar_alpha);");
    }

    if (alpha.type == connection_type::host) {
        gen.println("qdma_axis<{},0,0,0> scalar_beta;", bits);
        gen.println("scalar_beta.data = beta;");
        gen.println("scalar_beta.keep_all();");
        gen.println("scalar_beta.set_last(1);");
        gen.println("stream_beta.write(scalar_beta);");
    }
    gen.println<generator::DECREASE_BEFORE>("}}");
}

void gemv_generator::gen_s2mm(generator &gen) {
    unsigned bits = datatype_to_bits(k.type);

    if (out.type != connection_type::host) {
        throw std::runtime_error("Internal error: mm2s not needed");
    }

    gen.println<generator::INCREASE_AFTER>(
        "void {0}_s2mm(ap_int<{1}> *mem, int m, "
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
                                      "port = m bundle = control");
    gen.println<generator::NO_INDENT>("#pragma HLS interface s_axilite "
                                      "port = return bundle = control");
    gen.println();

    gen.println("// Retrieve data from out stream");
    gen.println<generator::INCREASE_AFTER>("for (int i = 0; i < m; i++) {{");
    gen.println<generator::NO_INDENT>("#pragma HLS pipeline II = 1");
    gen.println("qdma_axis<{},0,0,0> x = stream_out.read();", bits);
    gen.println("mem[i] = x.data;");
    gen.println<generator::DECREASE_BEFORE>("}}");
    gen.println<generator::DECREASE_BEFORE>("}}");
}

std::vector<pl_kernel_generator> gemv_generator::get_pl_generators() {
    std::vector<pl_kernel_generator> generators;
    if (A.type == connection_type::host) {
        generators.emplace_back("mm2s_A", [this](generator &gen) {
                this->gen_mm2s_A(gen);
            });
    }

    if (x.type == connection_type::host) {
        generators.emplace_back("mm2s_x", [this](generator &gen) {
                this->gen_mm2s_x(gen);
            });
    }

    if (y.type == connection_type::host) {
        generators.emplace_back("mm2s_y", [this](generator &gen) {
                this->gen_mm2s_y(gen);
            });
    }

    if (alpha.type == connection_type::host
        || beta.type == connection_type::host) {
        generators.emplace_back("mm2s_scalar", [this](generator &gen) {
                this->gen_mm2s_scalar(gen);
            });
    }

    if (out.type == connection_type::host) {
        generators.emplace_back("s2mm", [this](generator &gen) {
                this->gen_s2mm(gen);
            });
    }

    return generators;
}

void gemv_generator::gen_link(generator &gen) {
    if (A.type == connection_type::host) {
        gen.println("nk={0}_mm2s_A:1:{0}_mm2s_A", k.user_name);
    }
    if (x.type == connection_type::host) {
        gen.println("nk={0}_mm2s_x:1:{0}_mm2s_x", k.user_name);
    }
    if (y.type == connection_type::host) {
        gen.println("nk={0}_mm2s_y:1:{0}_mm2s_y", k.user_name);
    }
    if (alpha.type == connection_type::host
        || beta.type == connection_type::host) {
        gen.println("nk={0}_mm2s_scalar:1:{0}_mm2s_scalar", k.user_name);
    }
    if (out.type == connection_type::host) {
        gen.println("nk={0}_s2mm:1:{0}_s2mm", k.user_name);
    }
    gen.println();
    if (A.type == connection_type::host) {
        gen.println("sc={0}_mm2s_A.stream_A:ai_engine_0.{0}_A", k.user_name);
    }
    if (x.type == connection_type::host) {
        gen.println("sc={0}_mm2s_x.stream_x:ai_engine_0.{0}_x", k.user_name);
    }
    if (y.type == connection_type::host) {
        gen.println("sc={0}_mm2s_y.stream_y:ai_engine_0.{0}_y", k.user_name);
    }
    if (alpha.type == connection_type::host) {
        gen.println("sc={0}_mm2s_scalar.stream_alpha:ai_engine_0.{0}_alpha",
                    k.user_name);
    }
    if (beta.type == connection_type::host) {
        gen.println("sc={0}_mm2s_scalar.stream_beta:ai_engine_0.{0}_beta",
                    k.user_name);
    }
    if (out.type == connection_type::host) {
        gen.println("sc=ai_engine_0.{0}_out:{0}_s2mm.stream_out", k.user_name);
    }
    gen.println();
    if (A.type == connection_type::host) {
        gen.println("slr={0}_mm2s_A:SLR0", k.user_name);
    }
    if (x.type == connection_type::host) {
        gen.println("slr={0}_mm2s_x:SLR0", k.user_name);
    }
    if (y.type == connection_type::host) {
        gen.println("slr={0}_mm2s_y:SLR0", k.user_name);
    }
    if (alpha.type == connection_type::host
        || beta.type == connection_type::host) {
        gen.println("slr={0}_mm2s_scalar:SLR0", k.user_name);
    }
    if (out.type == connection_type::host) {
        gen.println("slr={0}_s2mm:SLR0", k.user_name);
    }
    gen.println();
    if (A.type == connection_type::host) {
        gen.println("sp={0}_mm2s_A.m_axi_gmem:MC_NOC0", k.user_name);
    }
    if (x.type == connection_type::host) {
        gen.println("sp={0}_mm2s_x.m_axi_gmem:MC_NOC0", k.user_name);
    }
    if (y.type == connection_type::host) {
        gen.println("sp={0}_mm2s_y.m_axi_gmem:MC_NOC0", k.user_name);
    }
    if (out.type == connection_type::host) {
        gen.println("sp={0}_s2mm.m_axi_gmem:MC_NOC0", k.user_name);
    }
}

} // generators
} // codegen
} // aieblas
