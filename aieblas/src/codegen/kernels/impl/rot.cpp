#include "aieblas/detail/util.hpp"
#include "aieblas/detail/codegen/kernels/rot.hpp"


namespace aieblas {
namespace codegen {
namespace generators {

void rot_generator::gen_kernel_glob(generator &gen) {
    const unsigned num_samples = k.wsize * 8 / datatype_to_bits(k.type);
    gen.println("#define NUM_SAMPLES {}", num_samples);
    if (!options.c.set || !options.s.set) {
        gen.println();
        const auto native_type = datatype_to_native_type(k.type, 3);
        gen.print("{0} chess_storage(%chess_alignof({1})) rot_storage[{2}] = "
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

void rot_generator::gen_kernel_args(generator &gen) {
    gen.print("input_window<{0}> *__restrict x, "
              "input_window<{0}> *__restrict y, "
              "output_window<{0}> *__restrict out_x, "
              "output_window<{0}> *__restrict out_y",
              datatype_to_str(k.type));

    if (!options.c.set) {
        gen.print(", input_stream<{0}> *__restrict c", datatype_to_str(k.type));
    }

    if (!options.s.set) {
        gen.print(", input_stream<{0}> *__restrict s", datatype_to_str(k.type));
    }
}

void rot_generator::gen_kernel_body(generator &gen) {
    if (!options.c.set) {
        gen.println("{0} *c_store = &rot_storage[0];", datatype_to_str(k.type));
    }
    if (!options.s.set) {
        gen.println("{0} *s_store = &rot_storage[0];", datatype_to_str(k.type));
    }
    if (!options.c.set || !options.s.set) {
        gen.println("{0} *rot_set = &rot_storage[1];", datatype_to_str(k.type));
        gen.println<generator::INCREASE_AFTER>("if (*rot_set == 0) {{");
        gen.println("*rot_set = 1;");
        if (!options.c.set) {
            gen.println("*c_store = readincr(c);");
        }
        if (!options.s.set) {
            gen.println("*s_store = readincr(s);");
        }
        gen.println<generator::DECREASE_BEFORE>("}}");
        gen.println();
    }
    gen.println("{0} vx, vy, vout_x, vout_y;", dtype);
    const char *loop_cond;
    if (k.vsize == 0) {
        if (options.c.set) {
            gen.println("{0} scalar_c = {1};", dtype, options.c.to_string());
        } else {
            gen.println("{0} scalar_c = *c_store;", dtype);
        }
        if (options.s.set) {
            gen.println("{0} scalar_s = {1};", dtype, options.s.to_string());
        } else {
            gen.println("{0} scalar_s = *s_store;", dtype);
        }
        loop_cond = "NUM_SAMPLES";
    } else {
        if (options.c.set) {
            gen.println("{0} scalar_c = aie::broadcast<{1}, {2}>({3});",
                        dtype, datatype_to_str(k.type), k.vsize,
                        options.c.to_string());
        } else {
            gen.println("{0} scalar_c = aie::broadcast<{1}, {2}>(*c_store);",
                        dtype, datatype_to_str(k.type), k.vsize);
        }

        if (options.s.set) {
            gen.println("{0} scalar_s = aie::broadcast<{1}, {2}>({3});",
                        dtype, datatype_to_str(k.type), k.vsize,
                        options.s.to_string());
        } else {
            gen.println("{0} scalar_s = aie::broadcast<{1}, {2}>(*s_store);",
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
        gen.println("vout_x = scalar_c * vx + scalar_s * vy;");
        gen.println("vout_y = scalar_c * vy - scalar_s * vx;");
    } else {
        gen.println("vx = window_readincr_v<{}>(x);", k.vsize);
        gen.println("vy = window_readincr_v<{}>(y);", k.vsize);
        gen.println("// out_x = c * x + s * y");
        gen.println("vout_x = aie::mac(aie::mul(scalar_c, vx), scalar_s, vy)"
                    ".to_vector<{}>();", datatype_to_str(k.type));
        gen.println("// out_y = c * y - s * x");
        gen.println("vout_y = aie::msc(aie::mul(scalar_c, vy), scalar_s, vx)"
                    ".to_vector<{}>();", datatype_to_str(k.type));
    }
    gen.println("window_writeincr(out_x, vout_x);");
    gen.println("window_writeincr(out_y, vout_y);");
    gen.println<generator::DECREASE_BEFORE>("}}");
}

std::vector<kernel_arg> get_rot_args() {
    return std::vector<kernel_arg>{{karg_type::input, "x", 1},
                                   {karg_type::input, "y", 1},
                                   {karg_type::output, "out_x", 1},
                                   {karg_type::output, "out_y", 1},
                                   {karg_type::input, "c", 0},
                                   {karg_type::input, "s", 0}};
}

bool rot_generator::need_mm2s() const {
    return x.type == connection_type::host
        || y.type == connection_type::host
        || c.type == connection_type::host
        || s.type == connection_type::host;
}

void rot_generator::gen_mm2s(generator &gen) {
    unsigned bits = datatype_to_bits(k.type);

    if (x.type != connection_type::host &&
        y.type != connection_type::host &&
        c.type != connection_type::host &&
        s.type != connection_type::host) {
        throw std::runtime_error("Internal error: mm2s not needed");
    }

    gen.print<generator::INCREASE_AFTER>("void {}_mm2s(", k.user_name);
    if (x.type == connection_type::host) {
        gen.print("ap_int<{0}> *mem_x, ", bits);
    }
    if (y.type == connection_type::host) {
        gen.print("ap_int<{0}> *mem_y, ", bits);
    }
    if (c.type == connection_type::host) {
        gen.print("ap_int<{0}> scalar_c, ", bits);
    }
    if (s.type == connection_type::host) {
        gen.print("ap_int<{0}> scalar_s, ", bits);
    }
    gen.print("int size");
    if (x.type == connection_type::host) {
        gen.print(", hls::stream<qdma_axis<{0}, 0, 0, 0>> &stream_x", bits);
    }
    if (y.type == connection_type::host) {
        gen.print(", hls::stream<qdma_axis<{0}, 0, 0, 0>> &stream_y", bits);
    }
    if (c.type == connection_type::host) {
        gen.print(", hls::stream<qdma_axis<{0}, 0, 0, 0>> &stream_c", bits);
    }
    if (s.type == connection_type::host) {
        gen.print(", hls::stream<qdma_axis<{0}, 0, 0, 0>> &stream_s", bits);
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
    if (c.type == connection_type::host) {
        gen.println<generator::NO_INDENT>("#pragma HLS interface axis "
                                          "port = stream_c");
        gen.println<generator::NO_INDENT>("#pragma HLS INTERFACE s_axilite "
                                          "port = scalar_c bundle = control");
    }
    if (s.type == connection_type::host) {
        gen.println<generator::NO_INDENT>("#pragma HLS interface axis "
                                          "port = stream_s");
        gen.println<generator::NO_INDENT>("#pragma HLS INTERFACE s_axilite "
                                          "port = scalar_s bundle = control");
    }

    gen.println<generator::NO_INDENT>("#pragma HLS INTERFACE s_axilite "
                                          "port = size bundle = control");
    gen.println<generator::NO_INDENT>("#pragma HLS interface s_axilite "
                                      "port = return bundle = control");
    gen.println();


    if (c.type == connection_type::host) {
        gen.println("// Send scalar over c stream");
        gen.println("qdma_axis<{},0,0,0> qdma_scalar_c;", bits);
        gen.println("qdma_scalar_c.data = (ap_int<{}>) scalar_c;", bits);
        gen.println("qdma_scalar_c.keep_all();");
        gen.println("qdma_scalar_c.set_last(1);");
        gen.println("stream_c.write(qdma_scalar_c);");
        gen.println();
    }

    if (s.type == connection_type::host) {
        gen.println("// Send scalar over c stream");
        gen.println("qdma_axis<{},0,0,0> qdma_scalar_s;", bits);
        gen.println("qdma_scalar_s.data = (ap_int<{}>) scalar_s;", bits);
        gen.println("qdma_scalar_s.keep_all();");
        gen.println("qdma_scalar_s.set_last(1);");
        gen.println("stream_s.write(qdma_scalar_s);");
        gen.println();
    }

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

bool rot_generator::need_s2mm() const {
    return out_x.type == connection_type::host
        || out_y.type == connection_type::host;
}

void rot_generator::gen_s2mm(generator &gen) {
    unsigned bits = datatype_to_bits(k.type);

    if (out_x.type != connection_type::host
        && out_y.type != connection_type::host) {
        throw std::runtime_error("Internal error: s2mm not needed");
    }

    gen.print<generator::INCREASE_AFTER>("void {}_s2mm(", k.user_name);
    if (out_x.type == connection_type::host) {
        gen.print("ap_int<{0}> *mem_out_x, ", bits);
    }
    if (out_y.type == connection_type::host) {
        gen.print("ap_int<{0}> *mem_out_y, ", bits);
    }
    gen.print("int size");
    if (out_x.type == connection_type::host) {
        gen.print(", hls::stream<qdma_axis<{0}, 0, 0, 0>> &stream_out_x", bits);
    }
    if (out_y.type == connection_type::host) {
        gen.print(", hls::stream<qdma_axis<{0}, 0, 0, 0>> &stream_out_y", bits);
    }
    gen.println(") {{");

    if (out_x.type == connection_type::host) {
        gen.println<generator::NO_INDENT>("#pragma HLS interface m_axi "
                                        "port = mem_out_x offset = slave");
        gen.println<generator::NO_INDENT>("#pragma HLS interface axis "
                                        "port = stream_out_x");
        gen.println<generator::NO_INDENT>("#pragma HLS INTERFACE s_axilite "
                                        "port = mem_out_x bundle = control");
    }
    if (out_y.type == connection_type::host) {
        gen.println<generator::NO_INDENT>("#pragma HLS interface m_axi "
                                        "port = mem_out_y offset = slave");
        gen.println<generator::NO_INDENT>("#pragma HLS interface axis "
                                        "port = stream_out_y");
        gen.println<generator::NO_INDENT>("#pragma HLS INTERFACE s_axilite "
                                        "port = mem_out_y bundle = control");
    }
    gen.println<generator::NO_INDENT>("#pragma HLS INTERFACE s_axilite "
                                      "port = size bundle = control");
    gen.println();
    gen.println<generator::NO_INDENT>("#pragma HLS interface s_axilite "
                                      "port = return bundle = control");
    gen.println();

    gen.println("// Retrieve data from out stream");
    gen.println<generator::INCREASE_AFTER>("for (int i = 0; i < size; i++) {{");
    gen.println<generator::NO_INDENT>("#pragma HLS pipeline II = 1");
    if (out_x.type == connection_type::host) {
        gen.println("qdma_axis<{},0,0,0> x = stream_out_x.read();", bits);
        gen.println("mem_out_x[i] = x.data;");
    }
    if (out_y.type == connection_type::host) {
        gen.println("qdma_axis<{},0,0,0> y = stream_out_y.read();", bits);
        gen.println("mem_out_y[i] = y.data;");
    }
    gen.println<generator::DECREASE_BEFORE>("}}");
    gen.println<generator::DECREASE_BEFORE>("}}");
}

std::vector<pl_kernel_generator> rot_generator::get_pl_generators() {
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

void rot_generator::gen_link(generator &gen) {
    if (x.type == connection_type::host ||
        y.type == connection_type::host) {
        gen.println("nk={0}_mm2s:1:{0}_mm2s", k.user_name);
    }
    if (out_x.type == connection_type::host ||
        out_y.type == connection_type::host) {
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
    if (c.type == connection_type::host) {
        gen.println("sc={0}_mm2s.stream_c:ai_engine_0.{0}_c", k.user_name);
    }
    if (s.type == connection_type::host) {
        gen.println("sc={0}_mm2s.stream_s:ai_engine_0.{0}_s",
                    k.user_name);
    }
    if (out_x.type == connection_type::host) {
        gen.println("sc=ai_engine_0.{0}_out_x:{0}_s2mm.stream_out_x",
                    k.user_name);
    }
    if (out_y.type == connection_type::host) {
        gen.println("sc=ai_engine_0.{0}_out_y:{0}_s2mm.stream_out_y",
                    k.user_name);
    }
    gen.println();
    if (x.type == connection_type::host ||
        y.type == connection_type::host) {
        gen.println("slr={0}_mm2s:SLR0", k.user_name);
    }
    if (out_x.type == connection_type::host ||
        out_y.type == connection_type::host) {
        gen.println("slr={0}_s2mm:SLR0", k.user_name);
    }
    gen.println();
    if (x.type == connection_type::host ||
        y.type == connection_type::host) {
        gen.println("sp={0}_mm2s.m_axi_gmem:MC_NOC0", k.user_name);
    }
    if (out_x.type == connection_type::host ||
        out_y.type == connection_type::host) {
        gen.println("sp={0}_s2mm.m_axi_gmem:MC_NOC0", k.user_name);
    }
}

} // generators
} // codegen
} // aieblas
