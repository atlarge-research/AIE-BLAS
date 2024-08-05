#include "aieblas/detail/util.hpp"
#include "aieblas/detail/codegen/kernels/iamax.hpp"


namespace aieblas {
namespace codegen {
namespace generators {

void iamax_generator::gen_kernel_glob(generator &gen) {
    const unsigned num_samples = k.wsize * 8 / datatype_to_bits(k.type);
    gen.println("#define NUM_SAMPLES {}", num_samples);
    gen.println();
    gen.println("uint64 chess_storage(%chess_alignof(v4int64)) counter[4] = "
                "{{0, 0, 0, 0}};");
    const auto native_type = datatype_to_native_type(k.type, 2);
    gen.print("{0} chess_storage(%chess_alignof({1})) result_storage[{2}] = {{",
              dtype, std::get<1>(native_type),
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

void iamax_generator::gen_kernel_args(generator &gen) {
    gen.print("input_stream<uint64> *__restrict in_size_n, "
              "input_window<{0}> *__restrict x, "
              "output_stream<{0}> *__restrict out", dtype);
}

void iamax_generator::gen_kernel_body(generator &gen) {
    gen.println("uint64 *num_cycles = &counter[0];");
    gen.println("uint64 *cycle = &counter[1];");
    gen.println<generator::INCREASE_AFTER>("if (*num_cycles == 0) {{");
    gen.println("*num_cycles = readincr(in_size_n) / NUM_SAMPLES;");
    gen.println<generator::DECREASE_BEFORE>("}}");
    gen.println("{0} &max = result_storage[0];", dtype);
    gen.println("{0} &index = result_storage[1];", dtype);
    gen.println();
    gen.println("{0} vx;", dtype);
    gen.println<generator::INCREASE_AFTER>("for (unsigned i = 0; "
                                           "i < NUM_SAMPLES; i++) {{");
    gen.println("vx = aie::abs(window_readincr(x));");
    gen.println<generator::INCREASE_AFTER>("if (vx > max) {{;");
    gen.println("max = vx;");
    gen.println("index = *cycle * NUM_SAMPLES + i;");
    gen.println<generator::DECREASE_BEFORE>("}}");
    gen.println<generator::DECREASE_BEFORE>("}}");
    gen.println();
    gen.println("*cycle += 1;");
    gen.println();
    gen.println<generator::INCREASE_AFTER>("if (*cycle == *num_cycles) {{");
    gen.println("writeincr(out, index, true);");
    gen.println<generator::DECREASE_BEFORE>("}}");
}

std::vector<kernel_arg> get_iamax_args() {
    return std::vector<kernel_arg>{{karg_type::input_index, "in_size_n", 0},
                                   {karg_type::input, "x", 1},
                                   {karg_type::output, "out", 0}};
}

bool iamax_generator::need_mm2s() const {
    return true;
}

void iamax_generator::gen_mm2s(generator &gen) {
    unsigned bits = datatype_to_bits(k.type);

    gen.print<generator::INCREASE_AFTER>("void {}_mm2s(", k.user_name);
    if (x.type == connection_type::host) {
        gen.print("ap_int<{0}> *mem, ", bits);
    }
    gen.print("int size, ");
    if (x.type == connection_type::host) {
        gen.print("hls::stream<qdma_axis<{0}, 0, 0, 0>> &stream_x, ", bits);
    }
    gen.print("hls::stream<qdma_axis<64, 0, 0, 0>> &stream_in_size_n");
    gen.println(") {{");

    if (x.type == connection_type::host) {
        gen.println<generator::NO_INDENT>("#pragma HLS interface m_axi "
                                          "port = mem offset = slave");
        gen.println<generator::NO_INDENT>("#pragma HLS interface axis "
                                          "port = stream_x");
        gen.println<generator::NO_INDENT>("#pragma HLS INTERFACE s_axilite "
                                          "port = mem bundle = control");
    }

    gen.println<generator::NO_INDENT>("#pragma HLS interface axis "
                                      "port = stream_in_size_n");
    gen.println<generator::NO_INDENT>("#pragma HLS INTERFACE s_axilite "
                                          "port = size bundle = control");
    gen.println<generator::NO_INDENT>("#pragma HLS interface s_axilite "
                                      "port = return bundle = control");
    gen.println();

    gen.println("qdma_axis<64,0,0,0> in_size;", bits);
    gen.println("in_size.data = size;");
    gen.println("in_size.keep_all();");
    gen.println("in_size.set_last(1);");
    gen.println("stream_in_size_n.write(in_size);");

    gen.println("// Send data over stream");
    gen.println<generator::INCREASE_AFTER>("for (int i = 0; i < size; i++) "
                                            "{{");
    gen.println<generator::NO_INDENT>("#pragma HLS pipeline II = 1");

    if (x.type == connection_type::host) {
        gen.println("qdma_axis<{},0,0,0> x;", bits);
        gen.println("x.data = mem[i];");
        gen.println("x.keep_all();");
        gen.println("stream_x.write(x);");
    }

    gen.println<generator::DECREASE_BEFORE>("}}");
    gen.println<generator::DECREASE_BEFORE>("}}");
}

bool iamax_generator::need_s2mm() const {
    return out.type == connection_type::host;
}

void iamax_generator::gen_s2mm(generator &gen) {
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
    gen.println("qdma_axis<{}, 0, 0, 0> x = stream_out.read();", bits);
    gen.println("*mem = x.data;");
    gen.println<generator::DECREASE_BEFORE>("}}");
}

std::vector<pl_kernel_generator> iamax_generator::get_pl_generators() {
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

void iamax_generator::gen_link(generator &gen) {
    gen.println("nk={0}_mm2s:1:{0}_mm2s", k.user_name);
    if (out.type == connection_type::host) {
        gen.println("nk={0}_s2mm:1:{0}_s2mm", k.user_name);
    }
    gen.println();
    gen.println("sc={0}_mm2s.stream_in_size_n:ai_engine_0.{0}_in_size_n",
                k.user_name);
    if (x.type == connection_type::host) {
        gen.println("sc={0}_mm2s.stream_x:ai_engine_0.{0}_x", k.user_name);
    }
    if (out.type == connection_type::host) {
        gen.println("sc=ai_engine_0.{0}_out:{0}_s2mm.stream_out", k.user_name);
    }
    gen.println();
    gen.println("slr={0}_mm2s:SLR0", k.user_name);
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
