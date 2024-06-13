#include "aieblas/detail/util.hpp"
#include "aieblas/detail/codegen/kernels/scale.hpp"


namespace aieblas {
namespace codegen {
namespace generators {

void scale_generator::gen_kernel_args(generator &gen) {
    gen.print("input_window<{0}> *in, output_window<{0}> *out, "
              "input_stream<{0}> *ctrl", datatype_to_str(k.type));
}

void scale_generator::gen_kernel_body(generator &gen) {
    gen.println("{0} c_in, c_out;", dtype);
    const char *loop_cond;
    if (k.vsize == 0) {
        gen.println("{0} scalar = readincr(ctrl);", dtype);
        loop_cond = "NUM_SAMPLES";
    } else {
        gen.println("{0} scalar = aie::broadcast<{1}, {2}>(readincr(ctrl));",
                    dtype, datatype_to_str(k.type), k.vsize);
        gen.println("constexpr unsigned NUM_LOOPS = {};",
                    num_samples / k.vsize);
        loop_cond = "NUM_LOOPS";
    }
    gen.println<generator::NO_INDENT>("");
    gen.println<generator::INCREASE_AFTER>("for (unsigned i = 0; "
                                           "i < {}; i++) {{", loop_cond);
    if (k.vsize == 0) {
        gen.println("window_readincr(in, c_in);");
        gen.println("c_out = c_in * scalar;");
    } else {
        gen.println("c_in = window_read_v<{}>(in);", k.vsize);
        gen.println("c_out = aie::mul(c_in, scalar).to_vector<{}>();",
                    datatype_to_str(k.type));
    }
    gen.println("window_writeincr(out, c_out);");
    gen.println<generator::DECREASE_BEFORE>("}}");
}

std::vector<kernel_arg> get_scale_args() {
    return std::vector<kernel_arg>{{karg_type::input_plio, "in", 128},
                                   {karg_type::input_plio, "ctrl", 0},
                                   {karg_type::output_plio, "out", 128}};
}

void scale_generator::gen_mm2s(generator &gen) {
    unsigned bits;
    switch (k.type) {
        case dtype::float32:
        case dtype::int32:
            bits = 32;
            break;
        case dtype::float64:
        case dtype::int64:
            bits = 64;
            break;
        default:
            bits = 0;
            break;
    }

    if (in.type != connection_type::host &&
        ctrl.type != connection_type::host) {
        return;
    }

    gen.print<generator::INCREASE_AFTER>("void {}_mm2s(", k.user_name);
    if (in.type == connection_type::host) {
        gen.print("ap_int<{0}> *mem, int size, "
                  "hls::stream<qdma_axis<{0}, 0, 0, 0>> &stream_in", bits);
    }
    if (ctrl.type == connection_type::host) {
        if (in.type == connection_type::host) {
            gen.print(", ");
        }

        gen.print("int scalar, "
                  "hls::stream<qdma_axis<{}, 0, 0, 0>> &stream_ctrl", bits);
    }
    gen.println(") {{");

    if (in.type == connection_type::host) {
        gen.println<generator::NO_INDENT>("#pragma HLS interface m_axi "
                                          "port = mem offset = slave");
        gen.println<generator::NO_INDENT>("#pragma HLS interface axis "
                                          "port = stream_in");
        gen.println<generator::NO_INDENT>("#pragma HLS INTERFACE s_axilite "
                                          "port = mem bundle = control");
        gen.println<generator::NO_INDENT>("#pragma HLS INTERFACE s_axilite "
                                          "port = size bundle = control");
    }
    if (ctrl.type == connection_type::host) {
        gen.println<generator::NO_INDENT>("#pragma HLS interface axis "
                                          "port = stream_ctrl");
        gen.println<generator::NO_INDENT>("#pragma HLS INTERFACE s_axilite "
                                          "port = scalar bundle = control");
    }
    gen.println<generator::NO_INDENT>("#pragma HLS interface s_axilite "
                                      "port = return bundle = control");
    gen.println("");

    if (ctrl.type == connection_type::host) {
        gen.println("// Send scalar over ctrl stream");
        gen.println("qdma_axis<{},0,0,0> qdma_scalar;", bits);
        gen.println("qdma_scalar.data = (ap_int<{}>) scalar;", bits);
        gen.println("qdma_scalar.keep_all();");
        gen.println("qdma_scalar.set_last(1);");
        gen.println("stream_ctrl.write(qdma_scalar);");
        gen.println("");
    }

    if (in.type == connection_type::host) {
        gen.println("// Send data over in stream");
        gen.println<generator::INCREASE_AFTER>("for (int i = 0; i < size; i++) "
                                               "{{");
        gen.println<generator::NO_INDENT>("#pragma HLS pipeline II = 1");
        gen.println("qdma_axis<{},0,0,0> x;", bits);
        gen.println("x.data = mem[i];");
        gen.println("x.keep_all();");
        gen.println("stream_in.write(x);");
        gen.println<generator::DECREASE_BEFORE>("}}");
    }
    gen.println<generator::DECREASE_BEFORE>("}}");
}

void scale_generator::gen_s2mm(generator &gen) {
    unsigned bits;
    switch (k.type) {
        case dtype::float32:
        case dtype::int32:
            bits = 32;
            break;
        case dtype::float64:
        case dtype::int64:
            bits = 64;
            break;
        default:
            bits = 0;
            break;
    }

    if (out.type != connection_type::host) {
        return;
    }

    gen.println<generator::INCREASE_AFTER>(
        "void {0}_s2mm(ap_int<{1}> *mem, int size, "
        "hls::stream<qdma_axis<{1}, 0, 0, 0>> &stream_out) {{",
        k.user_name, bits);

    gen.println<generator::NO_INDENT>("#pragma HLS interface m_axi "
                                      "port = mem offset = slave");
    gen.println<generator::NO_INDENT>("#pragma HLS interface axis "
                                      "port = stream_out");
    gen.println("");
    gen.println<generator::NO_INDENT>("#pragma HLS INTERFACE s_axilite "
                                      "port = mem bundle = control");
    gen.println<generator::NO_INDENT>("#pragma HLS INTERFACE s_axilite "
                                      "port = size bundle = control");
    gen.println<generator::NO_INDENT>("#pragma HLS interface s_axilite "
                                      "port = return bundle = control");
    gen.println("");

    gen.println("// Retrieve data from out stream");
    gen.println<generator::INCREASE_AFTER>("for (int i = 0; i < size; i++) {{");
    gen.println<generator::NO_INDENT>("#pragma HLS pipeline II = 1");
    gen.println("qdma_axis<{},0,0,0> x = stream_out.read();", bits);
    gen.println("mem[i] = x.data;");
    gen.println<generator::DECREASE_BEFORE>("}}");
    gen.println<generator::DECREASE_BEFORE>("}}");
}

void scale_generator::gen_link(generator &gen) {
    if (in.type == connection_type::host ||
        ctrl.type == connection_type::host) {
        gen.println("nk={0}_mm2s:1:{0}_mm2s", k.user_name);
    }
    if (out.type == connection_type::host) {
        gen.println("nk={0}_s2mm:1:{0}_s2mm", k.user_name);
    }
    gen.println("");
    if (in.type == connection_type::host) {
        gen.println("sc={0}_mm2s.stream_in:ai_engine_0.{0}_in", k.user_name);
    }
    if (ctrl.type == connection_type::host) {
        gen.println("sc={0}_mm2s.stream_ctrl:ai_engine_0.{0}_ctrl",
                    k.user_name);
    }
    if (out.type == connection_type::host) {
        gen.println("sc=ai_engine_0.{0}_out:{0}_s2mm.stream_out", k.user_name);
    } else if (out.type == connection_type::kernel) {
        gen.println("sc=ai_engine_0.{}_out:ai_engine_0.{}_{}", k.user_name,
                    out.kernel, out.parameter);
    }
    gen.println("");
    if (in.type == connection_type::host ||
        ctrl.type == connection_type::host) {
        gen.println("slr={0}_mm2s:SLR0", k.user_name);
    }
    if (out.type == connection_type::host) {
        gen.println("slr={0}_s2mm:SLR0", k.user_name);
    }
    gen.println("");
    if (in.type == connection_type::host ||
        ctrl.type == connection_type::host) {
        gen.println("sp={0}_mm2s.m_axi_gmem:MC_NOC0", k.user_name);
    }
    if (out.type == connection_type::host) {
        gen.println("sp={0}_s2mm.m_axi_gmem:MC_NOC0", k.user_name);
    }
}

} // generators
} // codegen
} // aieblas
