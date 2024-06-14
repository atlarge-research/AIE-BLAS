#include <fstream>
#include "aieblas/detail/util.hpp"
#include "aieblas/detail/codegen/generator.hpp"
#include "aieblas/detail/codegen/kernels.hpp"

namespace aieblas {
namespace codegen {
static inline void generate_graph_hdr(generator &gen) {
    gen.println("#include <adf.h>");
    for (const kernel &kernel : gen.get_data().kernels) {
        gen.println("#include \"kernels/{}.hpp\"", kernel.user_name);
    }

    gen.println("");
    gen.println("using namespace adf;");
    gen.println("");

    gen.println<generator::INCREASE_AFTER>("class simpleGraph : public graph "
                                           "{{");
    gen.println<generator::NO_INDENT>("private:");

    for (const kernel &kernel : gen.get_data().kernels) {
        gen.println("kernel {}k;", kernel.user_name);
    }

    gen.println("");
    gen.println<generator::NO_INDENT>("public:");

    for (const kernel &kernel : gen.get_data().kernels) {
        std::unique_ptr<kernel_generator> kernel_gen =
                get_kernel_generator(kernel);
        for (const kernel_arg &arg : kernel_gen->get_kernel_args()) {
            gen.println("{} {}_{};", kernel_arg_type_to_str(arg.type),
                        kernel.user_name, arg.name);
        }
    }

    gen.println("");
    gen.println<generator::INCREASE_AFTER>("simpleGraph() {{");

    for (const kernel &kernel : gen.get_data().kernels) {
        std::unique_ptr<kernel_generator> kernel_gen =
                get_kernel_generator(kernel);
        gen.println("// initialize {}", kernel.user_name);
        for (const kernel_arg &arg : kernel_gen->get_kernel_args()) {
            std::string plio_type;
            switch (kernel.type) {
                case dtype::float32:
                case dtype::int32:
                    plio_type = "plio_32_bits";
                    break;
                case dtype::float64:
                case dtype::int64:
                    plio_type = "plio_64_bits";
                    break;
                default:
                    plio_type = "unknown";
                    break;
            }

            gen.println("{1}_{2} = {0}::create(\"{1}_{2}\", {3}, \"data/"
                        "{1}_{2}.txt\");", kernel_arg_type_to_str(arg.type),
                        kernel.user_name, arg.name, plio_type);
        }

        gen.println("");
        gen.println("{0}k = kernel::create({0});", kernel.user_name);
        gen.println("");
    }

    unsigned net_count = 0;
    for (const kernel &kernel : gen.get_data().kernels) {
        unsigned in_count = 0;
        unsigned out_count = 0;
        std::unique_ptr<kernel_generator> kernel_gen =
                get_kernel_generator(kernel);
        gen.println("// connect {}", kernel.user_name);
        for (const kernel_arg &arg : kernel_gen->get_kernel_args()) {
            std::string type;
            std::string name;
            if (arg.dimensions == 0) {
                type = "stream";
                name = "";
            } else {
                type = std::format("window<{}>", kernel.wsize);
                name = std::format(" net{}", net_count);
                net_count++;
            }

            if (arg.type == karg_type::input_plio) {
                gen.println("connect<{}>{}({}_{}.out[0], {}k.in[{}]);",
                            type, name, kernel.user_name, arg.name,
                            kernel.user_name, in_count);
                in_count++;
            } else {
                gen.println("connect<{}>{}({}k.out[{}], {}_{}.in[0]);",
                            type, name, kernel.user_name, out_count,
                            kernel.user_name, arg.name);
                out_count++;
            }
        }

        gen.println("");
        gen.println("source({0}k) = \"kernels/{0}.cpp\";",
                    kernel.user_name);
        gen.println("runtime<ratio>({}k) = 0.9;", kernel.user_name);
        gen.println("");
    }

    gen.println<generator::DECREASE_BEFORE>("}}");
    gen.println<generator::DECREASE_BEFORE>("}};");
}

static inline void generate_graph_src(generator &gen) {
    gen.println("#include \"graph.hpp\"");
    gen.println("");
    gen.println("simpleGraph mygraph;");
    gen.println("");
    gen.println<generator::INCREASE_AFTER>("int main (void) {{");
    gen.println("adf::return_code ret;");
    gen.println("mygraph.init();");
    gen.println("");
    gen.println("ret = mygraph.run(1);");
    gen.println<generator::INCREASE_AFTER>("if (ret != adf::ok) {{");
    gen.println("printf(\"Run failed\\n\");");
    gen.println("return ret;");
    gen.println<generator::DECREASE_BEFORE>("}}");
    gen.println("");
    gen.println("ret = mygraph.end();");
    gen.println<generator::INCREASE_AFTER>("if (ret != adf::ok) {{");
    gen.println("printf(\"End failed\\n\");");
    gen.println("return ret;");
    gen.println<generator::DECREASE_BEFORE>("}}");
    gen.println("");
    gen.println("return 0;");
    gen.println<generator::DECREASE_BEFORE>("}}");
}

void generator::generate_graph() {
    fs::path aie_dir = out_dir / "aie";
    if (!fs::exists(aie_dir)) {
        log(log_level::error, "aie directory does not exist. "
                              "Call generate_kernels first.");
        throw std::runtime_error("aie directory does not exist");
    }

    fs::path graph_src = aie_dir / "graph.cpp";
    this->open(graph_src);
    generate_graph_src(*this);
    this->close();

    fs::path graph_hdr = aie_dir / "graph.hpp";
    this->open(graph_hdr);
    generate_graph_hdr(*this);
    this->close();
}
} // codegen
} // aieblas
