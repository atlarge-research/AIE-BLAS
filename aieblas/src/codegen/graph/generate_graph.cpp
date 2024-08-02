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

    gen.println();
    gen.println("using namespace adf;");
    gen.println();

    gen.println<generator::INCREASE_AFTER>("class simpleGraph : public graph "
                                           "{{");
    gen.println<generator::NO_INDENT>("private:");

    for (const kernel &kernel : gen.get_data().kernels) {
        gen.println("kernel {}k;", kernel.user_name);
    }

    gen.println();
    gen.println<generator::NO_INDENT>("public:");

    for (const kernel &kernel : gen.get_data().kernels) {
        std::unique_ptr<kernel_generator> kernel_gen =
                get_kernel_generator(kernel);
        for (const kernel_arg &arg : kernel_gen->get_kernel_args()) {
            if (kernel.connections.at(arg.name).type == connection_type::host) {
                gen.println("{} {}_{};", kernel_arg_type_to_str(arg.type),
                            kernel.user_name, arg.name);
            }
        }
    }

    gen.println();
    gen.println<generator::INCREASE_AFTER>("simpleGraph() {{");

    for (const kernel &kernel : gen.get_data().kernels) {
        std::unique_ptr<kernel_generator> kernel_gen =
                get_kernel_generator(kernel);
        gen.println("// initialize {}", kernel.user_name);
        for (const kernel_arg &arg : kernel_gen->get_kernel_args()) {
            if (kernel.connections.at(arg.name).type == connection_type::host) {
                std::string plio_type;
                // Manual override to keep index variables 64 bits
                if (arg.type == karg_type::input_index) {
                    plio_type = "plio_64_bits";
                } else {
                    plio_type= std::format("plio_{}_bits",
                                           datatype_to_bits(kernel.type));
                }

                gen.println("{1}_{2} = {0}::create(\"{1}_{2}\", {3}, \"data/"
                            "{1}_{2}.txt\");", kernel_arg_type_to_str(arg.type),
                            kernel.user_name, arg.name, plio_type);
            }
        }

        gen.println();
        gen.println("{0}k = kernel::create({0});", kernel.user_name);
        gen.println();
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
            std::string name = std::format(" net{}", net_count);
            net_count++;
            if (arg.dimensions == 0 || kernel.wsize == 0) {
                type = "stream";
            } else {
                type = std::format("window<{}>", kernel.wsize);
            }

            const auto &connection = kernel.connections.at(arg.name);

            if (connection.type == connection_type::host) { // mapping to PLIO
                if (arg.type == karg_type::input ||
                    arg.type == karg_type::input_index) {
                    std::string in_arg =
                            std::format("{}k.in[{}]", kernel.user_name,
                                        in_count);
                    if (arg.async) {
                        in_arg = std::format("async({})", in_arg);
                    }
                    gen.println("connect<{}>{}({}_{}.out[0], {});",
                                type, name, kernel.user_name, arg.name, in_arg);
                } else {
                    std::string out_arg =
                            std::format("{}k.out[{}]", kernel.user_name,
                                        out_count);
                    if (arg.async) {
                        out_arg = std::format("async({})", out_arg);
                    }
                    gen.println("connect<{}>{}({}, {}_{}.in[0]);",
                                type, name, out_arg, kernel.user_name,
                                arg.name);
                }
            } else if (arg.type == karg_type::output) { // mapping to ext kernel
                // find external kernel we are mapping against
                const auto &kernels = gen.get_data().kernels;
                auto cond = [&connection](const codegen::kernel &k) -> bool {
                    return k.user_name == connection.kernel;
                };
                auto it = std::find_if(kernels.begin(), kernels.end(), cond);
                if (it == kernels.end()) {
                    log(log_level::error, "Unknown kernel \"{}\"",
                        connection.kernel);
                    throw std::runtime_error("Unknown kernel");
                }
                unsigned ext_in_count = 0;
                unsigned ext_out_count = 0;
                bool ext_async = false;

                // find external kernel argument with correct in/out index
                for (const auto &ext_arg : get_kernel_args(it->operation)) {
                    if (ext_arg.name == connection.parameter) {
                        ext_async = ext_arg.async;
                        break;
                    }

                    if (ext_arg.type == karg_type::input ||
                        ext_arg.type == karg_type::input_index) {
                        ext_in_count++;
                    } else {
                        ext_out_count++;
                    }
                }

                std::string out_arg, in_arg;
                if (arg.type == karg_type::input ||
                    arg.type == karg_type::input_index) {
                    out_arg = std::format("{}k.out[{}]", it->user_name,
                                          ext_out_count);
                    if (ext_async) {
                        out_arg = std::format("async({})", out_arg);
                    }
                    in_arg = std::format("{}k.in[{}]", kernel.user_name,
                                         in_count);
                    if (arg.async) {
                        in_arg = std::format("async({})", in_arg);
                    }
                } else {
                    out_arg = std::format("{}k.out[{}]", kernel.user_name,
                                          out_count);
                    if (arg.async) {
                        out_arg = std::format("async({})", out_arg);
                    }
                    in_arg = std::format("{}k.in[{}]", it->user_name,
                                         ext_in_count);
                    if (ext_async) {
                        in_arg = std::format("async({})", in_arg);
                    }
                }

                gen.println("connect<{}>{}({}, {});", type, name, out_arg,
                            in_arg);
            } else if (connection.type == connection_type::none) {
                continue; // skip disabled argument
            }
            // Input connections to other kernels are defined by the other
            // kernel

            if (arg.type == karg_type::input ||
                arg.type == karg_type::input_index) {
                in_count++;
            } else {
                out_count++;
            }
        }

        gen.println();
        gen.println("source({0}k) = \"kernels/{0}.cpp\";",
                    kernel.user_name);
        gen.println("runtime<ratio>({}k) = 0.9;", kernel.user_name);
        if (kernel.tile_set) {
            gen.println("location<kernel>({}k) = tile({}, {});",
                        kernel.user_name, kernel.tile_x, kernel.tile_y);
        }
        gen.println();
    }

    gen.println<generator::DECREASE_BEFORE>("}}");
    gen.println<generator::DECREASE_BEFORE>("}};");
}

static inline void generate_graph_src(generator &gen) {
    gen.println("#include \"graph.hpp\"");
    gen.println();
    gen.println("simpleGraph mygraph;");
    gen.println();
    gen.println<generator::INCREASE_AFTER>("int main (void) {{");
    gen.println("adf::return_code ret;");
    gen.println("mygraph.init();");
    gen.println();
    gen.println("ret = mygraph.run(1);");
    gen.println<generator::INCREASE_AFTER>("if (ret != adf::ok) {{");
    gen.println("printf(\"Run failed\\n\");");
    gen.println("return ret;");
    gen.println<generator::DECREASE_BEFORE>("}}");
    gen.println();
    gen.println("ret = mygraph.end();");
    gen.println<generator::INCREASE_AFTER>("if (ret != adf::ok) {{");
    gen.println("printf(\"End failed\\n\");");
    gen.println("return ret;");
    gen.println<generator::DECREASE_BEFORE>("}}");
    gen.println();
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
