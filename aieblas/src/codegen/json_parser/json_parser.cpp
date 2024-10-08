#include <fstream>
#include <nlohmann/json.hpp>
#include <unordered_map>
#include <utility>
#include <vector>

#include "aieblas/detail/codegen/generator.hpp"
#include "aieblas/detail/codegen/json_parser.hpp"
#include "aieblas/detail/codegen/json_parser/get_kernel_options.hpp"
#include "aieblas/detail/codegen/kernels.hpp"
#include "aieblas/detail/util.hpp"

using json = nlohmann::json;

namespace aieblas {
namespace codegen {

void generator::parse_json(fs::path json_file) {
    std::ifstream f(json_file);
    json json_data = json::parse(f);

    if (!json_data.count("platform")) {
        throw parse_error("Top-level item platform is missing from json.");
    } else if (!json_data["platform"].is_string()) {
        throw parse_error("platform should be a string.");
    }

    this->d.platform = json_data["platform"].get<std::string>();

    if (json_data.count("profile")) {
        if (!json_data["profile"].is_boolean()) {
            throw parse_error("profile should be a boolean.");
        }
        this->d.profile = json_data["profile"].get<bool>();
    } else {
        this->d.profile = false;
    }

    std::unordered_map<kernel_parameter, kernel_parameter> connections;

    if (json_data.count("connections")) {
        if (!json_data["connections"].is_array()) {
            throw parse_error("connections should be an array.");
        }
        json &connections_json = json_data["connections"];
        std::size_t n_connections = connections_json.size();

        for (std::size_t i = 0; i < n_connections; ++i) {
            json &item = connections_json[i];
            if (!item.is_object()) {
                throw parse_error(
                    std::format("connection {} should be a dictionary", i));
            }

            if (!item.count("in")) {
                throw parse_error(
                    std::format("connection {} is missing in argument", i));
            } else if (!item["in"].is_object()) {
                throw parse_error(std::format(
                    "connection {} in argument should be a dictionary", i));
            } else if (!item["in"].count("kernel") ||
                       !item["in"]["kernel"].is_string()) {
                throw parse_error(
                    std::format("connection {} is missing in kernel", i));
            } else if (!item["in"].count("parameter") ||
                       !item["in"]["parameter"].is_string()) {
                throw parse_error(
                    std::format("connection {} is missing in parameter", i));
            }

            if (!item.count("out")) {
                throw parse_error(
                    std::format("connection {} is missing out argument", i));
            } else if (!item["out"].is_object()) {
                throw parse_error(std::format(
                    "connection {} out argument should be a dictionary", i));
            } else if (!item["out"].count("kernel") ||
                       !item["out"]["kernel"].is_string()) {
                throw parse_error(
                    std::format("connection {} is missing out kernel", i));
            } else if (!item["out"].count("parameter") ||
                       !item["out"]["parameter"].is_string()) {
                throw parse_error(
                    std::format("connection {} is missing out parameter", i));
            }

            kernel_parameter in{item["in"]["kernel"].get<std::string>(),
                                item["in"]["parameter"].get<std::string>()};
            kernel_parameter out{item["out"]["kernel"].get<std::string>(),
                                 item["out"]["parameter"].get<std::string>()};

            if (connections.contains(in)) {
                throw parse_error(
                    std::format("already defined a connection for {}.{}",
                                in.kernel, in.parameter));
            }

            connections.emplace(in, out);
        }
    }

    if (!json_data.count("kernels")) {
        throw parse_error("Top-level item kernels is missing from json.");
    } else if (!json_data["kernels"].is_array()) {
        throw parse_error("kernels should be an array.");
    }

    json &kernels = json_data["kernels"];
    std::size_t n_kernels = kernels.size();
    this->d.kernels.clear();
    this->d.kernels.reserve(n_kernels);

    for (std::size_t i = 0; i < n_kernels; ++i) {
        json &item = kernels[i];
        blas_op operation;
        std::string user_name;
        dtype type;
        unsigned vsize;
        unsigned wsize;
        bool tile_set;
        unsigned tile_x;
        unsigned tile_y;
        std::unordered_map<std::string, connection> connections_map;
        std::unique_ptr<kernel_options> extra_options;

        if (!item.is_object()) {
            throw parse_error(
                std::format("kernel {} should be a dictionary", i));
        }

        if (!item.count("blas_op")) {
            throw parse_error(std::format("kernel {} is missing 'blas_op'", i));
        } else if (!item["blas_op"].is_string()) {
            throw parse_error(
                std::format("blas_op should be a string in kernel {}.", i));
        }

        std::string blas_op_str = item["blas_op"].get<std::string>();
        operation = blas_op_from_str(blas_op_str);

        if (operation == blas_op::unknown) {
            throw parse_error(std::format(
                "blas_op '{}' is unknown in kernel {}.", blas_op_str, i));
        }

        if (!item.count("user_name")) {
            throw parse_error(
                std::format("kernel {} is missing 'user_name'", i));
        } else if (!item["user_name"].is_string()) {
            throw parse_error(
                std::format("user_name should be a string in kernel {}.", i));
        }

        user_name = item["user_name"].get<std::string>();

        if (!item.count("type")) {
            throw parse_error(std::format("kernel {} is missing 'type'", i));
        } else if (!item["type"].is_string()) {
            throw parse_error(
                std::format("type should be a string in kernel {}.", i));
        }

        std::string type_str = item["type"].get<std::string>();
        type = datatype_from_str(type_str);

        if (type == dtype::unknown) {
            throw parse_error(
                std::format("type '{}' is unknown in kernel {}.", type_str, i));
        }

        vsize = 0;
        if (item.count("vector_size")) {
            if (!item["vector_size"].is_number_unsigned()) {
                throw parse_error(std::format(
                    "vector_size should be an unsigned integer in kernel {}.",
                    i));
            }

            vsize = item["vector_size"].get<unsigned>();
        }

        wsize = 128;
        if (item.count("window_size")) {
            if (!item["window_size"].is_number_unsigned()) {
                throw parse_error(std::format(
                    "window_size should be an unsigned integer in kernel {}.",
                    i));
            }

            wsize = item["window_size"].get<unsigned>();
        }

        if (item.count("tile")) {
            if (!item["tile"].is_array() || item["tile"].size() < 2
                || !item["tile"][0].is_number_unsigned()
                || !item["tile"][1].is_number_unsigned()) {
                throw parse_error(std::format(
                    "tile should be an array of two unsigned integers in "
                    "kernel {}.", i));
            }

            tile_set = true;
            tile_x = item["tile"][0].get<unsigned>();
            tile_y = item["tile"][1].get<unsigned>();
        } else {
            tile_set = false;
            tile_x = 0;
            tile_y = 0;
        }

        if (item.count("extra")) {
            extra_options = get_kernel_options(operation, item["extra"]);
        } else {
            extra_options = get_kernel_options(operation, json::object());
        }

        const std::vector<kernel_arg> args = get_kernel_args(operation);

        for (const kernel_arg &arg : args) {
            const kernel_parameter &param = {user_name, arg.name};
            connection conn;

            if (extra_options->disabled_arg(arg.name)) {
                conn.type = connection_type::none;
            } else if (arg.type == karg_type::input ||
                       arg.type == karg_type::input_index) {
                // search by value
                auto it = std::find_if(
                    std::begin(connections), std::end(connections),
                    [&param](auto &&p) { return p.second == param; });

                if (it == std::end(connections)) {
                    conn.type = connection_type::host;
                } else {
                    conn.type = connection_type::kernel;
                    conn.kernel = it->first.kernel;
                    conn.parameter = it->first.parameter;
                }
            } else if (arg.type == karg_type::output) {
                // search by key
                if (connections.contains(param)) {
                    const kernel_parameter &p = connections[param];
                    conn.type = connection_type::kernel;
                    conn.kernel = p.kernel;
                    conn.parameter = p.parameter;
                } else {
                    conn.type = connection_type::host;
                }
            } else {
                throw parse_error("Unsupported karg_type");
            }

            connections_map[arg.name] = conn;
        }

        this->d.kernels.emplace_back(operation, std::move(user_name), type,
                                     vsize, wsize, tile_set, tile_x, tile_y,
                                     std::move(connections_map),
                                     std::move(extra_options));
    }
}

} // namespace codegen
} // namespace aieblas
