#include <fstream>
#include <nlohmann/json.hpp>
#include <vector>

#include "aieblas/detail/util.hpp"
#include "aieblas/detail/codegen/generator.hpp"
#include "aieblas/detail/codegen/json_parser.hpp"

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

    if (!json_data.count("kernels")) {
        throw parse_error("Top-level item kernels is missing from json.");
    } else if (!json_data["kernels"].is_array()) {
        throw parse_error("platform should be an array.");
    }

    this->d.platform = json_data["platform"].get<std::string>();

    json &kernels = json_data["kernels"];
    std::size_t n_kernels = kernels.size();
    this->d.kernels.clear();
    this->d.kernels.reserve(n_kernels);

    for (std::size_t i = 0; i < n_kernels; ++i) {
        json &item = kernels[i];
        kernel krnl;

        if (!item.is_object()) {
            throw parse_error(std::format("kernel {} should be a dictionary", i));
        }

        if (!item.count("blas_op")) {
            throw parse_error(std::format("kernel {} is missing 'blas_op'", i));
        } else if (!item["blas_op"].is_string()) {
            throw parse_error(std::format("blas_op should be a string in kernel {}.", i));
        }

        std::string blas_op_str = item["blas_op"].get<std::string>();
        krnl.operation = blas_op_from_str(blas_op_str);

        if (krnl.operation == blas_op::unknown) {
            throw parse_error(std::format("blas_op '{}' is unknown in kernel {}.", blas_op_str, i));
        }

        if (!item.count("user_name")) {
            throw parse_error(std::format("kernel {} is missing 'user_name'", i));
        } else if (!item["user_name"].is_string()) {
            throw parse_error(std::format("user_name should be a string in kernel {}.", i));
        }

        krnl.user_name = item["user_name"].get<std::string>();

        if (!item.count("type")) {
            throw parse_error(std::format("kernel {} is missing 'type'", i));
        } else if (!item["type"].is_string()) {
            throw parse_error(std::format("type should be a string in kernel {}.", i));
        }

        std::string type = item["type"].get<std::string>();
        krnl.type = datatype_from_str(type);

        if (krnl.type == dtype::unknown) {
            throw parse_error(std::format("type '{}' is unknown in kernel {}.", type, i));
        }

        this->d.kernels.push_back(krnl);
    }
}

} // codegen
} // aieblas
