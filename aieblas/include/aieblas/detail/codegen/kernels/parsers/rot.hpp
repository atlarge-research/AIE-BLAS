#pragma once
#include <nlohmann/json.hpp>

#include "aieblas/detail/codegen/kernels.hpp"
#include "aieblas/detail/codegen/kernels/rot.hpp"
#include "aieblas/detail/util.hpp"

namespace aieblas {
namespace codegen {
namespace generators {
inline std::unique_ptr<rot_options> get_rot_options(nlohmann::json json) {
    value c{};
    if (json.contains("c")) {
        auto cj = json["c"];
        switch (cj.type()) {
        case nlohmann::json::value_t::number_integer:
            c = value(cj.get<int64_t>());
            break;
        case nlohmann::json::value_t::number_unsigned:
            c = value(cj.get<uint64_t>());
            break;
        case nlohmann::json::value_t::number_float:
            c = value(cj.get<float>());
            break;
        default:
            throw std::runtime_error(
                std::format("Unsupported json type '{}'", cj.type_name()));
            break;
        }
    }

    value s{};
    if (json.contains("s")) {
        auto sj = json["s"];
        switch (sj.type()) {
        case nlohmann::json::value_t::number_integer:
            s = value(sj.get<int64_t>());
            break;
        case nlohmann::json::value_t::number_unsigned:
            s = value(sj.get<uint64_t>());
            break;
        case nlohmann::json::value_t::number_float:
            s = value(sj.get<float>());
            break;
        default:
            throw std::runtime_error(
                std::format("Unsupported json type '{}'", sj.type_name()));
            break;
        }
    }

    return std::make_unique<rot_options>(c, s);
}
} // namespace generators
} // namespace codegen
} // namespace aieblas
