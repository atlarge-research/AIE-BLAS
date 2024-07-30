#pragma once
#include <nlohmann/json.hpp>

#include "aieblas/detail/codegen/kernels.hpp"
#include "aieblas/detail/codegen/kernels/gemv.hpp"
#include "aieblas/detail/util.hpp"

namespace aieblas {
namespace codegen {
namespace generators {
inline std::unique_ptr<gemv_options> get_gemv_options(nlohmann::json json) {
    value alpha{};

    if (json.contains("alpha")) {
        auto a = json["alpha"];
        switch (a.type()) {
        case nlohmann::json::value_t::number_integer:
            alpha = value(a.get<int64_t>());
            break;
        case nlohmann::json::value_t::number_unsigned:
            alpha = value(a.get<uint64_t>());
            break;
        case nlohmann::json::value_t::number_float:
            alpha = value(a.get<float>());
            break;
        default:
            throw std::runtime_error(
                std::format("Unsupported json type '{}'", a.type_name()));
            break;
        }
    }

    value beta{};

    if (json.contains("beta")) {
        auto b = json["beta"];
        switch (b.type()) {
        case nlohmann::json::value_t::number_integer:
            beta = value(b.get<int64_t>());
            break;
        case nlohmann::json::value_t::number_unsigned:
            beta = value(b.get<uint64_t>());
            break;
        case nlohmann::json::value_t::number_float:
            beta = value(b.get<float>());
            break;
        default:
            throw std::runtime_error(
                std::format("Unsupported json type '{}'", b.type_name()));
            break;
        }
    }

    return std::make_unique<gemv_options>(alpha, beta);
}
} // namespace generators
} // namespace codegen
} // namespace aieblas
