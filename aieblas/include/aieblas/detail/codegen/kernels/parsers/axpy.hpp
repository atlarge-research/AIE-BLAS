#pragma once
#include <nlohmann/json.hpp>

#include "aieblas/detail/codegen/kernels.hpp"
#include "aieblas/detail/codegen/kernels/axpy.hpp"
#include "aieblas/detail/util.hpp"

namespace aieblas {
namespace codegen {
namespace generators {
inline std::unique_ptr<axpy_options> get_axpy_options(nlohmann::json json) {
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

    return std::make_unique<axpy_options>(alpha);
}
} // namespace generators
} // namespace codegen
} // namespace aieblas
