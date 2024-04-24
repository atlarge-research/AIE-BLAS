#pragma once

#include <vector>

#include "aieblas/detail/util.hpp"

namespace aieblas {
namespace codegen {

enum class blas_op : unsigned {
    unknown, dot, scale
};

enum class dtype : unsigned {
    unknown, int32, int64, float32, float64
};

struct kernel {
    blas_op operation;
    std::string user_name;
    dtype type;
};

struct data {
    std::string platform;
    std::vector<kernel> kernels;
};

constexpr inline const char *blas_op_to_str(blas_op op) {
    switch (op) {
    case blas_op::dot:
        return "dot";
    case blas_op::scale:
        return "scale";
    default:
        return "unknown";
    }
}

inline blas_op blas_op_from_str(const std::string_view str) {
    if (str == "dot") {
        return blas_op::dot;
    } else if (str == "scale") {
        return blas_op::scale;
    } else {
        return blas_op::unknown;
    }
}

constexpr inline const char *datatype_to_str(dtype type) {
    switch (type) {
    case dtype::int32:
        return "int32";
    case dtype::int64:
        return "int64";
    case dtype::float32:
        return "float";
    case dtype::float64:
        return "double";
    default:
        return "unknown";
    }
}

inline dtype datatype_from_str(const std::string_view str) {
    if (str == "int32") {
        return dtype::int32;
    } else if (str == "int64") {
        return dtype::int32;
    } else if (str == "float" || str == "float32") {
        return dtype::float32;
    } else if (str == "double" || str == "float64") {
        return dtype::float64;
    } else {
        return dtype::unknown;
    }
}

}
}
