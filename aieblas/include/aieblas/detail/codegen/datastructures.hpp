#pragma once

#include <unordered_map>
#include <vector>

#include "aieblas/detail/util.hpp"

namespace aieblas {
namespace codegen {

enum class blas_op : unsigned {
    unknown, dot, scal
};

enum class dtype : unsigned {
    unknown, int32, int64, float32, float64
};

enum class karg_type : unsigned {
    unknown, input_plio, output_plio
};

struct kernel_arg {
    karg_type type;
    std::string name;
    unsigned dimensions; // 0: scalar 1: vector 2: matrix
};

enum class connection_type : unsigned {
    host, kernel
};

struct connection {
    connection_type type;

    // kernel specific
    std::string kernel;
    std::string parameter;
};

struct kernel {
    blas_op operation;
    std::string user_name;
    dtype type;
    unsigned vsize;
    unsigned wsize;

    // maps parameter (of this kernel) to outside data port
    std::unordered_map<std::string, connection> connections;
};

struct data {
    bool profile;
    std::string platform;
    std::vector<kernel> kernels;
};

constexpr inline const char *blas_op_to_str(blas_op op) {
    switch (op) {
    case blas_op::dot:
        return "dot";
    case blas_op::scal:
        return "scal";
    default:
        return "unknown";
    }
}

inline blas_op blas_op_from_str(const std::string_view str) {
    if (str == "dot") {
        return blas_op::dot;
    } else if (str == "scal") {
        return blas_op::scal;
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

constexpr inline unsigned datatype_to_bits(dtype type) {
    switch (type) {
        case dtype::float32:
        case dtype::int32:
            return 32;
        case dtype::float64:
        case dtype::int64:
            return 64;
        default:
            return 0;
    }
}

constexpr inline const char *kernel_arg_type_to_str(karg_type karg) {
    switch (karg) {
    case karg_type::input_plio:
        return "input_plio";
    case karg_type::output_plio:
        return "output_plio";
    default:
        return "unknown";
    }
}

inline karg_type kernel_arg_type_from_str(const std::string_view str) {
    if (str == "input_plio") {
        return karg_type::input_plio;
    } else if (str == "output_plio") {
        return karg_type::output_plio;
    } else {
        return karg_type::unknown;
    }
}

}
}
