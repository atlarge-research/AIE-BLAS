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
    unknown, int8, int16, int32, int64, uint8, uint16, uint32, uint64, float32
};

enum class karg_type : unsigned {
    unknown, input, output
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
    case dtype::int8:
        return "int8";
    case dtype::int16:
        return "int16";
    case dtype::int32:
        return "int32";
    case dtype::int64:
        return "int64";
    case dtype::uint8:
        return "int8";
    case dtype::uint16:
        return "int16";
    case dtype::uint32:
        return "int32";
    case dtype::uint64:
        return "int64";
    case dtype::float32:
        return "float";
    default:
        return "unknown";
    }
}

inline dtype datatype_from_str(const std::string_view str) {
    if (str == "int8") {
        return dtype::int8;
    } else if (str == "int16") {
        return dtype::int16;
    } else if (str == "int32") {
        return dtype::int32;
    } else if (str == "int64") {
        return dtype::int64;
    } else if (str == "uint8") {
        return dtype::uint8;
    } else if (str == "uint16") {
        return dtype::uint16;
    } else if (str == "uint32") {
        return dtype::uint32;
    } else if (str == "uint64") {
        return dtype::uint64;
    } else if (str == "float" || str == "float32") {
        return dtype::float32;
    } else {
        return dtype::unknown;
    }
}

inline std::tuple<std::size_t, std::string>
datatype_to_native_type(dtype type, std::size_t min_vecsize) {
    const char *base = datatype_to_str(type);
    std::vector<std::size_t> sizes;
    switch (type) {
        case dtype::int8:
        case dtype::uint8:
            sizes = {16, 32, 64};
            break;
        case dtype::int16:
        case dtype::uint16:
            sizes = {16, 32, 64};
            break;
        case dtype::int32:
        case dtype::uint32:
            sizes = {4, 8, 16, 32};
            break;
        case dtype::float32:
            sizes = {8, 16, 32};
            break;
        case dtype::int64:
        case dtype::uint64:
            sizes = {4, 8, 16};
            break;
        default:
            base = "";
            sizes = {};
            break;
    }

    std::size_t size = 0;
    for (const auto &pos_size : sizes) {
        if (min_vecsize <= pos_size) {
            size = pos_size;
        }
    }

    return {size, std::format("v{}{}", size, base)};
}

constexpr inline const char *datatype_to_accum(dtype type) {
    switch (type) {
        case dtype::int8:
        case dtype::uint8:
        case dtype::int16:
        case dtype::uint16:
        case dtype::int32:
        case dtype::uint32:
            return "acc48";
        case dtype::float32:
            return "accfloat";
        case dtype::int64:
        case dtype::uint64:
            return "acc80";
        default:
            return 0;
    }
}

constexpr inline unsigned datatype_to_bits(dtype type) {
    switch (type) {
        case dtype::int8:
        case dtype::uint8:
            return 8;
        case dtype::int16:
        case dtype::uint16:
            return 16;
        case dtype::int32:
        case dtype::uint32:
        case dtype::float32:
            return 32;
        case dtype::int64:
        case dtype::uint64:
            return 64;
        default:
            return 0;
    }
}

constexpr inline const char *kernel_arg_type_to_str(karg_type karg) {
    switch (karg) {
    case karg_type::input:
        return "input_plio";
    case karg_type::output:
        return "output_plio";
    default:
        return "unknown";
    }
}

inline karg_type kernel_arg_type_from_str(const std::string_view str) {
    if (str == "input_plio") {
        return karg_type::input;
    } else if (str == "output_plio") {
        return karg_type::output;
    } else {
        return karg_type::unknown;
    }
}

}
}
