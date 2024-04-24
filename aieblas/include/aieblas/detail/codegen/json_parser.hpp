#pragma once

#include <stdexcept>

namespace aieblas {
namespace codegen {

class parse_error : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

} // codegen
} // aieblas
