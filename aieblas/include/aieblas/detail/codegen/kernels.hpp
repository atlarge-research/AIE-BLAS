#pragma once

#include <fstream>

#include "aieblas/detail/util.hpp"
#include "aieblas/detail/codegen/datastructures.hpp"

namespace aieblas {
namespace codegen {

void generate_kernel_src(generator &gen, const kernel &kernel);

} // codegen
} // aieblas
