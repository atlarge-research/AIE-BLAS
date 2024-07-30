#pragma once

#include <fstream>

#include "aieblas/detail/util.hpp"
#include "aieblas/detail/codegen/generator.hpp"
#include "aieblas/detail/codegen/kernels.hpp"

namespace aieblas {
namespace codegen {

void generate_graph(generator &gen);

} // codegen
} // aieblas
