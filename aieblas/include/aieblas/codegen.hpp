#pragma once

#include <filesystem>
#include <vector>

namespace aieblas {
namespace codegen {

void codegen(std::filesystem::path json, std::filesystem::path output);

} // codegen
} // aieblas
