#pragma once

#include <filesystem>

namespace aieblas {
namespace codegen {

void codegen(std::filesystem::path json, std::filesystem::path output);

} // codegen
} // aieblas
