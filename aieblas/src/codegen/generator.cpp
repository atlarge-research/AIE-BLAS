#include <fstream>
#include "aieblas/detail/util.hpp"
#include "aieblas/detail/codegen/generator.hpp"

namespace aieblas {
namespace codegen {
void generator::open(fs::path filename) {
    indent_ = 0;
    indented = false;
    if (cur_file_.is_open()) {
        this->close();
    }
    log(log_level::verbose, "Opening '{}'", filename.c_str());
    cur_file_ = std::ofstream{filename};
    cur_filename = filename;
}

void generator::close() {
    log(log_level::verbose, "Closing '{}'", cur_filename.c_str());
    cur_file_.close();
}

void generator::print_indent() {
    std::print(cur_file_, "{}", std::string(indent_ * indent_level, ' '));
    indented = true;
}
} // codegen
} // aieblas
