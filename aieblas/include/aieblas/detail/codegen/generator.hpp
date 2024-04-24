#pragma once

#include "aieblas/detail/util.hpp"
#include "aieblas/detail/codegen/datastructures.hpp"
#include "aieblas/detail/codegen/json_parser.hpp"

namespace aieblas {
namespace codegen {

class generator {
    public:
    generator(fs::path json, fs::path output) : out_dir(output) {
        try {
        parse_json(json);
        } catch (const parse_error &e) {
            throw parse_error(std::format("Parsing error from '{}': {}",
                                          json.c_str(), e.what()));
        }
    }

    private:
    void parse_json(fs::path json);

    data d;
    fs::path out_dir;
};

} // codegen
} // aieblas
