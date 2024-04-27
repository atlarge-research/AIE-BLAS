#pragma once

#include <cstddef>
#include <fstream>

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

    ~generator() {
        if (cur_file_.is_open()) {
            this->close();
        }
    }

    void generate_kernels();

    void open(fs::path filename);
    void close();

    std::ofstream &cur_file() {
        return cur_file_;
    }

    std::size_t indent() {
        return indent_;
    }

    void indent_incr(std::size_t levels = 1) {
        indent_ += levels;
    }

    void indent_decr(std::size_t levels = 1) {
        indent_ -= levels;
    }

    enum print_opts : unsigned {
        NO_OPTS = 0U,
        INCREASE_BEFORE = 1U,
        INCREASE_AFTER = 2U,
        DECREASE_BEFORE = 4U,
        DECREASE_AFTER = 8U
    };

    template <unsigned opts = NO_OPTS, typename... Args>
    void print(std::format_string<Args...> fmt, Args &&...args) {
        if (opts & INCREASE_BEFORE) {
            indent_incr();
        } else if (opts & DECREASE_BEFORE) {
            indent_decr();
        }

        if (!indented) {
            print_indent();
        }

        std::print(cur_file_, std::runtime_format(fmt),
                   std::forward<Args>(args)...);

        if (opts & INCREASE_AFTER) {
            indent_incr();
        } else if (opts & DECREASE_AFTER) {
            indent_decr();
        }
    }

    template <unsigned opts = NO_OPTS, typename... Args>
    void println(std::format_string<Args...> fmt, Args &&...args) {
        this->print<opts>(std::runtime_format(fmt),
                          std::forward<Args>(args)...);
        std::println(cur_file_, "");
        indented = false;
    }

    private:
    void parse_json(fs::path json);
    void generate_kernel(const kernel &kernel, fs::path kernel_dir);

    void print_indent();

    data d;
    fs::path out_dir;

    std::vector<fs::path> kernel_srcs;
    std::vector<fs::path> kernel_hdrs;

    std::ofstream cur_file_;
    fs::path cur_filename;
    std::size_t indent_;
    bool indented;
    const std::size_t indent_level = 4;
};

} // codegen
} // aieblas
