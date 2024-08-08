#pragma once

#include "cxx_compat.hpp"
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

namespace util {

static inline fs::path get_bench_dir() {
    fs::path dir = fs::path(__FILE__).parent_path();
    while (dir.filename().string() != "benchmark") {
        dir = dir.parent_path();
    }

    return dir;
}

static inline void write_results(fs::path &result_file, std::string &name,
                                 std::uint64_t size, double time) {
    std::ofstream result_stream;
    if (!fs::exists(result_file)) {
        result_stream = std::ofstream(result_file.c_str());
        std::println(result_stream, "name;size;time (ms)");
    } else {
        result_stream = std::ofstream(result_file.c_str(), std::ios::app);
    }

    std::println("writing results to {}", result_file.c_str());
    std::println(result_stream, "{};{};{}", name, size, time);
    result_stream.close();
}

static inline void write_result_sum(std::string &name, std::uint64_t size,
                                    double time) {
    fs::path result_file = get_bench_dir() / "results/sum.csv";
    write_results(result_file, name, size, time);
}

static inline void write_result_axpy(std::string &name, std::uint64_t size,
                                     double time) {
    fs::path result_file = get_bench_dir() / "results/single.csv";
    write_results(result_file, name, size, time);
}

static inline void write_result_gemv(std::string &name, std::uint64_t size,
                                     double time) {
    fs::path result_file = get_bench_dir() / "results/single.csv";
    write_results(result_file, name, size, time);
}

static inline void write_result_axpydot(std::string &name, std::uint64_t size,
                                        double time) {
    fs::path result_file = get_bench_dir() / "results/axpydot.csv";
    write_results(result_file, name, size, time);
}
}; // namespace util
