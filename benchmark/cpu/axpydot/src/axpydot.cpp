#include <chrono>
#include <cblas.h>
#include <cxxopts.hpp>
#include <filesystem>

#include "cxx_compat.hpp"
#include "timer.hpp"
#include "write_results.hpp"

namespace fs = std::filesystem;

using seconds = std::chrono::duration<double>;
using milliseconds = std::chrono::duration<double, std::milli>;
using microseconds = std::chrono::duration<double, std::micro>;

namespace {

struct arguments {
    std::uint64_t size;
};

inline void error(const std::string_view message) {
    std::println("Argument error: {}", message);
    std::fflush(stdout);
    exit(1);
}

struct arguments parse_args(int argc, const char* const* argv) {
    struct arguments args;
    try {
        cxxopts::Options options("host code",
                                "AIE engine host code");

        options.add_options()
            ("s,size", "Size of m",
             cxxopts::value<std::uint64_t>())
            ("h,help", "Print usage");

        auto results = options.parse(argc, argv);

        if (results.count("help")) {
            std::println("{}", options.help());
            std::fflush(stdout);
            exit(0);
        }

        std::uint64_t size = 64;
        if (results.count("size")) {
            size = results["size"].as<std::uint64_t>();
        }

        args = {size};
    } catch (const std::exception &e) {
        error(std::format("Unexpected exception: {}", e.what()));
    }

    return args;
}

template <typename Number>
constexpr void initialize_data(Number *w, Number *v, Number *u, Number &alpha,
                               Number &result, std::uint64_t size) {
    alpha = -2;
    result = 0;

    for (std::uint64_t i = 0; i < size; ++i) {
        w[i] = i;
        v[i] = i * 3;
        u[i] = 2;
        result += (v[i] + alpha * w[i]) * u[i];
    }
}
} // namespace

int main(int argc, char *argv[]) {
    struct arguments args = parse_args(argc, argv);
    Timer timer;

    float *w = new float[args.size];
    float *v = new float[args.size];
    float *u = new float[args.size];
    float alpha, result;

    initialize_data(w, v, u, alpha, result, args.size);

    timer.time_point("start");
    cblas_saxpy(args.size, alpha, w, 1, v, 1);
    float exp_result = cblas_sdot(args.size, w, 1, u, 1);
    timer.time_point("end");

    double exec_time = timer.time<milliseconds>("start", "end").count();

    std::println("Execution finished in {:.2f} ms!", exec_time);

    std::string name = std::format("cpu_axpydot");
    util::write_result_axpydot(name, args.size, exec_time);

    unsigned errors = 0;

    if (exp_result != result) {
        std::println("FAIL! ({} != {})", exp_result, result);

        errors += 1;
    } else {
        // std::println("{} (correct)", result[i]);
    }

    if (errors == 0) {
        std::println("OK!");
    } else {
        std::println("{} failures!", errors);
    }

    delete[] w;
    delete[] v;
    delete[] u;

    return errors == 0 ? 0 : 1;
}
