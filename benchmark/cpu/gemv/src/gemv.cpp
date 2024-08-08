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

constexpr const std::uint64_t n = 64;

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
constexpr void initialize_data(Number *A, Number &alpha, Number *x,
                               Number &beta, Number *y, Number *result,
                               std::uint64_t m) {
    alpha = 1;
    beta = 1;

    for (std::uint64_t i = 0; i < m; ++i) {
        y[i] = i;
        for (std::uint64_t j = 0; j < n; ++j) {
            A[i * n + j] = i * n + j;
        }
    }

    for (std::uint64_t j = 0; j < n; ++j) {
        x[j] = 1;
    }

    for (std::uint64_t i = 0; i < m; ++i) {
        Number temp = 0;
        for (std::uint64_t j = 0; j < n; ++j) {
            temp += A[i * n + j] * x[j];
        }
        result[i] = alpha * temp + beta * y[i];
    }
}
} // namespace

int main(int argc, char *argv[]) {
    struct arguments args = parse_args(argc, argv);
    Timer timer;

    float *A = new float[args.size * n];
    float *x = new float[n];
    float *y = new float[args.size];
    float *result = new float[args.size];
    float alpha, beta;

    initialize_data(A, alpha, x, beta, y, result, args.size);

    timer.time_point("start");
    cblas_sgemv(CblasRowMajor, CblasNoTrans, args.size, n, alpha, A, n, x, 1, beta, y, 1);
    timer.time_point("end");

    double exec_time = timer.time<milliseconds>("start", "end").count();

    std::println("Execution finished in {:.2f} ms!", exec_time);

    std::string name = std::format("cpu_gemv");
    util::write_result_gemv(name, args.size, exec_time);

    unsigned errors = 0;

    for (std::size_t i = 0; i < args.size; ++i) {
        if (y[i] != result[i]) {
            if (errors < 10) {
                std::println("FAIL! ({} != {})", y[i], result[i]);
            }
            if (errors == 10) {
                std::println("Hiding future errors...");
            }
            errors += 1;
        } else {
            // std::println("{} (correct)", result[i]);
        }
    }

    if (errors == 0) {
        std::println("OK!");
    } else {
        std::println("{} failures!", errors);
    }

    delete[] A;
    delete[] x;
    delete[] y;
    delete[] result;

    return errors == 0 ? 0 : 1;
}
