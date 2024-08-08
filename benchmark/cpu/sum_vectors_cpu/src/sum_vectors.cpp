#include <chrono>
#include <cblas.h>
#include <cxxopts.hpp>
#include <filesystem>
#include <thread>

#include "cxx_compat.hpp"
#include "timer.hpp"
#include "write_results.hpp"

namespace fs = std::filesystem;

using seconds = std::chrono::duration<double>;
using milliseconds = std::chrono::duration<double, std::milli>;
using microseconds = std::chrono::duration<double, std::micro>;

namespace {
struct arguments {
    std::uint64_t num_vectors;
    std::uint64_t size;
    bool axpy;
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
            ("v,vectors", "No. vectors",
             cxxopts::value<std::uint64_t>())
            ("a,axpy", "Run axpy benchmark")
            ("h,help", "Print usage");

        auto results = options.parse(argc, argv);

        if (results.count("help")) {
            std::println("{}", options.help());
            std::fflush(stdout);
            exit(0);
        }

        bool axpy = false;
        if (results.count("axpy")) {
            axpy = true;
        }

        std::uint64_t vectors = 2;
        if (results.count("vectors")) {
            vectors = results["vectors"].as<std::uint64_t>();
        }

        std::uint64_t size = 64;
        if (results.count("size")) {
            size = results["size"].as<std::uint64_t>();
        }

        args = {vectors, size, axpy};
    } catch (const std::exception &e) {
        error(std::format("Unexpected exception: {}", e.what()));
    }

    return args;
}

template <typename Number>
constexpr void initialize_data(Number **vectors, Number *result,
                               std::uint64_t size, std::uint64_t num_vectors) {
    for (std::uint64_t i = 0; i < size; ++i) {
        result[i] = 0;
        for (std::uint64_t j = 0; j < num_vectors; ++j) {
            vectors[j][i] = j;
            result[i] += j;
        }
    }
}
} // namespace

int main(int argc, char *argv[]) {
    struct arguments args = parse_args(argc, argv);
    Timer timer;

    float **vectors = new float*[args.num_vectors];
    float *result = new float[args.size];
    for (std::uint64_t i = 0; i < args.num_vectors; ++i) {
        vectors[i] = new float[args.size];
    }

    initialize_data(vectors, result, args.size, args.num_vectors);

    std::this_thread::sleep_for(std::chrono::seconds(1));

    timer.time_point("start");
    for (std::uint64_t i = 1; i < args.num_vectors; ++i) {
        cblas_saxpy(args.size, 1, vectors[i], 1, vectors[0], 1);
    }
    timer.time_point("end");

    double exec_time = timer.time<milliseconds>("start", "end").count();

    std::println("Execution finished in {:.2f} ms!", exec_time);

    if (args.axpy) {
        std::string name = "cpu_axpy";
        util::write_result_axpy(name, args.size, exec_time);
    } else {
        std::string name = std::format("cpu_sum_vectors_{}", args.num_vectors);
        util::write_result_sum(name, args.size, exec_time);
    }

    unsigned errors = 0;

    for (std::size_t i = 0; i < args.size; ++i) {
        if (vectors[0][i] != result[i]) {
            // std::println("FAIL! ({} != {})", result_device[i], result[i]);
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

    for (std::uint64_t i = 0; i < args.num_vectors; ++i) {
        delete[] vectors[i];
    }
    delete[] vectors;
    delete[] result;

    return errors == 0 ? 0 : 1;
}
