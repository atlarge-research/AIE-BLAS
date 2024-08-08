#include <chrono>
#include <cstdint>
#include <cxxopts.hpp>
#include <filesystem>
#include <thread>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_uuid.h>

#include "cxx_compat.hpp"
#include "timer.hpp"
#include "write_results.hpp"

namespace fs = std::filesystem;

using seconds = std::chrono::duration<double>;
using milliseconds = std::chrono::duration<double, std::milli>;
using microseconds = std::chrono::duration<double, std::micro>;

namespace {
struct arguments {
    fs::path xclbin;
    std::uint64_t size;
};

inline void error(const std::string_view message) {
    std::println("Argument error: {}", message);
    std::fflush(stdout);
    exit(1);
}

inline void error_required(const std::string_view missing_argument) {
    error(std::format("missing required positional argument {}\n"
                        "Run with -h to show usage.", missing_argument));
}

struct arguments parse_args(int argc, const char* const* argv) {
    struct arguments args;
    try {
        cxxopts::Options options("host code",
                                "AIE engine host code");

        options.add_options()
            ("xclbin", "XCLBIN file containing BLAS routines",
             cxxopts::value<std::string>())
            ("s,size", "Size of m",
             cxxopts::value<std::uint64_t>())
            ("h,help", "Print usage");

        options.positional_help("<xclbin>");
        options.parse_positional({"xclbin"});
        auto results = options.parse(argc, argv);

        if (results.count("help")) {
            std::println("{}", options.help());
            std::println("Positional arguments:");
            std::println("  <xclbin> XCLBIN file containing BLAS routines\n\n");
            std::fflush(stdout);
            exit(0);
        }

        if (!results.count("xclbin")) {
            error_required("xclbin");
        }

        std::uint64_t size = 64;
        if (results.count("size")) {
            size = results["size"].as<std::uint64_t>();
        }

        args = {fs::path(results["xclbin"].as<std::string>()), size};

        if (!fs::exists(args.xclbin)) {
            error(std::format("File '{}' does not exist",
                  args.xclbin.string()));
        }

        fs::path xclbin_file = fs::is_symlink(args.xclbin) ?
                               fs::read_symlink(args.xclbin) : args.xclbin;

        if (!fs::is_regular_file(xclbin_file)) {
            error(std::format("'{}' is not a regular file",
                              args.xclbin.string()));
        }
    } catch (const std::exception &e) {
        error(std::format("Unexpected exception: {}", e.what()));
    }

    return args;
}

template <typename Integer>
constexpr void initialize_data(Integer **vectors, Integer *result,
                               std::uint64_t size, std::uint64_t num_vectors) {
    for (std::uint64_t i = 0; i < size; ++i) {
        result[i] = 0;
        for (std::uint64_t j = 0; j < num_vectors; ++j) {
            vectors[j][i] = i;
            result[i] += i;
        }
    }
}

template <typename Integer>
constexpr void benchmark(Integer **vectors, Integer *result,
                         std::uint64_t size, std::uint64_t num_vectors) {
    for (std::uint64_t i = 0; i < size; ++i) {
        result[i] = 0;
        for (std::uint64_t j = 0; j < num_vectors; ++j) {
            result[i] += vectors[j][i];
        }
    }
}
} // namespace

int main(int argc, char *argv[]) {
    struct arguments args = parse_args(argc, argv);
    constexpr std::uint64_t num_vectors = 32;
    Timer timer;

    std::println("Loading XCLBIN...");
    xrt::device device{0};
    xrt::uuid uuid = device.load_xclbin(args.xclbin);
    std::println("XCLBIN loaded!");

    std::println("Loading kernels...");
    xrt::kernel mm2s = xrt::kernel(device, uuid, "mm2s");

    xrt::kernel s2mm = xrt::kernel(device, uuid, "red40_s2mm");
    std::println("Kernels loaded!");

    std::println("Allocating memory...");

    xrtMemoryGroup bank = mm2s.group_id(1);
    xrt::bo bo = xrt::bo(device, args.size * sizeof(std::int32_t), bank);

    xrtMemoryGroup bank_result = s2mm.group_id(0);
    xrt::bo bo_result{device, args.size * sizeof(std::int32_t), bank_result};
    std::println("Memory allocated!");

    std::println("Initializing memory...");
    std::int32_t **vectors = new std::int32_t *[num_vectors];
    std::int32_t *result = new std::int32_t[args.size];
    for (std::uint64_t i = 0; i < num_vectors; ++i) {
        if (i == 0) {
            vectors[i] = bo.map<std::int32_t *>();
        } else {
            vectors[i] = new std::int32_t[args.size];
        }
    }

    initialize_data(vectors, result, args.size, num_vectors);

    bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    std::println("Memory initialized!");

    std::println("Creating runners...");
    xrt::run run_mm2s = xrt::run(mm2s);
    run_mm2s.set_arg(0, args.size);
    run_mm2s.set_arg(1, bo);

    xrt::run run_s2mm(s2mm);
    run_s2mm.set_arg(0, bo_result);
    run_s2mm.set_arg(1, args.size);
    std::println("Runners created!");

    std::println("Starting PL kernels...");

    std::this_thread::sleep_for(std::chrono::seconds(1));

    timer.time_point("start");
    run_mm2s.start();
    run_s2mm.start();

    const ert_cmd_state state_mm2s = run_mm2s.wait(std::chrono::seconds(5));
    const ert_cmd_state state_s2mm = run_s2mm.wait(std::chrono::seconds(5));
    timer.time_point("end");

    if (state_mm2s == ERT_CMD_STATE_TIMEOUT) {
        std::println("Warning: mm2s timed out!");
    }
    if (state_s2mm == ERT_CMD_STATE_TIMEOUT) {
        std::println("Warning: s2mm timed out!");
    }

    double exec_time = timer.time<milliseconds>("start", "end").count();

    std::println("Execution finished in {:.2f} ms!", exec_time);

    std::string name = "sum_vectors_one_pl";
    util::write_result_sum(name, args.size, exec_time);

    std::int32_t *result_benchmark = new std::int32_t[args.size];
    timer.time_point("start_cpu");
    benchmark(vectors, result_benchmark, args.size, num_vectors);
    timer.time_point("end_cpu");
    std::println("CPU benchmark finished in {:.2f} ms!",
                 timer.time<milliseconds>("start_cpu", "end_cpu").count());

    std::println("Checking result...");
    bo_result.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    std::int32_t *result_device = bo_result.map<std::int32_t *>();

    unsigned errors = 0;

    for (std::size_t i = 0; i < args.size; ++i) {
        if (result_device[i] != result[i]) {
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

    delete[] vectors;
    delete[] result;
    delete[] result_benchmark;

    return errors == 0 ? 0 : 1;
}
