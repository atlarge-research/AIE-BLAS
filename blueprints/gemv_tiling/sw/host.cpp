#include <chrono>
#include <cstdint>
#include <cxxopts.hpp>
#include <filesystem>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_uuid.h>

#include "util/cxx_compat.hpp"
#include "util/timer.hpp"

namespace fs = std::filesystem;

using seconds = std::chrono::duration<double>;
using milliseconds = std::chrono::duration<double, std::milli>;
using microseconds = std::chrono::duration<double, std::micro>;

namespace {

std::uint64_t n;

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
constexpr void initialize_data(Integer *A, Integer &alpha, Integer *x,
                               Integer &beta, Integer *y, Integer *result,
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
        Integer temp = 0;
        for (std::uint64_t j = 0; j < n; ++j) {
            temp += A[i * n + j] * x[j];
        }
        result[i] = alpha * temp + beta * y[i];
    }
}

template <typename Integer>
constexpr void benchmark(Integer *A, Integer &alpha, Integer *x,
                         Integer &beta, Integer *y, Integer *result,
                         std::uint64_t m) {
    for (std::uint64_t i = 0; i < m; ++i) {
        Integer temp = 0;
        for (std::uint64_t j = 0; j < n; ++j) {
            temp += A[i * n + j] * x[j];
        }
        result[i] = alpha * temp + beta * y[i];
    }
}
} // namespace

int main(int argc, char *argv[]) {
    struct arguments args = parse_args(argc, argv);
    n = 2560;
    Timer timer;

    std::println("Loading XCLBIN...");
    xrt::device device{0};
    xrt::uuid uuid = device.load_xclbin(args.xclbin);
    std::println("XCLBIN loaded!");

    std::println("Loading kernels...");
    xrt::kernel mm2s_A = xrt::kernel(device, uuid, "gemv_mm2s_A");
    xrt::kernel mm2s_scalar = xrt::kernel(device, uuid, "gemv_mm2s_scalar");
    xrt::kernel mm2s_x = xrt::kernel(device, uuid, "gemv_mm2s_x");
    xrt::kernel mm2s_y = xrt::kernel(device, uuid, "gemv_mm2s_y");
    xrt::kernel s2mm = xrt::kernel(device, uuid, "gemv_s2mm");
    std::println("Kernels loaded!");

    std::println("Allocating memory...");
    // get memory bank groups for device buffers
    xrtMemoryGroup bank_A = mm2s_A.group_id(0);
    xrtMemoryGroup bank_x = mm2s_x.group_id(0);
    xrtMemoryGroup bank_y = mm2s_y.group_id(0);
    xrtMemoryGroup bank_result = s2mm.group_id(0);

    // Create device buffers
    xrt::bo bo_A{device, args.size * n * sizeof(std::int32_t), bank_A};
    xrt::bo bo_x{device, n * sizeof(std::int32_t), bank_x};
    xrt::bo bo_y{device, args.size * sizeof(std::int32_t), bank_y};
    xrt::bo bo_result{device, args.size * sizeof(std::int32_t), bank_result};
    std::println("Memory allocated!");

    std::println("Initializing memory...");
    std::int32_t *A = bo_A.map<std::int32_t *>();
    std::int32_t *x = bo_x.map<std::int32_t *>();
    std::int32_t *y = bo_y.map<std::int32_t *>();
    std::int32_t *result = new std::int32_t[args.size];
    std::int32_t alpha, beta;

    initialize_data(A, alpha, x, beta, y, result, args.size);

    bo_A.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_x.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_y.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    std::println("Memory initialized!");

    std::println("Creating runners...");
    xrt::run run_mm2s_A(mm2s_A);
    run_mm2s_A.set_arg(0, bo_A);
    run_mm2s_A.set_arg(1, args.size);
    run_mm2s_A.set_arg(2, n);

    xrt::run run_mm2s_scalar(mm2s_scalar);
    run_mm2s_scalar.set_arg(0, n);
    run_mm2s_scalar.set_arg(1, alpha);
    run_mm2s_scalar.set_arg(2, beta);

    xrt::run run_mm2s_x(mm2s_x);
    run_mm2s_x.set_arg(0, bo_x);
    run_mm2s_x.set_arg(1, args.size);
    run_mm2s_x.set_arg(2, n);

    xrt::run run_mm2s_y(mm2s_y);
    run_mm2s_y.set_arg(0, bo_y);
    run_mm2s_y.set_arg(1, args.size);

    xrt::run run_s2mm(s2mm);
    run_s2mm.set_arg(0, bo_result);
    run_s2mm.set_arg(1, args.size);
    std::println("Runners created!");

    std::println("Starting PL kernels...");
    timer.time_point("start");
    run_mm2s_A.start();
    run_mm2s_scalar.start();
    run_mm2s_x.start();
    run_mm2s_y.start();
    run_s2mm.start();

    const auto state_mm2s_A = run_mm2s_A.wait(std::chrono::seconds(5));
    const auto state_mm2s_scalar = run_mm2s_scalar.wait(std::chrono::seconds(5));
    const auto state_mm2s_x = run_mm2s_x.wait(std::chrono::seconds(5));
    const auto state_mm2s_y = run_mm2s_y.wait(std::chrono::seconds(5));
    const auto state_s2mm = run_s2mm.wait(std::chrono::seconds(5));
    timer.time_point("end");

    if (state_mm2s_A == ERT_CMD_STATE_TIMEOUT) {
        std::println("Warning: mm2s_A timed out!");
    }
    if (state_mm2s_scalar == ERT_CMD_STATE_TIMEOUT) {
        std::println("Warning: mm2s_scalar timed out!");
    }
    if (state_mm2s_x == ERT_CMD_STATE_TIMEOUT) {
        std::println("Warning: mm2s_x timed out!");
    }
    if (state_mm2s_y == ERT_CMD_STATE_TIMEOUT) {
        std::println("Warning: mm2s_y timed out!");
    }
    if (state_s2mm == ERT_CMD_STATE_TIMEOUT) {
        std::println("Warning: s2mm timed out!");
    }

    std::println("Execution finished in {:.2f} ms!",
                 timer.time<milliseconds>("start", "end").count());

    std::int32_t *results_benchmark = new std::int32_t[args.size];
    timer.time_point("start_cpu");
    benchmark(A, alpha, x, beta, y, results_benchmark, args.size);
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

    delete[] result;
    delete[] results_benchmark;


    return errors == 0 ? 0 : 1;
}
