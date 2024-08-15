#include <cblas.h>
#include <chrono>
#include <cstdint>
#include <cxxopts.hpp>
#include <filesystem>
#include <random>
#include <thread>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_uuid.h>

#include "cxx_compat.hpp"
#include "timer.hpp"

namespace fs = std::filesystem;

using seconds = std::chrono::duration<double>;
using milliseconds = std::chrono::duration<double, std::milli>;
using microseconds = std::chrono::duration<double, std::micro>;

namespace {

constexpr const std::uint64_t n = 64;

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

template <typename Number>
constexpr void initialize_data(Number *A, Number &alpha, Number *x,
                               Number &beta, Number *y, std::uint64_t m) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1000.0f, 1000.0f);

    alpha = dis(gen);
    beta = dis(gen);

    for (std::uint64_t i = 0; i < m; ++i) {
        y[i] = dis(gen);
        for (std::uint64_t j = 0; j < n; ++j) {
            A[i * n + j] = dis(gen);
        }
    }

    for (std::uint64_t j = 0; j < n; ++j) {
        x[j] = dis(gen);
    }
}
} // namespace

int main(int argc, char *argv[]) {
    struct arguments args = parse_args(argc, argv);
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
    xrt::bo bo_A{device, args.size * n * sizeof(float), bank_A};
    xrt::bo bo_x{device, n * sizeof(float), bank_x};
    xrt::bo bo_y{device, args.size * sizeof(float), bank_y};
    xrt::bo bo_result{device, args.size * sizeof(float), bank_result};
    std::println("Memory allocated!");

    std::println("Initializing memory...");
    float *A = bo_A.map<float *>();
    float *x = bo_x.map<float *>();
    float *y = bo_y.map<float *>();
    float *result = new float[args.size];
    float alpha, beta;

    initialize_data(A, alpha, x, beta, y, args.size);

    bo_A.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_x.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_y.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    std::println("Memory initialized!");

    std::println("Creating runners...");
    xrt::run run_mm2s_A(mm2s_A);
    run_mm2s_A.set_arg(0, bo_A);
    run_mm2s_A.set_arg(1, args.size);

    xrt::run run_mm2s_scalar(mm2s_scalar);
    run_mm2s_scalar.set_arg(0, alpha);
    run_mm2s_scalar.set_arg(1, beta);

    xrt::run run_mm2s_x(mm2s_x);
    run_mm2s_x.set_arg(0, bo_x);
    run_mm2s_x.set_arg(1, args.size);

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

    double exec_time = timer.time<milliseconds>("start", "end").count();

    std::println("Execution finished in {:.2f} ms!", exec_time);

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

    cblas_scopy(args.size, y, 1, result, 1);
    cblas_sgemv(CblasRowMajor, CblasNoTrans, args.size, n, alpha, A, n, x, 1, beta, result, 1);

    std::println("Checking result...");
    bo_result.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    float *result_device = bo_result.map<float *>();

    unsigned errors = 0;
    for (std::size_t i = 0; i < args.size; ++i) {
        if (std::fabs(result_device[i] - result[i]) >= 1e6) {
            std::println("FAIL! ({} != {})", result_device[i], result[i]);
            errors += 1;
        }
    }

    if (errors == 0) {
        std::println("OK!");
    } else {
        std::println("{} failures!", errors);
    }

    delete[] result;

    return errors == 0 ? 0 : 1;
}
