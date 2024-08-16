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
constexpr void initialize_data(Number *x, Number *y, Number &c, Number &s,
                               std::uint64_t size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1000.0f, 1000.0f);

    c = dis(gen);
    s = dis(gen);
    for (std::uint64_t i = 0; i < size; ++i) {
        x[i] = dis(gen);
        y[i] = dis(gen);
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
    xrt::kernel mm2s = xrt::kernel(device, uuid, "rot_mm2s");
    xrt::kernel s2mm = xrt::kernel(device, uuid, "rot_s2mm");
    std::println("Kernels loaded!");

    std::println("Allocating memory...");
    // get memory bank groups for device buffers
    xrtMemoryGroup bank_x = mm2s.group_id(0);
    xrtMemoryGroup bank_y = mm2s.group_id(1);
    xrtMemoryGroup bank_result_x = s2mm.group_id(0);
    xrtMemoryGroup bank_result_y = s2mm.group_id(1);

    // Create device buffers
    xrt::bo bo_x{device, args.size * sizeof(float), bank_x};
    xrt::bo bo_y{device, args.size * sizeof(float), bank_y};
    xrt::bo bo_result_x{device, args.size * sizeof(float), bank_result_x};
    xrt::bo bo_result_y{device, args.size * sizeof(float), bank_result_y};
    std::println("Memory allocated!");

    std::println("Initializing memory...");
    float *x = bo_x.map<float *>();
    float *y = bo_y.map<float *>();
    float *result_x = new float[args.size];
    float *result_y = new float[args.size];
    float c, s;

    initialize_data(x, y, c, s, args.size);

    bo_x.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_y.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    std::println("Memory initialized!");

    std::println("Creating runners...");
    xrt::run run_mm2s(mm2s);
    run_mm2s.set_arg(0, bo_x);
    run_mm2s.set_arg(1, bo_y);
    run_mm2s.set_arg(2, c);
    run_mm2s.set_arg(3, s);
    run_mm2s.set_arg(4, args.size);

    xrt::run run_s2mm(s2mm);
    run_s2mm.set_arg(0, bo_result_x);
    run_s2mm.set_arg(1, bo_result_y);
    run_s2mm.set_arg(2, args.size);
    std::println("Runners created!");

    std::println("Starting PL kernels...");

    std::this_thread::sleep_for(std::chrono::seconds(1));

    timer.time_point("start");
    run_mm2s.start();
    run_s2mm.start();

    const auto state_mm2s = run_mm2s.wait(std::chrono::seconds(5));
    const auto state_s2mm = run_s2mm.wait(std::chrono::seconds(5));
    timer.time_point("end");

    if (state_mm2s == ERT_CMD_STATE_TIMEOUT) {
        std::println("Warning: mm2s timed out!");
    }
    if (state_s2mm == ERT_CMD_STATE_TIMEOUT) {
        std::println("Warning: s2mm timed out!");
    }

    double exec_time = timer.time<milliseconds>("start", "end").count();

    std::println("Execution finished in {:.2f} ms!", exec_time);

    cblas_scopy(args.size, x, 1, result_x, 1);
    cblas_scopy(args.size, y, 1, result_y, 1);
    cblas_srot(args.size, result_x, 1, result_y, 1, c, s);

    std::println("Checking result...");
    bo_result_x.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_result_y.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    float *result_device_x = bo_result_x.map<float *>();
    float *result_device_y = bo_result_y.map<float *>();

    unsigned errors = 0;
    for (std::size_t i = 0; i < args.size; ++i) {
        if (std::fabs(result_device_x[i] - result_x[i]) >= std::fabs(result_x[i]) * 1e-4) {
            std::println("FAIL x[{}]! ({} != {})", i, result_device_x[i], result_x[i]);
            errors += 1;
        }
        if (std::fabs(result_device_y[i] - result_y[i]) >= std::fabs(result_y[i]) * 1e-4) {
            std::println("FAIL y[{}]! ({} != {})", i, result_device_y[i], result_y[i]);
            errors += 1;
        }
    }

    if (errors == 0) {
        std::println("OK!");
    } else {
        std::println("{} failures!", errors);
    }

    delete[] result_x;
    delete[] result_y;


    return errors == 0 ? 0 : 1;
}
