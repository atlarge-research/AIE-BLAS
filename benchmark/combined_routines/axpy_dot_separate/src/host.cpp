#include <chrono>
#include <cstdint>
#include <cxxopts.hpp>
#include <filesystem>
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
constexpr void initialize_data(Integer *w, Integer *v, Integer *u,
                               Integer &alpha, Integer &result,
                               std::uint64_t size) {
    alpha = -2;
    result = 0;
    for (std::uint64_t i = 0; i < size; ++i) {
        w[i] = i;
        v[i] = i * 3;
        u[i] = 2;

        result += (v[i] + alpha * w[i]) * u[i];
    }
}

template <typename Integer>
constexpr void benchmark(const Integer *w, const Integer *v, const Integer *u,
                         const Integer &alpha, Integer &result,
                         std::uint64_t size) {
    result = 0;
    for (std::uint64_t i = 0; i < size; ++i) {
        result += (v[i] + alpha * w[i]) * u[i];
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
    xrt::kernel axpy_mm2s = xrt::kernel(device, uuid, "axpy_mm2s");
    xrt::kernel axpy_s2mm = xrt::kernel(device, uuid, "axpy_s2mm");
    xrt::kernel dot_mm2s = xrt::kernel(device, uuid, "dot_mm2s");
    xrt::kernel dot_s2mm = xrt::kernel(device, uuid, "dot_s2mm");
    std::println("Kernels loaded!");

    std::println("Allocating memory...");
    // get memory bank groups for device buffers
    xrtMemoryGroup bank_w = axpy_mm2s.group_id(1);
    xrtMemoryGroup bank_v = axpy_mm2s.group_id(2);
    xrtMemoryGroup bank_z = axpy_s2mm.group_id(0);
    xrtMemoryGroup bank_u = dot_mm2s.group_id(0);
    xrtMemoryGroup bank_result = dot_s2mm.group_id(0);

    // Create device buffers
    xrt::bo bo_w{device, args.size * sizeof(std::int32_t), bank_w};
    xrt::bo bo_v{device, args.size * sizeof(std::int32_t), bank_v};
    xrt::bo bo_z{device, args.size * sizeof(std::int32_t), bank_z};
    xrt::bo bo_u{device, args.size * sizeof(std::int32_t), bank_u};
    xrt::bo bo_result{device, sizeof(std::int32_t), bank_result};
    std::println("Memory allocated!");

    std::println("Initializing memory...");
    std::int32_t *w = bo_w.map<std::int32_t *>();
    std::int32_t *v = bo_v.map<std::int32_t *>();
    std::int32_t *u = bo_u.map<std::int32_t *>();
    std::int32_t alpha, result;

    initialize_data(w, v, u, alpha, result, args.size);

    bo_w.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_v.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_u.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    std::println("Memory initialized!");

    std::println("Creating runners...");

    xrt::run run_axpy_mm2s(axpy_mm2s);
    run_axpy_mm2s.set_arg(0, args.size);
    run_axpy_mm2s.set_arg(1, bo_w);
    run_axpy_mm2s.set_arg(2, bo_v);
    run_axpy_mm2s.set_arg(3, alpha);

    xrt::run run_axpy_s2mm(axpy_s2mm);
    run_axpy_s2mm.set_arg(0, bo_z);
    run_axpy_s2mm.set_arg(1, args.size);

    xrt::run run_dot_mm2s(dot_mm2s);
    run_dot_mm2s.set_arg(0, bo_z);
    run_dot_mm2s.set_arg(1, bo_u);
    run_dot_mm2s.set_arg(2, args.size);

    xrt::run run_dot_s2mm(dot_s2mm);
    run_dot_s2mm.set_arg(0, bo_result);
    std::println("Runners created!");

    std::println("Starting PL kernels...");
    timer.time_point("start");
    run_axpy_mm2s.start();
    run_axpy_s2mm.start();

    const auto state_axpy_mm2s = run_axpy_mm2s.wait(std::chrono::seconds(5));
    const auto state_axpy_s2mm = run_axpy_s2mm.wait(std::chrono::seconds(5));

    timer.time_point("mid");

    run_dot_mm2s.start();
    run_dot_s2mm.start();
    const auto state_dot_mm2s = run_dot_mm2s.wait(std::chrono::seconds(5));
    const auto state_dot_s2mm = run_dot_s2mm.wait(std::chrono::seconds(5));
    timer.time_point("end");

    if (state_axpy_mm2s == ERT_CMD_STATE_TIMEOUT) {
        std::println("Warning: axpy mm2s timed out!");
    }
    if (state_axpy_s2mm == ERT_CMD_STATE_TIMEOUT) {
        std::println("Warning: axpy s2mm timed out!");
    }
    if (state_dot_mm2s == ERT_CMD_STATE_TIMEOUT) {
        std::println("Warning: dot mm2s timed out!");
    }
    if (state_dot_s2mm == ERT_CMD_STATE_TIMEOUT) {
        std::println("Warning: dot s2mm timed out!");
    }

    double exec_time = timer.time<milliseconds>("start", "end").count();

    std::println("Execution finished in {:.2f} ms! ({:.2f} ms & {:.2f} ms)",
                 exec_time,
                 timer.time<milliseconds>("start", "mid").count(),
                 timer.time<milliseconds>("mid", "end").count());

    std::string name = "axpydot_separate";
    util::write_result_axpydot(name, args.size, exec_time);

    std::int32_t result_benchmark;
    timer.time_point("start_cpu");
    benchmark(w, v, u, alpha, result_benchmark, args.size);
    timer.time_point("end_cpu");
    std::println("CPU benchmark finished in {:.2f} ms!",
                 timer.time<milliseconds>("start_cpu", "end_cpu").count());

    std::println("Checking result...");
    bo_result.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    std::int32_t *result_device = bo_result.map<std::int32_t *>();

    unsigned errors = 0;
    if (*result_device != result) {
        std::println("FAIL! ({} != {})", *result_device, result);
        errors += 1;
    } else {
        std::println("{} (correct)", result);
    }

    if (errors == 0) {
        std::println("OK!");
    } else {
        std::println("{} failures!", errors);
    }


    return errors == 0 ? 0 : 1;
}
