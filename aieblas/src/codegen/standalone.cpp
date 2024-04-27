#include <algorithm>
#include <cctype>
#include <cxxopts.hpp>

#include "aieblas/codegen.hpp"
#include "aieblas/detail/util.hpp"

namespace cg = aieblas::codegen;

namespace {
struct arguments {
    fs::path json;
    fs::path output;
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
        cxxopts::Options options("codegen",
                                "AIE engine code generator for aieblas");

        options.add_options()
            ("json", "JSON file containing BLAS routines",
             cxxopts::value<std::string>())
            ("output", "Output directory", cxxopts::value<std::string>())
            ("l,log-level", "Set the logging level",
             cxxopts::value<std::string>())
            ("h,help", "Print usage");

        options.positional_help("<json> <output>");
        options.parse_positional({"json", "output"});
        auto results = options.parse(argc, argv);

        if (results.count("help")) {
            std::println("{}", options.help());
            std::println("Positional arguments:");
            std::println("  <json>   JSON file containing BLAS routines");
            std::print  ("  <output> Output directory\n\n");
            std::fflush(stdout);
            exit(0);
        }

        if (results.count("log-level")) {
            std::string level_str = results["log-level"].as<std::string>();
            std::transform(level_str.begin(), level_str.end(),
                           level_str.begin(), ::tolower);
            aieblas::log_level level = aieblas::log_level_from_str(level_str);
            if (level == aieblas::log_level::unknown) {
                error(std::format("'{}' is not a supported log level",
                                  level_str));
            }
            aieblas::set_log_level(level);
        }

        if (aieblas::get_log_level() <= aieblas::log_level::debug) {
            std::println("C++ features:");
            cxx_compat::print_features();
            std::println("");
        }

        log(aieblas::log_level::verbose, "Log level: {}",
            aieblas::log_level_to_str(aieblas::get_log_level()));

        if (!results.count("json")) {
            error_required("json");
        }

        if (!results.count("output")) {
            error_required("output");
        }

        args = {fs::path(results["json"].as<std::string>()),
                fs::path(results["output"].as<std::string>())};

        if (!fs::exists(args.json)) {
            error(std::format("File '{}' does not exist", args.json.string()));
        }

        fs::path json_file = fs::is_symlink(args.json) ?
                            fs::read_symlink(args.json) : args.json;

        if (!fs::is_regular_file(json_file)) {
            error(std::format("'{}' is not a regular file",
                            args.json.string()));
        }

        if (!fs::exists(args.output)) {
            fs::path output_abs = fs::absolute(args.output);
            if (!output_abs.has_parent_path() ||
                    !fs::exists(output_abs.parent_path())) {
                error(std::format("Parent of '{}' does not exist",
                                args.output.string()));
            }

            fs::create_directory(args.output);
        }

        fs::path output_file = fs::is_symlink(args.output) ?
                            fs::read_symlink(args.output) : args.output;
        if (!fs::is_directory(output_file)) {
            error(std::format("'{}' is not a directory",
                            args.output.string()));
        }
    } catch (const std::exception &e) {
        error(std::format("Unexpected exception: {}", e.what()));
    }

    return args;
}
} // namespace


int main(int argc, char *argv[]) {
    struct arguments args = parse_args(argc, argv);

    // try {
    cg::codegen(args.json, args.output);
    // } catch (const std::exception &e) {
    //     std::println("Codegen error: {}", e.what());
    // }

    return 0;
}
