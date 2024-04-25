#pragma once
#include <iostream>
#include <string>
#include <sstream>

#if defined(__has_include) && __has_include(<version>)
#include <version>
#endif

/**
 * Create a definition for std::format from C++20.
 * Uses the {fmt} format library, which is mostly compatible with std::format.
 */
#ifdef __cpp_lib_format
#include <format>
#else
#include <fmt/format.h>
namespace std {
template <class... Args> using format_string = std::string_view;
template <typename... Args>
static inline std::string format(const std::string_view fmt, Args &&...args) {
    return fmt::format(fmt::runtime(fmt), std::forward<Args>(args)...);
}
} // namespace std
#endif

/**
 * Create a definition for std::runtime_format from C++26. In older versions
 * std::format already allows runtime format strings, so simply return the
 * string.
 */
#if !defined(__cpp_lib_format) || __cpp_lib_format < 202311L
#define CXX_COMPAT_NEEDS_RUNTIME_FORMAT 1
namespace std {
static inline std::string runtime_format(std::string_view fmt) {
    return std::string{fmt};
}
} // namespace std
#endif

/**
 * Create a definition for std::print and std::println from C++23.
 */
#ifdef __cpp_lib_print
#include <print>
#else
#define CXX_COMPAT_NEEDS_PRINT 1
namespace std {
template <typename... Args>
static inline void print(std::ostream &stream, std::format_string<Args...> fmt,
                         Args &&...args) {
    stream << std::format(std::runtime_format(fmt),
                          std::forward<Args>(args)...);
}

template <typename... Args>
static inline void print(std::format_string<Args...> fmt, Args &&...args) {
    print(std::cout, std::runtime_format(fmt), std::forward<Args>(args)...);
}

template <typename... Args>
static inline void println(std::ostream &stream,
                           std::format_string<Args...> fmt, Args &&...args) {
    stream << std::format(std::runtime_format(fmt),
                          std::forward<Args>(args)...) << "\n";
}

template <typename... Args>
static inline void println(std::format_string<Args...> fmt, Args &&...args) {
    println(std::cout, std::runtime_format(fmt), std::forward<Args>(args)...);
}
} // namespace std
#endif

namespace cxx_compat {
static inline void print_features() {
    bool supports_format = true;
#ifdef CXX_COMPAT_NEEDS_FORMAT
    supports_format = false;
#endif
    bool supports_runtime_format = true;
#ifdef CXX_COMPAT_NEEDS_RUNTIME_FORMAT
    supports_runtime_format = false;
#endif
    bool supports_print = true;
#ifdef CXX_COMPAT_NEEDS_PRINT
    supports_print = false;
#endif
    std::println("native {:<15}: {}", "format", supports_format);
    std::println("native {:<15}: {}", "runtime format",
                 supports_runtime_format);
    std::println("native {:<15}: {}", "print", supports_print);
}
}
