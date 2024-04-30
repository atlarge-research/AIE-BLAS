cmake_minimum_required(VERSION 3.22)
if (${CMAKE_VERSION} VERSION_GREATER "3.24")
# Set new FetchContent_Declare option DOWNLOAD_EXTRACT_TIMESTAMP to false
cmake_policy(SET CMP0135 NEW)
endif ()

project(aieblas CXX)

message(STATUS
    "Using compiler ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")

include(FetchContent)
# JSON library from https://github.com/nlohmann/json
FetchContent_Declare(json URL
    https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
    URL_HASH
    SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d
)
FetchContent_MakeAvailable(json)

# CLI options parser from https://github.com/jarro2783/cxxopts
FetchContent_Declare(cxxopts URL
    GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
    GIT_TAG 4bf61f08697b110d9e3991864650a405b3dd515d # v3.2.1
)
FetchContent_MakeAvailable(cxxopts)

set(AIEBLAS_SRC "${CMAKE_CURRENT_LIST_DIR}/src")
set(AIEBLAS_HDR "${CMAKE_CURRENT_LIST_DIR}/include")
set(AIEBLAS_INT_HDR "${AIEBLAS_HDR}/aieblas/detail")
set(AIEBLAS_COMMON_HEADERS
    "${AIEBLAS_INT_HDR}/util.hpp"
    "${AIEBLAS_INT_HDR}/util/cxx_compat.hpp"
    "${AIEBLAS_INT_HDR}/util/logging.hpp"
)

set(AIEBLAS_COMMON_SRC
    "${AIEBLAS_SRC}/util/logging.cpp"
)

set(AIEBLAS_CODEGEN_HEADERS
"${AIEBLAS_HDR}/aieblas/codegen.hpp"
"${AIEBLAS_INT_HDR}/codegen/datastructures.hpp"
"${AIEBLAS_INT_HDR}/codegen/generator.hpp"
"${AIEBLAS_INT_HDR}/codegen/json_parser.hpp"
"${AIEBLAS_INT_HDR}/codegen/kernels.hpp"
"${AIEBLAS_INT_HDR}/codegen/kernels/scale.hpp"
)

add_library(codegen STATIC
    "${AIEBLAS_SRC}/codegen/codegen.cpp"
    "${AIEBLAS_SRC}/codegen/generator.cpp"
    "${AIEBLAS_SRC}/codegen/json_parser.cpp"
    "${AIEBLAS_SRC}/codegen/kernels/generate_kernels.cpp"
    "${AIEBLAS_SRC}/codegen/kernels/get_kernel_generator.cpp"
    "${AIEBLAS_SRC}/codegen/kernels/impl/scale.cpp"
    ${AIEBLAS_COMMON_SRC}
    ${AIEBLAS_COMMON_HEADERS}
    ${AIEBLAS_CODEGEN_HEADERS}
)

add_executable(codegen_standalone
"${AIEBLAS_SRC}/codegen/standalone.cpp"
)

set_property(TARGET codegen_standalone PROPERTY OUTPUT_NAME codegen)

target_link_libraries(codegen PRIVATE
nlohmann_json::nlohmann_json
)

target_link_libraries(codegen_standalone PRIVATE
codegen
cxxopts::cxxopts
)

target_include_directories(codegen PUBLIC ${AIEBLAS_HDR})

target_compile_options(codegen PRIVATE
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:
        -Wall
        -Wextra
        -pedantic
        -Wno-unused-function
        -Wno-unused-parameter
        $<$<CONFIG:Debug>:-g3 -ggdb>
    >
)

target_compile_options(codegen_standalone PRIVATE
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:
        -Wall
        -Wextra
        -pedantic
        -Wno-unused-function
        -Wno-unused-parameter
        $<$<CONFIG:Debug>:-g3 -ggdb>
    >
)

if (cxx_std_23 IN_LIST CMAKE_CXX_COMPILE_FEATURES)
    message(STATUS "Using C++ 23 standard")
    target_compile_features(codegen PUBLIC cxx_std_23)
    set(DETECTED_CXX_STD "c++23")
elseif (cxx_std_20 IN_LIST CMAKE_CXX_COMPILE_FEATURES)
    message(STATUS "Using C++ 20 standard")
    target_compile_features(codegen PUBLIC cxx_std_20)
    set(DETECTED_CXX_STD "c++20")
else ()
    message(STATUS "Using C++ 17 standard")
    target_compile_features(codegen PUBLIC cxx_std_17)
    set(DETECTED_CXX_STD "c++17")
endif ()

include(cmake/check_source_code.cmake)
if (NOT CXX_HAS_SOURCE_LOCATION)
    message(FATAL_ERROR
        "AIE-BLAS requires a compiler with support for C++20 std::source_location")
endif ()

include(cmake/check_format.cmake)

if (NOT CXX_HAS_FORMAT)
    message(STATUS "Falling back to {fmt} library")
    # std::format alternative from https://github.com/fmtlib/fmt
    FetchContent_Declare(fmt URL
        https://github.com/fmtlib/fmt/releases/download/10.2.1/fmt-10.2.1.zip
        URL_HASH
        SHA256=312151a2d13c8327f5c9c586ac6cf7cddc1658e8f53edae0ec56509c8fa516c9
    )
    FetchContent_GetProperties(fmt)
    if (NOT fmt_POPULATED)
        FetchContent_Populate(fmt)
    endif ()
    add_library(fmt_core OBJECT EXCLUDE_FROM_ALL
                ${fmt_SOURCE_DIR}/src/format.cc)
    target_include_directories(fmt_core PUBLIC ${fmt_SOURCE_DIR}/include)
    target_compile_features(fmt_core PUBLIC cxx_std_17)
    target_link_libraries(codegen PRIVATE fmt_core)
    target_link_libraries(codegen_standalone PRIVATE fmt_core)
endif ()
