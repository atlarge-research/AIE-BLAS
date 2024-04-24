cmake_minimum_required(VERSION 3.11)

project(aieblas CXX)

message(STATUS "Using compiler ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")

include(FetchContent)
# JSON library from https://github.com/nlohmann/json
FetchContent_Declare(json URL
    https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
    DOWNLOAD_EXTRACT_TIMESTAMP FALSE
    URL_HASH
    SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d
)
FetchContent_MakeAvailable(json)

# CLI options parser from https://github.com/jarro2783/cxxopts
FetchContent_Declare(cxxopts URL
    GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
    GIT_TAG 4bf61f08697b110d9e3991864650a405b3dd515d # v3.2.1
    DOWNLOAD_EXTRACT_TIMESTAMP FALSE
)
FetchContent_MakeAvailable(cxxopts)

set(AIEBLAS_SRC "${CMAKE_CURRENT_LIST_DIR}/src")
set(AIEBLAS_HDR "${CMAKE_CURRENT_LIST_DIR}/include")
set(AIEBLAS_INT_HDR "${AIEBLAS_HDR}/aieblas/detail")
set(AIEBLAS_COMMON_HEADERS
    "${AIEBLAS_INT_HDR}/util.hpp"
    "${AIEBLAS_INT_HDR}/util/cxx_compat.hpp"
)

set(AIEBLAS_CODEGEN_HEADERS
    "${AIEBLAS_HDR}/aieblas/codegen.hpp"
    "${AIEBLAS_INT_HDR}/codegen/datastructures.hpp"
    "${AIEBLAS_INT_HDR}/codegen/generator.hpp"
    "${AIEBLAS_INT_HDR}/codegen/json_parser.hpp"
)

add_executable(codegen
    "${AIEBLAS_SRC}/codegen/standalone.cpp"
    "${AIEBLAS_SRC}/codegen/codegen.cpp"
    "${AIEBLAS_SRC}/codegen/json_parser.cpp"
    ${AIEBLAS_COMMON_HEADERS}
    ${AIEBLAS_CODEGEN_HEADERS}
)

target_link_libraries(codegen PRIVATE
    nlohmann_json::nlohmann_json
    cxxopts::cxxopts
)

target_include_directories(codegen PRIVATE ${AIEBLAS_HDR})
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

if (cxx_std_26 IN_LIST CMAKE_CXX_COMPILE_FEATURES)
    message(STATUS "Using C++ 26 standard")
    target_compile_features(codegen PUBLIC cxx_std_26)
elseif (cxx_std_23 IN_LIST CMAKE_CXX_COMPILE_FEATURES)
    message(STATUS "Using C++ 23 standard")
    target_compile_features(codegen PUBLIC cxx_std_23)
elseif (cxx_std_20 IN_LIST CMAKE_CXX_COMPILE_FEATURES)
    message(STATUS "Using C++ 20 standard")
    target_compile_features(codegen PUBLIC cxx_std_20)
else ()
    message(STATUS "Using C++ 17 standard")
    message(STATUS "Falling back to {fmt} library")
    # std::format alternative from https://github.com/fmtlib/fmt
    FetchContent_Declare(fmt URL
        https://github.com/fmtlib/fmt/releases/download/10.2.1/fmt-10.2.1.zip
        DOWNLOAD_EXTRACT_TIMESTAMP FALSE
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

    target_compile_features(codegen PUBLIC cxx_std_17)
    target_link_libraries(codegen PRIVATE fmt_core)
endif ()
