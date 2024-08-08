cmake_minimum_required(VERSION 3.22)
if (${CMAKE_VERSION} VERSION_GREATER "3.24")
# Set new FetchContent_Declare option DOWNLOAD_EXTRACT_TIMESTAMP to false
cmake_policy(SET CMP0135 NEW)
endif ()

project(sum_benchmark CXX)

message(STATUS
    "Using compiler ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")

#####################
# Load dependencies #
#####################
include(FetchContent)
# CLI options parser from https://github.com/jarro2783/cxxopts
FetchContent_Declare(cxxopts URL
    GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
    GIT_TAG 4bf61f08697b110d9e3991864650a405b3dd515d # v3.2.1
)
FetchContent_MakeAvailable(cxxopts)


###############
# Add targets #
###############
include(../OpenBLAS/lib/cmake/openblas/OpenBLASConfig.cmake)

add_executable(bench
    "src/sum_vectors.cpp"
    "include/cxx_compat.hpp"
    "include/timer.hpp"
)

target_include_directories(bench PRIVATE ${OpenBLAS_INCLUDE_DIRS} include)

target_link_libraries(bench PRIVATE cxxopts::cxxopts)
target_link_libraries(bench PRIVATE ${OpenBLAS_LIBRARIES})

###########################
# Set debug/warning flags #
###########################
target_compile_options(bench PRIVATE
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:
        -Wall
        -Wextra
        -Wno-unused-function
        -Wno-unused-parameter
        $<$<CONFIG:Debug>:-g3 -ggdb>
    >
)

######################################
# Set latest C++ versions to utilize #
# new features                       #
######################################
if (cxx_std_23 IN_LIST CMAKE_CXX_COMPILE_FEATURES)
    message(STATUS "Using C++ 23 standard")
    target_compile_features(bench PUBLIC cxx_std_23)
    set(DETECTED_CXX_STD "c++23")
elseif (cxx_std_20 IN_LIST CMAKE_CXX_COMPILE_FEATURES)
    message(STATUS "Using C++ 20 standard")
    target_compile_features(bench PUBLIC cxx_std_20)
    set(DETECTED_CXX_STD "c++20")
else ()
    message(STATUS "Using C++ 17 standard")
    target_compile_features(bench PUBLIC cxx_std_17)
    set(DETECTED_CXX_STD "c++17")
endif ()

#####################################
# Fall back to {fmt} if std::format #
# is not available                  #
#####################################
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
    target_link_libraries(bench PRIVATE fmt_core)
endif ()
