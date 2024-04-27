include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} -std=${DETECTED_CXX_STD}")

message(VERBOSE "Using CXX compiler flags: ${CMAKE_REQUIRED_FLAGS}")
check_cxx_source_compiles("
#include <iostream>
#include <format>

int main() {
    std::cout << std::format(\"{}\", \"hello world\") << std::endl;
    return 0;
}
" CXX_HAS_FORMAT
)
