include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} -std=${DETECTED_CXX_STD}")

message(VERBOSE "Using CXX compiler flags: ${CMAKE_REQUIRED_FLAGS}")
check_cxx_source_compiles("
#include <iostream>
#include <source_location>

int main() {
    const std::source_location location = std::source_location::current();
    std::cout << \"file: \"
              << location.file_name() << \"(\"
              << location.line() << \":\"
              << location.column() << \") \"
              << location.function_name() << std::endl;
    return 0;
}
" CXX_HAS_SOURCE_LOCATION
)
