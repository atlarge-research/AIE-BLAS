#include <fstream>
#include "aieblas/detail/util.hpp"
#include "aieblas/detail/codegen/generator.hpp"
#include "aieblas/detail/codegen/kernels.hpp"

namespace aieblas {
namespace codegen {

void generator::generate_cmake() {
    fs::path cmake_file = out_dir / "CMakeLists.txt";

    this->open(cmake_file, comment_type::HASHTAG);
    this->println("cmake_minimum_required(VERSION 3.22)");
    this->println();
    this->println("project(aieblas)");
    this->println();

    this->println<INCREASE_AFTER>("if (NOT EXISTS $ENV{{XILINX_VITIS}})");
    this->println("message(FATAL_ERROR \"Xilinx Vitis not found, make sure to "
                  "source setup.sh\")");
    this->println<DECREASE_BEFORE>("endif ()");
    this->println();

    this->println("set(PLATFORM /opt/xilinx/platforms/{0}/{0}.xpfm)",
                  this->d.platform);
    this->println("set(PL_FREQ 500)"); // TODO: Make variable?
    this->println();

    this->println<INCREASE_AFTER>("set(AIE_KERNELS");
    for (const fs::path &src : this->kernel_srcs) {
        this->println("{}", fs::relative(src, out_dir).native());
    }
    for (const fs::path &hdr : this->kernel_hdrs) {
        this->println("{}", fs::relative(hdr, out_dir).native());
    }
    this->println<DECREASE_BEFORE>(")");

    this->println<INCREASE_AFTER>("set(PL_KERNELS");
    for (const fs::path &krnl : this->pl_kernels) {
        this->println("{}", fs::relative(krnl, out_dir).native());
    }
    this->println<DECREASE_BEFORE>(")");
    this->println();

    this->println("set(VPP \"$ENV{{XILINX_VITIS}}/bin/v++\")");
    this->println("set(AIECC \"$ENV{{XILINX_VITIS}}/aietools/bin/"
                  "aiecompiler\")");
    this->println("set(AIESIM \"$ENV{{XILINX_VITIS}}/aietools/bin/"
                  "aiesimulator\")");
    this->println("set(X86SIM \"$ENV{{XILINX_VITIS}}/aietools/bin/"
                  "x86simulator\")");
    this->println();

    this->println("set(AIE_INCLUDE "
                  "\"-include=\\\"$ENV{{XILINX_VITIS}}/aietools/include\\\"\" "
                  "\"-include=\\\"${{CMAKE_CURRENT_SOURCE_DIR}}/aie\\\"\" "
                  "\"-include=\\\"${{CMAKE_CURRENT_SOURCE_DIR}}/data\\\"\" "
                  "\"-include=\\\"${{CMAKE_CURRENT_SOURCE_DIR}}/aie/"
                  "kernels\\\"\")");
    this->println("set(AIE_FLAGS ${{AIE_INCLUDE}} --platform ${{PLATFORM}} "
                  "--verbose=true --pl-freq=${{PL_FREQ}} -log-level=5 "
                  "--output=graph.json)");
    this->println("set(VPP_FLAGS --platform ${{PLATFORM}} --save-temps "
                  "--hls.jobs 8 --vivado.synth.jobs 8 --vivado.impl.jobs 8 "
                  "-g -t hw --log_dir log --temp_dir tmp --report_dir report)");
    this->println();

    this->println("#######");
    this->println("# AIE #");
    this->println("#######");
    this->println();
    this->println<INCREASE_AFTER>("add_custom_target(");
    this->println("xilinx");
    this->println("COMMAND mkdir -p xilinx");
    this->println("BYPRODUCTS xilinx");
    this->println("VERBATIM");
    this->println<DECREASE_BEFORE>(")");
    this->println();

    this->println("# HW / HW emu");
    this->println<INCREASE_AFTER>("add_custom_command(");
    this->println("OUTPUT xilinx/libadf.a");
    this->println("COMMAND ${{AIECC}} ${{AIE_FLAGS}} "
                  "--output-archive=libadf.a -workdir=./Work "
                  "--target=hw \"${{CMAKE_CURRENT_SOURCE_DIR}}/aie/"
                  "graph.cpp\"");
    this->println("MAIN_DEPENDENCY aie/graph.cpp");
    this->println("DEPENDS xilinx aie/graph.hpp ${{AIE_KERNELS}}");
    this->println("COMMENT \"Building ADF graph for HW/HW-emulation "
                  "xilinx/libadf_x86.a\"");
    this->println("WORKING_DIRECTORY xilinx");
    this->println("VERBATIM");
    this->println<DECREASE_BEFORE>(")");
    this->println();

    this->println("# SW emu");
    this->println<INCREASE_AFTER>("add_custom_command(");
    this->println("OUTPUT xilinx/libadf_x86.a");
    this->println("COMMAND ${{AIECC}} ${{AIE_FLAGS}} "
                  "--output-archive=libadf_x86.a -workdir=./Work_sw "
                  "--target=x86sim \"${{CMAKE_CURRENT_SOURCE_DIR}}/aie/"
                  "graph.cpp\"");
    this->println("MAIN_DEPENDENCY aie/graph.cpp");
    this->println("DEPENDS xilinx aie/graph.h ${{AIE_KERNELS}}");
    this->println("COMMENT \"Building ADF graph for SW-emulation "
                  "xilinx/libadf_x86.a\"");
    this->println("WORKING_DIRECTORY xilinx");
    this->println("VERBATIM");
    this->println<DECREASE_BEFORE>(")");
    this->println();

    this->println("######");
    this->println("# PL #");
    this->println("######");
    this->println();
    this->println("set(KERNEL_OBJECTS)");
    this->println("set(KERNEL_OBJECTS_PATH)");
    this->println<INCREASE_AFTER>("foreach(kernel ${{PL_KERNELS}})");
    this->println("get_filename_component(name ${{kernel}} NAME_WE)");
    this->println();
    this->println<INCREASE_AFTER>("add_custom_command(");
    this->println("OUTPUT xilinx/${{name}}.xo");
    this->println("COMMAND ${{VPP}} ${{VPP_FLAGS}} -c -k ${{name}} "
                  "${{CMAKE_CURRENT_SOURCE_DIR}}/${{kernel}} -o ${{name}}.xo");
    this->println("MAIN_DEPENDENCY ${{kernel}}");
    this->println("DEPENDS xilinx");
    this->println("COMMENT \"Building PL kernel xilinx/${{name}}.xo\"");
    this->println("WORKING_DIRECTORY xilinx");
    this->println("VERBATIM");
    this->println<DECREASE_BEFORE>(")");
    this->println("list(APPEND KERNEL_OBJECTS ${{name}}.xo)");
    this->println("list(APPEND KERNEL_OBJECTS_PATH xilinx/${{name}}.xo)");
    this->println<DECREASE_BEFORE>("endforeach()");
    this->println();

    this->println("##########");
    this->println("# XCLBIN #");
    this->println("##########");
    this->println();
    this->println<INCREASE_AFTER>("add_custom_command(");
    this->println("OUTPUT xilinx/aieblas.xsa");
    this->println("COMMAND ${{VPP}} ${{VPP_FLAGS}} -l ${{KERNEL_OBJECTS}} "
                  "libadf.a --config ${{CMAKE_CURRENT_SOURCE_DIR}}/link.cfg "
                  "-o aieblas.xsa");
    this->println("DEPENDS xilinx xilinx/libadf.a ${{KERNEL_OBJECTS_PATH}} "
                  "link.cfg");
    this->println("COMMENT \"Linking FPGA design xilinx/aieblas.xsa\"");
    this->println("WORKING_DIRECTORY xilinx");
    this->println("VERBATIM");
    this->println<DECREASE_BEFORE>(")");
    this->println();


    this->println<INCREASE_AFTER>("add_custom_command(");
    this->println("OUTPUT xilinx/aieblas.xclbin");
    this->println("COMMAND ${{VPP}} -p ${{VPP_FLAGS}} --package.boot_mode=ospi "
                  "aieblas.xsa libadf.a -o aieblas.xclbin");
    this->println("DEPENDS xilinx xilinx/aieblas.xsa xilinx/libadf.a");
    this->println("COMMENT \"Packaging FPGA design xilinx/aieblas.xclbin\"");
    this->println("WORKING_DIRECTORY xilinx");
    this->println("VERBATIM");
    this->println<DECREASE_BEFORE>(")");
    this->println();

    this->println();
    this->println<INCREASE_AFTER>("add_custom_target(aie");
    this->println("COMMAND cp --preserve --update aieblas.xclbin "
                  "${{CMAKE_BINARY_DIR}}/aieblas.xclbin");
    this->println("DEPENDS xilinx xilinx/aieblas.xclbin");
    this->println("WORKING_DIRECTORY xilinx");
    this->println("VERBATIM");
    this->println<DECREASE_BEFORE>(")");

    this->close();

}

} // codegen
} // aieblas
