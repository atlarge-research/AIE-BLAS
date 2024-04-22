
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <fstream>

#include "experimental/xrt_kernel.h"
#include "experimental/xrt_uuid.h"
#include "ground_truth.h"

int main(int argc, char **argv)
{
    //////////////////////////////////////////
    // Open xclbin
    //////////////////////////////////////////
    // const std::string device_str = "0000:86:00.1";
    unsigned int dev_index = 0;
    printf("Opening device %d (change if needed) and loading xclbin\n", dev_index);
    xrt::device device = xrt::device(dev_index); // <-- change if needed
    xrt::uuid xclbin_uuid = device.load_xclbin("a.xclbin");

    printf("XCLBIN loaded\n");
    int sizeIn = SAMPLES;
    int sizeOut = SAMPLES;
    int scalar = SCALAR;

    // create PL kernels
    printf("Creating kernels...\n");
    xrt::kernel s2mm_kernel = xrt::kernel(device, xclbin_uuid, "s2mm");
    xrt::kernel mm2s_kernel = xrt::kernel(device, xclbin_uuid, "mm2s");

    // get memory bank groups for device buffers
    xrtMemoryGroup bank_input = mm2s_kernel.group_id(0);
    xrtMemoryGroup bank_result = s2mm_kernel.group_id(0);

    // Create device buffers and copy input data

    printf("Allocating input buffer of size %ld (%d elements)\n", sizeIn * sizeof(int32_t), sizeIn);
    xrt::bo in_bohdl = xrt::bo(device, cInput, sizeIn * sizeof(int32_t), xrt::bo::flags::normal, bank_input);
    in_bohdl.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    xrt::bo out_bohdl = xrt::bo(device, sizeOut * sizeof(int32_t), xrt::bo::flags::normal, bank_result);

    // create kernel runners
    printf("Creating runners ...\n");
    xrt::run run_mm2s_kernel(mm2s_kernel);
    xrt::run run_s2mm_kernel(s2mm_kernel);

    // set kernel arguments
    run_mm2s_kernel.set_arg(0, in_bohdl);
    run_mm2s_kernel.set_arg(1, sizeIn);
    run_mm2s_kernel.set_arg(2, scalar);

    run_s2mm_kernel.set_arg(0, out_bohdl);
    run_s2mm_kernel.set_arg(1, sizeOut);

    // start kernels

    printf("Starting PL kernels ...\n");
    run_mm2s_kernel.start();
    run_s2mm_kernel.start();
    printf("...Started. Waiting for kernel termination...\n");

    // wait for kernels
    run_mm2s_kernel.wait();
    printf("First one finished\n");
    run_s2mm_kernel.wait();

    printf("Execution finished!\n");

    // copy back results
    out_bohdl.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    int32_t *results = out_bohdl.map<int32_t *>();

    // Comparing the execution data to the golden data

    int errorCount = 0;
    {
        for (int i = 0; i < sizeOut; i++)
        {
            if (results[i] != cGolden[i])
            {
                printf("Error found @ %d, %d != %d\n", i, results[i], cGolden[i]);
                errorCount++;
            }
        }

        if (errorCount)
            printf("Test failed with %d errors\n", errorCount);
        else
            printf("TEST PASSED\n");
    }

    return errorCount;
}
