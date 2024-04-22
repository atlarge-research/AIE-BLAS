
This mini application is inspired by the example shown in AMD/Xilinx Documentation (Chapter 3 of "Versal ACAP AI Engine Programming Environment User Guide (UG1076)" for v2022.1, or Chapter 7 of "AI Engine Kernel and Graph Programming Guide (UG1079)" for v2022.2+).



It is composed by two identical kernels computing over a sequence of complex numbers. Being `c1` the input number and `c2` the output number:
```C++
    c2.real = c1.real+c1.imag;
    c2.imag = c1.real-c1.imag;
```

Note: The kernels communicate through _windows_. From v2023.1 of Vitis, they are renamed as _buffer_ but the behavior is the same.

Resources:
- "AI Engine Kernel and Graph Programming Guide (UG1079)" 
- for more details Vitis examples https://github.com/Xilinx/Vitis-Tutorials/tree/2022.2/AI_Engine_Development/Feature_Tutorials/05-AI-engine-versal-integration. **Note:** that example is based on VCK190 and covers also *Software and Hardware Emulation* a fast functional simulation of the entire system via QEMU, including the AI Engine graph, the PL logic along with XRT-based host application to control the AI Engine, and PL. This is not covered by this example


## Configuration

The sample have been tested with Vitis `2022.2`.
Environment variables must be set accordingly to use XRT/Vivado/Vitis.

Furthermore we require you setup the platform path in the Makefile (`PLATFORM` variable). By default, it relies on an environment variable `PLATFORM_REPO_PATH` and targets the PCI-E Gen4x8 QDMA shell

```bash
export PLATFORM_REPO_PATH=/opt/xilinx/platforms
```

## Directory structure

The sample are organize as follows:
- the `aie` folder contains the AI-Engine source files (kernels and graph)
- the `pl_kernels` folder contains the PL kernels (data movers from DRAM to AIE and viceversa)
- the `sw` folder contains the host program
- the  `data` folder contains the input data and expected output



## Simulate on the server (X86 Simulation)

The `x86simulator`` is a fast-functional simulator as described in AMD/Xilinx documentation (UG1076). It
should be used to functionally simulate your AI Engine graph and is useful for functional
development and verification of kernels and graphs. It, however, does not provide timing,
resource, or performance information, nor guarantee that it will be functional on the Versal device.

Still it is a useful tool to inspect traces, printing with printf, run gdb or valgrind (see UG1076).



```bash
make aie
make sim
```
If successful the command should return a success message.
In this case the input data is loaded from file (under `data/input.txt`), used to execute the program that will
save the results under `x86simulator_output/data/data/output.txt`, that will in turn be compared with the expected results
(`data/golden.txt`).

The `aiecompiler` generates a `graph.aiecompile_summary` report, that can be inspected using `vitis_analizer`:

```bash
vitis_analyzer ./Work/graph.aiecompile_summary
```

## Simulate using AI Engine Simulator

While the x86 simulator is useful to have a quick assessment on the business logic of the program, it
is not a replacement for the AI Engine simulator, that  must still be run on all designs to verify behavior and obtain
performance information. The Versal® ACAP AI Engine SystemC simulator (`aiesimulator`) includes the modeling of the
global memory (DDR memory) and the network on chip (NoC) in addition to the AI Engine array.


Once that we are happy with the AI-Engine program we need to build the PL kernels, the host program and combine all together in a package
that will be then executed on the versal device.


To use the AI Engine simulator, we have first to compile the aie code for hardware

```bash
make aie TARGET=hw
make sim TARGET=hw
```

This will run the simulation. Also in this case the ouput data is written on a file (`aiesimulator_output/data/output.txt`) that can 
be inspected. By default this also include the timestamp (in nanosecond in this case) of when the corresponding PLIO port receives
the data. From this you can derive some performance figure (e.g., throughput), but requires external post-processing.

This time, the report produced by the `aiecompiler` will contain information about streams, DMA channels and so on.
Furthermore (by defaul in this example thanks to the `vcd` option), the `aiesimulator` produces a trace of the program execution that
can be inspected:

```bash
vitis_analyzer ./aiesimulator_output/default.aierun_summary 
```

For other capabalities of the AI Engine Simulator (e.g., stream stall detection) please refer to UG1076.



## Build for hardware    


For compiling the `mm2s` and `s2mm` HLS kernels run the following command:

```bash
make kernels TARGET=hw
```

The we can link together the AI Engine program with the PL kernels ([Xilinx Tutorial](https://github.com/Xilinx/Vitis-Tutorials/tree/2022.1/AI_Engine_Development/Feature_Tutorials/05-AI-engine-versal-integration#2-use-v-to-link-ai-engine-hls-kernels-with-the-platform))

```bash
make xsa TARGET=hw
make package TARGET=hw
```

We compile the host program with:

```bash
make host
```

And finally we can run it (inside the `sw` folder):

```bash
./host.exe
```
