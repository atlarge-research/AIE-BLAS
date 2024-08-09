## aieblas

This repository contains a BLAS implementation for the AMD AI Engine.

Note that the library is mostly a proof of concept and does not support all routines.

### Supported routines
Level 1:
- ASUM
- AXPY
- DOT
- IAMAX
- NRM2
- ROT
- SCAL

Level 2:
- GEMV

### Supported devices
- AMD Versal VCK5000

### Compiling the library
To compile the code generator, run `./build.sh` in the folder [`aieblas/`](./aieblas/).

### Running the benchmarks
To run the benchmarks, first compile the code generator, and then build the benchmarks by running `./build-all.sh` in the folder [`benchmark/util`](./benchmark/util).
