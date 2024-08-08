#!/bin/sh

# NOTE: Building all aieblas designs can take a long time (~3 hrs per design).
# Expect this script to take roughly 20 hrs to complete.

# get current directory
curdir=$( cd -P -- "$(dirname -- "$(command -v -- "$0")")" && pwd -P )

cd "$curdir"/.. && \
cd cpu && \
./install_blas.sh && \
cd sum_vectors_cpu && \
cmake -DCMAKE_BUILD_TYPE=Release -Bbuild && \
cd build && \
make && \
cd ../.. && \
cd gemv && \
cmake -DCMAKE_BUILD_TYPE=Release -Bbuild && \
cd build && \
make && \
cd ../.. && \
cd axpydot && \
cmake -DCMAKE_BUILD_TYPE=Release -Bbuild && \
cd build && \
make && \
cd ../../.. && \
cd single_routines && \
cd axpy && \
./build.sh && \
cd .. && \
cd gemv && \
./build.sh && \
cd ../.. && \
cd combined_routines && \
cd axpy_dot && \
./build.sh && \
cd .. && \
cd axpy_dot_separate && \
./build.sh && \
cd .. && \
cd sum_vectors && \
./build.sh && \
cd .. && \
cd sum_vectors_one_pl && \
./build.sh && \
cd .. && \
