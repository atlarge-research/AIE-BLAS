#!/bin/bash

# get current directory
curdir=$( cd -P -- "$(dirname -- "$(command -v -- "$0")")" && pwd -P )

cd "$curdir"

if ! [ -d $(readlink -f OpenBLAS) ]; then
    ./install_blas.sh
fi

build_program() {
    (cd "$1" && ./build.sh)
}

if command -v parallel &> /dev/null; then
    export -f build_program
    parallel -j 4 build_program ::: asum axpy dot gemv iamax nrm2 rot scal
else
    build_program asum
    build_program axpy
    build_program dot
    build_program gemv
    build_program iamax
    build_program nrm2
    build_program rot
    build_program scal
fi
