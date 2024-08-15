#!/bin/sh

# get current directory
curdir=$( cd -P -- "$(dirname -- "$(command -v -- "$0")")" && pwd -P )

cd "$curdir"/../benchmark/cpu && ./install_blas.sh
