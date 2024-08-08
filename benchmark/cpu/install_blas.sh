#!/bin/bash

wget https://github.com/OpenMathLib/OpenBLAS/releases/download/v0.3.27/OpenBLAS-0.3.27.tar.gz && \
tar --extract --file OpenBLAS-0.3.27.tar.gz && \
cd OpenBLAS-0.3.27 && \
make && \
make install PREFIX="$(cd "../"; pwd)/OpenBLAS" && \
cd .. && \
rm -r OpenBLAS-0.3.27 OpenBLAS-0.3.27.tar.gz
