#!/bin/sh

# get current directory
curdir=$( cd -P -- "$(dirname -- "$(command -v -- "$0")")" && pwd -P )

codegen="${curdir}/../../../aieblas/build/codegen"

if [ "$#" -ge 2 ]; then
    codegen="$1"
fi

# run codegen
(cd "${curdir}"; "${codegen}" -l verbose aieblas.json aieblas) && \
(cd "${curdir}"; cmake -DCMAKE_BUILD_TYPE=Release -Bbuild) && \
(cd "${curdir}/build"; make aie && make)
