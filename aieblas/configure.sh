#!/bin/sh

echo "Generating debug config in 'build/'"
cmake -B build -DCMAKE_BUILD_TYPE=debug .
echo "Done!"
echo
echo "Generating release config in 'build-release/'"
cmake -B build-release -DCMAKE_BUILD_TYPE=release .
echo "Done!"
