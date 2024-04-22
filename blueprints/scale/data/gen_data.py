#!/usr/bin/env python3
from pathlib import Path
import numpy as np

size = 32
scale = 5
input = np.random.randint(-100_000, 100_000, size=size).astype(np.int32)
golden = input * scale

with Path('input.txt').open('w') as f:
    for d in input:
        print(f"{d}", file=f)

with Path('ctrl.txt').open('w') as f:
    print(f"{scale}", file=f)

with Path('golden.txt').open('w') as f:
    for d in golden:
        print(f"{d}", file=f)

with Path('../sw/ground_truth.h').open('w') as f:
    print("#pragma once\n// DO NOT CHANGE THIS", file=f)
    print(f"#define SAMPLES {size}", file=f)
    print(f"#define SCALAR {scale}", file=f)
    print("\nstatic int32_t cInput[SAMPLES] __attribute__ ((__aligned__(4096))) = {", file=f)
    for i, d in enumerate(input):
        print(f"    {d}", end='', file=f)
        if i + 1 < size:
            print(',', end='', file=f)
        print(file=f)
    print("};\n\nstatic int32_t cGolden[SAMPLES] __attribute__ ((__aligned__(4096))) = {", file=f)
    for i, d in enumerate(golden):
        print(f"    {d}", end='', file=f)
        if i + 1 < size:
            print(',', end='', file=f)
        print(file=f)
    print("};", file=f)
