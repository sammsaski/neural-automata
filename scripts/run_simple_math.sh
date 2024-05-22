#!/bin/bash

current_dir=$(pwd)

if [[ "$current_dir" == *"/scripts" ]]; then
    # cd out of the scripts directory
    cd ..
fi

python examples/simple_math/simple_math.py


