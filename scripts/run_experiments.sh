#!/bin/bash

current_dir=$(pwd)

if [[ "$current_dir" == *"/scripts" ]]; then
    # cd out of the scripts directory and into examples/regex
    cd ../examples/regex
fi

# run the regex script
python regex.py

# cd into simple_math_vlm_comp for the arithmetic evaluation experiments
cd ../simple_math_vlm_comp

# run the arithmetic evaluation experiments
python simple_math_vlm_comp.py
