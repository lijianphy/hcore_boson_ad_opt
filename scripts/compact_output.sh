#!/bin/bash

mkdir -p output_compact
## for each output file (*.jsonl) in the directory, compact it to only include iteration, infidelity, and norm2_grad
for output_file in $(ls *.jsonl); do
    echo "Compact for file: $output_file"
    jq -c '{iteration, infidelity, norm2_grad}' $output_file > output_compact/$output_file
done
