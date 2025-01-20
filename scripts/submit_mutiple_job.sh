#!/bin/bash

export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH=$HOME/opt/petsc-3.22.1/linux-gnu-complex/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/opt/slepc-3.22.1/linux-gnu-complex/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/opt/OpenBLAS/lib:$LD_LIBRARY_PATH

# Check if at least one config file is provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <config_file1> [config_file2 ...]"
    exit 1
fi

# Loop through all provided config files
for config_file in "$@"; do
    # Get base name of config file for job naming
    base_name=$(basename "${config_file%.*}")
    job_name="job_${base_name}"
    # Extract cnt_stream and process_per_stream values from the config file
    cnt_stream=$(jq '.cnt_stream' $config_file)
    process_per_stream=$(jq '.process_per_stream' $config_file)
    # Calculate the total number of processes
    total_processes=$((cnt_stream * process_per_stream))
    echo "Submitting job for config = ${config_file} with total processes = ${total_processes}"
    
    sbatch --job-name="${job_name}" \
           --nodes=1 \
           --ntasks-per-node="${total_processes}" \
           --partition=batch \
           --output="${job_name}_%j.out" \
           --wrap="mpirun ./optimize_parallel ${config_file}"
done

echo "All jobs submitted!"

