#!/bin/bash

# Check if config file is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <config_file>"
    exit 1
fi

CONFIG_FILE=$1

# Define phases array
phases=(0.0  0.3141592653589793  0.6283185307179586  0.9424777960769379  1.2566370614359172  1.5707963267948966  1.8849555921538759  2.199114857512855  2.5132741228718345  2.827433388230814  3.141592653589793  3.455751918948772  3.7699111843077517  4.084070449666731  4.39822971502571  4.71238898038469  5.026548245743669  5.340707511102648  5.654866776461628  5.969026041820607)

# Loop through all phases
for phase in "${phases[@]}"; do
    # Submit job for each phase
    job_name="phase_${phase}"
    echo "Submitting job for phase = ${phase}"
    
    sbatch --job-name="${job_name}" \
           --nodes=1 \
           --ntasks-per-node=2 \
           --partition=batch \
           --output="${job_name}_%j.out" \
           --wrap="mpirun ./test_optimize ${CONFIG_FILE} ${phase}"
    
    # sleep for 2 seconds
    sleep 2
done

echo "All jobs submitted!"
