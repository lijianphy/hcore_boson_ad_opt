import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import json
from pathlib import Path

# each line in the JSON file is a dictionary of form:
# {
#     "iteration": 1,
#     "norm2_error": 1.0,
#     "norm2_grad": 0.1,
#     "coupling_strength": [0.1, 0.2, 0.3],
#     "gradient": [0.1, 0.2, 0.3]
# }

def load_and_process_files(file_paths):
    all_data = []
    current_max_iteration = 0
    
    # Sort files by timestamp in filename
    sorted_files = sorted(file_paths, key=lambda x: str(x))
    
    for file_path in sorted_files:
        records = []
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    records.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
        
        if records:
            df = pd.DataFrame(records)
            # Adjust iteration numbers
            df['iteration'] = df['iteration'] + current_max_iteration
            current_max_iteration = df['iteration'].max() + 1
            all_data.append(df)
    
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

def plot_iteration_data(data):
    plt.rcParams['figure.dpi'] = 200

    # Create subplots
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 5))

    # First subplot
    ax1.plot(data['iteration'], data['infidelity'], label='infidelity')
    ax1.plot(data['iteration'], data['norm2_grad'], label='Gradient')
    ax1.set_yscale('log')
    ax1.set(xlabel='Iteration', ylabel='Error', title='Error vs. Iteration')
    ax1.grid()
    ax1.legend()

    # # Second subplot
    # ax2.scatter(data['norm2_grad'], data['infidelity'], label='infidelity vs. Gradient', s=5, alpha=0.5)
    # ax2.set_xscale('log')
    # ax2.set_yscale('log')
    # ax2.set(xlabel='Gradient', ylabel='infidelity', title='infidelity vs. Gradient')
    # ax2.grid()
    # ax2.legend()

    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot iteration data from multiple JSON files')
    parser.add_argument('json_files', nargs='+', help='Paths to the JSON lines files')
    args = parser.parse_args()
    
    data = load_and_process_files(args.json_files)
    if data.empty:
        print("No valid data found in the input files")
        return
        
    plot_iteration_data(data)

if __name__ == '__main__':
    main()
