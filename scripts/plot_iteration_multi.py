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
            all_data.append((Path(file_path).name, df))
    
    return all_data

def plot_iteration_data(data_list):
    plt.rcParams['figure.dpi'] = 200
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 5))

    for idx, (filename, data) in enumerate(data_list):
        ax1.plot(data['iteration'], data['infidelity'], label=f'{idx}')

    ax1.set_yscale('log')
    ax1.set(xlabel='Iteration', ylabel='Error', title='Error vs. Iteration')
    ax1.grid()
    ax1.legend()

    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot iteration data from multiple JSON files')
    parser.add_argument('json_files', nargs='+', help='Paths to the JSON lines files')
    args = parser.parse_args()
    
    data_list = load_and_process_files(args.json_files)
    if not data_list:
        print("No valid data found in the input files")
        return
        
    plot_iteration_data(data_list)

if __name__ == '__main__':
    main()
