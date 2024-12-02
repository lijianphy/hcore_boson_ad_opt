import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import json
from pathlib import Path

def load_and_process_file(file_path):
    records = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                records.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    
    return pd.DataFrame(records) if records else pd.DataFrame()

def plot_comparison(data_list, labels):
    plt.rcParams['figure.dpi'] = 200
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Define line styles and markers for variety
    styles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'D', 'v', '>', '<', 'p']
    
    # Plot infidelity comparison
    for i, (data, label) in enumerate(zip(data_list, labels)):
        style = styles[i % len(styles)]
        marker = markers[i % len(markers)]
        ax1.plot(data['iteration'], data['infidelity'], label=label,
                 linestyle=style, marker=marker, markevery=max(1, len(data)//20),
                 markersize=6, alpha=0.8)
    ax1.set_yscale('log')
    ax1.set(xlabel='Iteration', ylabel='Infidelity', title='Infidelity Convergence Comparison')
    ax1.grid(True, alpha=0.3)
    ax1.legend(framealpha=0.8)

    # Plot gradient norm comparison
    for i, (data, label) in enumerate(zip(data_list, labels)):
        style = styles[i % len(styles)]
        marker = markers[i % len(markers)]
        ax2.plot(data['iteration'], data['norm2_grad'], label=label,
                 linestyle=style, marker=marker, markevery=max(1, len(data)//20),
                 markersize=6, alpha=0.8)
    ax2.set_yscale('log')
    ax2.set(xlabel='Iteration', ylabel='Gradient Norm', title='Gradient Norm Comparison')
    ax2.grid(True, alpha=0.3)
    ax2.legend(framealpha=0.8)

    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Compare convergence between multiple optimization runs')
    parser.add_argument('files', nargs='+', help='Paths to JSON lines files')
    parser.add_argument('--labels', nargs='+', help='Labels for the files')
    args = parser.parse_args()
    
    # Handle labels
    if args.labels and len(args.labels) != len(args.files):
        print("Error: Number of labels must match number of files")
        return
    labels = args.labels if args.labels else [f'Run {i+1}' for i in range(len(args.files))]
    
    # Load all data files
    data_list = []
    for file_path in args.files:
        data = load_and_process_file(file_path)
        if data.empty:
            print(f"Error: File {file_path} contains no valid data")
            return
        data_list.append(data)
        
    plot_comparison(data_list, labels)

if __name__ == '__main__':
    main()