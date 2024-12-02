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

def plot_comparison(data1, data2, label1, label2):
    plt.rcParams['figure.dpi'] = 200
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot infidelity comparison
    ax1.plot(data1['iteration'], data1['infidelity'], label=label1)
    ax1.plot(data2['iteration'], data2['infidelity'], label=label2)
    ax1.set_yscale('log')
    ax1.set(xlabel='Iteration', ylabel='Infidelity', title='Infidelity Convergence Comparison')
    ax1.grid(True)
    ax1.legend()

    # Plot gradient norm comparison
    ax2.plot(data1['iteration'], data1['norm2_grad'], label=label1)
    ax2.plot(data2['iteration'], data2['norm2_grad'], label=label2)
    ax2.set_yscale('log')
    ax2.set(xlabel='Iteration', ylabel='Gradient Norm', title='Gradient Norm Comparison')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Compare convergence between two optimization runs')
    parser.add_argument('file1', help='Path to first JSON lines file')
    parser.add_argument('file2', help='Path to second JSON lines file')
    parser.add_argument('--label1', default='Run 1', help='Label for first file')
    parser.add_argument('--label2', default='Run 2', help='Label for second file')
    args = parser.parse_args()
    
    data1 = load_and_process_file(args.file1)
    data2 = load_and_process_file(args.file2)
    
    if data1.empty or data2.empty:
        print("Error: One or both input files contain no valid data")
        return
        
    plot_comparison(data1, data2, args.label1, args.label2)

if __name__ == '__main__':
    main()