import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import argparse
import json
from pathlib import Path

def load_single_file(file_path):
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Calculate common limits
    x_min = min(data1['norm2_grad'].min(), data2['norm2_grad'].min())
    x_max = max(data1['norm2_grad'].max(), data2['norm2_grad'].max())
    y_min = min(data1['norm2_error'].min(), data2['norm2_error'].min())
    y_max = max(data1['norm2_error'].max(), data2['norm2_error'].max())

    # First 2D density plot
    h1 = ax1.hist2d(data1['norm2_grad'], data1['norm2_error'], range=[[x_min, x_max], [y_min, y_max]],
                    bins=50, cmap='viridis', norm=LogNorm())
    ax1.set(xlabel='Gradient', ylabel='Error', title=f'Density Plot: {label1}')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    fig.colorbar(h1[3], ax=ax1, label='Count')

    # Second 2D density plot
    h2 = ax2.hist2d(data2['norm2_grad'], data2['norm2_error'], range=[[x_min, x_max], [y_min, y_max]],
                    bins=50, cmap='viridis', norm=LogNorm())
    ax2.set(xlabel='Gradient', ylabel='Error', title=f'Density Plot: {label2}')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)
    fig.colorbar(h2[3], ax=ax2, label='Count')

    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Compare two optimization runs using density plots')
    parser.add_argument('file1', help='First JSON lines file')
    parser.add_argument('file2', help='Second JSON lines file')
    parser.add_argument('--label1', default='Run 1', help='Label for first file')
    parser.add_argument('--label2', default='Run 2', help='Label for second file')
    args = parser.parse_args()
    
    data1 = load_single_file(args.file1)
    data2 = load_single_file(args.file2)
    
    if data1.empty or data2.empty:
        print("Error: One or both input files contain no valid data")
        return
        
    plot_comparison(data1, data2, args.label1, args.label2)

if __name__ == '__main__':
    main()