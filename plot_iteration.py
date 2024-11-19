import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import json

def plot_iteration_data(data):
    plt.rcParams['figure.dpi'] = 200

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # First subplot
    ax1.plot(data['iteration'], data['norm2_error'], label='Error')
    ax1.plot(data['iteration'], data['norm2_grad'], label='Gradient')
    ax1.set_yscale('log')
    ax1.set(xlabel='Iteration', ylabel='Error', title='Error vs. Iteration')
    ax1.grid()
    ax1.legend()

    # Second subplot
    ax2.plot(data['norm2_grad'], data['norm2_error'], label='Error vs. Gradient')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set(xlabel='Gradient', ylabel='Error', title='Error vs. Gradient')
    ax2.grid()
    ax2.legend()

    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot iteration data from JSON file')
    parser.add_argument('json_file', help='Path to the JSON lines file', required=True)
    args = parser.parse_args()

    # Load data with error handling for incomplete last line
    records = []
    with open(args.json_file, 'r') as f:
        for line in f:
            try:
                records.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                # Skip incomplete/malformed lines
                continue
    
    data = pd.DataFrame(records)
    plot_iteration_data(data)

if __name__ == '__main__':
    main()
