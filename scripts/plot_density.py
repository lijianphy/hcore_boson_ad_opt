import json
import sys
import math
import matplotlib.pyplot as plt
import numpy as np
import argparse

def plot_density_evolution(filename, site_indices=None, total_time=math.pi, show_total=False):
    # Lists to store data
    time_steps = []
    densities = []
    
    # Read JSONL file
    with open(filename, 'r') as f:
        for line in f:
            data = json.loads(line)
            time_steps.append(data['time_step'])
            densities.append(data['density'])
    
    # Convert to numpy arrays for easier manipulation
    time_steps = np.array(time_steps)
    max_time = np.max(time_steps)
    time = time_steps / max_time * total_time
    densities = np.array(densities)
    
    # Create figure
    plt.rcParams['figure.dpi'] = 150
    plt.figure(figsize=(12, 8))
    
    # If no specific sites requested, plot all sites
    if site_indices is None:
        site_indices = range(densities.shape[1])
    
    # Plot requested sites' density evolution
    for site in site_indices:
        if site < densities.shape[1]:
            plt.plot(time, densities[:, site], label=f'Site {site}', 
                    linewidth=2, alpha=0.7)
    
    # Add total density plot if requested
    if show_total:
        total_density = np.sum(densities[:, site_indices], axis=1)
        plt.plot(time, total_density, '--', color='black', 
                label='Total', linewidth=2, alpha=0.8)
    
    plt.xlabel('Time')
    plt.ylabel('Density')
    plt.title('Density Evolution Over Time')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot density evolution from JSONL file.')
    parser.add_argument('jsonl_file', type=str, help='Input JSONL file containing density data')
    parser.add_argument('--total-time', '-t', type=float, default=math.pi,
                      help='Total time period (default: pi)')
    parser.add_argument('--sites', '-s', type=int, nargs='+',
                      help='Specific site indices to plot (default: all sites)')
    parser.add_argument('--total', action='store_true',
                      help='Show total density of selected sites')
    
    args = parser.parse_args()
    
    plot_density_evolution(args.jsonl_file, args.sites, args.total_time, args.total)
