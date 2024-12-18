import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Plot gap ratio of eigenvalue distributions')
    parser.add_argument('files', nargs='+', type=str, help='Input eigenvalue files')
    parser.add_argument('--bins', type=int, default=40, help='Number of histogram bins')
    return parser.parse_args()

def compute_gap_ratio(eigenvalues):
    """Compute the gap ratio of eigenvalues."""
    sorted_eigenvalues = np.sort(eigenvalues)
    diff = np.maximum(np.diff(sorted_eigenvalues), 1e-10)
    diff1 = diff[:-1]
    diff2 = diff[1:]
    r = np.minimum(diff1, diff2) / np.maximum(diff1, diff2)
    r = np.nan_to_num(r, nan=1.0)
    return r

def p_goe(x):
    """GOE probability density function."""
    return (27.0/4.0)*((x + x**2)/((1 + x + x**2)**2.5))

def p_pos(x):
    """Poisson probability density function."""
    return (4/np.pi) / (1.0 + x**2)

def main():
    args = parse_args()
    
    plt.rcParams['figure.dpi'] = 150
    plt.figure(figsize=(12, 7))
    colors = ['blue', 'red', 'green', 'purple', 'orange']  # Add more colors if needed
    
    for file_path, color in zip(args.files, colors):
        # Load eigenvalues from file
        eigenvalues = np.loadtxt(file_path)
        
        # Normalize eigenvalues
        gap_ratio = compute_gap_ratio(eigenvalues)
        
        # Plot histogram using step lines
        counts, bins, _ = plt.hist(gap_ratio, bins=args.bins, density=True, alpha=0)
        plt.step(bins[:-1], counts, where='post', color=color, alpha=0.7,
                label=f'{Path(file_path).stem}')
        
        # Plot kernel density estimation for smoother visualization
        # kde = gaussian_kde(gap_ratio)

        # plt.plot(x_range, kde(x_range), color=color, alpha=0.7)

    x_range = np.linspace(0, 1, 100)
    plt.plot(x_range, p_goe(x_range), color='red', alpha=0.7, label='GOE', linestyle='--')
    plt.plot(x_range, p_pos(x_range), color='blue', alpha=0.7, label='Poisson', linestyle='--')
    plt.xlabel('Gap Ratio')
    plt.ylabel('Density')
    plt.title('Distribution of Gap Ratios')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
