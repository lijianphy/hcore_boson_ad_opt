import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Plot eigenvalue distributions')
    parser.add_argument('files', nargs='+', type=str, help='Input eigenvalue files')
    parser.add_argument('--bins', type=int, default=40, help='Number of histogram bins')
    return parser.parse_args()

def normalize_eigenvalues(eigenvalues):
    """Normalize eigenvalues to have zero mean and unit variance."""
    return eigenvalues / np.max(np.abs(eigenvalues))

def main():
    args = parse_args()
    plt.rcParams['figure.dpi'] = 150
    plt.figure(figsize=(12, 7))
    colors = ['blue', 'red', 'green', 'purple', 'orange']  # Add more colors if needed
    
    for file_path, color in zip(args.files, colors):
        # Load eigenvalues from file
        eigenvalues = np.loadtxt(file_path)
        
        # Normalize eigenvalues
        normalized_eigenvalues = normalize_eigenvalues(eigenvalues)
        
        # Plot histogram
        plt.hist(normalized_eigenvalues, bins=args.bins, density=True, alpha=0.3, color=color,
                label=f'{Path(file_path).stem}\nMean: {np.mean(normalized_eigenvalues):.2f}\nStd: {np.std(normalized_eigenvalues):.2f}')
        
        # Plot kernel density estimation for smoother visualization
        kde = gaussian_kde(normalized_eigenvalues)
        x_range = np.linspace(normalized_eigenvalues.min(), normalized_eigenvalues.max(), 200)
        plt.plot(x_range, kde(x_range), color=color, alpha=0.7)

    plt.xlabel('Normalized Eigenvalue')
    plt.ylabel('Density')
    plt.title('Distribution of Normalized Eigenvalues')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
