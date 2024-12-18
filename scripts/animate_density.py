import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
import math

def create_animation(filename, total_time, m, n, output_file='animation.mp4'):
    # Read density data from JSONL file
    time_steps = []
    densities = []
    with open(filename, 'r') as f:
        for line in f:
            data = json.loads(line)
            time_steps.append(data['time_step'])
            densities.append(data['density'])

    # Convert to numpy arrays and calculate actual time values
    time_steps = np.array(time_steps)
    num_steps = len(time_steps)
    time = np.linspace(0, total_time, num_steps)  # Create evenly spaced time points
    densities = np.array(densities)

    # Verify the lattice size matches the data
    if m * n != densities.shape[1]:
        raise ValueError(f"m*n ({m*n}) does not match the number of sites in data ({densities.shape[1]})")

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.rcParams['figure.dpi'] = 150

    # Initialize heatmap
    heatmap = ax.imshow(
        densities[0].reshape(m, n),
        cmap='viridis',
        aspect='equal',
        vmin=0,
        vmax=1
    )

    # Add colorbar
    plt.colorbar(heatmap, label='Density')

    # Animation update function
    def update(frame):
        heatmap.set_data(densities[frame].reshape(m, n))
        ax.set_title(f'Time: {time[frame]:.2f}')
        return heatmap,

    # Create animation
    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(time_steps),
        interval=50,  # 50ms between frames
        blit=True
    )

    # Save animation instead of showing it
    anim.save(output_file, writer='ffmpeg', fps=20)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Animate density evolution on a 2D lattice.')
    parser.add_argument('jsonl_file', type=str, help='Input JSONL file containing density data')
    parser.add_argument('--total-time', '-t', type=float, default=math.pi,
                      help='Total time period (default: pi)')
    parser.add_argument('--rows', '-m', type=int, required=True,
                      help='Number of rows in the lattice')
    parser.add_argument('--cols', '-n', type=int, required=True,
                      help='Number of columns in the lattice')
    parser.add_argument('--output', '-o', type=str, default='animation.mp4',
                      help='Output video file (default: animation.mp4)')

    args = parser.parse_args()

    create_animation(args.jsonl_file, args.total_time, args.rows, args.cols, args.output)
