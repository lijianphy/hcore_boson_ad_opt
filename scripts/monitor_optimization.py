import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from collections import deque
import os

class OptimizationMonitor:
    def __init__(self, filename, buffer_size=1000):
        self.filename = filename
        self.buffer_size = buffer_size
        self.last_position = 0
        self.last_size = 0
        
        # Use deques for efficient fixed-size buffering
        self.iterations = deque(maxlen=buffer_size)
        self.norm2_errors = deque(maxlen=buffer_size)
        self.norm2_grads = deque(maxlen=buffer_size)
        
        # Statistics
        self.min_error = float('inf')
        self.min_grad = float('inf')
        
        # Setup the figure and subplots
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))
        # Add formatter function
        self.setup_plot()
        
    def setup_plot(self):
        # Setup error plot
        self.error_line, = self.ax1.semilogy([], [], 'b-', label='Current')
        self.error_min_line, = self.ax1.semilogy([], [], 'r.', label='Min')
        self.ax1.set_xlabel('Iteration')
        self.ax1.set_ylabel('infidelity')
        self.ax1.grid(True)
        self.ax1.set_title('Optimization Progress')
        self.ax1.legend()
        
        # Add text box for statistics
        self.stats_text = self.ax1.text(0.02, 0.98, '', transform=self.ax1.transAxes,
                                       verticalalignment='top', bbox=dict(boxstyle='round', 
                                       facecolor='white', alpha=0.8))
        
        # Setup gradient plot
        self.grad_line, = self.ax2.semilogy([], [], 'b-', label='Current')
        self.grad_min_line, = self.ax2.semilogy([], [], 'r.', label='Min')
        self.ax2.set_xlabel('Iteration')
        self.ax2.set_ylabel('norm2_grad')
        self.ax2.grid(True)
        self.ax2.legend()
    
    def update_data(self):
        try:
            # Check if file exists and has been modified
            current_size = os.path.getsize(self.filename)
            if current_size == self.last_size:
                return False
                
            with open(self.filename, 'r') as f:
                # Seek to last position if file wasn't truncated
                if current_size >= self.last_size:
                    f.seek(self.last_position)
                
                new_data = False
                for line in f:
                    try:
                        data = json.loads(line)
                        self.iterations.append(data['iteration'])
                        error = data['infidelity']
                        grad = data['norm2_grad']
                        self.norm2_errors.append(error)
                        self.norm2_grads.append(grad)
                        
                        # Update statistics
                        self.min_error = min(self.min_error, error)
                        self.min_grad = min(self.min_grad, grad)
                        new_data = True
                    except json.JSONDecodeError:
                        continue
                
                self.last_position = f.tell()
                self.last_size = current_size
                return new_data
                
        except FileNotFoundError:
            print(f"File {self.filename} not found. Waiting for file...")
            return False
    
    def animate(self, frame):
        if not self.update_data():
            return self.error_line, self.grad_line, self.error_min_line, self.grad_min_line, self.stats_text
            
        if not self.iterations:
            return self.error_line, self.grad_line, self.error_min_line, self.grad_min_line, self.stats_text
        
        # Update statistics text
        stats = f'Min error: {self.min_error:.6e}\nMin grad: {self.min_grad:.6e}\n'
        stats += f'Buffer size: {len(self.iterations)}'
        self.stats_text.set_text(stats)

        # Update error plot
        self.error_line.set_data(list(self.iterations), list(self.norm2_errors))
        self.error_min_line.set_data([self.iterations[-1]], [self.min_error])
        self.ax1.relim()
        self.ax1.autoscale_view()
        
        # Set x-axis limits and format for error plot
        xmin, xmax = min(self.iterations), max(self.iterations)
        self.ax1.set_xlim(xmin, xmax)
        
        # Update gradient plot
        self.grad_line.set_data(list(self.iterations), list(self.norm2_grads))
        self.grad_min_line.set_data([self.iterations[-1]], [self.min_grad])
        self.ax2.relim()
        self.ax2.autoscale_view()
        
        # Set x-axis limits and format for gradient plot
        self.ax2.set_xlim(xmin, xmax)
        
        return self.error_line, self.grad_line, self.error_min_line, self.grad_min_line, self.stats_text

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Monitor optimization progress')
    parser.add_argument('filename', type=str, help='Path to the JSON lines output file')
    parser.add_argument('--interval', type=int, default=1000,
                      help='Update interval in milliseconds (default: 1000)')
    parser.add_argument('--buffer', type=int, default=1000,
                      help='Number of points to keep in memory (default: 1000)')
    parser.add_argument('--save-count', type=int, default=20,  # reduced from 100 to 20
                      help='Number of animation frames to cache (default: 20)')
    args = parser.parse_args()

    plt.rcParams['figure.dpi'] = 200
    
    monitor = OptimizationMonitor(args.filename, buffer_size=args.buffer)
    ani = FuncAnimation(monitor.fig, monitor.animate, 
                       interval=args.interval, blit=True,
                       save_count=args.save_count)
    plt.show()

if __name__ == '__main__':
    main()
