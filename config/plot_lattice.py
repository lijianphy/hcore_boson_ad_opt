import networkx as nx
import matplotlib.pyplot as plt
import json
import numpy as np
import argparse
import sys

def parse_arguments():
    parser = argparse.ArgumentParser(description='Plot lattice coupling strengths')
    parser.add_argument('config_file', help='JSON configuration file path')
    parser.add_argument('m', type=int, help='Number of rows')
    parser.add_argument('n', type=int, help='Number of columns')
    return parser.parse_args()

def validate_dimensions(data, m, n):
    num_sites = data['cnt_site']
    if m * n != num_sites:
        print(f"Error: m*n ({m*n}) does not match the number of sites in config ({num_sites})")
        sys.exit(1)

def main():
    args = parse_arguments()
    
    # Read the JSON file
    with open(args.config_file, 'r') as f:
        data = json.load(f)
    
    m = args.m
    n = args.n
    # Validate dimensions
    validate_dimensions(data, m, n)
    
    # Create graph
    G = nx.Graph()
    
    # Add nodes
    for i in range(m * n):
        G.add_node(i)
    
    # Add edges with coupling strengths
    for bond, strength in zip(data['bonds'], data['coupling_strength']):
        G.add_edge(bond[0], bond[1], weight=strength)
    
    # Create position layout (m x n grid)
    pos = {}
    for i in range(m * n):
        row = i // n
        col = i % n
        pos[i] = (col, -row)
    
    # Get edge weights and determine symmetric limits
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    abs_max = max(abs(min(weights)), abs(max(weights)))
    vmin, vmax = -abs_max, abs_max

    plt.rcParams['figure.dpi'] = 200
    # Create figure with a specific size and layout for colorbar
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create node color list
    node_colors = ['lightgray'] * (m * n)
    initial_sites = data['initial_state']
    target_sites = data['target_state']
    
    for site in initial_sites:
        node_colors[site] = 'lightgreen'
    for site in target_sites:
        node_colors[site] = 'lightcoral'

    # Draw nodes with different colors
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                          node_size=600, alpha=0.9, ax=ax)

    # Draw edges with colors based on weights using diverging colormap
    edges = nx.draw_networkx_edges(G, pos, edge_color=weights, 
                                 edge_cmap=plt.cm.RdBu_r,  # reversed RdBu for conventional red(+)/blue(-) mapping
                                 edge_vmin=vmin, edge_vmax=vmax,
                                 width=3, ax=ax)

    # Add node labels
    nx.draw_networkx_labels(G, pos, ax=ax)

    # Add colorbar with centered diverging colors
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdBu_r, 
                              norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar = plt.colorbar(sm, ax=ax, label='Coupling Strength')
    cbar.ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)  # Add line at zero

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightgreen', label='Initial State'),
        Patch(facecolor='lightcoral', label='Target State'),
        Patch(facecolor='lightgray', label='Other Sites')
    ]
    ax.legend(handles=legend_elements, loc='upper left', 
              bbox_to_anchor=(1.15, 1.0))

    ax.set_title(f'{m}x{n} Lattice Coupling Strengths')
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
