#!/usr/bin/env python3

import argparse

def generate_2d_lattice_bonds(m: int, n: int) -> list:
    """
    Generate bonds for a 2D lattice of size m x n
    Args:
        m: number of rows
        n: number of columns
    Returns:
        list of bonds where each bond is [site1, site2]
    """
    bonds = []
    
    # Generate horizontal bonds
    for i in range(m):
        for j in range(n-1):
            site1 = i * n + j
            site2 = site1 + 1
            bonds.append([site1, site2])
    
    # Generate vertical bonds
    for i in range(m-1):
        for j in range(n):
            site1 = i * n + j
            site2 = site1 + n
            bonds.append([site1, site2])
            
    return bonds

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate bonds for a 2D lattice')
    parser.add_argument('-m', '--rows', type=int, required=True,
                        help='Number of rows')
    parser.add_argument('-n', '--columns', type=int, required=True,
                        help='Number of columns')
    
    args = parser.parse_args()
    
    bonds = generate_2d_lattice_bonds(args.rows, args.columns)
    print(f"Bonds for {args.rows}x{args.columns} lattice:")
    print(bonds)
    print(f"Number of bonds: {len(bonds)}")