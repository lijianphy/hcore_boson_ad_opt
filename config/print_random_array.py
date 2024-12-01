#!/usr/bin/env python3

# print an array of random numbers following Gaussian distribution

import random
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description='Print array of random numbers following Gaussian distribution')
    parser.add_argument('-a', type=float, help='Mean of the Gaussian distribution', required=True)
    parser.add_argument('-s', type=float, help='Standard deviation of the Gaussian distribution', required=True)
    parser.add_argument('-n', type=int, help='Number of random numbers to generate', required=True)
    
    args = parser.parse_args()
    
    if args.s < 1e-6:
        print("Error: Standard deviation must be positive")
        sys.exit(1)
        
    random_numbers = [random.gauss(args.a, args.s) for _ in range(args.n)]
    formatted_numbers = [f'{x:.4f}' for x in random_numbers]
    print('[' + ', '.join(formatted_numbers) + ']')

if __name__ == '__main__':
    main()
