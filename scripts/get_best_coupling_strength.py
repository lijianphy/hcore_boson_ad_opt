import json
import argparse
from typing import List, Tuple

def process_jsonl_file(filepath: str) -> Tuple[float, List[float]]:
    min_infidelity = float('inf')
    min_coupling_strength = None
    
    try:
        with open(filepath, 'r') as file:
            for line in file:
                data = json.loads(line)
                infidelity = data['infidelity']
                if infidelity < min_infidelity:
                    min_infidelity = infidelity
                    min_coupling_strength = data['coupling_strength']
        return min_infidelity, min_coupling_strength
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found")
        return float('inf'), None
    except json.JSONDecodeError:
        print(f"Error: File '{filepath}' contains invalid JSON")
        return float('inf'), None

def main():
    parser = argparse.ArgumentParser(description='Find coupling strength with minimum infidelity from JSONL files')
    parser.add_argument('files', nargs='+', help='One or more JSONL files to process')
    args = parser.parse_args()

    global_min_infidelity = float('inf')
    global_min_coupling = None
    global_min_file = None

    for filepath in args.files:
        min_infidelity, coupling_strength = process_jsonl_file(filepath)
        if min_infidelity < global_min_infidelity:
            global_min_infidelity = min_infidelity
            global_min_coupling = coupling_strength
            global_min_file = filepath

    if global_min_coupling is not None:
        print(f"File with minimum infidelity: {global_min_file}")
        print(f"Minimum infidelity: {global_min_infidelity:.10e}")
        print("Corresponding coupling strength:")
        print(global_min_coupling)

if __name__ == "__main__":
    main()