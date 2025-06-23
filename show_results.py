#!/usr/bin/env python3
"""
Script to read all .pkl files in ./data/ and display their phases and area values.
"""
import os
import pickle

def read_data_files(data_dir='./data'):
    """
    Reads all .pkl files in the given directory and returns a list of tuples:
    (filename, phases_list, area_value)
    """
    results = []
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    for fname in sorted(os.listdir(data_dir)):
        if not fname.lower().endswith('.pkl'):
            continue
        path = os.path.join(data_dir, fname)
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            phases = data.get('phases')
            area = data.get('area')
            results.append((fname, phases, area))
        except Exception as e:
            print(f"Failed to read {fname}: {e}")
    return results

if __name__ == '__main__':
    try:
        entries = read_data_files()
        for fname, phases, area in entries:
            print(f"{fname}:\n  phi = {[round(p, 4) for p in phases + [0]]}\n  omega_0 = {round(area, 4)}\n")
    except Exception as e:
        print(f"Error: {e}")
