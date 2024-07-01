# Copyright (c) 2024 Valeo
# See LICENSE.md for details.

import argparse
import math
import json


def compare_dicts(dict1, dict2, key_path='', *, tolerance=0.001, decimals=4, ignore_missing=False):
    if not isinstance(dict1, dict) or not isinstance(dict2, dict):
        raise ValueError("Both inputs should be dictionaries.")

    keys1, keys2 = set(dict1.keys()), set(dict2.keys())
    missing_in_dict2 = keys1 - keys2
    missing_in_dict1 = keys2 - keys1
    union_keys = keys1 | keys2

    formatter = "{:.%df}" % decimals

    def f(value):
        if isinstance(value, float):
            return formatter.format(value)
        if isinstance(value, str):
            return repr(value)
        return value

    for key in sorted(union_keys):
        new_path = f"{key_path}.{key}" if key_path else key

        if key in missing_in_dict1:
            if not ignore_missing:
                print(f"{new_path}: missing in first")
            continue

        if key in missing_in_dict2:
            if not ignore_missing:
                print(f"{new_path}: missing in second")
            continue

        value1, value2 = dict1[key], dict2[key]
        if isinstance(value1, dict) and isinstance(value2, dict):
            compare_dicts(value1, value2, new_path)
        elif isinstance(value1, float) and isinstance(value2, float):
            if math.isnan(value1) and math.isnan(value2):
                continue  # Both are NaN, considered equal.
            elif math.fabs(value1-value2) > tolerance or math.isnan(value1-value2):
                print(f"{new_path}: {f(value1)} vs {f(value2)}, difference: {f(value1 - value2)}")
        elif value1 != value2:
            print(f"{new_path}: {f(value1)} vs {f(value2)}")


def main():
    parser = argparse.ArgumentParser(description="Compare two results JSON files",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("json_file1", metavar='results1.json', type=str, help="Path to the first results JSON file")
    parser.add_argument("json_file2", metavar='results2.json', type=str, help="Path to the second results JSON file")
    parser.add_argument("--tolerance", type=float, default=0.001, help="tolerance for comparing floats")
    parser.add_argument("--decimals", type=int, default=4, help="number of decimals for displaying floats")
    parser.add_argument("--ignore-missing", action='store_true', help="ignores missing keys in either file")
    args = parser.parse_args()

    with open(args.json_file1, 'rt', encoding='utf-8') as file1:
        dict1 = json.load(file1)
    with open(args.json_file2, 'rt', encoding='utf-8') as file2:
        dict2 = json.load(file2)

    compare_dicts(dict1, dict2, tolerance=args.tolerance, decimals=args.decimals, ignore_missing=args.ignore_missing)


if __name__ == "__main__":
    main()
