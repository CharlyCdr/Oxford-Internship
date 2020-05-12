import argparse
import collections
import json

from economy import Economy as eco
from dynamics import Dynamics as dyn

if __name__ == '__main__':
    # parsing CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--parameters-file', help='Path to parameters file.')
    args = parser.parse_args()

    if args.parameters_file:
        try:
            with open(args.parameters_file, 'r') as f:
                params_dict = json.load(f)
        except Exception as e:
            print(f'Could not open parameters file {args.parameters_file}. Exiting.')
            print(e)
            exit(1)