from floor import Floor
from graph import segment_rooms

import numpy as np

import argparse
import config
import json

parser = argparse.ArgumentParser()
parser.add_argument('--floor_path', type=str)
parser.add_argument('--output_path', type=str, default='tmp')


def main():
    floor = Floor(0, 'sample_floor_name')
    floor.load(config.args.floor_path)

    arr = np.asarray(floor.pcd.points)
    l1 = np.quantile(arr[1], 0.05)
    l2 = np.quantile(arr[1], 0.95)

    floor.floor_zero_level = l1
    floor.floor_height = l2 - l1
    segment_rooms(floor=floor)
    

if __name__ == '__main__':
    config.args = parser.parse_args()
    print(config.args)
    with open('config.json', 'r') as f:
        config.config = json.load(f)
    
    main()