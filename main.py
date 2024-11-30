from segmentation.floor import Floor
from segmentation.graph import segment_rooms
import numpy as np

import argparse
import config
import json

parser = argparse.ArgumentParser()
parser.add_argument('--floor_path', type=str)
parser.add_argument('--output_path', type=str, default='tmp')
parser.add_argument('--save_path', type=str, default=None)


def main():
    floor = Floor(0, 'sample_floor_name')
    floor.load(config.args.floor_path)

    arr = np.asarray(floor.pcd.points)
    l1 = np.quantile(arr[1], config.config['floor_bounds_quantiles'][0])
    l2 = np.quantile(arr[1], config.config['floor_bounds_quantiles'][1])

    floor.floor_zero_level = l1
    floor.floor_height = l2 - l1
    segment_rooms(floor=floor)
    

if __name__ == '__main__':
    config.args = parser.parse_args()
    with open('configs/config.json', 'r') as f:
        config.config = json.load(f)
    
    main()