"""
    Class to represent the HOV-SG graph
"""

import os

import cv2
import numpy as np

import matplotlib.pyplot as plt

from segmentation.floor import Floor
from segmentation.utils import distance_transform

import config as config 

def segment_rooms(floor: Floor):
    """
    Segment the rooms from the floor point cloud
    :param floor: Floor, The floor object
    :param path: str, The path to save the intermediate results
    """
    tmp_floor_path = config.args.output_path
    parameters = config.config
    
    os.makedirs(tmp_floor_path, exist_ok=True)
    floor_pcd = floor.pcd
    xyz = np.asarray(floor_pcd.points)
    xyz_full = xyz.copy()
    floor_zero_level = floor.floor_zero_level
    floor_height = floor.floor_height
    ## Slice below the ceiling ##
    xyz = xyz[xyz[:, 1] < floor_zero_level + floor_height - parameters["below_ceiling"]]
    xyz = xyz[xyz[:, 1] >= floor_zero_level + parameters["above_floor"]]
    xyz_full = xyz_full[xyz_full[:, 1] < floor_zero_level + floor_height - parameters["below_ceiling_full"]]
    ## Slice above the floor and below the ceiling ##

    # project the point cloud to 2d
    pcd_2d = xyz[:, [0, 2]]
    xyz_full = xyz_full[:, [0, 2]]

    # define the grid size and resolution based on the 2d point cloud
    grid_size = (
        int(np.max(pcd_2d[:, 0]) - np.min(pcd_2d[:, 0])),
        int(np.max(pcd_2d[:, 1]) - np.min(pcd_2d[:, 1])),
    )
    grid_size = (grid_size[0] + 1, grid_size[1] + 1)
    resolution = parameters["resolution"]
    print("grid_size: ", resolution)

    # calc 2d histogram of the floor using the xyz point cloud to extract the walls skeleton
    num_bins = (int(grid_size[0] // resolution), int(grid_size[1] // resolution))
    num_bins = (num_bins[1] + 1, num_bins[0] + 1)
    hist, _, _ = np.histogram2d(pcd_2d[:, 1], pcd_2d[:, 0], bins=num_bins)
    if True:
        # plot the histogram
        plt.figure()
        plt.imshow(hist, interpolation="nearest", cmap="jet", origin="lower")
        plt.colorbar()
        plt.savefig(os.path.join(tmp_floor_path, "2D_histogram.png"))
    # applythresholding
    hist = cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    hist = cv2.GaussianBlur(hist, (5, 5), 1)
    hist_threshold = parameters["hist_threshold"] * np.max(hist)
    _, walls_skeleton = cv2.threshold(hist, hist_threshold, 255, cv2.THRESH_BINARY)

    # create a bigger image to avoid losing the walls
    walls_skeleton = cv2.copyMakeBorder(
        walls_skeleton, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0
    )

    # apply closing to the walls skeleton
    kernal = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    walls_skeleton = cv2.morphologyEx(
        walls_skeleton, cv2.MORPH_CLOSE, kernal, iterations=1
    )

    # extract outside boundary from histogram of xyz_full
    hist_full, _, _ = np.histogram2d(xyz_full[:, 1], xyz_full[:, 0], bins=num_bins)
    hist_full = cv2.normalize(hist_full, hist_full, 0, 255, cv2.NORM_MINMAX).astype(
        np.uint8
    )
    hist_full = cv2.GaussianBlur(hist_full, (21, 21), 2)
    _, outside_boundary = cv2.threshold(hist_full, 0, 255, cv2.THRESH_BINARY)

    # create a bigger image to avoid losing the walls
    outside_boundary = cv2.copyMakeBorder(
        outside_boundary, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0
    )

    # apply closing to the outside boundary
    kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    outside_boundary = cv2.morphologyEx(
        outside_boundary, cv2.MORPH_CLOSE, kernal, iterations=3
    )

    # extract the outside contour from the outside boundary
    contours, _ = cv2.findContours(
        outside_boundary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    outside_boundary = np.zeros_like(outside_boundary)
    cv2.drawContours(outside_boundary, contours, -1, (255, 255, 255), -1)
    outside_boundary = outside_boundary.astype(np.uint8)

    if True:
        plt.figure()
        plt.imshow(walls_skeleton, cmap="gray", origin="lower")
        plt.savefig(os.path.join(tmp_floor_path, "walls_skeleton.png"))

        plt.figure()
        plt.imshow(outside_boundary, cmap="gray", origin="lower")
        plt.savefig(os.path.join(tmp_floor_path, "outside_boundary.png"))

    # combine the walls skelton and outside boundary
    full_map = cv2.bitwise_or(walls_skeleton, cv2.bitwise_not(outside_boundary))

    # apply closing to the full map
    kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    full_map = cv2.morphologyEx(full_map, cv2.MORPH_CLOSE, kernal, iterations=2)

    if True:
        # plot the full map
        plt.figure()
        plt.imshow(full_map, cmap="gray", origin="lower")
        plt.savefig(os.path.join(tmp_floor_path, "full_map.png"))
    # apply distance transform to the full map
    room_vertices = distance_transform(full_map, resolution, tmp_floor_path)
    