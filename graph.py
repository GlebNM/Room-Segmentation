"""
    Class to represent the HOV-SG graph
"""

import os
import copy
from typing import Any, Dict, List, Set, Tuple, Union
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree

from scipy.ndimage import binary_erosion, median_filter
from scipy.spatial import Delaunay, Voronoi, voronoi_plot_2d

from sklearn.cluster import DBSCAN
from tqdm import tqdm
import networkx as nx

from floor import Floor
from room import Room
from utils import distance_transform, map_grid_to_point_cloud

import config 

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
    # xyz = xyz[xyz[:, 1] < floor_zero_level + 1.8]
    # xyz = xyz[xyz[:, 1] > floor_zero_level + 0.8]
    # xyz_full = xyz_full[xyz_full[:, 1] < floor_zero_level + 1.8]

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

    # using the 2D room vertices, map the room back to the original point cloud using KDTree
    # room_pcds = []
    # room_masks = []
    # room_2d_points = []
    # floor_tree = cKDTree(np.array(floor_pcd.points))
    # for i in tqdm(range(len(room_vertices)), desc="Assign floor points to rooms"):
    #     room = np.zeros_like(full_map)
    #     room[room_vertices[i][0], room_vertices[i][1]] = 255
    #     room_masks.append(room)
    #     room_m = map_grid_to_point_cloud(room, resolution, pcd_2d)
    #     room_2d_points.append(room_m)
    #     # extrude the 2D room to 3D room by adding z value from floor zero level to floor zero level + floor height, step by 0.1m
    #     z_levels = np.arange(
    #         floor_zero_level, floor_zero_level + floor_height, 0.05
    #     )
    #     z_levels = z_levels.reshape(-1, 1)
    #     z_levels *= -1
    #     room_m3dd = []
    #     for z in z_levels:
    #         room_m3d = np.hstack((room_m, np.ones((room_m.shape[0], 1)) * z))
    #         room_m3dd.append(room_m3d)
    #     room_m3d = np.concatenate(room_m3dd, axis=0)
    #     pcd = o3d.geometry.PointCloud()
    #     pcd.points = o3d.utility.Vector3dVector(room_m3d)
    #     # rotate floor pcd to align with the original point cloud
    #     T1 = np.eye(4)
    #     T1[:3, :3] = Rotation.from_euler("x", 90, degrees=True).as_matrix()
    #     pcd.transform(T1)
    #     # find the nearest point in the original point cloud
    #     _, idx = floor_tree.query(np.array(pcd.points), workers=-1)
    #     pcd = floor_pcd.select_by_index(idx)
    #     room_pcds.append(pcd)
    # # self.room_masks[floor.floor_id] = room_masks TODO

    # # compute the features of room: input a list of poses and images, output a list of embeddings list
    # room_index = 0
    # for i in range(len(room_2d_points)):
    #     room = Room(
    #         str(floor.floor_id) + "_" + str(room_index),
    #         floor.floor_id,
    #         name="room_" + str(room_index),
    #     )
    #     room.pcd = room_pcds[i]
    #     room.vertices = room_2d_points[i]
    #     # self.floors[int(floor.floor_id)].add_room(room) TODO
    #     room.room_height = floor_height
    #     room.room_zero_level = floor.floor_zero_level
    #     # self.rooms.append(room) TODO
    #     room_index += 1
    # print(
    #     "number of rooms in floor {} is {}".format(
    #         floor.floor_id, len(self.floors[int(floor.floor_id)].rooms)
    #     )
    # )
    # print("number of rooms:", len(room_2d_points))