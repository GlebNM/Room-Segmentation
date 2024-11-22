from collections import Counter, defaultdict
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple, Union

import cv2

import faiss
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import open3d as o3d
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree, distance
from sklearn.cluster import DBSCAN, KMeans
import torch
import torch.nn.functional as F
from torchmetrics.functional import pairwise_cosine_similarity
from tqdm import tqdm

import open3d as o3d

import config

def map_grid_to_point_cloud(occupancy_grid_map, resolution, point_cloud):
    """
    Map the occupancy grid back to the original coordinates in the point cloud.

    Parameters:
        occupancy_grid_map (numpy.array): Occupancy grid map as a 2D numpy array, where each cell is marked as either 0 (unoccupied) or 1 (occupied).
        grid_size (tuple): A tuple (width, height) representing the size of the occupancy grid map in meters.
        resolution (float): The resolution of each cell in the grid map in meters.
        point_cloud (numpy.array): 2D numpy array of shape (N, 2), where N is the number of points and each row represents a point (x, y).

    Returns:
        numpy.array: A subset of the original point cloud containing points that correspond to occupied cells in the occupancy grid.
    """

    # make sure image is binary
    occupancy_grid_map = (occupancy_grid_map > 0).astype(np.uint8)

    # Get the occupied cell indices
    y_cells, x_cells = np.where(occupancy_grid_map == 1)

    # Compute the corresponding point coordinates for occupied cells
    # NOTE: The coordinates are shifted by 10.5 cells to account for the padding added to the grid map
    mapped_x_coords = (x_cells - 10.5) * resolution + np.min(point_cloud[:, 0])
    mapped_y_coords = (y_cells - 10.5) * resolution + np.min(point_cloud[:, 1])

    # Stack the mapped x and y coordinates to form the mapped point cloud
    mapped_point_cloud = np.column_stack((mapped_x_coords, mapped_y_coords))

    return mapped_point_cloud

def distance_transform(occupancy_map, reselotion, tmp_path):
    """
        Perform distance transform on the occupancy map to find the distance of each cell to the nearest occupied cell.
        :param occupancy_map: 2D numpy array representing the occupancy map.
        :param reselotion: The resolution of each cell in the grid map in meters.
        :param path: The path to save the distance transform image.
        :return: The distance transform of the occupancy map.
    """

    print("occupancy_map shape: ", occupancy_map.shape)
    bw = occupancy_map.copy()
    full_map = occupancy_map.copy()

    # invert the image
    bw = cv2.bitwise_not(bw)

    # Perform the distance transform algorithm
    bw = np.uint8(bw)
    dist = cv2.distanceTransform(bw, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    print("range of dist: ", np.min(dist), np.max(dist))
    # so we can visualize and threshold it
    cv2.normalize(dist, dist, 0, 255, cv2.NORM_MINMAX)
    plt.figure()
    plt.imshow(dist, cmap="jet", origin="lower")
    plt.savefig(os.path.join(tmp_path, "dist.png"))

    dist = np.uint8(dist)
    # apply Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(dist, (11, 1), 10)
    plt.figure()
    plt.imshow(blur, cmap="jet", origin="lower")
    plt.savefig(os.path.join(tmp_path, "dist_blur.png"))
    _, dist = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    plt.figure()
    plt.imshow(dist, cmap="jet", origin="lower")
    plt.savefig(os.path.join(tmp_path, "dist_thresh.png"))

    # Create the CV_8U version of the distance image
    # It is needed for findContours()
    dist_8u = dist.astype("uint8")
    # Find total markers
    contours, _ = cv2.findContours(dist_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("number of seeds, aka rooms: ", len(contours))

    # print the area of each seed
    for i in range(len(contours)):
        print("area of seed {}: ".format(i), cv2.contourArea(contours[i]))

    # remove small seed contours
    min_area_m = config.config["min_area_m"]
    min_area = (min_area_m / reselotion) ** 2
    print("min_area: ", min_area)
    contours = [c for c in contours if cv2.contourArea(c) > min_area]
    print("number of contours after remove small seeds: ", len(contours))

    # Create the marker image for the watershed algorithm
    markers = np.zeros(dist.shape, dtype=np.int32)
    # Draw the foreground markers
    for i in range(len(contours)):
        cv2.drawContours(markers, contours, i, (i + 1), -1)
    # Draw the background marker
    circle_radius = 1  # in pixels
    cv2.circle(markers, (3, 3), circle_radius, len(contours) + 1, -1)

    # Perform the watershed algorithm
    full_map = cv2.cvtColor(full_map, cv2.COLOR_GRAY2BGR)
    cv2.watershed(full_map, markers)

    plt.figure()
    plt.imshow(markers, cmap="jet", origin="lower")
    plt.savefig(os.path.join(tmp_path, "markers.png"))

    # find the vertices of each room
    room_vertices = []
    for i in range(len(contours)):
        room_vertices.append(np.where(markers == i + 1))
    room_vertices = np.array(room_vertices, dtype=object).squeeze()
    print("room_vertices shape: ", room_vertices.shape)

    return room_vertices