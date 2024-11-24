"""
Room class to represent a room in a HOV-SGraph.
"""

import json
import os
from collections import defaultdict
from typing import Any, List

import numpy as np
import open3d as o3d


class Room:
    """
    Class to represent a room in a building.
    :param room_id: Unique identifier for the room
    :param floor_id: Identifier of the floor this room belongs to
    :param name: Name of the room (e.g., "Living Room", "Bedroom")
    """
    def __init__(self, room_id, floor_id, name=None):
        self.room_id = room_id  # Unique identifier for the room
        self.name = name  # Name of the room (e.g., "Living Room", "Bedroom")
        self.category = None  # placeholder for a GT category
        self.floor_id = floor_id  # Identifier of the floor this room belongs to
        self.objects = []  # List of objects inside the room
        self.vertices = []  # indices of the room in the point cloud 8 vertices
        self.embeddings = []  # List of tensors of embeddings of the room
        self.pcd = None  # Point cloud of the room
        self.room_height = None  # Height of the room
        self.room_zero_level = None  # Zero level of the room
        self.represent_images = []  # 5 images that represent the appearance of the room
        self.object_counter = 0

    def add_object(self, objectt):
        """
        Method to add objects to the room
        :param objectt: Object object to be added to the room
        """
        self.objects.append(objectt)  # Method to add objects to the room

    def load(self, path):
        """
        Load the room from folder as ply for the point cloud
        and json for the metadata
        """
        # load the point cloud
        self.pcd = o3d.io.read_point_cloud(os.path.join(path, str(self.room_id) + ".ply"))
        # load the metadata

    def __str__(self):
        return f"Room ID: {self.room_id}, Name: {self.name}, Floor ID: {self.floor_id}"
