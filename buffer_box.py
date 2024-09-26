import numpy as np
from octree import DynamicOctree, DynamicOctreeNode, OctreeConstructionParams
from objects import Object
import time
import json
import xml.etree.ElementTree as ET

class BufferBox:
    def __init__(self, min_coords, max_coords):
        self.min_coords = np.array(min_coords)
        self.max_coords = np.array(max_coords)

    def contains(self, position):
        """Check if a position is inside the buffer box."""
        return np.all(position >= self.min_coords) and np.all(position <= self.max_coords)

    def update_box(self, octree, new_min_coords, new_max_coords):
        """Update the buffer box with new coordinates."""
        self.min_coords = np.array(new_min_coords)
        self.max_coords = np.array(new_max_coords)

        # Recompute the root bounding box
        octree.compute_root_bounding_box(octree.root_node_id, octree.construction_params.get_slack_factor(),
                                    self.min_coords, self.max_coords)
        
        # Re-expand the octree nodes based on new root bounding box
        # octree.expand_octree_node(octree.root_node_id, list(range(octree.num_atoms)), [0] * octree.num_atoms,
        #                         0, octree.num_atoms - 1)