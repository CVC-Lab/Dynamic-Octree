import numpy as np
from octree import DynamicOctree, DynamicOctreeNode, OctreeConstructionParams
from objects import Object
import time
import json
import xml.etree.ElementTree as ET
import pdb

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
        # pdb.set_trace()
        # After updating the root, recursively update all non-root bounding boxes
        self.update_non_root_bounding_boxes(octree, octree.root_node_id)
        nodes = [node for node in octree.nodes if node is not None and node.parent_pointer == -1]
        for node in nodes:
            print(octree.nodes.index(node))
            self.update_non_root_bounding_boxes(octree, octree.nodes.index(node))

        # Re-expand the octree nodes based on new root bounding box
        # octree.expand_octree_node(octree.root_node_id, list(range(octree.num_atoms)), [0] * octree.num_atoms,
        #                         0, octree.num_atoms - 1)

    def update_non_root_bounding_boxes(self, octree, node_id):
        """Recursively update the bounding boxes for non-root nodes."""
        node = octree.nodes[node_id]

        # If the node is a leaf node, we don't need to update further down
        if node.is_leaf():
            return
        
        # print(f"Children of Node {node_id} are {node.child_pointer}")
        # Update bounding boxes for all children
        for i in range(8):
            child_id = node.child_pointer[i]
            if child_id != -1:  # If the child exists
                # print(f"The child id of the node {node_id} is {child_id}")
                octree.compute_non_root_bounding_box(child_id, i)
                # Recursively update children of this child
                self.update_non_root_bounding_boxes(octree, child_id)
