import random
import time
from octree import DynamicOctree, DynamicOctreeNode, OctreeConstructionParams
from objects import Object
import numpy as np

def test_dynamic_octree_with_time_series(time_series_data):
    """
    Tests the DynamicOctree with the given time series data and prints the nb_lists for each time step.
    
    Args:
        time_series_data (dict): A dictionary where the key is a timestamp and the value is a list of tuples,
                                 each tuple represents the position (x, y, z) of an object.
    """
    
    start_time = time.time()  # Record the start time
    
    # Create objects and add them to the octree at the 0th time step
    initial_positions = time_series_data[0]
    objects = []
    for i, pos in enumerate(initial_positions):
        obj = Object(position=pos, id=i)
        objects.append(obj)
        
    # Initialize objects from the first time step
    num_atoms = len(objects)
    construction_params = OctreeConstructionParams(max_leaf_size=2, max_leaf_dim=100, slack_factor=1.0)
    max_nodes = 10
    
    octree = DynamicOctree(objects, num_atoms, construction_params, verbose=True, max_nodes=max_nodes)
    
    # Build the octree
    octree.build_octree()
    
    # Process the time series data and update the octree
    for timestamp, positions in time_series_data.items():
        if timestamp == 0:
            continue  # Skip the 0th time step as it was already used to initialize

        print(f"\nTime step: {timestamp}\n")
        
        # Update the positions of objects in the octree
        for obj, new_pos in zip(objects, positions):
            octree.update_octree(obj, new_pos)
            
        print(f"\nNeighbor lists and distances at time stamp: {timestamp}")
        for atom_id in range(len(octree.atoms)):
            print(f"Atom {atom_id}: nb_list = {octree.nb_lists[atom_id]}")
            print(f"Atom {atom_id}: nb_list_with_dis = {octree.nb_lists_with_dist[atom_id]}")
    
    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time
    print(f"\nTotal time taken: {elapsed_time:.4f} seconds")

time_series = {
    0: [(1.0, 1.0, 1.0), (2.0, 2.0, 2.0), (2.0, 2.1, 2.1), (2.3, 2.3, 2.3), (2.5, 2.5, 2.5)],
    1: [(1.5, 1.5, 1.5), (2.5, 2.5, 2.1), (2.0, 2.2, 2.1), (2.3, 2.2, 2.3), (2.1, 2.1, 2.5)],
    2: [(2.0, 2.0, 2.0), (1.0, 1.0, 1.0), (2.2, 2.1, 2.2), (2.3, 2.4, 2.2), (2.1, 2.5, 2.3)],
}

test_dynamic_octree_with_time_series(time_series)
