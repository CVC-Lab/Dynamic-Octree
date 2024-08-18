import numpy as np
from octree import DynamicOctree, DynamicOctreeNode, OctreeConstructionParams
from objects import Object
import time

def generate_movement_function(initial_position, time_step):
    """
    Generate a new position based on a linear decrease function.
    
    Args:
        initial_position (tuple): The starting position (x, y, z).
        time_step (int): The current time step.
    
    Returns:
        tuple: The new position (x, y, z) after applying the movement function.
    """
    decrease_rate = 0.2  # The rate at which positions decrease
    new_position = tuple(coord - decrease_rate * time_step for coord in initial_position)
    return new_position

def generate_time_series_data(num_objects, num_time_steps):
    """
    Generate time series data for a given number of objects and time steps.
    
    Args:
        num_objects (int): Number of objects.
        num_time_steps (int): Number of time steps.
    
    Returns:
        dict: A dictionary with timestamps as keys and lists of positions as values.
    """
    time_series_data = {}

    initial_positions = [tuple(np.random.uniform(0.0, 15.0, 3)) for _ in range(num_objects)]
    
    for t in range(num_time_steps):
        positions_at_t = [generate_movement_function(pos, t) for pos in initial_positions]
        time_series_data[t] = positions_at_t
    
    return time_series_data

def test_dynamic_octree_with_time_series(time_series_data):
    """
    Tests the DynamicOctree with the given time series data and prints the nb_lists for each time step.
    
    Args:
        time_series_data (dict): A dictionary where the key is a timestamp and the value is a list of tuples,
                                 each tuple represents the position (x, y, z) of an object.
    """
    total_time_to_update = 0.0
    min_coords = np.array([float('inf')] * 3)
    max_coords = np.array([float('-inf')] * 3)
    initial_positions = time_series_data[0]
    objects = []
    for i, pos in enumerate(initial_positions):
        obj = Object(position=pos, id=i)
        objects.append(obj)
        
        min_coords = np.minimum(min_coords, pos)
        max_coords = np.maximum(max_coords, pos)
        
    num_atoms = len(objects)
    construction_params = OctreeConstructionParams(max_leaf_size=2, max_leaf_dim=100, slack_factor=1.0)
    max_nodes = 200

    start_time = time.time()
    octree = DynamicOctree(objects, num_atoms, construction_params, verbose=True, max_nodes=max_nodes)
    octree.build_octree()
    total_time_to_build = time.time() - start_time
    
    for timestamp, positions in time_series_data.items():
        if timestamp == 0:
            print(f"\n============Neighbor lists and distances at time stamp: {timestamp}============\n")
            for atom_id in range(len(octree.atoms)):
                print(f"Atom {atom_id}: nb_list = {octree.nb_lists[atom_id]}")
                print(f"      : nb_list_with_dis = {octree.nb_lists_with_dist[atom_id]}\n")
            continue  # Skip the 0th time step as it was already used to initialize

        print(f"\nTime step: {timestamp}\n")    

        start_time = time.time()
        for obj, new_pos in zip(objects, positions):
            octree.update_octree(obj, new_pos)
        update_time = time.time() - start_time
        total_time_to_update += update_time
            
        # print(f"Completed updating all objects in Time Step: {timestamp}")
        # print(f"Time taken to update octree: {update_time:.6f} seconds")
        print(f"Num nodes: {octree.num_nodes}")
        
        print(f"\n============Neighbor lists and distances at time stamp: {timestamp}============\n")
        for atom_id in range(len(octree.atoms)):
            print(f"Atom {atom_id}: nb_list = {octree.nb_lists[atom_id]}")
            print(f"      : nb_list_with_dis = {octree.nb_lists_with_dist[atom_id]}\n")


    total_time_for_trajectory = total_time_to_build + total_time_to_update
    average_time_to_update = total_time_to_update / (len(time_series_data) - 1)

    print("\n========= Final Results =========\n")
    print(f"Total time taken to build the octree: {total_time_to_build:.6f} seconds")
    print(f"Average time to update the octree: {average_time_to_update:.6f} seconds")
    print(f"Total time taken to complete the entire trajectory: {total_time_for_trajectory:.6f} seconds")

time_series_data = generate_time_series_data(num_objects=10, num_time_steps=10)
test_dynamic_octree_with_time_series(time_series_data)

