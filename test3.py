import numpy as np
from octree import DynamicOctree, OctreeConstructionParams
from objects import Object
import time
from scipy.spatial import KDTree

def generate_movement_function(initial_position, time_step):
    decrease_rate = 0.2  # The rate at which positions decrease
    new_position = tuple(coord - decrease_rate * time_step for coord in initial_position)
    return new_position

def generate_time_series_data(num_objects, num_time_steps):
    time_series_data = {}
    initial_positions = [tuple(np.random.uniform(0.0, 50.0, 3)) for _ in range(num_objects)]
    for t in range(num_time_steps):
        positions_at_t = [generate_movement_function(pos, t) for pos in initial_positions]
        time_series_data[t] = positions_at_t
    return time_series_data

def test_dynamic_octree(time_series_data, initial_bbox_coords):
    objects = {}
        
    # Initialize the octree with objects inside the bounding box
    initial_positions = time_series_data[0]
    for i, pos in enumerate(initial_positions):
        obj = Object(position=pos, id=i, timestamp=0)
        objects[i] = obj
    
    construction_params = OctreeConstructionParams(max_leaf_size=10, max_leaf_dim=100, slack_factor=1.0)
    octree = DynamicOctree(list(objects.values()), len(objects), construction_params, verbose=False, max_nodes=200)
    start_time = time.time()
    octree.build_octree(*initial_bbox_coords)
    total_time_to_build = time.time() - start_time
    
    start_time = time.time()
    for atom_index in range(octree.num_atoms):
        node = octree.get_node_containing_point(octree.atoms[atom_index])
        octree.update_nb_lists(atom_index, node)
    time_to_build_nb = time.time() - start_time

    # Write final results
    print("========= Octree Results =========")
    print(f"Total time to build octree: {total_time_to_build:.6f} seconds")
    print(f"Total time to build nblist: {time_to_build_nb:.6f} seconds\n")
    
    return octree


def test_dynamic_kdtree(time_series_data, initial_bbox_coords, search_radius):
    objects = {}
        
    # Initialize the KD-Tree with objects inside the bounding box
    initial_positions = time_series_data[0]
    for i, pos in enumerate(initial_positions):
        obj = Object(position=pos, timestamp=0, id=i)  # Initialize with timestamp and id
        objects[i] = obj
    
    positions_array = np.array([(obj.x, obj.y, obj.z) for obj in objects.values()])
    start_time = time.time()
    kdtree = KDTree(positions_array)
    total_time_to_build = time.time() - start_time

    start_time = time.time()
    neighbor_lists = {}
    for atom_index in range(len(objects)):
        neighbors = kdtree.query_ball_point(positions_array[atom_index], search_radius)
        neighbor_lists[atom_index] = neighbors
    time_to_build_nb = time.time() - start_time

    # Write final results
    print("\n========= KD-Tree Results =========")
    print(f"Total time to build KD-Tree: {total_time_to_build:.6f} seconds")
    print(f"Total time to build neighbor lists: {time_to_build_nb:.6f} seconds")
    
    return neighbor_lists


# Generate time series data
initial_boundingbox_coords = (np.array([0, 0, 0]), np.array([50, 50, 50]))

search_radius = 100

for i in [100, 1000, 5000, 10000]:
    time_series_data = generate_time_series_data(num_objects=i, num_time_steps=1)
    _ = test_dynamic_kdtree(time_series_data, initial_boundingbox_coords, search_radius)
    _ = test_dynamic_octree(time_series_data, initial_boundingbox_coords)

