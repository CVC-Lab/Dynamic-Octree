import numpy as np
from octree import DynamicOctree, OctreeConstructionParams
from objects import Object

def generate_movement_function(initial_position, time_step):
    decrease_rate = 0.2  # The rate at which positions decrease
    new_position = tuple(coord - decrease_rate * time_step for coord in initial_position)
    return new_position

def generate_time_series_data(num_objects, num_time_steps):
    time_series_data = {}
    initial_positions = [tuple(np.random.uniform(50.0, 10.0, 3)) for _ in range(num_objects)]
    for t in range(num_time_steps):
        positions_at_t = [generate_movement_function(pos, t) for pos in initial_positions]
        time_series_data[t] = positions_at_t
    return time_series_data

def test_dynamic_octree_with_time_series(time_series_data):
    initial_positions = time_series_data[0]
    objects = [Object(position=pos, id=i) for i, pos in enumerate(initial_positions)]
    num_atoms = len(objects)
    construction_params = OctreeConstructionParams(max_leaf_size=2, max_leaf_dim=100, slack_factor=1.0)
    max_nodes = 20
    octree = DynamicOctree(objects, num_atoms, construction_params, verbose=False, max_nodes=max_nodes)
    octree.build_octree()

    trajectory_data = {}
    nblist_data = {}

    for timestamp, positions in time_series_data.items():
        trajectory_data[timestamp] = {obj.id: pos for obj, pos in zip(objects, positions)}
        
        if timestamp > 0:
            for obj, new_pos in zip(objects, positions):
                octree.update_octree(obj, new_pos)
        
        nblist_data[timestamp] = {obj.id: octree.nb_lists[obj.id] for obj in objects}

    return trajectory_data, nblist_data

# Generate time series data
time_series_data = generate_time_series_data(num_objects=8, num_time_steps=35)
trajectory_data, nblist_data = test_dynamic_octree_with_time_series(time_series_data)
