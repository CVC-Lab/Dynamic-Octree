import numpy as np
import sys
import time
from octree import DynamicOctree, DynamicOctreeNode, OctreeConstructionParams
from objects import Object
import matplotlib.pyplot as plt

def get_memory_usage(obj, exclude_attributes=None):
    """
    Calculate the memory usage of an object and its members, optionally excluding specific attributes.
    
    Args:
        obj (object): The object to measure.
        exclude_attributes (list, optional): List of attribute names to exclude from memory calculation.
    
    Returns:
        int: Memory usage in bytes.
    """
    memory_usage = sys.getsizeof(obj)
    if hasattr(obj, '__dict__'):
        for key, value in obj.__dict__.items():
            if exclude_attributes and key in exclude_attributes:
                continue  # Skip the memory calculation for excluded attributes
            memory_usage += get_memory_usage(value, exclude_attributes)
    elif isinstance(obj, (list, tuple, set, dict)):
        for item in obj:
            memory_usage += get_memory_usage(item, exclude_attributes)
    return memory_usage

def generate_movement_function(initial_position, time_step):
    decrease_rate = 0.2
    new_position = tuple(coord - decrease_rate * time_step for coord in initial_position)
    return new_position

def generate_time_series_data(num_objects, num_time_steps):
    time_series_data = {}
    initial_positions = [tuple(np.random.uniform(10.0, 500.0, 3)) for _ in range(num_objects)]
    
    for t in range(num_time_steps):
        positions_at_t = [generate_movement_function(pos, t) for pos in initial_positions]
        time_series_data[t] = positions_at_t
    
    return time_series_data

def test_dynamic_octree_with_varying_objs(time_series_data):
    total_time_to_update = 0.0
    initial_positions = time_series_data[0]
    objects = [Object(position=pos, id=i) for i, pos in enumerate(initial_positions)]
    
    num_atoms = len(objects)
    construction_params = OctreeConstructionParams(max_leaf_size=2, max_leaf_dim=100, slack_factor=1.0)
    max_nodes = 50
    interaction_distance = 50
    
    start_time = time.time()
    octree = DynamicOctree(objects, num_atoms, construction_params, verbose=True, max_nodes=max_nodes, interaction_distance=interaction_distance)
    octree.build_octree()
    total_time_to_build = time.time() - start_time
    
    initial_memory_usage = get_memory_usage(octree)
    
    for timestamp, positions in time_series_data.items():
        if timestamp == 0:
            continue  # Skip the 0th time step as it was already used to initialize

        start_time = time.time()
        for obj, new_pos in zip(objects, positions):
            octree.update_octree(obj, new_pos)
        update_time = time.time() - start_time
        total_time_to_update += update_time
    
    total_memory_usage = get_memory_usage(octree, exclude_attributes=['nb_lists', 'nb_lists_with_dist'])
    
    total_time_for_trajectory = total_time_to_build + total_time_to_update
    average_time_to_update = total_time_to_update / (len(time_series_data) - 1)
    
    return total_memory_usage, total_time_to_build, average_time_to_update, total_time_for_trajectory

def bytes_to_mb(bytes_value):
    return bytes_value / 1024  # Convert bytes to KB

def plot_metrics_objs():
    object_counts = range(10, 5011, 500)  # From 10 to 10,000 in steps of 100
    memory_usages = []
    build_times = []
    avg_update_times = []
    total_times = []

    for num_objects in object_counts:
        print(f"Testing with {num_objects} objects...")
        time_series_data = generate_time_series_data(num_objects=num_objects, num_time_steps=100)
        total_memory_usage, total_time_to_build, average_time_to_update, total_time_for_trajectory = test_dynamic_octree_with_varying_objs(time_series_data)
        
        # Collect metrics
        memory_usages.append(bytes_to_mb(total_memory_usage)) 
        build_times.append(total_time_to_build)
        avg_update_times.append(average_time_to_update)
        total_times.append(total_time_for_trajectory)
    
    plt.figure(figsize=(10, 8))

    # Plot total memory usage
    plt.subplot(2, 2, 1)
    plt.plot(object_counts, memory_usages, marker='o')
    # plt.yscale('log')
    plt.xlabel('Number of Objects')
    plt.ylabel('Total Memory Usage (KB)')
    plt.title('Memory Usage vs. Number of Objects')
    plt.grid(True)

    # Plot total time to build
    plt.subplot(2, 2, 2)
    plt.plot(object_counts, build_times, marker='o', color='r')
    # plt.yscale('log')
    plt.xlabel('Number of Objects')
    plt.ylabel('Total Time to Build the Octree(seconds)')
    plt.title('Build Time vs. Number of Objects')
    plt.grid(True)

    # Plot average time to update
    plt.subplot(2, 2, 3)
    plt.plot(object_counts, avg_update_times, marker='o', color='g')
    # plt.yscale('log')  
    plt.xlabel('Number of Objects')
    plt.ylabel('Average Time to Update all Objects(seconds)')
    plt.title('Average Update Time vs. Number of Objects')
    plt.grid(True)

    # Plot total time for trajectory
    plt.subplot(2, 2, 4)
    plt.plot(object_counts, total_times, marker='o', color='m')
    # plt.yscale('log')  
    plt.xlabel('Number of Objects')
    plt.ylabel('Total Time for Trajectory (seconds)')
    plt.title('Total Trajectory Time vs. Number of Objects')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def test_dynamic_octree_varying_inter_dist(time_series_data, interaction_distance):
    total_time_to_update = 0.0
    initial_positions = time_series_data[0]
    objects = [Object(position=pos, id=i) for i, pos in enumerate(initial_positions)]
    
    num_atoms = len(objects)
    construction_params = OctreeConstructionParams(max_leaf_size=2, max_leaf_dim=100, slack_factor=1.0)
    max_nodes = 50
    
    start_time = time.time()
    octree = DynamicOctree(objects, num_atoms, construction_params, verbose=False, max_nodes=max_nodes, interaction_distance=interaction_distance)
    octree.build_octree()
    total_time_to_build = time.time() - start_time
    
    initial_memory_usage = get_memory_usage(octree)
    
    for timestamp, positions in time_series_data.items():
        if timestamp == 0:
            continue  # Skip the 0th time step as it was already used to initialize

        start_time = time.time()
        for obj, new_pos in zip(objects, positions):
            octree.update_octree(obj, new_pos)
        update_time = time.time() - start_time
        total_time_to_update += update_time
    
    total_memory_usage = get_memory_usage(octree)
    
    total_time_for_trajectory = total_time_to_build + total_time_to_update
    average_time_to_update = total_time_to_update / (len(time_series_data) - 1)
    
    return total_memory_usage, total_time_to_build, average_time_to_update, total_time_for_trajectory

def plot_metrics_with_varying_interaction_distances():
    interaction_distances = range(0, 501, 20)
    object_counts = 500
    memory_usages = []
    build_times = []
    avg_update_times = []
    total_times = []

    time_series_data = generate_time_series_data(num_objects=object_counts, num_time_steps=100)
    for distance in interaction_distances:
        print(f"Testing with interaction distance {distance}...")
        total_memory_usage, total_time_to_build, average_time_to_update, total_time_for_trajectory = test_dynamic_octree_varying_inter_dist(time_series_data, distance)
        
        # Collect metrics
        memory_usages.append(bytes_to_mb(total_memory_usage)) 
        build_times.append(total_time_to_build)
        avg_update_times.append(average_time_to_update)
        total_times.append(total_time_for_trajectory)
    
    plt.figure(figsize=(10, 8))

    # Plot total memory usage
    plt.subplot(2, 2, 1)
    plt.plot(interaction_distances, memory_usages, marker='o')
    plt.xlabel('Interaction Distance')
    plt.ylabel('Total Memory Usage (KB)')
    plt.title('Memory Usage vs. Interaction Distance')
    plt.grid(True)

    # Plot total time to build
    plt.subplot(2, 2, 2)
    plt.plot(interaction_distances, build_times, marker='o', color='r')
    plt.xlabel('Interaction Distance')
    plt.ylabel('Total Time to Build the Octree (seconds)')
    plt.title('Build Time vs. Interaction Distance')
    plt.grid(True)

    # Plot average time to update
    plt.subplot(2, 2, 3)
    plt.plot(interaction_distances, avg_update_times, marker='o', color='g')
    plt.xlabel('Interaction Distance')
    plt.ylabel('Average Time to Update all Objects (seconds)')
    plt.title('Average Update Time vs. Interaction Distance')
    plt.grid(True)

    # Plot total time for trajectory
    plt.subplot(2, 2, 4)
    plt.plot(interaction_distances, total_times, marker='o', color='m')
    plt.xlabel('Interaction Distance')
    plt.ylabel('Total Time for Trajectory (seconds)')
    plt.title('Total Trajectory Time vs. Interaction Distance')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def test_dynamic_octree_insertion(num_objects, num_inserts):
    initial_positions = [tuple(np.random.uniform(10.0, 500.0, 3)) for _ in range(num_objects)]
    objects = [Object(position=pos, id=i) for i, pos in enumerate(initial_positions)]
    
    num_atoms = len(objects)
    construction_params = OctreeConstructionParams(max_leaf_size=2, max_leaf_dim=100, slack_factor=1.0)
    max_nodes = 50
    interaction_distance = 50
    
    octree = DynamicOctree(objects, num_atoms, construction_params, verbose=False, max_nodes=max_nodes, interaction_distance=interaction_distance)
    octree.build_octree()
    
    insertion_times = []
    
    for i in range(num_inserts):
        new_position = tuple(np.random.uniform(10.0, 500.0, 3))
        new_obj = Object(position=new_position, id=num_objects + i)
        
        start_time = time.time()
        octree.insert_object(new_obj)
        insertion_time = time.time() - start_time
        insertion_times.append(insertion_time)
    
    average_insertion_time = np.mean(insertion_times)
    
    return average_insertion_time

def test_dynamic_octree_deletion(num_objects, num_deletes):
    initial_positions = [tuple(np.random.uniform(10.0, 500.0, 3)) for _ in range(num_objects)]
    objects = [Object(position=pos, id=i) for i, pos in enumerate(initial_positions)]
    
    num_atoms = len(objects)
    construction_params = OctreeConstructionParams(max_leaf_size=2, max_leaf_dim=100, slack_factor=1.0)
    max_nodes = 50
    interaction_distance = 50
    
    octree = DynamicOctree(objects, num_atoms, construction_params, verbose=False, max_nodes=max_nodes, interaction_distance=interaction_distance)
    octree.build_octree()
    
    deletion_times = []
    
    for i in range(num_deletes):
        obj_to_delete = objects[i % num_objects]
        
        start_time = time.time()
        octree.delete_atom(obj_to_delete)
        deletion_time = time.time() - start_time
        deletion_times.append(deletion_time)
    
    average_deletion_time = np.mean(deletion_times)
    
    return average_deletion_time

def plot_metrics_insertion():
    object_counts = range(10, 5011, 500)  # From 10 to 5000 in steps of 500
    avg_insertion_times = []

    for num_objects in object_counts:
        print(f"Testing insertion with {num_objects} objects...")
        avg_insertion_time = test_dynamic_octree_insertion(num_objects=num_objects, num_inserts=100)
        avg_insertion_times.append(avg_insertion_time)
    
    # plt.figure()
    # plt.plot(object_counts, avg_insertion_times, marker='o')
    # plt.xlabel("Number of Objects")
    # plt.ylabel("Average Insertion Time (seconds)")
    # plt.title("Average Insertion Time vs Number of Objects")
    # plt.grid(True)
    # plt.show()

def plot_metrics_deletion():
    object_counts = range(10, 5011, 500)  # From 10 to 5000 in steps of 500
    avg_deletion_times = []

    for num_objects in object_counts:
        print(f"Testing deletion with {num_objects} objects...")
        avg_deletion_time = test_dynamic_octree_deletion(num_objects=num_objects, num_deletes=100)
        avg_deletion_times.append(avg_deletion_time)
    
    # plt.figure()
    # plt.plot(object_counts, avg_deletion_times, marker='o')
    # plt.xlabel("Number of Objects")
    # plt.ylabel("Average Deletion Time (seconds)")
    # plt.title("Average Deletion Time vs Number of Objects")
    # plt.grid(True)
    # plt.show()


# plot_metrics_insertion()
plot_metrics_deletion()