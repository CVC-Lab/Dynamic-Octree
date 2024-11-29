import numpy as np
from octree import DynamicOctree, DynamicOctreeNode, OctreeConstructionParams
from objects import Object
import time
import plotly.graph_objects as go
import math
import random

def generate_movement_function(initial_position, time_step):
    """
    Generate a new position based on a linear decrease function.
    """
    decrease_rate = 2  # The rate at which positions decrease
    new_x = initial_position[0] + decrease_rate * time_step
    new_y = initial_position[1] + decrease_rate * time_step
    return (new_x, new_y, 0)  # Set z to 0


def generate_time_series_data(num_objects, num_time_steps):
    """
    Generate time series data for a given number of objects and time steps.
    """
    time_series_data = {}
    initial_positions = [(np.random.uniform(0.0, 100.0), np.random.uniform(0.0, 100.0), 0) for _ in range(num_objects)]

    for t in range(num_time_steps):
        positions_at_t = [(obj_id, list(map(int, generate_movement_function(pos, t)))) for obj_id, pos in enumerate(initial_positions)]
        time_series_data[t] = positions_at_t

    return time_series_data


def test_dynamic_octree(time_series_data, output_file, initial_bbox_coords, bbox_update_interval):
    total_time_to_update = 0.0
    total_time_for_trajectory = 0.0
    updates = 0
    
    objects = {}
    
    with open(output_file, 'w') as file:
        file.write(f"Initial Bounding Box: {initial_bbox_coords[0]} to {initial_bbox_coords[1]}\n\n")
        
        # Initialize the octree with objects inside the bounding box
        initial_positions = time_series_data[0]
        for i, pos in initial_positions:
            obj = Object(position=pos, id=i)
            objects[i] = obj
        
        construction_params = OctreeConstructionParams(max_leaf_size=15, max_leaf_dim=100, slack_factor=1.0)
        octree = DynamicOctree(list(objects.values()), len(objects), construction_params, verbose=True, max_nodes=200)
        start_time = time.time()
        octree.build_octree(*initial_bbox_coords)
        total_time_to_build = time.time() - start_time
        
        n_objects_total = 0
        n_upds_total = 0
        n_dels_total = 0

        for timestamp, positions in time_series_data.items():
            if timestamp == 0:
                continue

            start_time = time.time()
            local_n_upds, local_n_dels, local_n_objects = 0, 0, 0
            
            # Update positions of objects within the bounding box
            for id, new_pos in positions:
                file.write(f"New position of object {id}: {new_pos}\n")
                updates += 1
                
                if id in objects:  # Update existing object
                    obj = objects[id]
                    local_n_upds += 1
                    prev_node = octree.object_to_node_map[obj]
                    target_atom, target_node = octree.update_octree(obj, new_pos)
                    if prev_node != target_node:
                        local_n_dels += 1
                        nb_list = octree.update_nb_lists_local(target_atom, target_node)
                else:
                    # Insert new object inside the bounding box
                    local_n_objects += 1
                    obj = Object(position=new_pos, id=id)
                    objects[id] = obj
                    octree.insert_object(obj)
            
            n_objects_total += local_n_objects
            n_upds_total += local_n_upds
            n_dels_total += local_n_dels
            total_time_for_trajectory += time.time() - start_time

            nb_list = octree.nb_lists_with_dist
            file.write("\nAfter Updating position of all the atoms:\n")
            for i, nb in enumerate(nb_list):
                if nb:  # Check if the list is not empty
                    file.write(f"Atom {i}: {nb}\n")
                    
        # Write final results
        file.write("\n========= Final Results =========\n")
        file.write(f"Total objects inserted: {n_objects_total}, atoms updated: {n_upds_total}, atoms deleted: {n_dels_total}.\n\n")
        file.write(f"Total time to build octree: {total_time_to_build:.6f} seconds\n")
        file.write(f"Total time to process trajectories: {total_time_for_trajectory:.6f} seconds\n")

        return octree


neighbor_cutoff = 10  # Set the radius for neighbors

initial_boundingbox_coords = (np.array([0, 0, 0]), np.array([100, 100, 0]))
bbox_update_interval = 10

time_series_data = generate_time_series_data(num_objects=200, num_time_steps=1)

octree = test_dynamic_octree(time_series_data, 'results_synthetic.txt', initial_boundingbox_coords, bbox_update_interval)

def visualize_objects_and_boxes(octree):
    """
    Visualize the octree nodes and atoms.

    Args:
        octree (DynamicOctree): The octree instance containing nodes and objects.
    """
    # Create a figure
    fig = go.Figure()

    # Plot all atoms (objects) in the octree
    for obj_id, obj in enumerate(octree.atoms):
        obj_pos = obj.get_position()

        # Plot the object without labels initially
        fig.add_trace(go.Scatter(
            x=[obj_pos[0]], 
            y=[obj_pos[1]],
            mode='markers',
            marker=dict(color='blue', size=8),
            name=f'Atom {obj_id}',
            showlegend=False
        ))

    # Draw octree nodes (bounding boxes)
    for node in octree.nodes:
        # print(octree.nodes.index(node))
        node_min, node_max = node.get_node_bounds()

        # Draw the bounding box for each node
        fig.add_trace(go.Scatter(
            x=[node_min[0], node_max[0], node_max[0], node_min[0], node_min[0]],
            y=[node_min[1], node_min[1], node_max[1], node_max[1], node_min[1]],
            mode='lines',
            name=f'Node {node}',
            line=dict(color='green', width=2, dash='dash'),
            showlegend=False
        ))

    # Set up layout for the figure
    fig.update_layout(
        title='Octree Nodes and Atoms',
        xaxis_title='X-axis',
        yaxis_title='Y-axis',
        width=800,
        height=800,
        xaxis=dict(showgrid=True, zeroline=True),
        yaxis=dict(showgrid=True, zeroline=True)
    )

    fig.show()


# Call the visualization function
visualize_objects_and_boxes(octree)
