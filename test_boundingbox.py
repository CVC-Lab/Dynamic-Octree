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
        
        construction_params = OctreeConstructionParams(max_leaf_size=10, max_leaf_dim=100, slack_factor=1.0)
        octree = DynamicOctree(list(objects.values()), len(objects), construction_params, verbose=False, max_nodes=200)
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


def visualize_objects_and_boxes(objects, initial_bbox_coords):
    min_bbox, max_bbox = initial_bbox_coords
    
    fig = go.Figure()

    # Draw the entire bounding box
    fig.add_trace(go.Scatter(
        x=[min_bbox[0], max_bbox[0], max_bbox[0], min_bbox[0], min_bbox[0]],
        y=[min_bbox[1], min_bbox[1], max_bbox[1], max_bbox[1], min_bbox[1]],
        mode='lines',
        name='Bounding Box',
        line=dict(color='black', width=2),
        fill='toself',
        fillcolor='rgba(0, 0, 0, 0.1)',
    ))

    # Plot the objects
    for obj_id, pos in objects:
        fig.add_trace(go.Scatter(
            x=[pos[0]], 
            y=[pos[1]],
            mode='markers+text',
            marker=dict(color='blue', size=8),
            text=[str(obj_id)],
            textposition="top center",
            showlegend=False
        ))

    fig.update_layout(
        title='Objects and Bounding Boxes in 2D',
        xaxis_title='X-axis',
        yaxis_title='Y-axis',
        width=800,
        height=800,
        xaxis=dict(showgrid=True, zeroline=True),
        yaxis=dict(showgrid=True, zeroline=True),
    )
    
    fig.show()


#========================================================================================================


def visualize_objects_and_variable_circles(octree, num_objects=4):
    """
    Visualize all objects, and randomly select num_objects to draw circles around them
    with a variable radius for each object. Highlight and label objects within each circle.
    
    Args:
        octree (DynamicOctree): The octree instance containing objects.
        num_objects (int): The number of random objects to draw circles around.
    """
    fig = go.Figure()

    # Plot all objects in the bounding box
    for obj_id, obj in enumerate(octree.atoms):
        obj_pos = obj.get_position()
        
        # Plot the object without labels initially
        fig.add_trace(go.Scatter(
            x=[obj_pos[0]], 
            y=[obj_pos[1]],
            mode='markers',
            marker=dict(color='blue', size=8),
            showlegend=False
        ))

    # Randomly select num_objects objects to draw circles around
    selected_obj_ids = random.sample(range(len(octree.atoms)), num_objects)

    for selected_obj_id in selected_obj_ids:
        selected_obj = octree.atoms[selected_obj_id]
        selected_obj_pos = selected_obj.get_position()

        # Generate a random cutoff for each object
        variable_cutoff = random.uniform(5, 20)

        # Draw the circle around the selected object
        theta = np.linspace(0, 2 * np.pi, 100)
        circle_x = selected_obj_pos[0] + variable_cutoff * np.cos(theta)
        circle_y = selected_obj_pos[1] + variable_cutoff * np.sin(theta)

        fig.add_trace(go.Scatter(
            x=circle_x,
            y=circle_y,
            mode='lines',
            name=f'Neighbor cutoff for {selected_obj_id} (r={variable_cutoff:.2f})',
            line=dict(color='green', width=2),
        ))

        # Highlight the selected object
        fig.add_trace(go.Scatter(
            x=[selected_obj_pos[0]], 
            y=[selected_obj_pos[1]],
            mode='markers+text',
            marker=dict(color='red', size=12),
            text=[f'Selected: {selected_obj_id}'],
            textposition="top center",
            showlegend=False
        ))

        # Label and highlight objects inside the variable cutoff circle
        for obj_id, obj in enumerate(octree.atoms):
            obj_pos = obj.get_position()
            dist = np.linalg.norm(np.array(obj_pos[:2]) - np.array(selected_obj_pos[:2]))

            if dist <= variable_cutoff and obj_id != selected_obj_id:
                # Highlight and label the objects within the cutoff
                fig.add_trace(go.Scatter(
                    x=[obj_pos[0]], 
                    y=[obj_pos[1]],
                    mode='markers+text',
                    marker=dict(color='yellow', size=10),
                    text=[f'{obj_id}'],
                    textposition="top center",
                    showlegend=False
                ))

        # Print the nb_list (neighbor list) for the selected object
        nb_list = [i for i, obj in enumerate(octree.atoms) if np.linalg.norm(np.array(obj.get_position()[:2]) - np.array(selected_obj_pos[:2])) <= variable_cutoff]
        print(f"Neighbor list of object {selected_obj_id} (cutoff = {variable_cutoff:.2f}): {nb_list}")

    fig.update_layout(
        title=f'All Objects with Variable Neighbor Cutoff Circles for {num_objects} Random Objects',
        xaxis_title='X-axis',
        yaxis_title='Y-axis',
        width=1000,
        height=1000,
        xaxis=dict(showgrid=True, zeroline=True),
        yaxis=dict(showgrid=True, zeroline=True),
    )

    fig.show()


neighbor_cutoff = 10  # Set the radius for neighbors

initial_boundingbox_coords = (np.array([0, 0, 0]), np.array([100, 100, 0]))
bbox_update_interval = 10

time_series_data = generate_time_series_data(num_objects=1000, num_time_steps=1)

octree = test_dynamic_octree(time_series_data, 'results_synthetic.txt', initial_boundingbox_coords, bbox_update_interval)

visualize_objects_and_variable_circles(octree, neighbor_cutoff)
