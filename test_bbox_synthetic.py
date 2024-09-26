import numpy as np
from octree import DynamicOctree, DynamicOctreeNode, OctreeConstructionParams
from objects import Object
from buffer_box import BufferBox
import time
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D

def generate_movement_function(initial_position, time_step):
    """
    Generate a new position based on a linear decrease function.
    
    Args:
        initial_position (tuple): The starting position (x, y, z).
        time_step (int): The current time step.
    
    Returns:
        tuple: The new position (x, y, 0) after applying the movement function.
    """
    decrease_rate = 2  # The rate at which positions decrease
    new_x = initial_position[0] + decrease_rate * time_step
    new_y = initial_position[1] + decrease_rate * time_step
    return (new_x, new_y, 0)  # Set z to 0

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
    
    # Initialize positions for objects, setting z to 0
    initial_positions = [(np.random.uniform(0.0, 50.0), np.random.uniform(0.0, 50.0), 0) for _ in range(num_objects)]
    
    for t in range(num_time_steps):
        positions_at_t = [(obj_id, list(map(int, generate_movement_function(pos, t)))) for obj_id, pos in enumerate(initial_positions)]
        time_series_data[t] = positions_at_t
    
    return time_series_data

#========================================================================================================

def test_dynamic_octree(time_series_data, output_file, initial_bbox_coords, bbox_update_interval, buffer_box):
    total_time_to_update = 0.0
    total_time_to_update_nb_lists = 0.0
    total_time_for_trajectory = 0.0
    updates = 0
    # Initialize the buffer box with initial coordinates
    min_coords, max_coords = initial_bbox_coords
    # buffer_box = BufferBox(min_coords, max_coords)
    
    objects = {}
    
    with open(output_file, 'w') as file:
        file.write(f"Initial Buffer Box: {min_coords} to {max_coords}\n\n")

        # Initialize the octree with objects inside the initial buffer box
        initial_positions = time_series_data[0]
        for i, pos in initial_positions:
            if buffer_box.contains(pos):
                obj = Object(position=pos, id=i)
                objects[i] = obj
        
        construction_params = OctreeConstructionParams(max_leaf_size=10, max_leaf_dim=100, slack_factor=1.0)
        octree = DynamicOctree(list(objects.values()), len(objects), construction_params, verbose=False, max_nodes=200)
        start_time = time.time()
        octree.build_octree(min_coords, max_coords)
        total_time_to_build = time.time() - start_time
        
        n_objects = len(objects)
        n_upds, n_dels = 0, 0
        n_objects_total = 0
        n_upds_total = 0
        n_dels_total = 0

        for timestamp, positions in time_series_data.items():
            # Flag to check if bbox was updated in this iteration
            bbox_updated = False

            if timestamp == 0:
                continue
            # Update the buffer box every `bbox_update_interval`
            if timestamp % bbox_update_interval == 0:
                # Log the bbox update
                file.write(f"\nUpdated Buffer Box at timestamp {timestamp}: {buffer_box.min_coords} to {buffer_box.max_coords}\n")
                file.write(f"Updated Root Bounding Box of DO: {octree.nodes[octree.root_node_id].get_lx()}, {octree.nodes[octree.root_node_id].get_ly()}, {octree.nodes[octree.root_node_id].get_lz()} and dimensions: {octree.nodes[octree.root_node_id].get_dim()}\n")
                start_time = time.time()

                # Update buffer box coordinates (example logic)
                # new_min_coords = buffer_box.min_coords + np.random.randint(0, 100, size=3)
                # new_max_coords = buffer_box.max_coords + np.random.randint(0, 100, size=3)

                # Update only the x and y coordinates while leaving the z coordinate unchanged
                new_min_coords = buffer_box.min_coords.copy()
                new_max_coords = buffer_box.max_coords.copy()

                # Randomly update x and y coordinates (indices 0 and 1) but leave z (index 2) unchanged
                # new_min_coords[:2] += np.random.randint(0, 10, size=2)  # Update x, y for min_coords
                # new_max_coords[:2] += np.random.randint(0, 10, size=2)  # Update x, y for max_coords

                buffer_box.update_box(octree, new_min_coords, new_max_coords)
                bbox_updated = True
                
                # Remove objects outside the buffer box
                objects_to_remove = [id for id, obj in objects.items() if not buffer_box.contains(obj.get_position())]
                file.write(f"\n{objects_to_remove}\n")
                for obj_id in objects_to_remove:
                    if objects[obj_id] in octree.atoms:
                        # pdb.set_trace()
                        file.write(f"Deleting Object {obj_id}\n")
                        octree.delete_object(objects[obj_id])
                        del objects[obj_id]
                # pdb.set_trace()
                # Calculate time for this bbox update
                total_time_to_update += time.time() - start_time

            start_time = time.time()
            local_n_upds, local_n_dels, local_n_objects = 0, 0, 0
            # Update positions of objects within the buffer box (even when bbox is still)
            for id, new_pos in positions:
                file.write(f"New position of object {id}: {new_pos}\n")
                updates += 1
                
                if id in objects:  # Update existing object
                    obj = objects[id]
                    if buffer_box.contains(new_pos):
                        local_n_upds += 1
                        prev_node = octree.object_to_node_map[obj]
                        target_atom, target_node = octree.update_octree(obj, new_pos)
                        if prev_node != target_node:
                            local_n_dels += 1
                            nb_list = octree.update_nb_lists_local(target_atom, target_node)
                    else:
                        file.write(f'Object {id} went outside the buffer box\n')
                        octree.delete_object(obj)
                        del objects[id]
                else:
                    if buffer_box.contains(new_pos):  # Insert new object inside bbox
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

            # Write the stats even when bbox is still
            if bbox_updated:
                file.write("\n========= Results for BBox Update =========\n")
            else:
                file.write("\n========= Results for BBox Without Update =========\n")
            
            file.write(f"At timestamp {timestamp}, bbox update: {bbox_updated}\n")
            file.write(f"{local_n_objects} new atoms inserted, {local_n_upds} atoms updated, {local_n_dels} atoms deleted.\n")
            file.write(f"Total objects now inside the buffer box: {len(objects)}\n")
            file.write(f"Total time to update octree: {total_time_to_update:.6f} seconds\n")
            file.write("\n")
            break

        # Compute averages over all bbox updates
        average_time_to_update = total_time_to_update / (updates if updates > 0 else 1)

        # Write the final results after processing all timestamps
        file.write("\n========= Final Results =========\n")
        file.write(f"After {updates} operations, there were {n_objects_total} new atoms inserted, {n_upds_total} atoms updated in their positions, and {n_dels_total} atoms deleted.\n\n")
        file.write(f"Total time taken to build the octree: {total_time_to_build:.6f} seconds\n")
        file.write(f"Average time to update the octree: {average_time_to_update:.6f} seconds\n")
        file.write(f"Total time taken to complete all the trajectories: {total_time_for_trajectory:.6f} seconds\n")

        # visualize_objects_in_buffer_box(octree.atoms, octree, buffer_box)
        return octree

def visualize_objects_and_boxes(objects, buffer_box, initial_bbox_coords):
    min_bbox, max_bbox = initial_bbox_coords
    buffer_min, buffer_max = buffer_box.min_coords, buffer_box.max_coords
    
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

    # Draw the buffer box
    fig.add_trace(go.Scatter(
        x=[buffer_min[0], buffer_max[0], buffer_max[0], buffer_min[0], buffer_min[0]],
        y=[buffer_min[1], buffer_min[1], buffer_max[1], buffer_max[1], buffer_min[1]],
        mode='lines',
        name='Buffer Box',
        line=dict(color='red', width=2),
        fill='toself',
        fillcolor='rgba(255, 0, 0, 0.2)',
    ))

    # Plot the objects
    for obj_id, pos in objects:
        color = 'blue' if buffer_box.contains(pos) else 'orange'
        fig.add_trace(go.Scatter(
            x=[pos[0]], 
            y=[pos[1]],
            mode='markers+text',
            marker=dict(color=color, size=8),
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

def visualize_objects_in_buffer_box(objects, octree, buffer_box):
    fig = go.Figure()
    
    # Draw the buffer box
    buffer_min, buffer_max = buffer_box.min_coords, buffer_box.max_coords
    fig.add_trace(go.Scatter(
        x=[buffer_min[0], buffer_max[0], buffer_max[0], buffer_min[0], buffer_min[0]],
        y=[buffer_min[1], buffer_min[1], buffer_max[1], buffer_max[1], buffer_min[1]],
        mode='lines',
        name='Buffer Box',
        line=dict(color='red', width=2),
        fill='toself',
        fillcolor='rgba(255, 0, 0, 0.2)',
    ))

    # Plot the objects within the buffer box
    for obj in objects:
        pos = obj.get_position()
        obj_id = octree.atoms.index(obj)
        if buffer_box.contains(pos):
            nb_list_value = octree.nb_lists[obj_id]
            fig.add_trace(go.Scatter(
                x=[pos[0]], 
                y=[pos[1]],
                mode='markers+text',
                marker=dict(color='blue', size=8),
                text=[f'{nb_list_value}'],
                textposition="top left",
                showlegend=False,
            ))

    fig.update_layout(
        title='Objects in Buffer Box with nb_lists',
        xaxis_title='X-axis',
        yaxis_title='Y-axis',
        width=800,
        height=800,
        xaxis=dict(showgrid=True, zeroline=True),
        yaxis=dict(showgrid=True, zeroline=True),
    )
    
    fig.show()

#========================================================================================================

initial_boundingbox_coords = (np.array([0, 0, 0]), np.array([50, 50, 0]))
initial_bbox_coords = (np.array([0, 0, 0]), np.array([50, 50, 0]))
bbox_update_interval = 10
buffer_box = BufferBox(np.array([0, 0, 0]), np.array([50, 50, 0]))
time_series_data = generate_time_series_data(num_objects=1000, num_time_steps=1)
visualize_objects_and_boxes(time_series_data[0], buffer_box, initial_boundingbox_coords)
octree = test_dynamic_octree(time_series_data, 'results_synthetic.txt', initial_bbox_coords, bbox_update_interval, buffer_box)

# visualize_objects_in_buffer_box(objects, octree, buffer_box)