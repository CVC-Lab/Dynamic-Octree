import numpy as np
from octree import DynamicOctree, DynamicOctreeNode, OctreeConstructionParams
from objects import Object
from buffer_box import BufferBox
import time
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D
import pdb

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

def visualize_moving_buffer_box(octrees, objects, buffer_box, initial_bbox_coords, num_time_steps, buffer_box_path):
    """
    Visualize the buffer box moving along a defined path and highlight objects within it.

    Args:
        objects (dict): Dictionary containing object ids and their positions.
        buffer_box (BufferBox): The initial buffer box.
        initial_bbox_coords (tuple): The initial bounding box coordinates.
        num_time_steps (int): The number of time steps in the simulation.
        buffer_box_path (list): List of (min_coords, max_coords) for each time step defining the movement of the buffer box.
    """
    min_bbox, max_bbox = initial_bbox_coords
    
    # Create a figure
    fig = go.Figure()

    # Loop over time steps to create frames for animation
    frames = []
    for t in range(num_time_steps):
        # print(t)
        min_coords, max_coords = buffer_box_path[t]  # Get buffer box position at time step t
        buffer_box.update_box(octrees[t], min_coords, max_coords)  # Update the buffer box coordinates
        
        # Create scatter traces for the objects and buffer box at the current time step
        frame_data = []
        
        # Draw the entire bounding box (static across time)
        frame_data.append(go.Scatter(
            x=[min_bbox[0], max_bbox[0], max_bbox[0], min_bbox[0], min_bbox[0]],
            y=[min_bbox[1], min_bbox[1], max_bbox[1], max_bbox[1], min_bbox[1]],
            mode='lines',
            name='Bounding Box',
            line=dict(color='black', width=2),
            fill='toself',
            fillcolor='rgba(0, 0, 0, 0.1)',
        ))

        # Draw the buffer box (moving)
        frame_data.append(go.Scatter(
            x=[min_coords[0], max_coords[0], max_coords[0], min_coords[0], min_coords[0]],
            y=[min_coords[1], min_coords[1], max_coords[1], max_coords[1], min_coords[1]],
            mode='lines',
            name=f'Buffer Box (Time {t})',
            line=dict(color='red', width=2),
            fill='toself',
            fillcolor='rgba(255, 0, 0, 0.2)',
        ))

        # Plot the objects, highlighting those inside the buffer box
        for obj_id, obj in objects.items():
            # print(obj)
            pos = obj#.get_position()
            color = 'blue' if buffer_box.contains(pos) else 'orange'
            frame_data.append(go.Scatter(
                x=[pos[0]], 
                y=[pos[1]],
                mode='markers+text',
                marker=dict(color=color, size=8),
                text=[str(obj_id)],
                textposition="top center",
                showlegend=False
            ))

        for node in octrees[t].nodes:
            if node is not None:
                node_min, node_max = node.get_node_bounds()
                # print(octree.nodes.index(node))
                # Draw the bounding box for each node
                fig.add_trace(go.Scatter(
                    x=[node_min[0], node_max[0], node_max[0], node_min[0], node_min[0]],
                    y=[node_min[1], node_min[1], node_max[1], node_max[1], node_min[1]],
                    mode='lines',
                    name=f'Node {node}',
                    line=dict(color='green', width=2, dash='dash'),
                    showlegend=False
                ))

        # Append this frame's data to the frames list
        frames.append(go.Frame(data=frame_data, name=str(t)))
    
    # Add initial traces to the figure (from the first time step)
    fig.add_trace(frames[0].data[0])  # Static bounding box
    fig.add_trace(frames[0].data[1])  # Initial buffer box
    for scatter_trace in frames[0].data[2:]:
        fig.add_trace(scatter_trace)  # Initial objects

    # Set up layout for animation
    fig.update_layout(
        title='Moving Buffer Box with Atoms',
        xaxis_title='X-axis',
        yaxis_title='Y-axis',
        width=800,
        height=800,
        xaxis=dict(showgrid=True, zeroline=True),
        yaxis=dict(showgrid=True, zeroline=True),
        updatemenus=[{
            'type': 'buttons',
            'buttons': [{
                'label': 'Play',
                'method': 'animate',
                'args': [None, {
                    'frame': {'duration': 5000, 'redraw': True},
                    'fromcurrent': True
                }]
            }, {
                'label': 'Pause',
                'method': 'animate',
                'args': [[None], {'frame': {'duration': 0}, 'mode': 'immediate'}]
            }]
        }]
    )

    # Add frames to the figure
    fig.frames = frames

    fig.show()


# def visualize_moving_buffer_box_with_nodes(objects, buffer_box, initial_bbox_coords, num_time_steps, buffer_box_path, octrees):
#     """
#     Visualize the buffer box moving along a defined path, highlight objects within it, 
#     and visualize the octree nodes at each timestep.

#     Args:
#         objects (dict): Dictionary containing object ids and their positions.
#         buffer_box (BufferBox): The initial buffer box.
#         initial_bbox_coords (tuple): The initial bounding box coordinates.
#         num_time_steps (int): The number of time steps in the simulation.
#         buffer_box_path (list): List of (min_coords, max_coords) for each time step defining the movement of the buffer box.
#         octree (DynamicOctree): The octree containing nodes and objects.
#     """
#     min_bbox, max_bbox = initial_bbox_coords
    
#     # Create a figure
#     fig = go.Figure()

#     # Loop over time steps to create frames for animation
#     frames = []
#     for t in range(num_time_steps):
#         min_coords, max_coords = buffer_box_path[t]  # Get buffer box position at time step t
#         buffer_box.update_box(octrees, min_coords, max_coords)  # Update the buffer box coordinates
        
#         # Create scatter traces for the objects, buffer box, and octree nodes at the current time step
#         frame_data = []
        
#         # Draw the entire bounding box (static across time)
#         frame_data.append(go.Scatter(
#             x=[min_bbox[0], max_bbox[0], max_bbox[0], min_bbox[0], min_bbox[0]],
#             y=[min_bbox[1], min_bbox[1], max_bbox[1], max_bbox[1], min_bbox[1]],
#             mode='lines',
#             name='Bounding Box',
#             line=dict(color='black', width=2),
#             fill='toself',
#             fillcolor='rgba(0, 0, 0, 0.1)',
#         ))

#         # Draw the buffer box (moving)
#         frame_data.append(go.Scatter(
#             x=[min_coords[0], max_coords[0], max_coords[0], min_coords[0], min_coords[0]],
#             y=[min_coords[1], min_coords[1], max_coords[1], max_coords[1], min_coords[1]],
#             mode='lines',
#             name=f'Buffer Box (Time {t})',
#             line=dict(color='red', width=2),
#             fill='toself',
#             fillcolor='rgba(255, 0, 0, 0.2)',
#         ))

#         # Draw octree nodes
#         for node in octrees.nodes:
#         # for node in octrees[t].nodes:
#             node_min, node_max = node.get_node_bounds()
            
#             frame_data.append(go.Scatter(
#                 x=[node_min[0], node_max[0], node_max[0], node_min[0], node_min[0]],
#                 y=[node_min[1], node_min[1], node_max[1], node_max[1], node_min[1]],
#                 mode='lines',
#                 name=f'Node Boundary (Time {t})',
#                 line=dict(color='green', width=2, dash='dash'),
#                 showlegend=False,
#             ))

#         # Plot the objects, highlighting those inside the buffer box
#         for obj_id, pos in objects.items():
#             color = 'blue' if buffer_box.contains(pos) else 'orange'
#             frame_data.append(go.Scatter(
#                 x=[pos[0]], 
#                 y=[pos[1]],
#                 mode='markers+text',
#                 marker=dict(color=color, size=8),
#                 text=[str(obj_id)],
#                 textposition="top center",
#                 showlegend=False
#             ))

#         # Append this frame's data to the frames list
#         frames.append(go.Frame(data=frame_data, name=str(t)))
    
#     # Add initial traces to the figure (from the first time step)
#     fig.add_trace(frames[0].data[0])  # Static bounding box
#     fig.add_trace(frames[0].data[1])  # Initial buffer box
#     for scatter_trace in frames[0].data[2:]:
#         fig.add_trace(scatter_trace)  # Initial objects and node boundaries

#     # Set up layout for animation
#     fig.update_layout(
#         title='Moving Buffer Box with Octree Nodes and Atoms',
#         xaxis_title='X-axis',
#         yaxis_title='Y-axis',
#         width=800,
#         height=800,
#         xaxis=dict(showgrid=True, zeroline=True),
#         yaxis=dict(showgrid=True, zeroline=True),
#         updatemenus=[{
#             'type': 'buttons',
#             'buttons': [{
#                 'label': 'Play',
#                 'method': 'animate',
#                 'args': [None, {
#                     'frame': {'duration': 1000, 'redraw': True},
#                     'fromcurrent': True
#                 }]
#             }, {
#                 'label': 'Pause',
#                 'method': 'animate',
#                 'args': [[None], {'frame': {'duration': 0}, 'mode': 'immediate'}]
#             }]
#         }]
#     )

#     # Add frames to the figure
#     fig.frames = frames
#     fig.show()


def test_dynamic_octree(objects_initial, time_series_data, output_file, initial_bbox_coords, bbox_update_interval, path_points, buffer_box):
    total_time_to_update = 0.0
    total_time_for_trajectory = 0.0
    updates = 0
    min_coords, max_coords = initial_bbox_coords
    
    objects = {}
    octrees = []
    
    # Path-related variables
    path_index = 1
    num_path_points = len(path_points)

    # Initialize a buffer box path for visualization
    buffer_box_path = [(initial_bbox_coords[0], initial_bbox_coords[1])]
    
    with open(output_file, 'w') as file:
        initial_positions = time_series_data[0]
        # Initialize objects inside the buffer box
        for i, pos in initial_positions:
            if buffer_box.contains(pos):
                obj = Object(position=pos, id=i)
                objects[i] = obj
        
        construction_params = OctreeConstructionParams(max_leaf_size=5, max_leaf_dim=100, slack_factor=1.0)
        octree = DynamicOctree(list(objects.values()), len(objects), construction_params, verbose=False, max_nodes=200)
        
        start_time = time.time()
        octree.build_octree(min_coords, max_coords)
        total_time_to_build = time.time() - start_time
        
        file.write(f"\n=========================Statistics for TIMESTAMP: 0=========================\n")
        file.write(f"\nThere are {octree.num_atoms} atoms in the buffer box:\n")
        for i in range(len(octree.atoms)):
            file.write(f"Atom {i}: {octree.atoms[i].id} with coordinates: {octree.atoms[i].x, octree.atoms[i].y, octree.atoms[i].z}\n")
        
        for i, nb in enumerate(octree.nb_lists):
            if nb:
                file.write(f"Atom {i}: {nb}\n")

        # Tracking updates and deletions
        n_objects_total = 0
        n_upds_total = 0
        n_dels_total = 0
        octrees = []
        octrees.append(octree)

        # visualize_moving_buffer_box(octree, objects, buffer_box, initial_bbox_coords, num_time_steps, buffer_box_path)

        for timestamp, positions in time_series_data.items():
            octree.reset_nb_lists()
            bbox_updated = False

            if timestamp == 0:
                continue
            
            file.write(f"\n=========================Statistics for TIMESTAMP: {timestamp}=========================\n")
            print(f"\n=========================Statistics for TIMESTAMP: {timestamp}=========================\n")
            
            # Update the buffer box along the path
            if timestamp % bbox_update_interval == 0 and path_index < num_path_points:
                # print(path_points[path_index])
                new_min_coords = path_points[path_index][0]
                new_max_coords = path_points[path_index][1]  # assuming pairs of points
                buffer_box.update_box(octree, new_min_coords, new_max_coords)
                path_index = path_index + 1
                bbox_updated = True

                # Add the updated buffer box path for visualization
                buffer_box_path.append((new_min_coords, new_max_coords))
                
                # Remove objects outside the buffer box
                objects_to_remove = [id for id, obj in objects.items() if not buffer_box.contains(obj.get_position())]
                file.write(f"\nObjects outside buffer box: {objects_to_remove}\n")
                for obj_id in objects_to_remove:
                    octree.delete_object(objects[obj_id])
                    del objects[obj_id]
                # octree.contract_empty_leaf_nodes()
                total_time_to_update += time.time() - start_time
                

            # Process object positions at each timestamp
            start_time = time.time()
            local_n_upds, local_n_dels, local_n_objects = 0, 0, 0
            for id, new_pos in positions:
                updates += 1
                
                if id in objects:  # Update existing object
                    obj = objects[id]
                    if buffer_box.contains(new_pos):
                        local_n_upds += 1
                        prev_node = octree.object_to_node_map[obj]
                        target_atom, target_node = octree.update_octree(obj, new_pos)
                        if prev_node != target_node:
                            local_n_dels += 1
                    else:
                        file.write(f'Object {id} moved outside buffer box\n')
                        octree.delete_object(obj)
                        del objects[id]
                else:  # Insert new object
                    if buffer_box.contains(new_pos):
                        local_n_objects += 1
                        obj = Object(position=new_pos, id=id)
                        objects[id] = obj
                        octree.insert_object(obj)
            
            octrees.append(octree)

            visualize_moving_buffer_box(octrees, objects_initial, buffer_box, initial_bbox_coords, len(buffer_box_path), buffer_box_path)

            n_objects_total += local_n_objects
            n_upds_total += local_n_upds
            n_dels_total += local_n_dels
            total_time_for_trajectory += time.time() - start_time
            # octree.print_all_atoms_in_nodes()
            # Write atom info
            file.write(f"\nThere are {octree.num_atoms} atoms in the buffer box:\n")
            for i in range(len(octree.atoms)):
                if octree.atoms[i] is not None:
                    file.write(f"Atom {i}: {octree.atoms[i].id} with coordinates: {octree.atoms[i].x, octree.atoms[i].y, octree.atoms[i].z}\n")
            
            nb_list = octree.nb_lists
            file.write("\nNeighbourhood lists of the above atoms:\n")
            for i, nb in enumerate(nb_list):
                if nb:
                    file.write(f"Atom {i}: {nb}\n")

            if bbox_updated:
                file.write("\n========= Results for BBox Update =========\n")
            else:
                file.write("\n========= Results for BBox Without Update =========\n")
            
            file.write(f"At timestamp {timestamp}, bbox update: {bbox_updated}\n")
            file.write(f"{local_n_objects} new atoms inserted, {local_n_upds} atoms updated, {local_n_dels} atoms deleted.\n")
            file.write(f"Total objects now inside the buffer box: {len(objects)}\n")
            file.write(f"Total time to update octree: {total_time_to_update:.6f} seconds\n")
            file.write("\n")

        # Call visualization at the end of the loop (or during updates if needed)
        # visualize_moving_buffer_box_with_nodes(objects_initial, buffer_box, initial_bbox_coords, len(buffer_box_path), buffer_box_path, octree)

        # average_time_to_update = total_time_to_update / (updates if updates > 0 else 1)
        return octree



# Example initializations
initial_bbox_coords = (np.array([80, 80, 0]), np.array([100, 100, 0]))  # Initial bounding box coordinates
num_time_steps = 2

# Generate a synthetic path for the buffer box
buffer_box_path = []
start_x, start_y = 80, 80  # Starting position of the buffer box
end_x, end_y = 0, 20       # Ending position of the buffer box

for t in range(num_time_steps):
    current_x = start_x + (end_x - start_x) * (t / (num_time_steps - 1))
    current_y = start_y - (start_y - end_y) * (t / (num_time_steps - 1))
    
    min_coords = list(np.array([current_x, current_y, 0]))  # Min coordinates of the buffer box
    max_coords = list(np.array([current_x + 20, current_y + 20, 0]))  # Max coordinates of the buffer box
    buffer_box_path.append((min_coords, max_coords))

# buffer_box_path = list(buffer_box_path)

objects = {i: (np.random.uniform(0, 100), np.random.uniform(0, 100), 0) for i in range(50)}

# Example time series data (dictionary with timestamps as keys and object positions as values)
time_series_data = {t: [(i, list(np.array(objects[i]))) for i in objects] for t in range(num_time_steps)}

# Initialize the BufferBox
buffer_box = BufferBox(np.array([80, 80, 0]), np.array([100, 100, 0]))

# Output file to write results to
output_file = "bbox_simulation_output.txt"

# Call the function
octree = test_dynamic_octree(
    objects,
    time_series_data=time_series_data,
    output_file=output_file,
    initial_bbox_coords=initial_bbox_coords,
    bbox_update_interval=1,  # Assuming update happens every timestamp
    path_points=buffer_box_path,  # The path for the buffer box to follow
    buffer_box=buffer_box
)

# visualize_moving_buffer_box(objects, buffer_box, initial_bbox_coords, num_time_steps, buffer_box_path)
# visualize_moving_buffer_box_with_nodes(objects, buffer_box, initial_bbox_coords, num_time_steps, buffer_box_path, octree)
