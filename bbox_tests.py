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

def visualize_moving_buffer_box(objects, buffer_box, initial_bbox_coords, num_time_steps, buffer_box_path):
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
        min_coords, max_coords = buffer_box_path[t]  # Get buffer box position at time step t
        buffer_box.update_box(None, min_coords, max_coords)  # Update the buffer box coordinates
        
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
        for obj_id, pos in objects.items():
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
                    'frame': {'duration': 50, 'redraw': True},
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




# Example usage
initial_bbox_coords = (np.array([80, 80, 0]), np.array([100, 100, 0]))  # Example initial coordinates
num_time_steps = 16

# Generate a synthetic path for the buffer box to follow diagonally
buffer_box_path = []
start_x, start_y = 80, 80  # Starting coordinates (top left)
end_x, end_y = 20, 0       # Ending coordinates (bottom right)

for t in range(num_time_steps):
    # Calculate the current position based on linear interpolation
    current_x = start_x + (end_x - start_x) * (t / (num_time_steps - 1))
    current_y = start_y - (start_y - end_y) * (t / (num_time_steps - 1))
    
    min_coords = np.array([current_x, current_y, 0])        # Min coordinates of the buffer box
    max_coords = np.array([current_x + 20, current_y + 20, 0])  # Max coordinates of the buffer box
    buffer_box_path.append((min_coords, max_coords))


# Example of object positions (you can generate your own)
objects = {i: (np.random.uniform(0, 100), np.random.uniform(0, 100), 0) for i in range(200)}

buffer_box = BufferBox(np.array([0, 0, 0]), np.array([20, 20, 0]))
visualize_moving_buffer_box(objects, buffer_box, initial_bbox_coords, num_time_steps, buffer_box_path)

