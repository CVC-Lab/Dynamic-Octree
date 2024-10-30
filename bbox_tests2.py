import numpy as np
from objects import Object
from buffer_box import BufferBox
import time
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D

def generate_movement_function(initial_position, time_step):
    """
    Generate a new position based on a random curved movement function.
    
    Args:
        initial_position (tuple): The starting position (x, y, z).
        time_step (int): The current time step.
    
    Returns:
        tuple: The new position (x, y, 0) after applying the movement function.
    """
    # Random parameters for the curve
    amplitude_x = np.random.uniform(5, 35)  # Random amplitude for x movement
    amplitude_y = np.random.uniform(5, 35)  # Random amplitude for y movement
    frequency = np.random.uniform(0.5, 0.5)  # Random frequency
    
    # Curved motion using sine and cosine
    new_x = initial_position[0] + amplitude_x * np.cos(frequency * time_step)
    new_y = initial_position[1] + amplitude_y * np.sin(frequency * time_step)
    
    return (new_x, new_y, 0)  # Set z to 0



def generate_time_series_data(num_objects, num_time_steps, lead_object_id, moving_fraction=0.5):
    """
    Generate time series data for a given number of objects and time steps.
    
    Args:
        num_objects (int): Number of objects.
        num_time_steps (int): Number of time steps.
        moving_fraction (float): Fraction of objects that should move.
    
    Returns:
        dict: A dictionary with object ids as keys and lists of positions as values.
    """
    time_series_data = {}

    # Initialize positions for objects, setting z to 0
    initial_positions = [(np.random.uniform(0.0, 100.0), np.random.uniform(0.0, 100.0), 0) for _ in range(num_objects)]
    
    # Mark a fraction of the objects to move
    num_moving_objects = int(moving_fraction * num_objects)
    moving_object_ids = np.random.choice(range(num_objects), size=num_moving_objects, replace=False)
    moving_object_ids = list(moving_object_ids)

    if lead_object_id not in moving_object_ids:
        moving_object_ids.append(0)

    for obj_id in range(num_objects):
        positions = []
        for t in range(num_time_steps):
            if obj_id in moving_object_ids:
                # Moving objects follow the movement function
                positions.append(list(map(int, generate_movement_function(initial_positions[obj_id], t))))
            else:
                # Non-moving objects stay in their initial positions
                positions.append(list(map(int, initial_positions[obj_id])))
        time_series_data[obj_id] = positions
    
    return time_series_data

def visualize_moving_buffer_box(time_series_data, buffer_box, lead_object_id, buffer_size, num_time_steps):
    """
    Visualize the buffer box following a lead object as both move, and highlight objects within it.

    Args:
        time_series_data (dict): Dictionary containing object ids and their positions over time.
        buffer_box (BufferBox): The initial buffer box.
        lead_object_id (int): ID of the object that the buffer box should follow.
        buffer_size (float): Size of the buffer box.
        num_time_steps (int): The number of time steps in the simulation.
    """
    # Create a figure
    fig = go.Figure()

    # Initialize a list to store lead object path
    lead_object_path_x = []
    lead_object_path_y = []

    # Circle parameters
    circle_radius = 5
    theta = np.linspace(0, 2 * np.pi, 100)  # 100 points for a smooth circle

    # Loop over time steps to create frames for animation
    frames = []
    for t in range(num_time_steps):
        lead_object_pos = time_series_data[lead_object_id][t]  # Get the lead object's position at time t
        
        # Append current lead object position to path lists
        lead_object_path_x.append(lead_object_pos[0])
        lead_object_path_y.append(lead_object_pos[1])

        # Update the buffer box to center around the lead object
        min_coords = np.array([lead_object_pos[0] - buffer_size/2, lead_object_pos[1] - buffer_size/2, 0])
        max_coords = np.array([lead_object_pos[0] + buffer_size/2, lead_object_pos[1] + buffer_size/2, 0])
        buffer_box.update_box(None, min_coords, max_coords)  # Update buffer box position
        
        # Create scatter traces for the objects and buffer box at the current time step
        frame_data = []
        
        # Draw the buffer box (moving with the lead object)
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
        for obj_id, pos_at_t in time_series_data.items():
            pos = pos_at_t[t]
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

            # Add the path of the lead object only up to the current time step
            if t > 0:  # Only add a path if t > 0
                frame_data.append(go.Scatter(
                    x=lead_object_path_x,
                    y=lead_object_path_y,
                    mode='lines',
                    name='Lead Object Path',
                    line=dict(color='green', width=2, shape='spline'),
                    marker=dict(size=5),
                    showlegend=False
                ))

            # Add a circle around the lead object
            circle_x = lead_object_pos[0] + circle_radius * np.cos(theta)
            circle_y = lead_object_pos[1] + circle_radius * np.sin(theta)
            
            frame_data.append(go.Scatter(
                x=circle_x,
                y=circle_y,
                mode='lines',
                name='Lead Object Circle',
                line=dict(color='purple', width=1),
                showlegend=False
            ))

        # Append this frame's data to the frames list
        frames.append(go.Frame(data=frame_data, name=str(t)))
    
    # Add initial traces to the figure (from the first time step)
    fig.add_trace(frames[0].data[0])  # Initial buffer box
    for scatter_trace in frames[0].data[1:]:
        fig.add_trace(scatter_trace)  # Initial objects

    # Set up layout for animation
    fig.update_layout(
        title='Buffer Box Following a Moving Object',
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
                    'frame': {'duration': 100, 'redraw': True},
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
num_time_steps = 16
buffer_size = 20

# Choose a lead object (e.g., object with id 0)
lead_object_id = 0

# Generate time series data with moving objects
time_series_data = generate_time_series_data(num_objects=200, num_time_steps=num_time_steps, lead_object_id=lead_object_id, moving_fraction=0.3)

# Create a buffer box object
initial_pos = time_series_data[lead_object_id][0]
buffer_box = BufferBox(np.array([initial_pos[0] - buffer_size/2, initial_pos[1] - buffer_size/2, 0]), 
                       np.array([initial_pos[0] + buffer_size/2, initial_pos[1] + buffer_size/2, 0]))

# Visualize the buffer box following the lead object
visualize_moving_buffer_box(time_series_data, buffer_box, lead_object_id, buffer_size, num_time_steps)
