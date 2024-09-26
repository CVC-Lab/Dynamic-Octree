import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import plotly.graph_objects as go
from PIL import Image
from octree import DynamicOctree, DynamicOctreeNode, OctreeConstructionParams
from objects import Object
from buffer_box import BufferBox
import time
from mpl_toolkits.mplot3d import Axes3D

# Load the image
image_path = '1.jpg'

# List of bounding boxes (x_min, y_min, x_max, y_max)
bboxes = [
    (97, 166, 132, 266),
    (161, 144, 195, 238),
    (180, 138, 215, 240),
    (324, 125, 361, 212),
    (448, 144, 494, 247),
    (489, 140, 546, 236),
    (563, 308, 607, 419),
    (583, 286, 622, 390)
]

# def visualize_bounding_boxes(image, bboxes):
#     fig, ax = plt.subplots()
#     ax.imshow(image)

#     # Loop through bounding boxes and add them to the plot
#     for i, bbox in enumerate(bboxes):
#         # Calculate width and height of the rectangle
#         width = bbox[2] - bbox[0]
#         height = bbox[3] - bbox[1]

#         # Create a Rectangle patch
#         rect = patches.Rectangle((bbox[0], bbox[1]), width, height, linewidth=2, edgecolor='r', facecolor='none')
#         ax.add_patch(rect)

#         # Calculate center coordinates
#         center_x = (bbox[0] + bbox[2]) / 2
#         center_y = (bbox[1] + bbox[3]) / 2
        
#         # Add center coordinates as text label
#         ax.text(center_x, center_y, f'({center_x:.1f}, {center_y:.1f})', fontsize=10, color='blue', ha='center')
#         print(center_x, center_y)

#     plt.axis('off')  # Hide axes
#     plt.savefig('bbox_with_centers.png')

def visualize_bounding_boxes(image_path, bboxes):
    # Load the image
    img = Image.open(image_path)
    img_width, img_height = img.size

    # Create a Plotly figure with the image
    fig = go.Figure()

    # Add the image as a background
    fig.add_layout_image(
        dict(
            source=img,
            x=0,
            y=img_height,
            xref="x",
            yref="y",
            sizex=img_width,
            sizey=img_height,
            sizing="stretch",
            opacity=1,
            layer="below"
        )
    )

    # Loop through bounding boxes and add them to the plot
    for bbox in bboxes:
        # Create a rectangle for the bounding box
        fig.add_trace(go.Scatter(
            x=[bbox[0], bbox[2], bbox[2], bbox[0], bbox[0]],
            y=[img_height - bbox[1], img_height - bbox[1], img_height - bbox[3], img_height - bbox[3], img_height - bbox[1]],
            mode='lines',
            line=dict(color='red', width=2),
            fill='none'
        ))

        # Calculate center coordinates
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2

        # Adjust for Plotly's Y-axis
        adjusted_center_y = img_height - center_y

        # Add a dot at the center of the bounding box
        fig.add_trace(go.Scatter(
            x=[center_x],
            y=[adjusted_center_y],
            mode='markers',
            marker=dict(color='yellow', size=10),
            showlegend=False
        ))

    fig.update_layout(
        title='Bounding Boxes with Center Dots',
        xaxis=dict(showgrid=False, zeroline=False, range=[0, img_width]),
        yaxis=dict(showgrid=False, zeroline=False, range=[0, img_height]),
        yaxis_scaleanchor="x",  # Maintain aspect ratio
        yaxis_scaleratio=1,
        width=img_width,
        height=img_height,
        showlegend=False
    )

    fig.show()

visualize_bounding_boxes(image_path, bboxes)

#======================================================================================================

def visualize_objects_and_boxes_on_image(image_path, objects, buffer_box):
    # Load the image
    img = Image.open(image_path)
    img_width, img_height = img.size

    # Set initial bounding box coordinates to image dimensions
    initial_bbox_coords = (np.array([0, 0]), np.array([img_width, img_height]))
    
    buffer_min, buffer_max = buffer_box.min_coords, buffer_box.max_coords
    
    # Create a figure with the image
    fig = go.Figure()
    fig.add_layout_image(
        dict(
            source=img,
            x=0,
            y=img_height,
            xref="x",
            yref="y",
            sizex=img_width,
            sizey=img_height,
            sizing="stretch",
            opacity=1,
            layer="below"
        )
    )

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
        title='Objects and Buffer Box',
        xaxis_title='X-axis',
        yaxis_title='Y-axis',
        width=img_width,
        height=img_height,
        xaxis=dict(showgrid=False, zeroline=False, range=[0, img_width]),
        yaxis=dict(showgrid=False, zeroline=False, range=[0, img_height]),
        yaxis_scaleanchor="x",  # Maintain aspect ratio
        yaxis_scaleratio=1,
    )

    fig.show()

# Example usage
image_path = '1.jpg'
buffer_box = BufferBox(np.array([100, 100, 0]), np.array([300, 300, 0]))
objects = [
    (0, (114.5, 264.0, 0)), 
    (1, (178.0, 289.0, 0)), 
    (2, (197.5, 291.0, 0)), 
    (3, (471.0, 284.5, 0)), 
    (4, (342.5, 311.5, 0)), 
    (5, (517.5, 292.0, 0)), 
    (6, (585.0, 116.5, 0)), 
    (7, (602.5, 142.0, 0))
]

visualize_objects_and_boxes_on_image(image_path, objects, buffer_box)


def test_dynamic_octree(data, buffer_box):
    updates = 0
    # Initialize the buffer box with initial coordinates
    min_coords, max_coords = np.array([100, 100, 0]), np.array([300, 300, 0])
    
    objects = {}
    
    # Initialize the octree with objects inside the initial buffer box
    initial_positions = data
    for i, pos in initial_positions:
        if buffer_box.contains(pos):
            obj = Object(position=pos, id=i)
            objects[i] = obj
    
    construction_params = OctreeConstructionParams(max_leaf_size=10, max_leaf_dim=100, slack_factor=1.0)
    octree = DynamicOctree(list(objects.values()), len(objects), construction_params, verbose=False, max_nodes=200)
    octree.build_octree(min_coords, max_coords)
    local_n_upds, local_n_dels, local_n_objects = 0, 0, 0
    # Update positions of objects within the buffer box (even when bbox is still)
    for id, new_pos in data:
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
                octree.delete_object(obj)
                del objects[id]
        else:
            if buffer_box.contains(new_pos):  # Insert new object inside bbox
                local_n_objects += 1
                obj = Object(position=new_pos, id=id)
                objects[id] = obj
                octree.insert_object(obj)
    
    print(octree.nb_lists_with_dist)
    # visualize_objects_in_buffer_box('1.jpg', objects, octree, buffer_box)
    

def visualize_objects_in_buffer_box(image_path, objects, octree, buffer_box):
    # Load the image
    img = Image.open(image_path)
    img_width, img_height = img.size

    fig = go.Figure()

    # Add the image as a background
    fig.add_layout_image(
        dict(
            source=img,
            x=0,
            y=img_height,
            xref="x",
            yref="y",
            sizex=img_width,
            sizey=img_height,
            sizing="stretch",
            opacity=1,
            layer="below"
        )
    )

    # Draw the buffer box
    buffer_min, buffer_max = buffer_box.min_coords, buffer_box.max_coords
    fig.add_trace(go.Scatter(
        x=[buffer_min[0], buffer_max[0], buffer_max[0], buffer_min[0], buffer_min[0]],
        # y=[img_height - buffer_min[1], img_height - buffer_min[1], img_height - buffer_max[1], img_height - buffer_max[1], img_height - buffer_min[1]],
        y=[buffer_min[1], buffer_min[1], buffer_max[1], buffer_max[1], buffer_min[1]],
        
        mode='lines',
        name='Buffer Box',
        line=dict(color='red', width=2),
        fill='toself',
        fillcolor='rgba(255, 0, 0, 0.2)',
    ))

    # Plot the objects within the buffer box
    for obj_id, obj in objects.items():
        pos = obj.get_position()
        obj_id = octree.atoms.index(obj)

        if buffer_box.contains(pos):
            nb_list_value = octree.nb_lists[obj_id]
            # Adjust the y-coordinate for Plotly
            adjusted_y = img_height - pos[1]
            fig.add_trace(go.Scatter(
                x=[pos[0]], 
                y=[adjusted_y],
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
        width=img_width,
        height=img_height,
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        yaxis_scaleanchor="x",  # Maintain aspect ratio
        yaxis_scaleratio=1,
        showlegend=False
    )

    fig.show()

octree = test_dynamic_octree(objects, buffer_box)
