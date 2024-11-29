import numpy as np
from octree import DynamicOctree, DynamicOctreeNode, OctreeConstructionParams
from objects import Object
import time
import json
import xml.etree.ElementTree as ET
from buffer_box import BufferBox
import pdb

def parse_xml_file(file_path):
    try:
        # Parse the XML file
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        result_dict = {}
        
        # Iterate over each frame in the XML
        for frame in root.findall('frame'):
            frame_number = int(frame.get('number'))
            objects = frame.find('objectlist').findall('object')
            
            # Extract object details
            frame_data = []
            for obj in objects:
                obj_id = int(obj.get('id'))  # Convert ID to integer
                box = obj.find('box')
                xc = int(float(box.get('xc')))  # Convert xc to integer
                yc = int(float(box.get('yc')))  # Convert yc to integer
                frame_data.append((obj_id, (xc, yc, 0)))
            
            # Store in dictionary
            result_dict[frame_number] = frame_data
        
        return result_dict

    except ET.ParseError as e:
        print(f"Error parsing XML: {e}")
        return None
    except FileNotFoundError:
        print("File not found.")
        return None
    except IOError:
        print("Error reading file.")
        return None

def load_time_series_data(file_path):
    """
    Loads time series data from a JSON file.
    
    Args:
        file_path (str): Path to the JSON file containing the time series data.
        
    Returns:
        dict: A dictionary where the key is a timestamp and the value is a list of tuples.
    """
    with open(file_path, 'r') as f:
        time_series_data = json.load(f)
    # Convert list of lists to tuple of tuples
    result = {}
    for timestamp, entries in time_series_data.items():
        result[int(timestamp)] = [(int(obj_id), list(map(int, coord))) for obj_id, coord in entries]
    
    return result

def test_dynamic_octree_with_ETH(time_series_data, output_file, initial_bbox_coords, bbox_update_interval):
    total_time_to_update = 0.0
    total_time_to_update_nb_lists = 0.0
    total_time_for_trajectory = 0.0
    updates = 0
    # Initialize the buffer box with initial coordinates
    min_coords, max_coords = initial_bbox_coords
    buffer_box = BufferBox(min_coords, max_coords)
    
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
                new_min_coords[:2] += np.random.randint(0, 100, size=2)  # Update x, y for min_coords
                new_max_coords[:2] += np.random.randint(0, 100, size=2)  # Update x, y for max_coords

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

        # Compute averages over all bbox updates
        average_time_to_update = total_time_to_update / (updates if updates > 0 else 1)

        # Write the final results after processing all timestamps
        file.write("\n========= Final Results =========\n")
        file.write(f"After {updates} operations, there were {n_objects_total} new atoms inserted, {n_upds_total} atoms updated in their positions, and {n_dels_total} atoms deleted.\n\n")
        file.write(f"Total time taken to build the octree: {total_time_to_build:.6f} seconds\n")
        file.write(f"Average time to update the octree: {average_time_to_update:.6f} seconds\n")
        file.write(f"Total time taken to complete all the trajectories: {total_time_for_trajectory:.6f} seconds\n")


file_path = 'ETHdata.xml'
parsed_data = parse_xml_file(file_path)

initial_bbox_coords = (np.array([0, 0, 0]), np.array([500, 500, 0]))
# Define the interval at which the buffer box updates (every 10 time steps in this case)
bbox_update_interval = 10

test_dynamic_octree_with_ETH(parsed_data, 'results_ETH_with_bbox.txt', initial_bbox_coords, bbox_update_interval)
