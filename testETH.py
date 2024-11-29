import numpy as np
from octree import DynamicOctree, DynamicOctreeNode, OctreeConstructionParams
from objects import Object
import time
import json
import xml.etree.ElementTree as ET

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

def test_dynamic_octree_with_Edinburgh(time_series_data, output_file):
    """
    Tests the DynamicOctree with the given time series data and writes the nb_lists for each time step to a file.
    
    Args:
        time_series_data (dict): A dictionary where the key is a timestamp and the value is a list of tuples,
                                 each tuple represents the position (x, y, z) of an object.
        output_file (str): Path to the file where results will be written.
    """
    total_time_to_update = 0.0
    total_time_to_update_nb_lists = 0.0
    min_coords = np.array([float('inf')] * 3)
    max_coords = np.array([float('-inf')] * 3)
    
    if not time_series_data:
        with open(output_file, 'w') as file:
            file.write("No data provided.\n")
        return
    
    initial_positions = time_series_data[0]
    objects = []
    
    for i, pos in initial_positions:
        obj = Object(position=pos, id=i+1)
        objects.append(obj)

    for entries in time_series_data.values():
        for id, pos in entries:
            min_coords = np.minimum(min_coords, pos)
            max_coords = np.maximum(max_coords, pos)
    
    with open(output_file, 'w') as file:
        file.write(f"Minimum coordinates: {min_coords}\n")
        file.write(f"Maximum coordinates: {max_coords}\n\n")
        
        num_atoms = len(objects)
        construction_params = OctreeConstructionParams(max_leaf_size=100, max_leaf_dim=10000, slack_factor=1.0)
        max_nodes = 10000

        # Build the octree
        start_time = time.time()
        octree = DynamicOctree(objects, num_atoms, construction_params, verbose=False, max_nodes=max_nodes)
        octree.build_octree()
        total_time_to_build = time.time() - start_time

        for timestamp, positions in time_series_data.items():
            # if timestamp == 0:
            #     file.write(f"\n============Neighbor lists and distances at time stamp: {timestamp}============\n")
            #     for atom_id in range(len(octree.atoms)):
            #         file.write(f"Atom {atom_id}: nb_list = {octree.nb_lists[atom_id]}\n")
            #         file.write(f"      : nb_list_with_dis = {octree.nb_lists_with_dist[atom_id]}\n")
            #     continue  # Skip the 0th time step as it was already used to initialize

            # file.write(f"\nTime step: {timestamp}\n")    

            start_time = time.time()
            for obj, new_pos in positions:
                object = objects[obj]
                target_atom, target_node = octree.update_octree(object, new_pos)
                nb_lists_start_time = time.time()
                octree.update_nb_lists_local(target_atom, target_node)
                nb_lists_update_time = time.time() - nb_lists_start_time
                total_time_to_update_nb_lists += nb_lists_update_time
            update_time = time.time() - start_time
            total_time_to_update += update_time
            
            # file.write(f"Num nodes: {octree.num_nodes}\n")
            
            # file.write(f"\n============Neighbor lists and distances at time stamp: {timestamp}============\n")
            # for atom_id in range(len(octree.atoms)):
            #     file.write(f"Atom {atom_id}: nb_list = {octree.nb_lists[atom_id]}\n")
            #     file.write(f"      : nb_list_with_dis = {octree.nb_lists_with_dist[atom_id]}\n")

        total_time_for_trajectory = total_time_to_build + total_time_to_update
        average_time_to_update = total_time_to_update / (len(time_series_data) - 1) if len(time_series_data) > 1 else 0
        average_time_to_update_nblists = total_time_to_update_nb_lists / (len(time_series_data) - 1) if len(time_series_data) > 1 else 0

        file.write("\n========= Final Results =========\n")
        file.write(f"Total time taken to build the octree: {total_time_to_build:.6f} seconds\n")
        file.write(f"Average time to update the octree: {average_time_to_update:.6f} seconds\n")
        file.write(f"Average time to update the nblists: {average_time_to_update_nblists:.6f} seconds\n")
        file.write(f"Total time taken to complete all the trajectories: {total_time_for_trajectory:.6f} seconds\n")

def test_dynamic_octree_with_ETH(time_series_data, initial_bbox_coords, output_file):
    """
    Tests the DynamicOctree with the given time series data and writes the nb_lists for each time step to a file.
    
    Args:
        time_series_data (dict): A dictionary where the key is a timestamp and the value is a list of tuples,
                                 each tuple represents the position (x, y, z) of an object.
        output_file (str): Path to the file where results will be written.
    """
    total_time_to_update = 0.0
    total_time_to_update_nb_lists = 0.0
    min_coords = np.array([float('inf')] * 3)
    max_coords = np.array([float('-inf')] * 3)
    
    if not time_series_data:
        with open(output_file, 'w') as file:
            file.write("No data provided.\n")
        return
    
    objects = {}
    nb_lists_dict = {}  # Dictionary to store non-empty nb_list values with their timestamps

    for entries in time_series_data.values():
        for id, pos in entries:
            min_coords = np.minimum(min_coords, pos)
            max_coords = np.maximum(max_coords, pos)
    
    with open(output_file, 'w') as file:
        file.write(f"Minimum coordinates: {min_coords}\n")
        file.write(f"Maximum coordinates: {max_coords}\n\n")
        
        construction_params = OctreeConstructionParams(max_leaf_size=2, max_leaf_dim=100, slack_factor=1.0)
        max_nodes = 200

        initial_positions = time_series_data[0]
        for i, pos in initial_positions:
            obj = Object(position=pos, id=i)
            objects[i] = obj
            
        # Build the octree
        start_time = time.time()
        octree = DynamicOctree(list(objects.values()), len(objects), construction_params, verbose=False, max_nodes=max_nodes)
        # octree.build_octree()
        octree.build_octree(*initial_bbox_coords)
        updates = 0
        n_objects = len(objects)
        n_upds, n_dels = 0, 0
        total_time_to_build = time.time() - start_time

        for timestamp, positions in time_series_data.items():
            file.write(f"\n================\nAt Timestamp: {timestamp}\n================\n")
            # print(f"\n================\nAt Timestamp: {timestamp}\n================\n")
            
            start_time = time.time()
            for id, new_pos in positions:
                updates += 1
                if id not in objects: # If the object is new, create and insert it
                    n_objects += 1
                    obj = Object(position=new_pos, id=id)
                    objects[id] = obj
                    octree.insert_object(obj)
                    file.write("\ninserting an atom\n")
                else:
                    object = objects[id]
                    n_upds += 1
                    prev_node = octree.object_to_node_map[object]
                    target_atom, target_node = octree.update_octree(object, new_pos)
                    if prev_node != target_node:
                        n_dels += 1
                        nb_lists_start_time = time.time()
                        nb_list = octree.update_nb_lists_local(target_atom, target_node)
                        nb_lists_update_time = time.time() - nb_lists_start_time
                        total_time_to_update_nb_lists += nb_lists_update_time
                        
                        file.write("\nAfter Updating position of an atom\n")
                        for i, nb in enumerate(nb_list):
                            if nb:  # Check if the list is not empty
                                file.write(f"Atom {i}: {nb}\n")
                                # print(f"Atom {i}: {nb}\n")
                            
            update_time = time.time() - start_time
            total_time_to_update += update_time
            
        total_time_for_trajectory = total_time_to_build + total_time_to_update
        average_time_to_update = total_time_to_update / (len(time_series_data) - 1) if len(time_series_data) > 1 else 0
        average_time_to_update_nblists = total_time_to_update_nb_lists / (len(time_series_data) - 1) if len(time_series_data) > 1 else 0

        file.write("\n========= Final Results =========\n")
        file.write(f"After {updates} operations, there were {n_objects} new atoms inserted, {n_upds} atoms updated in their positions, and {n_dels} atoms deleted. In total, there are now {n_objects} atoms.\n\n")
        file.write(f"Total time taken to build the octree: {total_time_to_build:.6f} seconds\n")
        file.write(f"Average time to update the octree: {average_time_to_update:.6f} seconds\n")
        file.write(f"Average time to update the nblists: {average_time_to_update_nblists:.6f} seconds\n")
        file.write(f"Total time taken to complete all the trajectories: {total_time_for_trajectory:.6f} seconds\n")
        
        # Write nb_list results to the file
        # file.write("\n========= Non-Empty nb_list Values =========\n")
        # for timestamp, nb_list in nb_lists_dict.items():
        #     file.write(f"Timestamp {timestamp}:\n")
        #     file.write(f"nb_list: {nb_list}\n\n")



file_path = 'ETHdata.xml'
parsed_data = parse_xml_file(file_path)
initial_boundingbox_coords = (np.array([-5, -5, 0]), np.array([900, 900, 0]))
# test_dynamic_octree_with_ETH(parsed_data, initial_boundingbox_coords, 'results_ETH.txt')
test_dynamic_octree_with_Edinburgh(parsed_data, 'results_Edinburg.txt')