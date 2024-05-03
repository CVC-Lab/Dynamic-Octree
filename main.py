from Octree import DynamicOctree, DynamicOctreeNode, DynamicOctreeNodeAttr, OctreeConstructionParams
import random
from objects import Object
import sys

def generate_random_objects(num_objects):
    objects = []
    for _ in range(num_objects):
        position = [random.uniform(0, 100), random.uniform(0, 100), random.uniform(0, 100)]
        obj = Object(position)
        objects.append(obj)
    return objects

def test_dynamic_octree_node():
    # Generate random objects
    num_objects = 4
    objects = generate_random_objects(num_objects)

    # Initialize a DynamicOctreeNode
    node = DynamicOctreeNode()

    # Test node initialization
    assert node.lx == 0.0
    assert node.ly == 0.0
    assert node.lz == 0.0
    assert node.dim == 0.0
    assert node.num_atoms == 0
    assert node.n_fixed == 0
    assert node.id_cap == 5
    assert node.id_num == 0
    assert node.atom_indices == []
    assert node.parent_pointer == -1
    assert node.child_pointer == [-1] * 8
    assert node.leaf is True

    # Test combining and setting attributes
    all_child_attributes = [DynamicOctreeNodeAttr() for _ in range(8)]
    node.combine_and_set_attribs(all_child_attributes)

    # Test computing own attributes
    node.compute_own_attribs(objects)

    # Test updating attributes
    obj_to_add = Object([50.0, 50.0, 50.0])  # For this test, the position is not relevant
    node.update_attribs(obj_to_add, add=True)

    assert node.num_atoms == 1
    assert node.n_fixed == 0

    obj_to_remove = Object([25.0, 25.0, 25.0])  # For this test, the position is not relevant
    node.update_attribs(obj_to_remove, add=False)

    assert node.num_atoms == 0
    assert node.n_fixed == 0

    print("DynamicOctreeNode test passed!")

    # Print octree status
    print("Octree status:")
    print("Number of objects:", node.num_atoms)
    print("Number of fixed objects:", node.n_fixed)

def test_dynamic_octree():
    # Initialize objects
    atoms = [Object([random.uniform(0, 100), random.uniform(0, 100), random.uniform(0, 100)]) for _ in range(10)]
    num_atoms = len(atoms)
    construction_params = OctreeConstructionParams(max_leaf_size=5, max_leaf_dim=10, slack_factor=1.0)
    max_nodes = 100

    # Initialize DynamicOctree
    octree = DynamicOctree(atoms, num_atoms, construction_params, verbose=True, max_nodes=max_nodes)

    # Test building the octree
    assert octree.build_octree() == True

    # Test printing the octree status
    octree.print_octree()

    # Additional tests for other methods/functions if needed

    print("All tests passed!")

# TODO: Check if parent_pointers are correctly assigned

def test_add_remove_objects():
    # Initialize objects
    atoms = [Object([random.uniform(0, 100), random.uniform(0, 100), random.uniform(0, 100)]) for _ in range(10)]
    num_atoms = len(atoms)
    construction_params = OctreeConstructionParams(max_leaf_size=5, max_leaf_dim=10, slack_factor=1.0)
    max_nodes = 100

    # Initialize DynamicOctree
    octree = DynamicOctree(atoms, num_atoms, construction_params, verbose=True, max_nodes=max_nodes)

    # Test building the octree
    assert octree.build_octree() == True

    print("Initial octree status:")
    print("Number of objects:", octree.num_atoms)
    print("Number of Nodes:", octree.num_nodes)

    # Add new object
    new_object = Object([50.0, 50.0, 50.0])
    octree.add_atom_to_non_leaf(0, len(atoms))  # Assuming adding to the root node for simplicity
    octree.add_atom_to_leaf(0, len(atoms))  # Assuming adding to the root node for simplicity

    print("Octree status after adding new object:")
    print("Number of objects:", octree.num_atoms)
    print("Number of Nodes:", octree.num_nodes)

    # Remove an object
    octree.remove_atom_from_non_leaf(0, len(atoms) - 1)  # Assuming removing from the root node for simplicity
    octree.remove_atom_from_leaf(0, len(atoms) - 1)  # Assuming removing from the root node for simplicity

    print("Octree status after removing an object:")
    print("Number of objects:", octree.num_atoms)
    print("Number of Nodes:", octree.num_nodes)

    print("All tests passed!")

def test_traverse_octree():
    # Initialize objects
    atoms = [Object([random.uniform(0, 100), random.uniform(0, 100), random.uniform(0, 100)]) for _ in range(20)]
    num_atoms = len(atoms)
    construction_params = OctreeConstructionParams(max_leaf_size=5, max_leaf_dim=10, slack_factor=1.0)
    max_nodes = 100

    # Initialize DynamicOctree
    octree = DynamicOctree(atoms, num_atoms, construction_params, verbose=True, max_nodes=max_nodes)

    # Test building the octree
    assert octree.build_octree() == True

    # Test traversing the octree
    octree.traverse_octree(0)  # Assuming starting traversal from the root node

    print("Traversal completed!")


def print_node_details(self, node_id, indent=""):
    """
    Print details of a node and its children recursively.

    Args:
        node_id (int): The ID of the node to print.
        indent (str): Indentation string for better visualization.
    """
    node = self.nodes[node_id]
    print(indent + f"Node ID: {node_id}")
    print(indent + f"Parent Pointer: {node.parent_pointer}")
    print(indent + f"Node Attributes: {node.attribs}")  # Print any other relevant node attributes
    
    if node.leaf:
        print(indent + "Leaf Node")
        print(indent + f"Atom Indices: {node.atom_indices}")
    else:
        print(indent + "Non-Leaf Node")
        print(indent + f"Child Pointers: {node.child_pointer}")

        # Recursively print details of children nodes
        for child_id in node.child_pointer:
            if child_id != -1:
                self.print_node_details(child_id, indent + "  ")

# Modify the test_dynamic_octree() function to include printing the octree
def test_dynamic_octree():
    # Initialize objects and octree
    atoms = [Object([random.uniform(0, 100), random.uniform(0, 100), random.uniform(0, 100)]) for _ in range(125)]
    num_atoms = len(atoms)
    construction_params = OctreeConstructionParams(max_leaf_size=5, max_leaf_dim=10, slack_factor=1.0)
    max_nodes = 100
    octree = DynamicOctree(atoms, num_atoms, construction_params, verbose=True, max_nodes=max_nodes)

    # Redirecting prints to a text file
    sys.stdout = open('output.txt', 'w')

    # Test building the octree
    assert octree.build_octree() == True

    # Print octree status
    # octree.print_octree()

    print("All tests passed!")

    # Closing the file after writing
    sys.stdout.close()

def test_object_to_node_map():
    # Initialize objects
    atoms = [Object([random.uniform(0, 100), random.uniform(0, 100), random.uniform(0, 100)]) for _ in range(10)]
    num_atoms = len(atoms)
    construction_params = OctreeConstructionParams(max_leaf_size=5, max_leaf_dim=10, slack_factor=1.0)
    max_nodes = 100

    # Initialize DynamicOctree
    octree = DynamicOctree(atoms, num_atoms, construction_params, verbose=True, max_nodes=max_nodes)

    # Test building the octree
    assert octree.build_octree() == True

    print("Initial octree status:")
    print("Number of objects:", octree.num_atoms)
    print("Number of Nodes:", octree.num_nodes)

    # Print object_to_node_map before adding new object
    print("object_to_node_map before adding new object:", octree.object_to_node_map)

    # Add new object
    new_object = Object([50.0, 50.0, 50.0])
    octree.add_atom_to_non_leaf(0, len(atoms))  # Assuming adding to the root node for simplicity
    octree.add_atom_to_leaf(0, len(atoms))  # Assuming adding to the root node for simplicity

    # Print object_to_node_map after adding new object
    print("object_to_node_map after adding new object:", octree.object_to_node_map)
    
    # Verify if the object_to_node_map is updated properly after adding an object
    # assert new_object in octree.object_to_node_map
    # assert octree.object_to_node_map[new_object] == 0  # Assuming added to the root node

    print("Octree status after adding new object:")
    print("Number of objects:", octree.num_atoms)
    print("Number of Nodes:", octree.num_nodes)

    # Print object_to_node_map before removing an object
    print("object_to_node_map before removing an object:", octree.object_to_node_map)

    # Remove an object
    octree.remove_atom_from_non_leaf(0, len(atoms) - 1)  # Assuming removing from the root node for simplicity
    octree.remove_atom_from_leaf(0, len(atoms) - 1)  # Assuming removing from the root node for simplicity

    # Print object_to_node_map after removing an object
    print("object_to_node_map after removing an object:", octree.object_to_node_map)

    # Verify if the object_to_node_map is updated properly after removing an object
    assert new_object not in octree.object_to_node_map

    print("Octree status after removing an object:")
    print("Number of objects:", octree.num_atoms)
    print("Number of Nodes:", octree.num_nodes)

    print("All tests passed!")


if __name__ == "__main__":
    # test_dynamic_octree_node()  # This test function is only testing the methods of the DynamicOctreeNode class, 
    # test_dynamic_octree()       # This test function is testing if the octree nodes are generated and expanded correctly or not.
    # test_add_remove_objects()
    # test_traverse_octree()  #TODO: This traversal shows if the parents and chilren are connected correctly. Check and Verify
    # test_dynamic_octree()
    test_object_to_node_map()