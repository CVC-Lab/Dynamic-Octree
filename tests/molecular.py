import sys, os

# Adding Octree root to sys.path
sys.path.insert(0, '../')
sys.path.insert(0, "../data/molecular/")
# File paths
ATOMS_PATH = "../data/molecular/1BDD/atoms/initial.txt"
CHANGES_PATH = "../data/molecular/1BDD/atoms/changes.txt"

# For cleaning
ATOM_DIR = "1BDD/atoms/"

from Octree import DynamicOctree, OctreeConstructionParams
from objects import Object
from utils import cleanDirectory

def populateInitial(atom_path):
    r"""
    Populates a list of initial atom_ids and positions as well as a
    dictionary of atoms and their current positions

    args:
        atom_path(str): Directory of all initial atom positions
    
    returns:
        * atoms(dict{atom_id:Object()): Dictionary of all atom objects and their ID
        * atom_list(lst[Object]): List of atom Objects
    """
    # Current atoms and their respective positions
    # {atom_id:Object()}
    atoms = {}
    # List of atoms and starting positions
    # Used for construction of the octree
    # Populated with Object()
    atom_list = []

    # Populating 
    with open(atom_path, 'r') as path:
        data = [line.rstrip('\n').split() for line in path]
        for i in data:
            atom_id = int(i[0])-1 # Atom id, 0th indexing
            atom_x, atom_y, atom_z = float(i[1]), float(i[2]), float(i[3]) # xyz
            position = [atom_x, atom_y, atom_z]
            atom = Object(position=position)
            atom.set_id(atom_id)

            # Populating list of atoms
            atom_list.append(atom)
            # Populating inital octree state
            atoms[atom_id] = atom

    return atoms, atom_list

def populateChanges(changes_path)->list:
    r"""
    Reads a changes file and creates a list of changes for the octree to update

    args:
        * atom_pos(dict{atom_id:(x,y,z)}): Initial atom positions
        * changes_path(str): Path to changes file

    returns:
        changes(list[Object]): List of atom objects with new positions (not exclusive)
    """
    id_changes = []
    position_changes = []

    with open(changes_path, 'r') as c:
        data = [line.rstrip('\n').split() for line in c]
        for i in data:
            id = int(i[0])-1 # atom id, 0th indexing
            position = (float(i[1]), float(i[2]), float(i[3])) # xyz
            id_changes.append(id)
            position_changes.append(position)

    return id_changes, position_changes

def testOctreeMolecular(initial_path, changes_path, verbose=True):
    r"""
    Testing octree with a short molecular trajectory path

    args:
        * initial_path(str): Path to initial atom positions file
        * changes_path(str): Path to changes in atom positions
    """
    atoms, initial_atoms = populateInitial(initial_path)
    print()
    id_changes, position_changes = populateChanges(changes_path=changes_path)
    num_atoms = len(initial_atoms)

    construction_params = OctreeConstructionParams(max_leaf_size=5,
                                                   max_leaf_dim=10,
                                                   slack_factor=1.0)
    max_nodes = num_atoms

    # Initialize DynamicOctree
    octree = DynamicOctree(initial_atoms, num_atoms,
                           construction_params,
                           verbose=False,
                           max_nodes=max_nodes)

    # Test building the octree
    assert octree.build_octree() == True

    # Testing insertion and deletion of each atom
    # octree.object_to_node_map: {Object:node_id}
    # ! TODO
    for index, value in enumerate(id_changes):
        # Getting atom object from an id
        atom_object = atoms[value]

        # Node and atom_id for the remove_atom/add_atom functions
        node_id = octree.object_to_node_map[atom_object]
        atom_id = value

        try:
            #! Ideally we would use
            #! octree.update(atom_object)
            #* Not sure when to use non-leaf vs leaf
            # First removes the atom from the Octree, modifies
            # the atom with a new position, and readds the atom

            if verbose:
                print(node_id, atom_id)
                print("Atom_object xyz before change")
                print(atom_object.x, atom_object.y, atom_object.z)

            octree.remove_atom_from_non_leaf(node_id, atom_id)

            position = position_changes[index] # Get updated position of atom
            atom_object.set_position(position) # Set new atom position

            if verbose:
                print(f"Updated atom {atom_id} to {position}")

            #* Not sure when to use non-leaf vs leaf
            # Readds the atom to the Octree
            octree.add_atom_to_non_leaf(node_id, atom_id)

            if verbose:
                print("Atom_object xyz after change")
                print(atom_object.x, atom_object.y, atom_object.z)
                print("")
                print("")

        except:
            print(f"Error at {node_id}, {atom_id}")

if __name__ == "__main__":
    cleanDirectory(ATOM_DIR)
    testOctreeMolecular(initial_path=ATOMS_PATH, changes_path=CHANGES_PATH)