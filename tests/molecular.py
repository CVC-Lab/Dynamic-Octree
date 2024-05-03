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
            atom_id = float(i[0]) # Atom id
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
    changes = []

    with open(changes_path, 'r') as c:
        data = [line.rstrip('\n').split() for line in c]
        for i in data:
            id = float(i[0]) # atom id
            position = (i[1], i[2], i[3]) # xyz
            atom = Object(position=position)
            atom.set_id(id)
            changes.append(atom)

    return changes

def testOctreeMolecular(initial_path, changes_path):
    r"""
    Testing octree with a short molecular trajectory path

    args:
        * initial_path(str): Path to initial atom positions file
        * changes_path(str): Path to changes in atom positions
    """
    atoms, initial_atoms = populateInitial(initial_path)
    print()
    changes = populateChanges(changes_path=changes_path)
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
    # ! TODO
    # octree.object_to_node_map: {Object:node_id}
    for i in octree.object_to_node_map.items():
        print(i)

if __name__ == "__main__":
    cleanDirectory(ATOM_DIR)
    testOctreeMolecular(initial_path=ATOMS_PATH, changes_path=CHANGES_PATH)