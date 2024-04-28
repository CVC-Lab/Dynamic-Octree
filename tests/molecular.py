import sys, os

# Adding Octree root to sys.path
sys.path.insert(0, '../')
sys.path.insert(0, "../data/molecular/")
ATOMS_PATH = "../data/molecular/1BDD/atoms/initial.txt"
ATOM_DIR = "1BDD/atoms/"
CHANGES_PATH = "../data/molecular/1BDD/atoms/changes.txt"

from Octree import DynamicOctree, OctreeConstructionParams
from objects import Object
from utils import cleanDirectory

def populateData(atom_path):
    r"""
    Populates a list of initial atom_ids and positions as well as a
    dictionary of atoms and their current positions

    args:
        atom_path(str): Directory of all initial atom positions
    
    returns:
        * all_atoms(dict{atom_id:(x,y,z)}): Dictionary of all atoms and their 
        initial positions
        * atom_list(lst[Object]): List of atom Objects
    """
    # Current atoms and their respective positions
    # {atom_id:tuple(pos_x,pos_y,pos_z)}
    all_atoms = {}
    # List of atoms and starting positions
    # Used for construction of the octree
    # Populated with Object()
    atom_list = []

    # Populating 
    with open(atom_path, 'r') as atoms:
        data = [line.rstrip('\n').split() for line in atoms]
        for i in data:
            atom_id = float(i[0]) # Atom id
            atom_x, atom_y, atom_z = float(i[1]), float(i[2]), float(i[3]) # xyz
            position = [atom_x, atom_y, atom_z]
            atom = Object(position=position, id=atom_id)

            # Populating list of atoms
            atom_list.append(atom)
            # Populating inital octree state
            all_atoms[atom_id] = position

    return all_atoms, atom_list

def testOctreeMolecular(atom_path=ATOMS_PATH, changes_path=CHANGES_PATH):
    atom_changes, initial_atoms = populateData(atom_path)
    num_atoms = len(initial_atoms)

    construction_params = OctreeConstructionParams(max_leaf_size=5,
                                                   max_leaf_dim=10,
                                                   slack_factor=1.0)
    max_nodes = 300 # ~264 atoms

    # Initialize DynamicOctree
    octree = DynamicOctree(initial_atoms, num_atoms,
                           construction_params,
                           verbose=False,
                           max_nodes=max_nodes)

    # Test building the octree
    assert octree.build_octree() == True

    octree.traverse_octree(0)


if __name__ == "__main__":
    cleanDirectory(ATOM_DIR)
    testOctreeMolecular()