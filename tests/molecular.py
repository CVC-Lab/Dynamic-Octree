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

def populateInitial(atom_path):
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
    atom_pos = {}
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
            atom = Object(position=position)
            atom.set_id(atom_id)

            # Populating list of atoms
            atom_list.append(atom)
            # Populating inital octree state
            atom_pos[atom_id] = position
    
    print("Atom positions")
    print(atom_pos)
    print()
    print("Atom List")
    for i in atom_list:
        print(f"{i.id} ({i.x},{i.y},{i.z})")

    return atom_pos, atom_list

def populateChanges(atom_pos, changes_path)->list:
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
    atom_pos, initial_atoms = populateInitial(initial_path)
    print()
    changes = populateChanges(atom_pos=atom_pos, changes_path=changes_path)
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
    for i in changes:
        octree.update_octree(i)

if __name__ == "__main__":
    cleanDirectory(ATOM_DIR)
    testOctreeMolecular(initial_path=ATOMS_PATH, changes_path=CHANGES_PATH)