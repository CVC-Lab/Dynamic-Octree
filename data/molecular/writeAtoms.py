# Converts each PDB in a series to atom coordinates:
# ATOM_NUM X Y Z
import os
from utils import cleanDirectory

FILE_DIR = "1BDD/trajectories"
CHANGES_FILE = "1BDD/atoms/changes.txt"
ATOM_POS_FILE = "1BDD/atoms/initial.txt"
ATOM_DIR = "1BDD/atoms/"
INITIAL_PDB = "1BDD/1bdd.pdb"
new_line = '\n'

def writeInitial(initial_path, write_atoms):
    r"""
    Given an initial PDB, writes the position of each atom to atoms.txt and
    saves the position into a dictionary (atoms)

    args:
        * initial_path(str): Path of initial PDB file
        * write_atoms(str): Path to write the atoms.txt file

    returns:
        * atoms(dict{atom_id:(x, y, z)}): Dict of each atom's position
    """
    atoms = {}
    with open(initial_path, 'r') as pdb:
        lines = [line.rstrip('\n') for line in pdb]
        for i in lines:
            if (len(i) > 0) and (i.split()[0] == "ATOM"):
                atom_data = i.split()
                # Atom number
                atom_id = int(atom_data[1])
                # Atom x, y, z
                atom_pos = (float(atom_data[6]), float(atom_data[7]), float(atom_data[8]))
                # Saving initial atom positions
                atoms[atom_id] = atom_pos

                # Writing initial data to atoms.txt
                with open(os.path.join(write_atoms), 'a') as a:
                    a.write(f"{atom_id} {atom_pos[0]} {atom_pos[1]} {atom_pos[2]}{new_line}")

    return atoms

def writeAtoms(trajectory_path, write_atoms, write_changes):
    r"""
    Given a trajectory path, returns the final positions of each atom

    args:
        trajectory_path: Path to PDB trajectories directory
        write_path: Path to write each atom_coords file (Should be directory)
    """
    # Generate initial postitions into atoms.txt
    atoms = writeInitial(initial_path=INITIAL_PDB, write_atoms=ATOM_POS_FILE)

    # Updates the current position of each atom given each trajectory snapshot
    for trajectory in os.listdir(trajectory_path):
        with open(os.path.join(trajectory_path, trajectory), 'r') as pdb:
            lines = [line.rstrip('\n') for line in pdb]
            atoms = writeChanges(lines, atoms, write_atoms, write_changes)

    return atoms

def writeChanges(lines, atoms, write_atoms, write_changes):
    r"""
    Returns Atom_nums and XYZ for each atom
    in PDB file line

    args:
        lines(lst): List of all lines in the PDB
        atoms(dict{atom_id:(x, y, z)}): Current position of each atom 
    returns:
        atoms(lst): List of atom information in the form of [ATOM_NUM, X, Y, Z]
    """

    for i in lines:
        if (len(i) > 0) and (i.split()[0] == "ATOM"):
            atom_data = i.split()
            # Atom number
            atom_id = int(atom_data[1])
            # Atom x, y, z
            atom_pos = (float(atom_data[6]), float(atom_data[7]), float(atom_data[8]))
            
            # Changes.txt
            # Atom position changed
            if atom_id in atoms.keys():
                if atoms[atom_id] != atom_pos:
                    atoms[atom_id] = atom_pos

                    # Write atom differences to changes.txt
                    with open(os.path.join(write_changes), 'a') as changes:
                        changes.write(f"{atom_id} {atoms[atom_id][0]} {atoms[atom_id][1]} {atoms[atom_id][2]}{new_line}")

            # atoms.txt
            # Intitial creation of atoms
            else:
                atoms[atom_id] = atom_pos
                with open(os.path.join(write_atoms), 'a') as a:
                    a.write(f"{atom_id} {atoms[atom_id][0]} {atoms[atom_id][1]} {atoms[atom_id][2]}{new_line}")

    return atoms

if __name__ == "__main__":
    cleanDirectory(ATOM_DIR)
    writeAtoms(trajectory_path=FILE_DIR, 
               write_atoms=ATOM_POS_FILE, 
               write_changes=CHANGES_FILE)