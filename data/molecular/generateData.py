from pyrosetta import pose_from_pdb, get_fa_scorefxn, init, Pose
from pyrosetta.rosetta.protocols.minimization_packing import MinMover
from pyrosetta.rosetta.core.scoring import CA_rmsd
from pyrosetta.rosetta.protocols.simple_moves import SmallMover
from pyrosetta.rosetta.core.pack.task import TaskFactory
from pyrosetta.rosetta.core.pack.task.operation import RestrictToRepacking
from pyrosetta.rosetta.core.select.movemap import MoveMapFactory

init() # Pyrosetta init

# Constant for Energy thresholds
ENERGY_THRESHOLD = 0.5 # 20%
# Constant for RMSD threshold
RMSD_THRESHOLD = 2 # 2 angstroms

# Import desired PDB (Change this)
DATA_DIR = "1BDD/"
pose = pose_from_pdb(DATA_DIR + "1bdd.pdb")
# Where to save trajectory data
DUMP_DIR = DATA_DIR + "trajectories/"


# Movement factory w.r.t. energy
min_mover = MinMover()
# Since movements are extremely small by default, set up
# a mover that allows for more movement
tf = TaskFactory()
tf.push_back(RestrictToRepacking())  # No design, only repacking
mmf = MoveMapFactory()
mmf.all_bb(True)  # Allow backbone movements
mmf.all_chi(True)  # Allow side chain movements
small_mover = SmallMover()  # 1.0 = magnitude of perturbation, 1 = number of moves
small_mover.angle_max('H', 180)  # Increase max angle for helices
small_mover.angle_max('E', 180)  # Increase max angle for strands
small_mover.angle_max('L', 180)  # Increase max angle for loops

# Criterion for saving: %Change in Energy or change in RMSD
# Energy
scorefxn = get_fa_scorefxn()
previous_energy = scorefxn(pose) # Getting energy
# RMSD:
previous_RMSD_pose = Pose()
previous_RMSD_pose.assign(pose)

for i in range(1, 100):  #
    small_mover.apply(pose)
    min_mover.apply(pose)
    # Getting the energy from molecule from last change and now
    current_energy = scorefxn(pose)
    # RMSD between last significant change and now
    current_RMSD = CA_rmsd(pose, previous_RMSD_pose)

    # Energy difference threshold
    d_energy = abs(current_energy - previous_energy)
    percent_d_energy = d_energy / previous_energy

    # If change meets any of the current requirements, save the PDB
    if (d_energy >= ENERGY_THRESHOLD):
        pose.dump_pdb(f"{DUMP_DIR}trajectory_frame_{i}_energy_{round(d_energy, 3)}.pdb")
    elif (current_RMSD >= RMSD_THRESHOLD):
        pose.dump_pdb(f"{DUMP_DIR}trajectory_frame_{i}_RMSD_{round(current_RMSD, 3)}.pdb")
        previous_RMSD_pose = pose

    previous_energy = current_energy
