import pyrosetta
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--input-pdb', type=str, help='input pdb file')
args = parser.parse_args()


"""
This is essentially skeleton code for how you would run ProteinMPNN in PyRosetta (except I can't actually run the last command right now since I didn't compile with extra=pytorch)
"""


# initialize pyrosetta
pyrosetta.init()


input_pdb = args.input_pdb


# instantiate mover
mover = pyrosetta.rosetta.protocols.protein_mpnn.ProteinMPNNMover()


# load pose to apply mover to
pose = pyrosetta.rosetta.core.import_pose.pose_from_file(input_pdb)


# set input pose
mover.set_input_pose(pose)


# create chainselector object
chainA = pyrosetta.rosetta.core.select.residue_selector.ChainSelector('A')
sele = pyrosetta.rosetta.core.select.residue_selector.ResidueSelector(chainA)


# set designable sequences
mover.set_design_selector_rs(sele)


# apply the mover (where the issue arose)
mover.apply(pose)
