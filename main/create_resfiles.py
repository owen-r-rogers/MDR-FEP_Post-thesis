import argparse
import re
import pyrosetta
import pyrosetta.rosetta as rosetta
import pandas as pd
import numpy as np
from os import path, environ
from pyrosetta.rosetta.core.pack.task import TaskFactory
from pyrosetta.rosetta.core.pack.task import operation
from pyrosetta.rosetta.core.select.residue_selector import NeighborhoodResidueSelector


"""
This is the precursor step to running MDR. This will create 
the necessary resfiles. Previously the resfiles were not being recognized some
of the time since they were being overwritten and created on each SLURM array.

This will make a resfile, assuming SSM is what you want, of the format:
{residue_to_mutate}.resfile

Also, it will avoid design for each one.
"""


def parse_pdb_for_resfile(pdb_file):
    """
    Reads a pdb file and converts it to a string.
    :param pdb_file - PDB file of interest
    :return: PDB file information as a string
    """
    pdb_string = ''

    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                pdb_string += line
    return pdb_string


def string_to_pose_for_resfile(pdb_string):
    """
    Takes the output of parse_pdb and converts it to a PyRosetta pose object.
    :param pdb_string: Output of parse_pdb()
    :return: pose object
    """

    # convert to rosetta naming conventions
    pdb_string = re.sub("ATOM  (.....)  CD  ILE", "ATOM  \\1  CD1 ILE", pdb_string)
    pdb_string = re.sub("ATOM  (.....)  OC1", "ATOM  \\1  O  ", pdb_string)
    pdb_string = re.sub("ATOM  (.....)  OC2", "ATOM  \\1  OXT", pdb_string)

    # create pose
    pose = rosetta.core.pose.Pose()
    rosetta.core.import_pose.pose_from_pdbstring(pose, pdb_string)

    return pose


def create_ssm_dict_for_resfile(ref_pose, chain_id):
    """
    Make a dictionary in the format: [seqpos to mutate]: [acceptable AA mutations]
    :param ref_pose: Reference pose object
    :param chain_id: The chain to mutate
    :return: SSM dictionary
    """

    alphabet = 'ACDEFGHIKLMNPQRSTVWY'
    ssm_dict = {}
    chain_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6}
    chain_num = chain_mapping[chain_id]

    '''returns a vector of the last residue number in each chain'''
    chain_termini = rosetta.core.pose.chain_end_res(ref_pose)

    # determine residues of each chain
    end = chain_termini[chain_num]
    if chain_num == 1:
        start = 1
    else:
        chain_length = chain_termini[chain_num] - chain_termini[chain_num - 1]
        start = end - chain_length + 1

    # fill with 20 (-1) residues to saturate
    for pos in range(start, end + 1):
        ssm_dict[pos] = [aa for aa in list(alphabet) if aa != ref_pose.residue(pos).name1()]

    return ssm_dict


def create_resfile(blueprint_pose, seqpos, repacking_radius, resfile_dir):
    """

    :param blueprint_pose: The pose to use as a reference. All the distances for repacking will be
    based on this pose.
    :param seqpos: Sequence position to build the .resfile around. If there is no
    distance given then this seqpos will be the only one that undergoes any changes.
    :param repacking_radius: Radius around the seqpos in Å to repack.
    :param resfile_dir: The directory where the resfiles will go.
    :return: None
    """

    seqpos = int(seqpos)
    orig_aa = blueprint_pose.residue(seqpos).name1()
    wt_pose = blueprint_pose.clone()
    wt_pose.update_residue_neighbors()

    ''' Reference TaskFactory to parse to resfiles '''
    ref_tf = TaskFactory()
    ref_tf.push_back(operation.InitializeFromCommandline())
    ref_tf.push_back(operation.RestrictToRepacking())
    ref_tf.push_back(operation.ExtraRotamersGeneric())
    ref_tf.push_back(operation.NoRepackDisulfides())
    ref_tf.push_back(operation.UseMultiCoolAnnealer())
    repack_sphere = NeighborhoodResidueSelector()
    repack_sphere.set_focus(str(seqpos))
    repack_sphere.set_distance(repacking_radius)
    repack_sphere.set_include_focus_in_subset(True)
    prevent_repacking = operation.PreventRepackingRLT()
    restrict_to_focus = operation.OperateOnResidueSubset(prevent_repacking, repack_sphere, True)
    ref_tf.push_back(restrict_to_focus)
    blueprint_packer_task = ref_tf.create_task_and_apply_taskoperations(wt_pose)

    ''' Format resides to operate on, a bit convoluted '''
    blueprint_task_file = path.join(resfile_dir, f'{orig_aa}{seqpos}_packer_task.txt')
    with open(blueprint_task_file, 'w') as f:
        f.write(str(blueprint_packer_task))
    pack_task = []
    with open(blueprint_task_file, 'r') as f:
        for line in f:
            if line[0].isnumeric():
                split = line.split()
                pack_task.append([split[0], split[1], split[2]])
    pack_df = pd.DataFrame(pack_task, columns=['residue', 'pack?', 'design?'])
    residues_to_repack = pack_df[pack_df['pack?'] == 'TRUE']['residue'].tolist()
    residues_to_write_to_resfile = []
    num_chains = blueprint_pose.num_chains()
    if num_chains == 1:
        start_chain_a = blueprint_pose.chain_begin(1)
        end_chain_a = blueprint_pose.chain_end(1)
        chain_a_range = np.arange(start_chain_a, end_chain_a + 1)
        for res in residues_to_repack:
            if int(res) in chain_a_range:
                adjusted_resid = res
                residues_to_write_to_resfile.append(f'{adjusted_resid},A')
    elif num_chains == 2:
        start_chain_a = blueprint_pose.chain_begin(1)
        end_chain_a = blueprint_pose.chain_end(1)
        start_chain_b = blueprint_pose.chain_begin(2)
        end_chain_b = blueprint_pose.chain_end(2)
        chain_a_range = np.arange(start_chain_a, end_chain_a + 1)
        chain_b_range = np.arange(start_chain_b, end_chain_b + 1)
        for res in residues_to_repack:
            if int(res) in chain_a_range:
                adjusted_resid = res
                residues_to_write_to_resfile.append(f'{adjusted_resid},A')
            elif int(res) in chain_b_range:
                adjusted_resid = int(res) - end_chain_a
                residues_to_write_to_resfile.append(f'{adjusted_resid},B')
    elif num_chains == 3:
        start_chain_a = blueprint_pose.chain_begin(1)
        end_chain_a = blueprint_pose.chain_end(1)
        start_chain_b = blueprint_pose.chain_begin(2)
        end_chain_b = blueprint_pose.chain_end(2)
        start_chain_c = blueprint_pose.chain_begin(3)
        end_chain_c = blueprint_pose.chain_end(3)
        chain_a_range = np.arange(start_chain_a, end_chain_a + 1)
        chain_b_range = np.arange(start_chain_b, end_chain_b + 1)
        chain_c_range = np.arange(start_chain_c, end_chain_c + 1)
        for res in residues_to_repack:
            if int(res) in chain_a_range:
                adjusted_resid = res
                residues_to_write_to_resfile.append(f'{adjusted_resid},A')
            elif int(res) in chain_b_range:
                adjusted_resid = int(res) - end_chain_a
                residues_to_write_to_resfile.append(f'{adjusted_resid},B')
            elif int(res) in chain_c_range:
                adjusted_resid = int(res) - end_chain_b
                residues_to_write_to_resfile.append(f'{adjusted_resid},C')

    ''' Write resfiles and PackerTasks '''
    resfile_index_file = path.join(resfile_dir, f'{seqpos}_resfile_index')
    resfile = path.join(resfile_dir, f'{seqpos}.resfile')
    with open(resfile_index_file, 'w') as f:
        for seqpos_and_chain in residues_to_write_to_resfile:
            f.write(f'{seqpos_and_chain}\n')
    with open(resfile, 'w') as rf:
        rf.write('NATRO\n')
        rf.write('start\n')
        rf.write('\n')
        with open(resfile_index_file, 'r') as f:
            for line in f:
                split = line.strip().split(',')
                aa = split[0]
                chain = split[1]
                rf.write(f'{aa} {chain} NATAA\n')


def generate_resfiles(blueprint_pose, mutation_dictionary, repack_radius, resfile_dir):
    """
    Apply create_resfile() to every seqpos
    :param mutation_dictionary: output of create_ssm_mutations_for_resfile
    :return: None
    """

    # iterate through the sequence and allowable mutations (contained in mutation_dictionary)
    for seqpos in mutation_dictionary.keys():
        create_resfile(blueprint_pose, seqpos, repack_radius, resfile_dir)

if __name__ == '__main__':

    # initialize pyrosetta
    pyrosetta.init()

    # initialize and parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--repack-within', default=0.0, type=float, help='The SAME distance as for MDRFEP')
    parser.add_argument('--chain', type=str, help='The SAME chain as for MDRFEP, usually chain A')
    args = parser.parse_args()

    # define the input and resfiles environmental values
    environ['PDB_DIR'] = './input'
    environ['RF_DIR'] = './resfiles'

    pdb_house = environ.get('PDB_DIR')
    resfile_house = environ.get('RF_DIR')

    # define the packing sphere
    packing_sphere = args.repack_within

    # define the reference pose
    ref_pdb = path.join(pdb_house, "1QYS.pdb")
    ref_string = parse_pdb_for_resfile(ref_pdb)
    ref_pose = string_to_pose_for_resfile(ref_string)

    # generate the mutation dictionary and use that to generate the resfiles
    ssm_mutations = create_ssm_dict_for_resfile(ref_pose, args.chain)
    generate_resfiles(ref_pose, ssm_mutations, packing_sphere, resfile_house)
