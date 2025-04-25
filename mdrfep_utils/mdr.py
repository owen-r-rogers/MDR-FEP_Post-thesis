import argparse
import fnmatch as fm
import re
import pyrosetta
import pyrosetta.rosetta as rosetta
import numpy as np
import sys
from os import listdir, path, environ, getcwd
from collections import defaultdict
from natsort import natsorted
from pyrosetta import MoveMap
from pyrosetta import init
from pyrosetta.toolbox.mutants import mutate_residue
from pyrosetta.rosetta.protocols import minimization_packing as pack_min
from pyrosetta.rosetta.core.pack.task import TaskFactory
from pyrosetta.rosetta.core.pack.task import operation

"""
MDR
Parallelized, much faster
for in silico deep mutagenesis
"""


def parse_pdb(pdb_file):
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


def string_to_pose(pdb_string):
    """
    Takes the output of parse_pdb and converts it to a PyRosetta pose object. Necessary because Rosetta and PDB have slightly different naming conventions.
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


def create_ssm_dict(ref_pose, chain_id):
    """
    Make a dictionary in the format: [seqpos to mutate]: [acceptable AA mutations]. This will be processed later and dictate which amino acid mutations are carried out.
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


def score_pose(pose, score_function):
    """
    Simple function, scores the pose using the given score function.
    """
    return score_function(pose)


def minimize_sidechains(pose, score_function):
    """
    Optional function. Creates a MoveMap and then performs gradient-based energy minimization, keeping the backbone fixed.
    Results in the code being significantly slower.
    :param pose: Pose object
    :param score_function: Score function to evaluate
    :return: None
    """

    # create MoveMap
    move_map = MoveMap()
    move_map.set_bb(False)  # Do NOT move backbone torsion angles
    move_map.set_chi(True)

    # Apply the mover
    min_mover = rosetta.protocols.minimization_packing.MinMover()
    min_mover.set_movemap(move_map)
    min_mover.score_function(score_function)
    min_mover.min_type("lbfgs_armijo_nonmonotone")
    min_mover.tolerance(1e-6)
    min_mover.apply(pose)


def pack_rotamers(pose, seqpos, mutation, score_function, seqpos_resfile, minimize):
    """
    This is the core function of the R part of MDR-FEP.
    :param pose: Pose object
    :param seqpos: Sequence position of interest
    :param mutation: AA to mutate current WT to
    :param score_function
    :param seqpos_resfile: The .resfile corresponding to this seqpos
    :param minimize: Boolean
    :return: The ∆Energy of introducing this mutation at this residue.
    """

    # clone the WT and MUT pose so they are packed independently
    wt_pose = pose.clone()
    mut_pose = pose.clone()

    # number of iterations to apply the mover
    nloops = 50

    print(f'Processing {seqpos_resfile} for position {seqpos}', flush=True)
    print(f'Native structure score: {score_pose(wt_pose, score_function)}', flush=True)

    # initialize taskfactory
    task_factory = TaskFactory()
    task_factory.push_back(operation.InitializeFromCommandline()) # make sure that command line arguments are parsed
    task_factory.push_back(operation.RestrictToRepacking()) # no designing
    task_factory.push_back(operation.ExtraRotamersGeneric()) # including extra rotamers
    task_factory.push_back(operation.NoRepackDisulfides()) # avoid disulfides
    task_factory.push_back(operation.UseMultiCoolAnnealer(states=10)) # like it sounds

    # read the given resfile
    parse_rf = operation.ReadResfile(seqpos_resfile)
    task_factory.push_back(parse_rf)

    # create a packer task
    wt_packer_task = task_factory.create_task_and_apply_taskoperations(wt_pose)

    # create and apply a mover
    wt_mover = pack_min.PackRotamersMover(score_function, wt_packer_task, nloop=nloops)
    wt_mover.apply(wt_pose)

    print(wt_mover.info(), flush=True)
    print(f'WT score after packing: {score_pose(wt_pose, score_function)}', flush=True)

    # optionally, minimize the WT side chains
    if minimize:

        minimize_sidechains(wt_pose, score_function)
        print(f'WT score after minimization: {score_pose(wt_pose, score_function)}', flush=True)

    # score the WT
    score_wt = score_pose(wt_pose, score_function)

    # Do the same for the mutant
    mutate_residue(mut_pose, seqpos, mutation)

    # create and apply a mutant mover
    mut_packer_task = task_factory.create_task_and_apply_taskoperations(mut_pose)
    mut_mover = pack_min.PackRotamersMover(score_function, mut_packer_task, nloop=nloops)
    mut_mover.apply(mut_pose)

    print(mut_mover.info(), flush=True)
    print(f'MUT score after packing: {score_pose(mut_pose, score_function)}', flush=True)

    # again (optionally) minimize the MUT side chains.
    if minimize:
        minimize_sidechains(mut_pose, score_function)
        print(f'MUT score after minimization: {score_pose(mut_pose, score_function)}', flush=True)

    print(f'NUMBER OF ITERATIONS USED FOR REPACKING: {mut_mover.nloop()}', flush=True)

    # record the MUT score
    score_mut = score_pose(mut_pose, score_function)

    # produce a change in energy
    delta_energy = score_mut - score_wt

    return delta_energy


def deep_mut(pose, resfile_dir, mutation_dictionary, minimize):
    """
    Uses the pose, resfile directory and the mutation dictionary to carry out deep mutagenesis. Essentially this iterates through the mutation dictionary and performs the pack_rotamers function, storing the results in the delta_energy_dict.
    :param pose
    :param resfile_dir: Directory containing the resfiles. Set as an environmental variable later on.
    :param mutation_dictionary: Output of create_ssm_dict()
    :param minimize: Boolean
    :return: Dictionary of delta-energy values for the entire pose.
    """

    print(f'Was this minimized? {minimize}', flush=True)

    # initialize ∆E dictionary for storage
    delta_energy_dict = defaultdict(list)

    # iterate through the SSM dictionary and guide the
    # packing function
    for seqpos, mutations in mutation_dictionary.items():

        # WT-AA 1 letter code
        orig_aa = pose.residue(int(seqpos)).name1()

        # find .resfile for that residue
        seqpos_resfile = path.join(resfile_dir, f'{seqpos}.resfile')

        # iterate through mutations that are permitted for that AA
        for mut in mutations:
            print(f'Processing mutation {mut} at position {seqpos}', flush=True)

            # PACK the residue
            delta_energy = pack_rotamers(pose, seqpos, mut, rosetta.core.scoring.get_score_function(is_fullatom=True), seqpos_resfile=seqpos_resfile, minimize=minimize)

            # unique ID for storing
            # form of [WT name1] [seqpos] [MUT name 1]
            unique_id = f'{orig_aa}{seqpos}{mut}'

            # add entry to dictionary
            delta_energy_dict[unique_id].append(delta_energy)

    return delta_energy_dict


def save_delta_energy_array(delta_energy_dict, frame_num):
    """
    Save the delta_energy_dict that is output
    by deep_mut() for EACH frame.
    Each frame corresponds to a timepoint in the MD run.
    This .npz file will contain a single value for each (num AA * 19) mutation.
    :param delta_energy_dict: Output of deep_mut()
    :param frame_num: timepoint of MD
    :return: None
    """

    de_array = {key: np.array(values) for key, values in delta_energy_dict.items()}
    processed_frame = f'{int(frame_num):04}'
    np.savez(f'{processed_frame}.npz', **de_array)


def split_fileset(file_list, block_size):
    """
    Split the .pdb files according to the block (batch is probably the correct term)
    that will be processed by the SLURM array
    :param file_list: List of .pdb files to partition
    :param block_size: The amount of files to divide into each block/batch
    :return: List of .pdb files by block
    """

    return [file_list[i:i + block_size] for i in range(0, len(file_list), block_size)]


def process_frame(pdb_dir, file, resfile_dir, mutation_dictionary, minimize):
    """
    Performs a per-frame deep mutagenesis.
    :param pdb_dir
    :param file: PDB file that is being mutated
    :param resfile_dir: Directory full of .resfile
    :param mutation_dictionary: create_ssm_dict() output
    :param minimize: Boolean
    :return: None, because this function is executed within process_ensemble()
    """

    pdb_string = parse_pdb(path.join(pdb_dir, file))
    pose = string_to_pose(pdb_string)
    delta_energy_dict = deep_mut(pose, resfile_dir, mutation_dictionary, minimize)
    frame_num = path.basename(file).replace('frame', '').replace('.pdb', '')
    save_delta_energy_array(delta_energy_dict, frame_num)


def process_ensemble(pdb_dir, resfile_dir, mutation_dictionary, block_size, minimize):
    """
    Processes the directory of .pdb files. Takes in a directory full of .pdb and .resfile, a mutation
    dictionary that says what residues to saturate, a block size
    detailing how to partition the files for SLURM, and a minimize boolean.
    :param pdb_dir
    :param resfile_dir
    :param mutation_dictionary: create_ssm_dict() output
    :param block_size: num files // block_size = number of SLURM arrays to request
    :param minimize: Boolean
    :return: None
    """

    # get list of files and assign a task ID
    all_files = natsorted(listdir(pdb_dir))
    files = [f for f in all_files if fm.fnmatch(f, 'frame*.pdb')]
    blocks = split_fileset(files, block_size)

    # this function will only be executed on the files corresponding to this task ID
    task_id = int(environ.get('SLURM_ARRAY_TASK_ID'))

    # task_id is 0-indexed
    if task_id < len(blocks):

        # get the batch of files
        block_to_process = blocks[task_id]

        # iterate through them
        for file in block_to_process:
            print(f'Processing {file}', flush=True)

            # main function
            process_frame(pdb_dir, file, resfile_dir, mutation_dictionary, minimize)

    else:
        sys.exit('SLURM_ARRAY_TASK_ID is greater than the number of blocks')


def compile_array(directory, name):
    """
    A (slightly convoluted) function that saves the output
    of the script.
    :param directory: Directory containing the output files
    :param name: Name of the final output file.
    :return: None
    """
    files = []
    for file in listdir(directory):
        if fm.fnmatch(file, '????.npz'):
            files.append(file)
    files.sort()
    array_dict = {}
    for file in files:
        data = np.load(file)
        if path.basename(file).replace('.npz', '') == '0000':
            for key in data.files:
                array_dict[key] = []
        else:
            pass
    for file in files:
        data = np.load(file)
        for key, value in array_dict.items():
            values = data[key]
            value.append(values)
    for key, _ in array_dict.items():
        array_dict[key] = np.concatenate(array_dict[key])
    np.savez(f'{name}', **array_dict)


def get_functional_state(pose):
    """
    Assigns extensions to the output files for convenience.
    :param pose
    :return: monomer/dimer etc based on chains present
    """
    chain_count = pose.num_chains()
    if chain_count == 1:
        fxn = 0
        ext = 'monomer'
        return fxn, ext
    elif chain_count == 2:
        fxn = 1
        ext = 'dimer'
        return fxn, ext
    elif chain_count == 3:
        fxn = 1
        ext = 'trimer'
        return fxn, ext


def init_pyros(beta=False,
               talaris=False,
               beta_genpot=False,
               fatal=False,
               error=False,
               warning=False,
               info=False,
               debug=False,
               trace=False,
               soft_rep=False):
    """
    Initializes PyRosetta based on parsed arguments.
    :param beta: score_function argument: Use the beta_nov16 function.
    :param talaris: score_function argument:
    :param beta_genpot: score_function argument:
    :param fatal: verbosity - least verbose (I)
    :param error: II
    :param warning: III
    :param info: IV
    :param debug: verbosity - second most verbose (V)
    :param trace: verbosity - most verbose. Will lead to comically large output files. Not recommended. (VI)
    :param soft_rep: score_function argument: If passed will use the soft-rep where repulsive interactions are dampened.
    :return: None
    """

    # standard flags
    flags = "-ex1 \
    -ex2 \
    -ignore_waters \
    -packing:multi_cool_annealer 10 \
    -out:nstruct 50 \
    -in:path:database /smithlab/opt/anaconda/envs/pyrosetta2024.39_py3.12/lib/python3.12/site-packages/pyrosetta/database \
    -corrections:score:dun10_dir /smithlab/opt/anaconda/envs/pyrosetta2024.39_py3.12/lib/python3.12/site-packages/pyrosetta/database/rotamer/ExtendedOpt1-5"

    if soft_rep:
        flags += " -score:weights soft_rep_design"

    if beta:
        flags += " -beta_nov16"
    elif talaris:
        flags += " -restore_talaris_behavior"
    elif beta_genpot:
        flags += " -beta"

    if fatal:
        flags += " -out:level 0"
    elif error:
        flags += " -out:level 100"
    elif warning:
        flags += " -out:level 200"
    elif info:
        flags += " -out:level 300"
    elif debug:
        flags += " -out:level 400"
    elif trace:
        flags += " -out:level 500"

    init(flags)


def main(pirate_noises):

    # pyrosetta initialization arguments
    init_pyros(pirate_noises.beta,
               pirate_noises.talaris,
               pirate_noises.beta_gen_pot,
               pirate_noises.fatal,
               pirate_noises.error,
               pirate_noises.warning,
               pirate_noises.info,
               pirate_noises.debug,
               pirate_noises.trace,
               pirate_noises.soft_rep)

    sfxn_name = pyrosetta.rosetta.core.scoring.get_score_functionName()
    print(f'Using the scorefunction: {sfxn_name}', flush=True)

    """ Define variables """
    output_name = pirate_noises.name

    # set the environmental variables for the PDB_DIR and the RF_DIR
    # PDB_DIR - where the .pdb files to process are
    # RF_DIR - where the .resfiles are
    environ['PDB_DIR'] = './input'
    environ['RF_DIR'] = './resfiles'

    pdb_house = environ.get('PDB_DIR')
    resfile_house = environ.get('RF_DIR')

    # chain to saturate
    chain_to_mutate = pirate_noises.chain

    # reference .pdb files to create a saturation dictionary
    ref_pdb = path.join(pdb_house, "frame0.pdb")
    ref_string = parse_pdb(ref_pdb)
    ref_pose = string_to_pose(ref_string)

    # get the naming extension based on the number of chains present
    _, chain_extension = get_functional_state(ref_pose)

    # reverse engineer the chain number based on chain letter passed
    chain_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6}
    chain_num = chain_mapping[chain_to_mutate]

    # get the AA sequence of the chain to saturate
    sequence = ref_pose.chain_sequence(chain_num)

    # write and save a .seq file
    with open(f'{output_name}_{chain_extension}.seq', 'w') as f:
        f.write(str(sequence))

    # the name of the output file to later save after processing mutations
    output_data_array = f'{output_name}_{chain_extension}.npz'

    # processing variables
    block_size = pirate_noises.block_size

    # create the dictionary of mutations to perform
    ssm_mutations = create_ssm_dict(ref_pose, chain_to_mutate)

    # main function execution
    if pirate_noises.minimize:
        minimize = True
    else:
        minimize = False

    process_ensemble(pdb_house, resfile_house, ssm_mutations, block_size, minimize)

    # concatenate arrays into list
    files = [f for f in listdir(getcwd()) if fm.fnmatch(f, '????.npz')]

    # num_files is by default the number of files returned by ls *.pdb
    if pirate_noises.num_files:

        # only compile the array once the job is finished
        if len(files) == pirate_noises.num_files:
            compile_array(getcwd(), f'{output_data_array}')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='In silico mutagenesis with MDR')

    # misc and packing control
    parser.add_argument("-n", '--name', default='', type=str, help='The name of the simulation, kind of arbitrary')
    parser.add_argument('--minimize', action='store_true', default=False, help='Perform sidechain minimization')

    # scoring control
    parser.add_argument('--beta', action='store_true', default=False, help='Pass if you want to use the beta_nov16 score function instead of ref2015')
    parser.add_argument('--beta-gen-pot', action='store_true', default=False, help='Pass if you want to use the most recent beta score function')
    parser.add_argument('--talaris', action='store_true', default=False, help='Pass if you want to use the talaris2013 score function')
    parser.add_argument("--chain", default='A', type=str, help='The chain to be mutated.')
    parser.add_argument('--soft-rep', action='store_true', default=False, help='Pass to dampen repulsive terms. Useful for cases with limited backbone flexibility')

    # output control
    parser.add_argument('--block-size', default=50, type=int, help='The batch size to use')
    parser.add_argument('--num-files', type=int, default=501, help='The number of files to process')

    # output verbosity control
    parser.add_argument('--fatal', action='store_true', default=False, help='Log fatal errors only')
    parser.add_argument('--error', action="store_true", default=False, help='Log errors and below')
    parser.add_argument('--warning', action="store_true", default=False, help='Log warnings and below')
    parser.add_argument('--info', action="store_true", default=False, help='Log info and below')
    parser.add_argument('--debug', action="store_true", default=False, help='Log debug info and below')
    parser.add_argument('--trace', action="store_true", default=False, help='Log everything that is output. Will make HUGE files')

    args = parser.parse_args()

    main(args)
