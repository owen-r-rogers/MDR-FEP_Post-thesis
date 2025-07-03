import argparse
from mdrfep_utils.fep import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experimental-data', type=str, default='ssm_correlation_for_plotting.sc', help='The experimental data from Cao et al. (2022)')
    parser.add_argument('--beta-ub', type=float, default=1, help='The upper bound of the beta parameter')
    parser.add_argument('--beta-step', type=float, default=0.000001, help='The amount to increase the beta parameter by which each step of the grid search')

    args = parser.parse_args()

    # first, create a way to generate all 40 'conditions' of running MDRFEP
    scorefxns = ['hardrep', 'softrep']
    minimizations = ['min', 'nomin']
    repacking_radii = [5, 15]

    runs = []

    for scorefxn in scorefxns:
        for mini in minimizations:
            for rr in repacking_radii:
                run = f'{scorefxn}__{mini}__{rr}'
                runs.append(run)

    # get the conditions that have complete data
    names = []
    for name in runs:
        if len(os.listdir(f'./results/all_proteins/{name}')) == 10:
            names.append(name)

    # assign each run to a core
    task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID'))

    name = names[task_id]
    print(f'SLURM_ARRAY_TASK_ID number {task_id} is processing {name}', flush=True)

    # make sure that what you requested exists
    assert os.path.exists(f'./results/all_proteins/{name}'), f'The directory for {name} does not exist'

    # make sure that you have enough files in there
    assert len(os.listdir(f'./results/all_proteins/{name}')) == 10, f'You dont have data for all of the proteins in {name} run'

    files = os.listdir(f'./results/all_proteins/{name}')

    for file in files:
        protein = file.split('__')[0]
        assert protein in protein_list, f'You dont have {protein} in the list of proteins'

    # load all the predicted data
    all_data = []

    scorefxn = name.split('__')[0]
    minimizations = name.split('__')[1]
    repacking_radius = name.split('__')[2]

    for protein in protein_list:

        assert os.path.exists(f'./results/all_proteins/{name}/{protein}__{scorefxn}__{minimizations}__{repacking_radius}__dimer.npz'), f'{protein}__{scorefxn}__{minimizations}__{repacking_radius}__dimer.npz does not exist'

        assert os.path.exists(f'./results/all_proteins/{name}/{protein}__{scorefxn}__{minimizations}__{repacking_radius}__monomer.npz'), f'{protein}__{scorefxn}__{minimizations}__{repacking_radius}__monomer.npz does not exist'

        rosetta_df = parse_rosetta_data(f'./results/all_proteins/{name}/{protein}__{scorefxn}__{minimizations}__{repacking_radius}__dimer.npz', f'./results/all_proteins/{name}/{protein}__{scorefxn}__{minimizations}__{repacking_radius}__monomer.npz', protein)

        all_data.append(rosetta_df)

    all_rosetta_data = pd.concat(all_data)

    # load all the experimental data
    all_exp_data = []

    for protein in protein_list:

        assert os.path.exists(
            f'./natives/{protein}.pdb'), 'The protein {protein}.pdb does not exist or is incorrectly named'
        assert os.path.exists(
            f'./sequences/{protein}.seq'), 'The protein {protein}.seq does not exist or is incorrectly named'

        exp_df = parse_exp_data(args.experimental_data, protein, f'./natives/{protein}.pdb', f'./sequences/{protein}.seq')

        all_exp_data.append(exp_df)

    all_experimental_data = pd.concat(all_exp_data)

    all_data = combine_dfs(all_rosetta_data, all_experimental_data)

    # carry out the grid search
    _ = grid_search(all_data, name, beta_ub=args.beta_lb, beta_step=args.beta_step)
