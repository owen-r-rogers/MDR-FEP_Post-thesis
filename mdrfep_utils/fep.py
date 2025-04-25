import pandas as pd
import numpy as np
import seaborn as sns
import pandas.testing as pdt
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from pyrosetta import *
from pyrosetta.rosetta import *

pd.set_option('future.no_silent_downcasting', True)

init()

"""
Grid-search of MDR-FEP data.

For optimizing over ALL proteins, use the following directory setup:

    [(softrep or hardrep)]__[(min or nomin)]__[(5 or 15)]   

Where each directory has the .npz files for both the monomer and dimer of EACH minibinder. 
    
"""

# This dictionary is mostly for cosmetics, it's referenced for the titles of the graphs since the design names are super long
protein_dict = {
    'bcov_v3_r3_ems_3hC_436_0002_000000017_0001_0001_47_64_H_.._ems_p1-15H-GBL-16H-GABBL-16H_0382_0001_0001_0001_0001_0001_0001_0001_0001': 'IL-7ra',
    'Motif1400_ems_3hM_482_0001_7396_0001': 'FGFR2',
    'ems_3hC_1642_000000001_0001': 'VirB8',
    'longxing_CationpiFixed_HHH_eva_0229_000000001_0001_0001': 'TrkA',
    'NewR1_ems_ferrM_2623_0002_000000011_0001_0001_0004_crosslinked_1': 'CD3_delta'
}

protein_list = list(protein_dict.keys())


def parse_exp_data(affinity_estimate, protein, protein_dimer_pdb, protein_seq):
    """
    Function to process the experimental data from Cao et al. (2022), https://doi.org/10.1038/s41586-022-04654-9.
    :param affinity_estimate: ssm_correlation_for_plotting.sc output file from supp data.
    :param validation: validation_score.sc output file from supp data.
    :param protein: The string of the design name.
    :param protein_dimer_pdb: .pdb of the protein as it appears in supp data.
    :param protein_seq: .seq file of the protein sequence in FASTA format.
    :return Dataframe of parsed experimental data.
    """

    # read experimental data files
    ssm_correlation_for_plotting = pd.read_csv(affinity_estimate, sep=r'\s+')

    # create an experimental dataframe
    exp_df = ssm_correlation_for_plotting[ssm_correlation_for_plotting['ssm_parent'] == protein].copy()

    # create and store a sequence dictionary
    seq_dict = {}

    with open(protein_seq, 'r') as f:
        for line in f:

            print(f'{protein} sequence: {line}', flush=True)

            seq_dict[protein] = line

    # add a column for parent_letter
    parent_letters = []

    sequence = seq_dict.get(protein)
    for _, row in exp_df.iterrows():

        wt_aa = sequence[row['ssm_seqpos'] - 1]

        parent_letters.append(wt_aa)

    exp_df.loc[:, 'parent_letter'] = parent_letters

    # filter the dataframe based on the sequence dictionary
    # (remove wild-type residues from mutation dataset)
    filtered_rows = []

    for _, row in exp_df.iterrows():

        wt_aa = sequence[row['ssm_seqpos'] - 1]
        if row['ssm_letter'] != wt_aa:
            filtered_rows.append(row)

    # create new dataframe
    exp_data = pd.DataFrame(filtered_rows)
    exp_data = exp_data.reset_index(drop=True)

    # add an 'is_loop' column for later fitting
    pose = pose_from_file(protein_dimer_pdb)

    print(f'Pose object created from {protein}:', flush=True)
    print(pose, flush=True)

    monomer = pose.split_by_chain()[1]
    dssp = f'x{core.scoring.dssp.Dssp(monomer).get_dssp_secstruct()}'

    is_loops = []

    for _, row in exp_data.iterrows():
        secstruct = dssp[row['ssm_seqpos']]
        is_loop = (secstruct == 'L')
        is_loops.append(is_loop)

    exp_data.loc[:, 'is_loop'] = is_loops

    # Motif1400... has slightly different experimental data (no mutations to C)
    if protein != 'Motif1400_ems_3hM_482_0001_7396_0001':
        for seqpos, seqpos_group in exp_data.groupby('ssm_seqpos'):
            assert len(seqpos_group) == 19, 'There are more than the allowed amino acid mutations in one or more seqpos groups'

    # remove unnecessary columns
    # actually don't do this, need to compare to their rosetta data
    # exp_data.drop(columns=['delta_rosetta_lb', 'delta_rosetta_ub'], inplace=True)

    print(f'Structure of experimental data for {protein}:', flush=True)
    print(exp_data.shape, flush=True)

    return exp_data


def zwanzig(beta, array):
    """
    Applies FEP using the Zwanzig equation over an array for a single mutation.
    :param beta: (pseudo) inverse temperature factor
    :param array: ∆Energy values from Roseta.
    :return: ∆G (REU) point average from array.
    """
    arr = np.array(array, dtype='float64')

    delta_g = (-1 / beta) * np.log(np.mean(np.exp(-beta * arr)))

    return delta_g


def parse_rosetta_data(dimer_npz, monomer_npz, protein):
    """
    Process data from MDR-FEP .npz files.
    :param dimer_npz: .npz file from running MDR on the dimer form.
    :param monomer_npz: .npz file from running MDR on the monomer.
    :param protein: SSM parent string
    :return Dataframe of predicted data.
    """

    print(f'Processing files for {protein}:', flush=True)
    print(f'Dimer - {dimer_npz}', flush=True)
    print(f'Monomer - {monomer_npz}', flush=True)

    # read data
    dimer_data = np.load(dimer_npz)
    monomer_data = np.load(monomer_npz)

    # create a dimer and monomer dataframe
    dim = []
    for mut in dimer_data.files:
        dim_data = {'ssm_parent': protein,
                    'native_aa': str(mut[0]),
                    'ssm_seqpos': int(mut[1:-1]),
                    'ssm_letter': f'{mut[-1]}',
                    'dimer_de': dimer_data[mut]}
        dim.append(dim_data)
    dimer_df = pd.DataFrame(dim)

    mon = []
    for mut in monomer_data.files:
        mon_data = {'ssm_parent': protein,
                    'native_aa': str(mut[0]),
                    'ssm_seqpos': int(mut[1:-1]),
                    'ssm_letter': f'{mut[-1]}',
                    'monomer_de': monomer_data[mut]}
        mon.append(mon_data)
    monomer_df = pd.DataFrame(mon)

    # concatenate the data
    ros_df = monomer_df.merge(dimer_df, 'left', ['ssm_seqpos', 'ssm_letter', 'ssm_parent', 'native_aa'])
    ros_df = ros_df[['dimer_de', 'monomer_de', 'ssm_seqpos', 'ssm_letter', 'ssm_parent', 'native_aa']]

    return ros_df


def combine_dfs(rosetta_df, experimental_df):
    """
    Function to merge the predicted and experimental data.
    :param rosetta_df: parse_rosetta_data() output
    :param experimental_df: parse_exp_data() output
    :return: A (filtered) dataframe containing all the predicted and experimental data.
    """

    all_df = rosetta_df.merge(experimental_df, 'left', ['ssm_seqpos', 'ssm_letter', 'ssm_parent'])

    # had the issue where this column wasn't being processed as a boolean!!!!
    all_df['is_loop'] = all_df['is_loop'].astype(bool)
    all_df['low_conf'] = all_df['low_conf'].astype(bool)
    all_df['avid_doesnt_agree'] = all_df['avid_doesnt_agree'].astype(bool)

    return all_df


def produce_delta_g(filt_df, beta):
    """
    Takes in the output of combine_dfs() and applies the Zwanzig equation over the monomer and dimer ∆E (Rosetta) values.
    :param filt_df: Output of combine_dfs()
    :param beta: (pseudo) inverse temperature factor
    :return: Dataframe containing FEP output.
    """

    dimer_arrays = [np.array(filt_df['dimer_de'].iloc[i]) for i in range(0, len(filt_df))]
    monomer_arrays = [np.array(filt_df['monomer_de'].iloc[i]) for i in range(0, len(filt_df))]

    dimer_dgs = [zwanzig(beta, array) for array in dimer_arrays]
    monomer_dgs = [zwanzig(beta, array) for array in monomer_arrays]

    dg_df = filt_df.copy()

    dg_df.loc[:, 'dimer_dg'] = dimer_dgs
    dg_df.loc[:, 'monomer_dg'] = monomer_dgs

    dg_df['ddg'] = dg_df['dimer_dg'] - dg_df['monomer_dg']

    return dg_df


def dg_fold_to_p_fold(dg_fold):
    """
    Robbed straight from the supplementary data of Cao et al. (2022)
    :param dg_fold: The (estimated) free energy of folding for the WT monomer.
    :return sigmoid(dg_fold) --> probability of folding.
    """

    p_fold = 1 / (1 + np.exp(np.clip(dg_fold / 0.6, -500, 500)))
    return p_fold


def p_fold_effect(p_ref, p_new, fit_point=0.125):
    """
    Robbed straight from the supplementary data of Cao et al. (2022)
    :param p_ref: The WT probability of folding
    :param p_new: The probability of folding after mutating to X.
    :param fit_point: Derived value for the fraction of miniprotein bound in solution.
    :return Essentially a KD multiplier when you perturb P(fold) in some way.
    """

    p_fold_eff = p_ref / p_new * (1 - fit_point * p_ref) / (1 - fit_point * p_new)
    return p_fold_eff


def ddg_monomer_from_dg_fold_and_delta_e(dg_fold, delta_g_monomer):
    """
    Robbed straight from the supplementary data of Cao et al. (2022)
    :param dg_fold: Estimated free energy of WT folding.
    :param delta_g_monomer: Zwanzig(∆E_monomer) output
    :return P(fold)_effect when you perturb the monomer with X mutation.
    """

    p_fold = dg_fold_to_p_fold(dg_fold)

    new_p_fold = dg_fold_to_p_fold(dg_fold + delta_g_monomer)

    ddg_monomer = np.log(p_fold_effect(p_fold, new_p_fold)) * 0.6

    return ddg_monomer


def filter_for_dg_fold_fitting(df):
    """
    Filter the dataframe before fitting with dg_fold.
    :param df: Output of combine_dfs()
    :return: Filtered dataframe.
    """
    try:
        df['is_loop'] = df['is_loop'].astype(bool)
        df['low_conf'] = df['low_conf'].astype(bool)
        df['avid_doesnt_agree'] = df['avid_doesnt_agree'].astype(bool)

        dg_fold_fitting_df = df[(df['is_monomer_core'] | df['is_monomer_boundary'] | df['is_monomer_surface'])
                                & ((df['ssm_letter'] != 'P') & (df['ssm_letter'] != 'C') & (df['ssm_letter'] != 'G'))
                                & ((df['parent_letter'] != 'P') & (df['parent_letter'] != 'C') & (df['parent_letter'] != 'G'))
                                & (~df['is_loop'])
                                & (~df['delta_exp_ddg_ub'].isnull() & ~df['delta_exp_ddg_lb'].isnull())
                                & (np.isfinite(df['delta_exp_ddg_ub']) & np.isfinite(df['delta_exp_ddg_lb']))
                                & (~df['low_conf'] & ~df['avid_doesnt_agree'])
                                ]

        return dg_fold_fitting_df

    except ValueError as e:
        print(f'Error {e}', flush=True)
        return None


def fit_dg_fold(mdrfep_dg_monomer_values, y_lb, y_ub, dg_fold_ci_width_avg_dev=0.25, steps=1000, dg_fold_lower_clip=-10,
                dg_fold_upper_clip=10):
    """
    Use least-squares regression and a grid search to estimate the free energy of folding.
    :param mdrfep_dg_monomer_values: Array of Zwanzig(∆E array) output
    :param y_lb: Experimental lower bound for binding affinity change.
    :param y_ub: Experimental upper bound for binding affinity change.
    :param dg_fold_ci_width_avg_dev: Buffer essentially
    :param steps: Number of fold fitting steps to perform
    :param dg_fold_lower_clip: Lower bound for ∆G_fold
    :param dg_fold_upper_clip: Upper bound for ∆G_fold
    :return: ∆G_fold (center, lower, and upper bounds).
    """

    # create array of dg_folds to try for the grid search
    dg_fold = np.linspace(dg_fold_lower_clip, dg_fold_upper_clip, steps)

    # calculate all the possible ∆∆G_monomer values you could see
    ddg_monomer = ddg_monomer_from_dg_fold_and_delta_e(dg_fold[None, :], mdrfep_dg_monomer_values[:, None])

    # Mark areas where the ddg_monomer is outside the confidence interval
    data_is_too_high = ddg_monomer < y_lb[:, None]
    data_is_too_low = ddg_monomer > y_ub[:, None]

    # drop out stabilizing mutations
    data_valid = y_ub > 0

    # mark which scores to keep.
    # keep the scores that fall outside the confidence intervals that are valid,
    # this way you can perform the fit on data that actually needs fitting
    keep_scores = (data_is_too_high | data_is_too_low) & data_valid[:, None]

    # make a reference values matrix
    # empty at first
    lookup_vals = np.zeros(ddg_monomer.shape)

    # broadcast mutational values so that they are repeated across each row (which represents one mutation)
    lookup_vals[data_is_too_high] = np.repeat(y_lb, steps).reshape(len(y_lb), steps)[data_is_too_high]
    lookup_vals[data_is_too_low] = np.repeat(y_ub, steps).reshape(len(y_ub), steps)[data_is_too_low]

    # finally, set invalid mutations to 0
    lookup_vals[~data_valid] = 0

    # initialize matrix containing the residuals (from the ideal ∆∆G_monomer vs (∆G_fold + ∆E) curve)
    delta = np.zeros(ddg_monomer.shape)

    # create calibration curves to find the ∆G_fold
    x_steps = 800
    standard_dg_monomers = np.linspace(mdrfep_dg_monomer_values.min(), mdrfep_dg_monomer_values.max(), x_steps)

    # create a matrix storing all combinations of (ordered) predicted mutation values and ∆G_fold values to try
    # this is essentially the transpose of the lookup_vals except the amount of mutational values are tweaked
    # such that they span the range of min, max and are not the ACTUAL mutational values besides min, max
    standard_ddg_monomer_curves = ddg_monomer_from_dg_fold_and_delta_e(dg_fold[:, None], standard_dg_monomers[None, :])

    # now iterate through and add to delta (essentially an empty grid filled during a grid search)
    for i_curve in range(len(standard_ddg_monomer_curves)):
        curve = standard_ddg_monomer_curves[i_curve]

        # search left and right for the closest point
        hits = np.searchsorted(curve, lookup_vals[:, i_curve]).clip(0, len(curve) - 1)
        x_hits = standard_dg_monomers[hits]

        # calculate the delta value
        # for each actual ∆G_monomer value, how far is it from the values that fit into the standard curve?
        offsets = (mdrfep_dg_monomer_values - x_hits)

        # take the horizontal slice results and put it into
        # delta as a vertical slice
        delta[:, i_curve] = offsets

    # set values where the score is invalid (stabilizing mutations) to 0
    delta[~keep_scores] = 0

    # RMSE
    scores = np.sqrt(
        np.sum(
            np.square(delta), axis=0
        ) / len(mdrfep_dg_monomer_values)
    )

    # You want the confidence interval to cover all scores within dg_fold_ci_width_avg_dev of the best
    for_conf = scores - scores.min() - dg_fold_ci_width_avg_dev

    # produce center and lower and upper bounds
    center_i = np.argmin(scores)
    lb_i = (for_conf < 0).argmax(axis=-1)
    ub_i = np.cumsum((for_conf < 0).astype(int), axis=-1).argmax(axis=-1)

    if np.all(for_conf < 0):
        lb_i = 0
        ub_i = len(for_conf) - 1
        center_i = ub_i // 2

    return dg_fold[center_i], dg_fold[lb_i], dg_fold[ub_i]


cats = ['is_interface_core', 'is_interface_boundary', 'is_monomer_core',
        'is_monomer_boundary', 'is_monomer_surface']


def plot_results(plotting_df, file_name, protein_title_name, ideal_beta, max_corr):
    """
    Function to plot the output of the grid_search().
    """

    # create figure
    f, a = plt.subplots(figsize=(10, 10), dpi=300)
    alpha = 0.9

    # plot a smooth distribution for the background
    sns.kdeplot(x=plotting_df['delta_exp_ddg_center'], y=plotting_df['ddg_rosetta'], alpha=alpha / 3, ax=a, fill=True, color='red', levels=20)

    # plot scatterplot of raw data
    sns.scatterplot(x=plotting_df['delta_exp_ddg_center'], y=plotting_df['ddg_rosetta'], alpha=alpha, ax=a, edgecolor='black', **{'facecolor': 'none'})

    # make borders thicker
    a.spines['top'].set_linewidth(1.5)
    a.spines['right'].set_linewidth(1.5)
    a.spines['bottom'].set_linewidth(1.5)
    a.spines['left'].set_linewidth(1.5)

    # plot the lines at x, y = 0
    a.axhline(y=0, linestyle='-', color='black')
    a.axvline(x=0, linestyle='-', color='black')

    # plot a line of best fit
    x_min, x_max = plotting_df['delta_exp_ddg_center'].min(), plotting_df['delta_exp_ddg_center'].max()
    x_min -= 2
    x_max += 2
    x = np.linspace(x_min, x_max, len(plotting_df))

    try:
        fit = np.poly1d(np.polyfit(plotting_df['delta_exp_ddg_center'], plotting_df['ddg_rosetta'], 1))
        a.plot(x, fit(x), linestyle='--', color='blue', label=max_corr)

    except np.linalg.LinAlgError as e:
        print(e)

    a.set_xlim(x_min, x_max)

    # make the plot look nice
    plt.legend()
    plt.title(f'{protein_title_name}, beta = {ideal_beta}', fontsize=24)
    plt.xlabel(r'$∆∆G_{\text{Experimental}}$', fontsize=24)
    plt.ylabel(r'$∆∆E_{\text{MDRFEP}}$', fontsize=24)
    plt.grid(False)
    plt.savefig(f'{file_name}.png', dpi=300, transparent=True)
    plt.close()


def grid_search(all_protein_df, conditions_name, beta_lb=0, beta_ub=0.1, beta_step=0.0001, metric_of_interest='correlation'):
    """
    Performs a 1-D grid search over all specified values of Beta.
    :param all_protein_df: pd.DataFrame object containing MDR-FEP and experimental data for all miniprotein binders
    :param conditions_name: The name of the Rosetta conditions that produced the MDR-FEP values, dictates the name of the output files
    :param beta_lb: Lower bound of the Beta parameter
    :param beta_ub: Upper bound of the Beta parameter
    :param beta_step: dBeta over the course of the grid search
    :param metric_of_interest: This will search through the complete results of the grid search and find the Beta value that produced the best value of this
    :return: A dataframe containing (mostly) complete results. I say mostly only because this outputs so many files as is, so usually I don't do anything with this dataframe
    """
    # save all_protein_df for later plotting
    all_protein_df.to_csv(f'all_data_{conditions_name}.csv', index=False)

    # big, big dictionary
    correlations = {
        'beta': [],

        # overall correlation
        'correlation': [],
        'correlation_intcore': [],
        'correlation_intbound': [],
        'correlation_moncore': [],
        'correlation_monbound': [],
        'correlation_monsurf': [],
        'correlation_bcov': [],
        'correlation_motif': [],
        'correlation_ems': [],
        'correlation_longxing': [],
        'correlation_newr1': [],
        'accuracy': [],
        'accuracy_intcore': [],
        'accuracy_intbound': [],
        'accuracy_moncore': [],
        'accuracy_monbound': [],
        'accuracy_monsurf': [],
        'accuracy_bcov': [],
        'accuracy_motif': [],
        'accuracy_ems': [],
        'accuracy_longxing': [],
        'accuracy_newr1': []
    }

    for beta in np.arange(beta_lb, beta_ub + beta_step, beta_step):

        if beta == 0:
            beta += 1e-9

        print(f'Fitting (beta of {beta})', flush=True)

        # create a dataframe storing output of zwanzig equation
        dg_df = produce_delta_g(all_protein_df, beta)
        print(f'Length of dg_df before fitting ∆G_fold:')
        print(len(dg_df))

        # filter to feed into ∆G_fold fitting
        dg_folds = []
        for name, subdf in dg_df.groupby('ssm_parent'):

            for_dg_fold = filter_for_dg_fold_fitting(subdf)

            # create inputs for ∆G_fold fitting
            monomer_dgs = for_dg_fold['monomer_dg'].values
            exp_ddg_lbs = for_dg_fold['delta_exp_ddg_lb'].values
            exp_ddg_ubs = for_dg_fold['delta_exp_ddg_ub'].values

            # fit ∆G_fold
            dg_fold_center, dg_fold_lb, dg_fold_ub = fit_dg_fold(monomer_dgs, exp_ddg_lbs, exp_ddg_ubs)

            dg_folds.append(
                {'ssm_parent': name, 'dg_fold': dg_fold_center, 'dg_fold_lb': dg_fold_lb, 'dg_fold_ub': dg_fold_ub})

        fold_df = pd.DataFrame(dg_folds)

        df = dg_df.merge(fold_df, 'inner', 'ssm_parent')
        print(f'Length of dg_df after fitting ∆G_fold:')
        print(len(df))

        # now calculate rosetta's ability to predict mutations
        to_assess = df.copy()

        # work around the bug
        to_assess['low_conf'] = to_assess['low_conf'].astype(bool)
        to_assess['avid_doesnt_agree'] = to_assess['avid_doesnt_agree'].astype(bool)
        to_assess['is_loop'] = to_assess['is_loop'].astype(bool)

        # do some filtering to mimic how they filtered
        print(f'Length of to_assess before filtering: {len(to_assess)}')
        to_assess = to_assess[(True & (to_assess['ssm_letter'] != 'C')
                                   & (to_assess['ssm_letter'] != 'P')
                                   & (to_assess['parent_letter'] != 'C')
                                   & (~to_assess['low_conf'] & ~to_assess['avid_doesnt_agree'])
                                   & ~(to_assess['delta_exp_ddg_lb'].isnull() & to_assess['delta_exp_ddg_ub'].isnull())
                                   & (np.isfinite(to_assess['delta_exp_ddg_ub']) | np.isfinite(to_assess['delta_exp_ddg_lb']))
                                   & (~to_assess['is_loop'])
                                   )]
        print(f'Length of to_assess after filtering: {len(to_assess)}')

        to_assess = to_assess[['parent_letter', 'ssm_parent', 'dg_fold', 'dg_fold_lb', 'dg_fold_ub', 'monomer_dg', 'dimer_dg', 'ddg',
                                'delta_exp_ddg_lb', 'delta_exp_ddg_center', 'delta_exp_ddg_ub', 'ssm_seqpos', 'ssm_letter', 'delta_rosetta_lb', 'delta_rosetta_ub']
                                + cats].copy()

        # fill to_assess df with ∆∆G_rosetta values
        to_assess['ddg_monomer'] = ddg_monomer_from_dg_fold_and_delta_e(to_assess['dg_fold'], to_assess['monomer_dg'])
        to_assess['ddg_rosetta'] = to_assess['ddg'] + to_assess['ddg_monomer'].clip(0, None)

        # ignore mutations to C because they don't have experimental data
        to_correlate = to_assess[~(to_assess['ssm_letter'] == 'C')]
        corr = pearsonr(to_correlate['ddg_rosetta'], to_correlate['delta_exp_ddg_center'])

        print(f'Correlation with beta {beta}:', flush=True)
        print(corr.statistic, flush=True)

        correlations['beta'].append(beta)
        correlations['correlation'].append(corr.statistic)

        # now go through and get correlation data for each REGION, and then each protein
        intcore = to_correlate[to_correlate['is_interface_core'] == True]
        intcore_corr = pearsonr(intcore['ddg_rosetta'], intcore['delta_exp_ddg_center'])
        correlations['correlation_intcore'].append(intcore_corr.statistic)

        intbound = to_correlate[to_correlate['is_interface_boundary'] == True]
        intbound_corr = pearsonr(intbound['ddg_rosetta'], intbound['delta_exp_ddg_center'])
        correlations['correlation_intbound'].append(intbound_corr.statistic)

        moncore = to_correlate[to_correlate['is_monomer_core'] == True]
        moncore_corr = pearsonr(moncore['ddg_rosetta'], moncore['delta_exp_ddg_center'])
        correlations['correlation_moncore'].append(moncore_corr.statistic)

        monbound = to_correlate[to_correlate['is_monomer_boundary'] == True]
        monbound_corr = pearsonr(monbound['ddg_rosetta'], monbound['delta_exp_ddg_center'])
        correlations['correlation_monbound'].append(monbound_corr.statistic)

        monsurf = to_correlate[to_correlate['is_monomer_surface'] == True]
        monsurf_corr = pearsonr(monsurf['ddg_rosetta'], monsurf['delta_exp_ddg_center'])
        correlations['correlation_monsurf'].append(monsurf_corr.statistic)

        # now go through and get correlation data for each PROTEIN
        bcov_df = to_correlate[to_correlate['ssm_parent'] == protein_list[0]]
        bcov_corr = pearsonr(bcov_df['ddg_rosetta'], bcov_df['delta_exp_ddg_center'])
        correlations['correlation_bcov'].append(bcov_corr.statistic)

        motif_df = to_correlate[to_correlate['ssm_parent'] == protein_list[1]]
        motif_corr = pearsonr(motif_df['ddg_rosetta'], motif_df['delta_exp_ddg_center'])
        correlations['correlation_motif'].append(motif_corr.statistic)

        ems_df = to_correlate[to_correlate['ssm_parent'] == protein_list[2]]
        ems_corr = pearsonr(ems_df['ddg_rosetta'], ems_df['delta_exp_ddg_center'])
        correlations['correlation_ems'].append(ems_corr.statistic)

        longxing_df = to_correlate[to_correlate['ssm_parent'] == protein_list[3]]
        longxing_corr = pearsonr(longxing_df['ddg_rosetta'], longxing_df['delta_exp_ddg_center'])
        correlations['correlation_longxing'].append(longxing_corr.statistic)

        newr1_df = to_correlate[to_correlate['ssm_parent'] == protein_list[4]]
        newr1_corr = pearsonr(newr1_df['ddg_rosetta'], newr1_df['delta_exp_ddg_center'])
        correlations['correlation_newr1'].append(newr1_corr.statistic)

        # get ddg_mdrfep upper and lower bounds
        data = {'ssm_parent': [],
                'ssm_letter': [],
                'ssm_seqpos': [],
                'ddg_mdrfep_lb': [],
                'ddg_mdrfep_ub': []}

        for name, subdf in to_correlate.groupby('ssm_parent'):
            dg_fold_lin = np.linspace(subdf['dg_fold_lb'].iloc[0], subdf['dg_fold_ub'].iloc[0], 100)

            monomer_dgs = subdf['monomer_dg'].values

            ddg_bind_monomer = ddg_monomer_from_dg_fold_and_delta_e(dg_fold_lin[None, :], monomer_dgs[:, None])

            ddg_monomer_lb = np.min(ddg_bind_monomer, axis=-1)
            ddg_monomer_ub = np.max(ddg_bind_monomer, axis=-1)

            interface_effect = subdf['ddg'].values

            ddg_mdrfep_lb = ddg_monomer_lb + interface_effect
            ddg_mdrfep_ub = ddg_monomer_ub + interface_effect

            data['ssm_parent'].extend(subdf['ssm_parent'].values)
            data['ssm_letter'].extend(subdf['ssm_letter'].values)
            data['ssm_seqpos'].extend(subdf['ssm_seqpos'].values)
            data['ddg_mdrfep_lb'].extend(ddg_mdrfep_lb)
            data['ddg_mdrfep_ub'].extend(ddg_mdrfep_ub)

        bounds_data = pd.DataFrame(data)

        # now you have a dataframe that will hold all the data
        complete_df = to_correlate.merge(bounds_data, 'left', ['ssm_parent', 'ssm_letter', 'ssm_seqpos'])

        # fill it with data on whether it agrees or not
        my_within_bounds = ((complete_df['ddg_mdrfep_ub'].values + 1 > complete_df['delta_exp_ddg_lb'].values)
                            | np.isnan(complete_df['delta_exp_ddg_lb'].values))
        my_within_bounds &= ((complete_df['ddg_mdrfep_lb'] - 1 < complete_df['delta_exp_ddg_ub'].values)
                             | np.isnan(complete_df['delta_exp_ddg_ub'].values))
        complete_df['mdrfep_agrees'] = my_within_bounds

        # add to dictionary
        correlations['accuracy'].append(np.sum(complete_df['mdrfep_agrees']) / len(complete_df))

        correlations['accuracy_intcore'].append(np.sum(complete_df[complete_df['is_interface_core'] == True]['mdrfep_agrees']) / len(complete_df[complete_df['is_interface_core'] == True]))

        correlations['accuracy_intbound'].append(np.sum(complete_df[complete_df['is_interface_boundary'] == True]['mdrfep_agrees']) / len(complete_df[complete_df['is_interface_boundary'] == True]))

        correlations['accuracy_moncore'].append(np.sum(complete_df[complete_df['is_monomer_core'] == True]['mdrfep_agrees']) / len(complete_df[complete_df['is_monomer_core'] == True]))

        correlations['accuracy_monbound'].append(np.sum(complete_df[complete_df['is_monomer_boundary'] == True]['mdrfep_agrees']) / len(complete_df[complete_df['is_monomer_boundary'] == True]))

        correlations['accuracy_monsurf'].append(np.sum(complete_df[complete_df['is_monomer_surface'] == True]['mdrfep_agrees']) / len(complete_df[complete_df['is_monomer_surface'] == True]))

        correlations['accuracy_bcov'].append(np.sum(complete_df[complete_df['ssm_parent'] == protein_list[0]]['mdrfep_agrees']) / len(complete_df[complete_df['ssm_parent'] == protein_list[0]]))

        correlations['accuracy_motif'].append(np.sum(complete_df[complete_df['ssm_parent'] == protein_list[1]]['mdrfep_agrees']) / len(complete_df[complete_df['ssm_parent'] == protein_list[1]]))

        correlations['accuracy_ems'].append(np.sum(complete_df[complete_df['ssm_parent'] == protein_list[2]]['mdrfep_agrees']) / len(complete_df[complete_df['ssm_parent'] == protein_list[2]]))

        correlations['accuracy_longxing'].append(np.sum(complete_df[complete_df['ssm_parent'] == protein_list[3]]['mdrfep_agrees']) / len(complete_df[complete_df['ssm_parent'] == protein_list[3]]))

        correlations['accuracy_newr1'].append(np.sum(complete_df[complete_df['ssm_parent'] == protein_list[4]]['mdrfep_agrees']) / len(complete_df[complete_df['ssm_parent'] == protein_list[4]]))

    corr_df = pd.DataFrame(correlations)
    corr_df.to_csv(f'correlations_{conditions_name}.sc', index=False)

    # go through the rest and save that plot
    ''' Go through one 'unit' of the above grid search with the ideal beta value '''

    max_corr_idx = np.argmax(corr_df[metric_of_interest])
    max_corr = corr_df.iloc[max_corr_idx][metric_of_interest]
    ideal_beta = corr_df.iloc[max_corr_idx]['beta']

    # using the ideal beta and frame_cutoff, make a dataframe containing MDR-FEP data
    delta_g_df = produce_delta_g(all_protein_df, ideal_beta)
    dg_fold = []

    for name, subdf in delta_g_df.groupby('ssm_parent'):
        for_dg_fold = filter_for_dg_fold_fitting(subdf)
        mon_dgs = for_dg_fold['monomer_dg'].values
        exp_ddg_l = for_dg_fold['delta_exp_ddg_lb'].values
        exp_ddg_u = for_dg_fold['delta_exp_ddg_ub'].values

        dgf_center, dgf_l, dgf_u = fit_dg_fold(mon_dgs, exp_ddg_l, exp_ddg_u)

        print(f'∆G_fold of {name} for ideal beta:')
        print(dgf_center)

        dg_fold.append(
            {
                'ssm_parent': name,
                'dg_fold': dgf_center,
                'dg_fold_lb': dgf_l,
                'dg_fold_ub': dgf_u
            }
        )

    dg_fold_df = pd.DataFrame(dg_fold)
    ideal_df = delta_g_df.merge(dg_fold_df, 'inner', 'ssm_parent')

    ideal_df = ideal_df[(True & (ideal_df['ssm_letter'] != 'C')
                             & (ideal_df['ssm_letter'] != 'P')
                             & (ideal_df['parent_letter'] != 'C')
                             & (~ideal_df['low_conf'] & ~ideal_df['avid_doesnt_agree'])
                             & ~(ideal_df['delta_exp_ddg_lb'].isnull() & ideal_df['delta_exp_ddg_ub'].isnull())
                             & (np.isfinite(ideal_df['delta_exp_ddg_ub']) | np.isfinite(ideal_df['delta_exp_ddg_lb']))
                             & (~ideal_df['is_loop'])
                             )]

    ideal_data = {'ssm_parent': [],
                'ssm_letter': [],
                'ssm_seqpos': [],
                'ddg_mdrfep_lb': [],
                'ddg_mdrfep_ub': []}

    for name, subdf in ideal_df.groupby('ssm_parent'):
        dg_fold_lin = np.linspace(subdf['dg_fold_lb'].iloc[0], subdf['dg_fold_ub'].iloc[0], 100)
        monomer_dgs = subdf['monomer_dg'].values
        ddg_bind_monomer = ddg_monomer_from_dg_fold_and_delta_e(dg_fold_lin[None, :], monomer_dgs[:, None])
        ddg_monomer_lb = np.min(ddg_bind_monomer, axis=-1)
        ddg_monomer_ub = np.max(ddg_bind_monomer, axis=-1)
        interface_effect = subdf['ddg'].values
        ddg_mdrfep_lb = ddg_monomer_lb + interface_effect
        ddg_mdrfep_ub = ddg_monomer_ub + interface_effect
        ideal_data['ssm_parent'].extend(subdf['ssm_parent'].values)
        ideal_data['ssm_letter'].extend(subdf['ssm_letter'].values)
        ideal_data['ssm_seqpos'].extend(subdf['ssm_seqpos'].values)
        ideal_data['ddg_mdrfep_lb'].extend(ddg_mdrfep_lb)
        ideal_data['ddg_mdrfep_ub'].extend(ddg_mdrfep_ub)

    ideal_bounds_data = pd.DataFrame(ideal_data)
    for_output = ideal_df.merge(ideal_bounds_data, 'left', ['ssm_parent', 'ssm_letter', 'ssm_seqpos'])
    mdrfep_agrees = ((for_output['ddg_mdrfep_ub'].values + 1 > for_output['delta_exp_ddg_lb'].values)
    | np.isnan(for_output['delta_exp_ddg_lb'].values))
    mdrfep_agrees &= ((for_output['ddg_mdrfep_lb'].values - 1 < for_output['delta_exp_ddg_ub'].values)
    | np.isnan(for_output['delta_exp_ddg_ub'].values))
    for_output['mdrfep_agrees'] = mdrfep_agrees

    for_output.to_csv(f'{conditions_name}_mdrfep_output.sc', index=False)

    to_plot = ideal_df.copy()
    to_plot = to_plot[
        ['ssm_parent', 'dg_fold', 'dg_fold_lb', 'dg_fold_ub', 'monomer_dg', 'dimer_dg', 'ddg', 'delta_exp_ddg_lb',
         'delta_exp_ddg_center', 'delta_exp_ddg_ub', 'ssm_seqpos', 'ssm_letter', 'delta_rosetta_lb',
         'delta_rosetta_ub'] + cats].copy()
    to_plot['ddg_monomer'] = ddg_monomer_from_dg_fold_and_delta_e(to_plot['dg_fold'], to_plot['monomer_dg'])
    to_plot['ddg_rosetta'] = to_plot['ddg'] + to_plot['ddg_monomer'].clip(0, None)

    # take out cysteines if not already done
    to_plot = to_plot[~(to_plot['ssm_letter'] == 'C')].copy()
    to_plot.to_csv(f'for_plotting_{conditions_name}.sc', index=False)

    intcore_to_plot = to_plot[to_plot['is_interface_core'] == True]
    intcore_corr = pearsonr(intcore_to_plot['ddg_rosetta'], intcore_to_plot['delta_exp_ddg_center'])
    plot_results(intcore_to_plot, f'{conditions_name}_intcore', f'Interface core', ideal_beta,
                     f'{intcore_corr.statistic}')

    intbound_to_plot = to_plot[to_plot['is_interface_boundary'] == True]
    intbound_corr = pearsonr(intbound_to_plot['ddg_rosetta'], intbound_to_plot['delta_exp_ddg_center'])
    plot_results(intbound_to_plot, f'{conditions_name}_intbound', f'Interface boundary', ideal_beta,
                     f'{intbound_corr.statistic}')

    moncore_to_plot = to_plot[to_plot['is_monomer_core'] == True]
    moncore_corr = pearsonr(moncore_to_plot['ddg_rosetta'], moncore_to_plot['delta_exp_ddg_center'])
    plot_results(moncore_to_plot, f'{conditions_name}_moncore', f'Monomer core', ideal_beta,
                     f'{moncore_corr.statistic}')

    monbound_to_plot = to_plot[to_plot['is_monomer_boundary'] == True]
    monbound_corr = pearsonr(monbound_to_plot['ddg_rosetta'], monbound_to_plot['delta_exp_ddg_center'])
    plot_results(monbound_to_plot, f'{conditions_name}_monbound', f'Monomer boundary', ideal_beta,
                     f'{monbound_corr.statistic}')

    monsurf_to_plot = to_plot[to_plot['is_monomer_surface'] == True]
    monsurf_corr = pearsonr(monsurf_to_plot['ddg_rosetta'], monsurf_to_plot['delta_exp_ddg_center'])
    plot_results(monsurf_to_plot, f'{conditions_name}_monsurf', f'Monomer surface', ideal_beta,
                     f'{monsurf_corr.statistic}')

    plotting_df = to_plot[['ddg_rosetta', 'delta_exp_ddg_center', 'delta_exp_ddg_lb', 'delta_exp_ddg_ub']].copy()

    plot_results(plotting_df, conditions_name, conditions_name, ideal_beta, max_corr)

    return corr_df
