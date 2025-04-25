import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pyrosetta
from matplotlib.colors import ListedColormap


protein_dict = {
    'bcov_v3_r3_ems_3hC_436_0002_000000017_0001_0001_47_64_H_.._ems_p1-15H-GBL-16H-GABBL-16H_0382_0001_0001_0001_0001_0001_0001_0001_0001': 'IL-7ra',
    'Motif1400_ems_3hM_482_0001_7396_0001': 'FGFR2',
    'ems_3hC_1642_000000001_0001': 'VirB8',
    'longxing_CationpiFixed_HHH_eva_0229_000000001_0001_0001': 'TrkA',
    'NewR1_ems_ferrM_2623_0002_000000011_0001_0001_0004_crosslinked_1': 'CD3_delta'
}


def get_sequence(pdb):
    # initialize pyrosetta
    pyrosetta.init()

    # create a pose
    pose = pyrosetta.pose_from_file(pdb)
    pose.update_residue_neighbors()

    # split the monomer
    monomer = pose.split_by_chain()[1]

    # get the sequence
    seq = monomer.sequence()

    return seq



"""

Plotting ∆G as a function of the distance from the interface. 
This will make two plots. One will be a scatterplot and one will be a bunch of lines
corresponding to a range of beta values that will show the proportion of mutations that are below a certain
threshold.

"""


# makes a list of distances so that you can see how energy changes as a function
# of distance from the interface
def make_distance_list(pdb):

    # initialize pyrosetta
    pyrosetta.init()

    # create a pose
    pose = pyrosetta.pose_from_file(pdb)
    pose.update_residue_neighbors()

    # split the monomer
    monomer = pose.split_by_chain()[1]

    # create interface residues list
    chain_a = pyrosetta.rosetta.core.select.residue_selector.ChainSelector('A')
    chain_b = pyrosetta.rosetta.core.select.residue_selector.ChainSelector('B')

    interface_on_a = pyrosetta.rosetta.core.select.residue_selector.NeighborhoodResidueSelector(chain_b, 10.0, False)
    interface_on_b = pyrosetta.rosetta.core.select.residue_selector.NeighborhoodResidueSelector(chain_a, 10.0, False)

    interface_by_vector = pyrosetta.rosetta.core.select.residue_selector.InterGroupInterfaceByVectorSelector(interface_on_a, interface_on_b)
    interface_by_vector.cb_dist_cut(5.5)
    interface_by_vector.vector_angle_cut(75)   # figure out what all of these do before thesis
    interface_by_vector.vector_dist_cut(9)

    interface_residues = interface_by_vector.get_residues(pose)
    interface_residues = [
        res for res in interface_residues if res <= monomer.total_residue()
    ]

    # create a distance matrix
    ca_xyz = np.array([monomer.residue(r).xyz('CA') for r in range(1, monomer.total_residue() + 1)])
    distance_matrix = np.linalg.norm(ca_xyz[:, None, :] - ca_xyz[None, :, :], axis=2)

    seq_dist_dict = {
        seqpos: min(distance_matrix[seqpos - 1, [r - 1 for r in interface_residues]])
        for seqpos in range(1, monomer.total_residue() + 1)
    }

    seq_dist_dict = dict(sorted(seq_dist_dict.items(), key=lambda item: item[1]))

    return distance_matrix, seq_dist_dict, interface_residues


def plot_dg_vs_interface(dataframe, seq_dist_dict, betas, colormap, scalar_mapping, threshold=-1, save=False):

    # define x axis
    x = np.linspace(
        min(list(seq_dist_dict.values())),
        max(list(seq_dist_dict.values())),
        len(seq_dist_dict)
    )

    # create figure and axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(25, 20), dpi=300)

    for i, beta in enumerate(betas):
        y = [
            np.mean((dataframe[dataframe['ssm_seqpos'] == seqpos][f'ddg_{beta}'] <= threshold))
            for seqpos in range(1, len(seq_dist_dict) + 1)
        ]

        means = [
            np.mean(dataframe[dataframe['ssm_seqpos'] == seqpos][f'ddg_{beta}'])
            for seqpos in range(1, len(seq_dist_dict) + 1)
        ]

        y = np.array(y)

        b, a = np.polyfit(x, y, 1)

        ax2.plot(x, a + b * x, color=colormap[i], lw=2.5, label=f'Beta {beta}')
        ax1.scatter(x, means, s=400, color=colormap[i], alpha=0.25)

    # titles and labels
    ax1.set_ylabel(f'∆∆G', fontsize=32)
    ax1.set_title('', fontsize=32, pad=10)

    ax2.set_xlabel('Distance from interface in Å', fontsize=28)
    ax2.set_ylabel(f'Proportion below {threshold} kJ per mol', fontsize=32)

    # Customize axis appearance
    linew = 3
    ax1.spines['right'].set_linewidth(linew)
    ax1.spines['top'].set_linewidth(linew)
    ax1.spines['left'].set_linewidth(linew)
    ax1.spines['bottom'].set_linewidth(linew)

    ax2.spines['right'].set_linewidth(linew)
    ax2.spines['top'].set_linewidth(linew)
    ax2.spines['left'].set_linewidth(linew)
    ax2.spines['bottom'].set_linewidth(linew)

    color = 'black'
    ax1.spines['right'].set_color(color)
    ax1.spines['top'].set_color(color)
    ax1.spines['left'].set_color(color)
    ax1.spines['bottom'].set_color(color)

    ax2.spines['right'].set_color(color)
    ax2.spines['top'].set_color(color)
    ax2.spines['left'].set_color(color)
    ax2.spines['bottom'].set_color(color)

    # Adjust tick parameters
    ax1.tick_params(axis='both', which='major', labelsize=28, pad=2)
    ax2.tick_params(axis='both', which='major', labelsize=28, pad=2)

    # turn off grid
    ax1.grid(False)
    ax2.grid(False)

    cbar = plt.colorbar(scalar_mapping, ax=[ax1, ax2], orientation='vertical', fraction=0.05, pad=.02)
    cbar.set_label('Beta values', fontsize=12)

    if save:
        fig.savefig(f'dg_vs_interface_{threshold}_cd3dsr.png', dpi=300, transparent=True)


"""

Creating a heatmap similar to Cao et al. 2022 (Nature).



"""


def interp(ddg):
    cmap_size = 100
    cmap_specials = 1
    seismic = plt.colormaps['seismic_r']
    cmap = np.zeros((cmap_size * 2 + cmap_specials, 4))
    cmap[:cmap_size * 2] = seismic(np.linspace(0, 1, cmap_size * 2))
    CMAP_BLACK = len(cmap) - 1
    cmap[CMAP_BLACK] = np.array([0, 0, 0, 1])

    low_value = -10
    high_value = 10

    interp_ddg = int(np.interp(ddg, [low_value, high_value], [0, cmap_size * 2 - 1]))

    return interp_ddg


def fill_heatmap(monomer_seq, dimer_seq, dimer_npz, monomer_npz, beta=0.1):
    """
    Megafxn
    """
    # color info because I can't figure it out
    cmap_size = 100
    cmap_specials = 1
    seismic = plt.colormaps['seismic_r']
    cmap = np.zeros((cmap_size * 2 + cmap_specials, 4))
    cmap[:cmap_size * 2] = seismic(np.linspace(0, 1, cmap_size * 2))
    CMAP_BLACK = len(cmap) - 1
    grey = 0.5
    cmap[CMAP_BLACK] = np.array([0, 0, 0, 1])
    colorbar = matplotlib.colors.ListedColormap(cmap)

    # extract aa sequences and make sure you ran MDR correctly.
    monomer_sequence = []
    with open(monomer_seq) as ms:
        for line in ms:
            strip = line.strip().split()
            monomer_sequence.append(strip[0])

    dimer_sequence = []
    with open(dimer_seq) as ds:
        for line in ds:
            strip = line.strip().split()
            dimer_sequence.append(strip[0])

    assert len(monomer_sequence) == len(dimer_sequence)
    ref_sequence = monomer_sequence[0]

    # load dimer and monomer npz file
    dimer = np.load(dimer_npz)
    monomer = np.load(monomer_npz)

    # helper functions
    def zwanzig_dg(mut, beta):
        de_dimer = dimer[mut]
        de_monomer = monomer[mut]

        dg_dimer = (-1 / beta) * np.log(np.mean(np.exp(-beta * de_dimer)))
        dg_monomer = (-1 / beta) * np.log(np.mean(np.exp(-beta * de_monomer)))

        ddg = dg_dimer - dg_monomer

        return ddg

    def zwanzig_per_mut(mut, beta):
        de_dimer = dimer[mut]
        de_monomer = monomer[mut]

        dg_dimer = (-1 / beta) * np.log(np.mean(np.exp(-beta * de_dimer)))
        dg_monomer = (-1 / beta) * np.log(np.mean(np.exp(-beta * de_monomer)))

        ddg = dg_dimer - dg_monomer

        return interp(ddg)

    def create_heatmap(design_seq):
        aa = ['C', 'P', 'G', 'A', 'V', 'I', 'M', 'L', 'F', 'Y', 'W', 'S', 'T', 'N', 'Q', 'D', 'E', 'R', 'K', 'H']
        heatmap = pd.DataFrame(np.full((20, len(design_seq)), np.nan), index=aa,
                               columns=[x[1] + str(x[0] + 1) for x in enumerate(design_seq)])
        label = pd.DataFrame(index=aa, columns=[x[1] + str(x[0] + 1) for x in enumerate(design_seq)])

        for col in heatmap.columns:
            orig_aa = col[0]
            heatmap.at[orig_aa, col] = CMAP_BLACK
            label.at[orig_aa, col] = orig_aa
        label.fillna("", inplace=True)

        return heatmap, label

    def make_dataframe(dict):
        df = pd.DataFrame.from_dict(dict, orient='index', columns=['color'])
        return df

    # create a heatmap and label
    heatmap, label = create_heatmap(ref_sequence)
    assert dimer.files == monomer.files

    # create a list of mutations and a dataframe of ddg colors
    mut_list = [mut for mut in dimer.files]
    ddg_color_data = {mut: zwanzig_per_mut(mut, beta) for mut in mut_list}
    ddg_color_df = make_dataframe(ddg_color_data)

    # also create a dataframe of raw values for use with the colorbar
    ddg_data = [zwanzig_dg(mut, beta) for mut in mut_list]

    # fill heatmap with the color values from the zwanzig function
    for seqpos in range(1, len(ref_sequence) + 1):
        orig_aa = ref_sequence[seqpos - 1]
        for idx, row in ddg_color_df.iterrows():
            mutated_seqpos = int(idx[1:-1])
            mutated_to = idx[-1]
            if seqpos == mutated_seqpos:
                heatmap.at[mutated_to, f'{orig_aa}{seqpos}'] = row['color']

    return heatmap, label, ddg_data


def plot_heatmap(heatmap, title, label, ddg_df, save_name=None):
    """
    Plots the heatmap and label objects that are output by the fill_heatmap() function.
    """

    # same color issue as above
    cmap_size = 100
    cmap_specials = 1
    seismic = plt.colormaps['seismic_r']
    cmap = np.zeros((cmap_size * 2 + cmap_specials, 4))
    cmap[:cmap_size * 2] = seismic(np.linspace(0, 1, cmap_size * 2))
    CMAP_BLACK = len(cmap) - 1
    cmap[CMAP_BLACK] = np.array([0, 0, 0, 1])

    # set fontscale
    sns.set(font_scale=1.0)

    # create fig and 2 axes
    f, (ax, ax2) = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={'width_ratios': [45, 1]}, dpi=300)

    g = sns.heatmap(heatmap, linewidths=0.5, fmt="", annot=label, annot_kws={"size": 15}, cmap=ListedColormap(cmap),
                    ax=ax, vmin=0, vmax=len(cmap) - 1, cbar=False)

    g.set_xticklabels(g.get_xmajorticklabels(), fontsize=12)
    g.set_yticklabels(g.get_ymajorticklabels(), fontsize=12)

    ax.set_title(f'Predicted change in binding affinity for {title}', fontsize=32)

    # add colormap
    norm = matplotlib.colors.Normalize(vmin=min(ddg_df), vmax=max(ddg_df))
    cbar = f.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=ListedColormap(cmap)), cax=ax2, orientation='vertical')
    cbar.set_label("$∆∆G_{bind}$ upon mutation", fontsize=16)

    f.tight_layout()

    if not save_name is None:
        os.makedirs("./figs", exist_ok=True)
        plt.savefig(f"./figs/{save_name}.png", dpi=300, facecolor='w', edgecolor='w', orientation='portrait',
                    bbox_inches='tight')

    plt.show()


