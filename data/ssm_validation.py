#!/usr/bin/env python

# A script that will tell you how likely it is that your protein design model
#  and your SSM data agree with each other.
# Specifically, it reports the p-value that you could have gotten your data by chance
#  It does this by shuffling your experimental data and recalculating the validation
#    metrics. Then it makes a normal distribution and finds the p-value of your data.

# From Cao 2021

# Brian Coventry 2021





import os
import sys
import pandas as pd
import xarray as xr
import numpy as np
import argparse
import scipy.stats
import warnings
import re





parser = argparse.ArgumentParser(description="")
parser.add_argument("pooled_counts.list", type=str, help="Same as estimate_affinity_from_ngs.py")
parser.add_argument("sorts.csv", type=str, help="Same as estimate_affinity_from_ngs.py")
parser.add_argument("affinity_estimate_out", type=str, help="The output from estimate_affinity_from_ngs.py. Your SSM mutations must be labeled"
                                                            +" such that they are <parent_name>__<seqpos>__<letter>. And the parent design should"
                                                            +" be labeled <parent_name>__native. See sorting_ngs_data/*_ssm/pooled_counts.list"
                                                            +" for examples.")
parser.add_argument("fast_cart_fastrelax_out", type=str, help="The scorefile from your fast_cart_fastrelax trajectories. The file should"
                                                            +" contain at least one output from everything that came from make_all_ssm_mutations.py"
                                                            +" If there are multiple outputs per mutation, the lowest scoring one is selected.")
parser.add_argument("parent_pdbs", type=str, nargs="+", help="The pdb files you wish to perform this analysis on. Name must coordinate with"
                                                            +" affinity_estimate_out and fast_cart_fastrelax_out.")

parser.add_argument("--random_copies", type=int, default="100", help="How many shuffled datasets should be produced. More is always better"
                                                                    +" but is slower. Fewer leads to more noise.")
parser.add_argument("--rosetta_kcal_buffer", type=float, default=1, help="How close does the Rosetta estimation need to get to the experimental"
                                                                        +" effect to be considered correct. There's likely a sweetspot for this"
                                                                        +" number to maximize the predictive power of this script.")
parser.add_argument("--dont_estimate_dg_fold", action="store_true", help="Skip the dg_fold estimation step. Although this step makes sense"
                                                                        +" in principle, one really has to wonder how well you can actually"
                                                                        +" estimate this given typical data noise. The correlation is typically"
                                                                        +" better with estimation, however, often the dg_fold estimates are"
                                                                        +" -4 kcal/mol +- 4kcal/mol")
parser.add_argument("--dg_fold_ci_width_avg_dev", type=float, default=0.25, help="dg_fold is fitted with least-squares regression. The final"
                                                                        +" fit parameter is a root-mean-squared error between the Rosetta scores"
                                                                        +" and the experimental values in kcal/mol. This value represents how"
                                                                        +" far away the edges of the confidence interval will be.")
parser.add_argument("--entropy_pool_conc_factor", type=float, default=0.1, help="Which pool should we use for sequence entropy?"
                                                                        +" The default setting of 0.1 will try to pick the pool closest to 10-fold"
                                                                        +" lower than the parent Yeast-KD. 1.0 would pick the pool closest to the"
                                                                        +" Yeast-KD, and 10 10-fold above.")
parser.add_argument("--print_shuffle_copy_info", action="store_true", help="Print the calculated values for all of the random_copies too.")
parser.add_argument("--dump_correlation_data", action="store_true", help="Dump all the data that goes into these calculations. Helpful if you "
                                                                        +" want to make a graph of predicted vs experimental energy for instance.")
parser.add_argument("--dump_shuffled_data", action="store_true", help="Used with --dump_correlation_data. Use this to dump all the random copies too.")

parser.add_argument("--output_file", default="ssm_validation.sc", type=str, help="The output file.")

args = parser.parse_args(sys.argv[1:])

print("Loading pyrosetta")

from pyrosetta import *
from pyrosetta.rosetta import *
init("-mute all -beta_nov16")

def fix_scorefxn(sfxn):
    opts = sfxn.energy_method_options()
    opts.hbond_options().decompose_bb_hb_into_pair_energies(True)
    sfxn.set_energy_method_options(opts)
scorefxn = get_fa_scorefxn()
fix_scorefxn(scorefxn)



def assign_rounds(sorts):
    # Assing round by starting from expression and moving forwards
    sorts['expression_parent'] = ""
    sorts.loc[sorts['special'] == 'naive', 'round'] = -1
    expression_mask = sorts['special'] == 'expression'
    sorts.loc[expression_mask, 'round'] = 0
    sorts.loc[expression_mask, 'expression_parent'] = sorts[expression_mask]['pool_name']
    missing = True
    iterr = 0
    while ( missing ):
        missing = False
        iterr +=1
        for idx, row in sorts.iterrows():
            parent = row['parent_pool']
            if ( parent == "" ):
                continue
            if ( np.isnan(sorts.loc[parent]['round']) ):
                missing = True
                continue
            sorts.loc[idx, 'round'] = sorts.loc[parent]['round'] + 1
            if ( row['expression_parent'] == "" ):
                sorts.loc[idx, 'expression_parent'] = sorts.loc[parent]['expression_parent']
            
        if ( iterr > 100 ):
            sys.exit("Could not assign rounds. Parent structure does not make sense. All pools must be derived from expression"
                    +" (With the exception of a naive pool)")



def load_pose_data(pdbs):

    chainA = core.select.residue_selector.ChainSelector("A")
    chainB = core.select.residue_selector.ChainSelector("B")
    interface_on_A = core.select.residue_selector.NeighborhoodResidueSelector(chainB, 10.0, False)
    interface_on_B = core.select.residue_selector.NeighborhoodResidueSelector(chainA, 10.0, False)
    interface_by_vector = core.select.residue_selector.InterGroupInterfaceByVectorSelector(interface_on_A, interface_on_B)
    interface_by_vector.cb_dist_cut(11)
    interface_by_vector.cb_dist_cut(5.5)
    interface_by_vector.vector_angle_cut(75)
    interface_by_vector.vector_dist_cut(9)

    pose_dats = []
    sequences = {}

    # We're loading the poses to figure out their sequence as well as to determine
    # The core/boundary/surface and interface/monomer
    for pdb in pdbs:
        print("    " + pdb)

        name = os.path.basename(pdb)
        if ( name.endswith(".gz") ):
            name = name[:-len(".gz")]
        if ( name.endswith(".pdb") ):
            name = name[:-len(".pdb")]

        pose = pose_from_file(pdb)
        monomer = pose.split_by_chain()[1]
        sequence = monomer.sequence()
        dssp = "x" + core.scoring.dssp.Dssp(monomer).get_dssp_secstruct()

        sc_neighbors = core.select.util.SelectResiduesByLayer()
        sc_neighbors.use_sidechain_neighbors( True )
        sc_neighbors.compute(pose, "", True)

        atomic_depth = core.scoring.atomic_depth.AtomicDepth( pose, 2.3, False, 0.5 )
        atomic_depth_monomer = core.scoring.atomic_depth.AtomicDepth( monomer, 2.3, False, 0.5 )
        type_set = pose.residue(1).type().atom_type_set()

        probe_size = 2.8
        per_atom_sasa = core.id.AtomID_Map_double_t()
        rsd_sasa = utility.vector1_double()
        core.scoring.calc_per_atom_sasa(pose, per_atom_sasa, rsd_sasa, 2.8, False)

        scorefxn(pose)
        scorefxn(monomer)

        interface_subset = interface_by_vector.apply(pose)

        for seqpos in range(1, monomer.size()+1):

            data = {"ssm_parent":name, "ssm_seqpos":seqpos}

            data['sc_neighbors'] = sc_neighbors.rsd_sasa(seqpos)
            data['is_loop'] = dssp[seqpos] == "L"
            data['by_vector'] = interface_subset[seqpos]

            res = pose.residue(seqpos)
            monomer_res = monomer.residue(seqpos)
            data['depth'] = atomic_depth.calcdepth(res.atom(res.nbr_atom()), type_set)
            data['depth_monomer'] = atomic_depth_monomer.calcdepth(monomer_res.atom(monomer_res.nbr_atom()), type_set)

            data['ddg'] = 2*(pose.energies().residue_total_energy(seqpos) - monomer.energies().residue_total_energy(seqpos))

            sc_sasa = 0
            for i in range(res.first_sidechain_atom(), res.nheavyatoms()+1):
                sc_sasa += per_atom_sasa(seqpos, i)
                for j in range(res.attached_H_begin(i), res.attached_H_end(i)+1):
                    sc_sasa += per_atom_sasa(seqpos, j)

            data['sc_sasa'] = sc_sasa

            pose_dats.append(data)

        sequences[name] = sequence

    pose_df = pd.DataFrame(pose_dats)

    # These should be option flags. But this calculation is so complicated that unless you're looking at the code, you're not going
    #  to specify this stuff correctly.
    # So either change the hardcoded stuff, or make the option flags yourself. (Or even change the calculation here)
    has_ddg = np.abs(pose_df['ddg']) > 1
    is_core = ((pose_df['sc_neighbors'] > 5.2) | (pose_df['depth'] - pose_df['depth_monomer'] > 1 )) & (pose_df['sc_sasa'] < 5)

    pose_df['is_interface_core'] = has_ddg & is_core
    pose_df['is_interface_boundary'] = ( has_ddg | pose_df['by_vector'] ) & ~pose_df['is_interface_core']
    pose_df['is_monomer_core'] = ~has_ddg & (pose_df['sc_neighbors'] >= 5.2) & ~pose_df['is_interface_boundary']
    pose_df['is_monomer_boundary'] = ~has_ddg & (pose_df['sc_neighbors'] < 5.2) & (pose_df['sc_neighbors'] >= 2.0) & ~pose_df['by_vector']
    pose_df['is_monomer_surface'] = ~has_ddg & (pose_df['sc_neighbors'] < 2.0) & ~pose_df['by_vector']

    # Make sure all positions are in exactly 1 category
    assert( np.all( pose_df[['is_interface_core', 'is_interface_boundary', 'is_monomer_core', 'is_monomer_boundary',
                            'is_monomer_surface']].sum(axis=1) == 1) )

    pose_df['ssm_seqpos'] = pose_df['ssm_seqpos'].astype(str)

    return sequences, pose_df




print("Loading data files")


kd_df = pd.read_csv(args.affinity_estimate_out, sep=r"\s+")

if ( "target" in list(kd_df) ):
    target = kd_df['target'].iloc[0]
else:
    target = "none"
assert("kd_lb" in list(kd_df))
assert("kd_ub" in list(kd_df))
assert("lowest_conc" in list(kd_df))
assert("highest_conc" in list(kd_df))
assert("low_conf" in list(kd_df))
assert("avid_doesnt_agree" in list(kd_df))
assert("description" in list(kd_df))
kd_df = kd_df[['kd_lb', 'kd_ub', 'lowest_conc', 'highest_conc', 'low_conf', 'avid_doesnt_agree', 'description' ]].copy()

kd_df['description'] = kd_df['description'].str.replace("([^_])_([0-9]+)_([A-Z])$", r"\1__\2__\3")

score_df = pd.read_csv(args.fast_cart_fastrelax_out, sep=r"\s+")
assert("ddg_no_repack" in list(score_df))
assert("total_score_monomer" in list(score_df))
assert("total_score_target" in list(score_df))
assert("total_score" in list(score_df))
assert("description" in list(score_df))
score_df = score_df[['ddg_no_repack', 'total_score_monomer', 'total_score_target', 'total_score', 'description']].copy()
score_df['ddg_no_repack'] = score_df['ddg_no_repack'].astype(float)
score_df['total_score_monomer'] = score_df['total_score_monomer'].astype(float)
score_df['total_score_target'] = score_df['total_score_target'].astype(float)
score_df['total_score'] = score_df['total_score'].astype(float)

score_df['description'] = score_df['description'].str.replace("([^_])_([0-9]+)_([A-Z])$", r"\1__\2__\3")


sorts = pd.read_csv(args.__getattribute__("sorts.csv"), sep=",")
counts = pd.read_csv(args.__getattribute__("pooled_counts.list"), sep=r"\s+")
sorts.index = sorts['pool_name']

counts['description'] = counts['description'].str.replace("([^_])_([0-9]+)_([A-Z])$", r"\1__\2__\3")

# change nan to empty string
sorts['parent_pool'] = sorts['parent_pool'].fillna("")
sorts['special'] = sorts['special'].fillna("")
sorts['avidity'] = sorts['avidity'].fillna("")
sorts['notes'] = sorts['notes'].fillna("").astype(str)
sorts.loc[sorts['avidity'] == 'avi', 'avidity'] = 'avid'
sorts['round'] = np.nan
if ( "num_cells" in list(sorts) and "collected_cells" not in list(sorts)):
    sorts['collected_cells'] = sorts['num_cells']


if ( (sorts['special'] == 'expression').sum() == 0 ):
    sys.exit("You must have a value in sorts.csv where the special column == \"expression\"")

if ( ~np.all((sorts['avidity'] == 'avid') | (sorts['avidity'] == "") )):
    sys.exit("Valid choices for the avidity column are \"avid\" and \"\"")


# Give each sort a name and assign round 1, 2, 3, etc
assign_rounds(sorts)

# Only keep expression and non-avdity standard sorts
useful_mask = (sorts['avidity'] == "") & ((sorts['special'] == "") | (sorts['special'] == "expression"))

# If two sorts are at the same concentration. We want the later one.
sorts = sorts[useful_mask].sort_values('round')
sorts = sorts.drop_duplicates('concentration', keep='last')

# Remake counts but in terms of concentrations
new_df = pd.DataFrame(index=range(len(counts)))
for idx, row in sorts.iterrows():
    if ( row['special'] == 'expression'):
        continue
    
    name = "%.3f"%(row['concentration'])
    
    new_df[name + "_counts"] = counts[row['pool_name']]
    new_df[name + "_enrich"] = counts[row['pool_name']] / counts[row['expression_parent']]

new_df['description'] = counts['description']
counts = new_df

# Merge new counts into kd_df
to_print = "We have %i points with KD. %i with counts. And %%i with both."%(len(kd_df), len(counts))
kd_df = kd_df.merge(counts, 'inner', 'description')
print(to_print%len(kd_df))


print("Loading pdbs")

sequences, pose_df = load_pose_data(args.parent_pdbs)


# Make the KD rows for the parent
parent_rows = []
for idx, row in kd_df[kd_df['description'].str.contains("_+native$")].iterrows():
    name = re.sub( "_+native$", "", row['description'])
    if ( name not in sequences ):
        continue
    seq = sequences[name]
    
    for il, let in enumerate(seq):
        seqpos = il + 1
        
        new_row = row.copy()
        new_row['native'] = True
        new_row['description'] = name + "__%i__%s"%(seqpos, let)
        parent_rows.append(new_row)

del counts
del sorts
        
extra_kd_df = pd.DataFrame(parent_rows)

# Add parent rows to the kd_df and then remove the originals
kd_df['native'] = False
kd_df = pd.concat((kd_df, extra_kd_df))
old_size = len(kd_df)
kd_df = kd_df.drop_duplicates(subset=['description'], keep='last')
if ( len(kd_df) != old_size ):
    print("!!!!!!!!!!!!!!!!!! Warning !!!!!!!!!!!!!!!!!")
    print("  There were duplicate names in the experimental dataframe")
    print("  Either you truly do have duplicate names in your experimental data")
    print("  Or you have experimental datapoints for mutations that were native")
    print("  This can also trigger if you have the wrong sequence")
    print("")
kd_df = kd_df[~kd_df['description'].str.contains("_+native$")].copy()

parts = kd_df['description'].str.extract(r"(.*[^_])_+([0-9]+)_+([A-Z])$")
kd_df['ssm_parent'] = parts[0]
kd_df['ssm_seqpos'] = parts[1].astype(str)
kd_df['ssm_letter'] = parts[2]


print("We have %i datapoints with KD and counts. We gathered %i positions from pdbs (%i total mutations)." % (len(kd_df), len(pose_df), len(pose_df)*20 ))
save_kd_df = kd_df
kd_df = kd_df.merge(pose_df, 'inner', ['ssm_parent', 'ssm_seqpos'])
print("  After merge. We have %i datapoints with KD and counts. (%i positions)"%
             (len(kd_df), len((kd_df['ssm_parent'] + "@" + kd_df['ssm_seqpos']).unique()) ))

if ( len(kd_df) == 0 ):
    print("Error! None of your experimental data files matched your pdb files!")
    sys.exit()


# Combine scores with kd_df. But don't drop things without scores.
to_print = "We have %i datapoints to consider. %i datapoints with scores. And %%i with both."%(len(kd_df), len(score_df))
test_kd_df = kd_df.merge(score_df, 'left', 'description')
save_kd_df = kd_df

if ( (~test_kd_df['ddg_no_repack'].isnull()).sum() == 0 ):
    score_df['description'] = score_df['description'].str.slice(None, -5)
    test_kd_df = kd_df.merge(score_df, 'left', 'description')
    if ( len(test_kd_df) > 0 ):
        print("Removed _0001 from your score file")

kd_df = test_kd_df

print(to_print%(~kd_df['ddg_no_repack'].isnull()).sum())

if ( (~kd_df['ddg_no_repack'].isnull() & ~kd_df['native']).sum() == 0 ):
    print("")
    print("Error! None of your scorefile names matched your KD names. The names must match nearly exactly. This script will trim"
            +" a single _000X from your score-file if needed and will fix _A_1 vs __A__1 for you. But other than that,"
            +" you need exact matches.")
    sys.exit()

del score_df

if ( kd_df['description'].duplicated().sum() > 0 ):
    print("Detected multiple scores for same binders. Taking best by total_score")
    kd_df = kd_df.sort_values('total_score')
    kd_df = kd_df.drop_duplicates('description', keep="first")
    print("Now we have %i data points"%len(kd_df))

print("")
print("Mutational coverage")
print("  ------- Scores -------    ------- Counts -------")

for parent, subdf in kd_df.groupby('ssm_parent'):
    expected_number = 20 * len(sequences[parent])
    has_score = (~subdf['ddg_no_repack'].isnull()).sum()
    has_counts = len(subdf)
    print("    %5i / %5i (%2i%%)  --   %5i / %5i (%2i%%) -- %s"%(has_score, expected_number, has_score/expected_number*100,
                                                            has_counts, expected_number, has_counts/expected_number*100, parent))


# Take the median of all of the parent scores.
for ssm_parent, subdf in kd_df.groupby('ssm_parent'):
    subdf = subdf[subdf['native']]
    
    for term in list(subdf):
        if ( "ssm" in term ):
            continue
        if ( subdf[term].dtype == bool):
            continue
        try:
            float(subdf[term].iloc[0])
        except:
            continue

        sub = subdf[~subdf[term].isnull()]
        
        median = np.median(sub[term])
        kd_df.loc[subdf.index, term] = median

print("Concentrations we are calculating entropy from")

# Figure out what pool we're going to use for entropy calculations and then gather data
kd_df['chosen_counts'] = 0
kd_df['chosen_enrich'] = 0
kd_df['chosen_ratio'] = 0
kd_df['parent_kd'] = 0

for parent, subdf in kd_df.groupby('ssm_parent'):
    natives = subdf[subdf['native'].astype(bool)]
    if ( len(natives) == 0 ):
        print("Error! No natives detected for " + parent)
        continue
    parent_kd = np.clip(np.sqrt(natives['kd_ub'].mean() * natives['kd_lb'].mean()), natives['lowest_conc'].iloc[0]/10, 
                                                                                        natives['highest_conc'].iloc[0]*1e8)

    # Find all the pool concentrations we could use to perform the entropy calculation.
    # Originally this script accepted multiple experiments, in practice, all the pools here will be valid
    options = []
    for name in list(subdf):
        if ( not name.endswith("_counts") ):
            continue
        if ( name.startswith("chosen")):
            continue
        valids = (~subdf[name].isnull()).sum()
        if ( valids / len(subdf) > 0.2 ):
            options.append(name)
            
    look_kd = parent_kd * args.entropy_pool_conc_factor
    
    conces = np.array([float(x.split("_")[0]) for x in options])
    
    # Error is the ratio of the concentration to the look_concentration
    errors = conces / look_kd
    mask = errors < 1
    errors[mask] = 1 / errors[mask]
    
    smallest = np.argmin(errors)
    picked = options[smallest]
    my_ratio = conces[smallest] / parent_kd

    kd_df.loc[subdf.index, 'chosen_counts'] = subdf[picked]
    kd_df.loc[subdf.index, 'chosen_enrich'] = subdf[picked.replace("counts", "enrich")]
    kd_df.loc[subdf.index, 'chosen_ratio'] = my_ratio
    
    
    print("  %8.3f -- %s"%(conces[smallest], parent))
    
    

print("Generating %i shuffled-data decoys for each parent"%args.random_copies)

new_dfs = []

for parent, subdf in kd_df.groupby('ssm_parent'):

    # Don't swap the parents around
    natives = subdf[subdf['native'].astype(bool)]
    subdf = subdf[~subdf['native'].astype(bool)]

    ubs = subdf['kd_ub'].values
    lbs = subdf['kd_lb'].values
    low = subdf['lowest_conc'].values
    high = subdf['highest_conc'].values
    conf = subdf['low_conf'].values
    avid = subdf['avid_doesnt_agree'].values
    counts = subdf['chosen_counts'].values
    enrich = subdf['chosen_enrich'].values

    for icopy in range(args.random_copies):
        indices = np.arange(len(subdf))
        np.random.shuffle(indices)

        copy_df = subdf.copy()

        copy_df['kd_ub'] = ubs[indices]
        copy_df['kd_lb'] = lbs[indices]
        copy_df['lowest_conc'] = low[indices]
        copy_df['highest_conc'] = high[indices]
        copy_df['low_conf'] = conf[indices]
        copy_df['avid_doesnt_agree'] = avid[indices]
        copy_df['chosen_counts'] = counts[indices]
        copy_df['chosen_enrich'] = enrich[indices]

        copy_df = pd.concat((copy_df, natives))

        copy_df['description'] = "rand%04i_"%icopy + copy_df['description'] 
        copy_df['ssm_parent'] = "rand%04i_"%icopy + copy_df['ssm_parent'] 

        new_dfs.append(copy_df)

    

new = pd.concat(new_dfs)
kd_df = pd.concat((kd_df, new))

print("With the shuffled decoys. We're up to %i datapoints"%(len(kd_df)))

kd_df['kd_center'] = np.sqrt(kd_df['kd_lb'].clip(kd_df['lowest_conc']/10, kd_df['highest_conc']*1e8)
                            * kd_df['kd_ub'].clip(kd_df['lowest_conc']/10, kd_df['highest_conc']*1e8))

# xarray can't do strings
kd_df['p_letter'] = kd_df['ssm_letter'].values.astype('U1').view(np.int32)


delta_df = kd_df[['ssm_parent', 'ssm_seqpos', 'ssm_letter', 'native', 'p_letter',
                    'kd_lb', 'kd_ub', 'kd_center', 'chosen_enrich',
                    'ddg_no_repack', 'total_score_monomer', 'total_score_target', 'total_score'
                ]].reset_index(drop=True)

delta_df['native'] = delta_df['native'].astype(float)

# These work like dataframes except you can have multiple indices (i.e. the shape here has len 3)
ds = pd.pivot_table(delta_df, index=['ssm_parent', 'ssm_letter', 'ssm_seqpos']).to_xarray()

# Turn everything non-native into nan then convert to native value
native_values = ds.where(ds['native'] > 0)
native_values = native_values.ffill(dim="ssm_letter").bfill(dim="ssm_letter")

# Subtractions differences
ds['delta_ddg_no_repack'] = ds['ddg_no_repack'] - native_values['ddg_no_repack']
ds['delta_total_score_monomer'] = ds['total_score_monomer'] - native_values['total_score_monomer']
ds['delta_total_score_target'] = ds['total_score_target'] - native_values['total_score_target']
ds['delta_total_score'] = ds['total_score'] - native_values['total_score']

# Division differences
ds['delta_kd_center'] = ds['kd_center'] / native_values['kd_center']
ds['delta_chosen_enrich'] = ds['chosen_enrich'] / native_values['chosen_enrich']

# Passthrough
ds['parent_letter'] = native_values['p_letter']

# We're going to set the kd_ub and kd_lb to be the minimum and maximum changes we could
#   possibly see given the confidence intervals

ds['delta_kd_ub'] = ds['kd_ub'] / native_values['kd_lb']
ds['delta_kd_lb'] = ds['kd_lb'] / native_values['kd_ub']
ds['delta_kd_center'] = ds['kd_center'] / native_values['kd_center']


# Now we calculate some derived numbers (0.6 ~= boltzmann * 273)
with np.errstate(divide='ignore'):
    ds['delta_exp_ddg_lb'] = (np.log(ds['delta_kd_lb'])*0.6)
    ds['delta_exp_ddg_ub'] = (np.log(ds['delta_kd_ub'])*0.6)
    ds['delta_exp_ddg_center'] = (np.log(ds['delta_kd_center'])*0.6)


# Back into dataframe land
df_vs_native = ds.to_dataframe().reset_index()

# There are tons of nan now. We use kd_center because there's no way this can be nan under normal circumstances
df_vs_native = df_vs_native[~df_vs_native['kd_center'].isnull()].copy()

df_vs_native['parent_letter'] = df_vs_native['parent_letter'].astype(np.int32).view('U1')

to_merge = df_vs_native[['delta_ddg_no_repack', 'delta_total_score_monomer', 'delta_total_score_target', 'delta_total_score',
                        'delta_kd_center', 'delta_kd_ub', 'delta_kd_lb', 'delta_chosen_enrich',
                        'delta_exp_ddg_lb', 'delta_exp_ddg_ub', 'delta_exp_ddg_center',
                        'parent_letter',
                        'ssm_parent', 'ssm_letter', 'ssm_seqpos'
                        ]]


assert(len(to_merge) == len(kd_df))

df = kd_df.merge(to_merge, 'inner', ['ssm_parent', 'ssm_letter', 'ssm_seqpos'])

assert(len(df) == len(to_merge))


# Everything above this point is more-or-less data munging


#############################################################################################################


#                                      Begin SSM Validation Section



#############################################################################################################


# Everything below this point is the SSM Validation technique






############################# Estimating ΔG_fold from the experimental data #############################


# What is the effect of P(fold) on fitted KD?
#   -- And how much does the point on the KD-fitting-sigmoid you choose effect this? (frac_bound)

# fitted_kd = conc * ( 1 - frac_bound ) / frac_bound

# Let P1 and P2 be two different P(fold) values
# fitted_kd1 / fitted_kd2 = conc/conc ( 1 - frac_bound * P1 ) / frac_bound / P1 / ( 1 - frac_bound * P2 ) * frac_bound * P2
# fitted_kd1 / fitted_kd2 =  P2 / P1 * ( 1 - frac_bound * P1 ) / ( 1 - frac_bound * P2 )

# Using wolfram, we see that this function depends very little on frac_bound
# Set frac_bound = 0.12

# fitted_kd1 / fitted_kd2 =  P2 / P1 * ( 1 - 0.12 * P1 ) / ( 1 - 0.12 * P2 )

# Returns KD multiplier when you go from p_ref to p_new
def p_fold_effect(p_ref, p_new, fit_point=0.125):
    return p_ref / p_new * ( 1 - fit_point * p_ref ) / ( 1 - fit_point * p_new )

def dg_fold_to_p_fold(dg_fold):
    return 1 / (1 + np.exp(dg_fold/0.6))

def ddg_bind_from_dg_fold_and_delta_e( dg_fold, delta_e ):
    p_fold = dg_fold_to_p_fold(dg_fold)
    new_p_fold = dg_fold_to_p_fold(dg_fold + delta_e)
    
    return np.log(p_fold_effect( p_fold, new_p_fold )) * 0.6


dg_fold_upper_clip = 15
dg_fold_lower_clip = -15

# Hold on tight. This function is doing least-squares regression minimizing the error on the x-axis (not the y-axis like normal)
#  Also the y-axis has error bars.

# This returns a confidence interval that includes all dg_folds where the root mean squared error of the fit
#  is less than ci_width_avg_dev (in kcal/mol)
def my_dg_fold_fit(xs, y_lb, y_ub, ci_width_avg_dev=0.25):

    if ( len(xs) == 0 ):
        return 0, dg_fold_lower_clip, dg_fold_upper_clip

    steps = 1000
    
    # dg_folds to try. We're just going to do all and pick the best
    dg_fold = np.linspace(dg_fold_lower_clip, dg_fold_upper_clip, steps)
    
    # Calculated y-values when we affect dg_fold by x
    y_fit = ddg_bind_from_dg_fold_and_delta_e( dg_fold[None,:], xs[:,None])
    
    # Figure out if the y_fit falls outside of our confidence interval
    data_is_too_high = y_fit < y_lb[:,None]
    data_is_too_low = y_fit > y_ub[:,None]

    # We drop out stabilizing mutations because they're rare and they give huge penalties
    #  to any dg_fold < 0
    data_valid = y_ub > 0
    
    keep_scores = (data_is_too_high | data_is_too_low) & data_valid[:,None]
    
    # When we search left and right for the closest y_fit. Use the part of the confidence interval
    #  that was closest to start with
    lookup_vals = np.zeros(y_fit.shape)
    lookup_vals[data_is_too_high] = np.repeat(y_lb, steps).reshape(len(y_lb), steps)[data_is_too_high]
    lookup_vals[data_is_too_low]  = np.repeat(y_ub, steps).reshape(len(y_ub), steps)[data_is_too_low]
    lookup_vals[~data_valid]  = 0
    
    # how far you have to go left or right to hit the curve
    delta = np.zeros(y_fit.shape)
    
    # these are the curves we're going to use to calculate the deltas
    #  one curve for each dg_fold step
    x_steps = 800
    standard_xs = np.linspace(xs.min(), xs.max(), x_steps)
    standard_curves = ddg_bind_from_dg_fold_and_delta_e( dg_fold[:,None], standard_xs[None,:])
    
    for i_curve in range(len(standard_curves)):
        curve = standard_curves[i_curve]
        
        # actually search left and right for the closest point
        hits = np.searchsorted(curve, lookup_vals[:,i_curve]).clip(0, len(curve)-1)
        x_hits = standard_xs[hits]
        
        offsets = (xs - x_hits)
        delta[:,i_curve] = offsets
        
    delta[~keep_scores] = 0
    
    # Standard y-axis least-squares fit
    # y_axis_error = lookup_vals - y_fit
    # y_axis_error[~data_valid] = 0
    


    # Standard root mean squared error
    # scores = np.sqrt( np.sum(np.square(delta) + np.square(y_axis_error), axis=0) / len(xs))
    scores = np.sqrt( np.sum(np.square(delta), axis=0) / len(xs))
    
    # We want our CI to cover all scores within ci_width_avg_dev of the best
    for_conf = scores - scores.min() - ci_width_avg_dev

    # print(for_conf)
    
    center_i = np.argmin(scores)
    lb_i = (for_conf < 0).argmax(axis=-1)
    ub_i = np.cumsum((for_conf < 0).astype(int), axis=-1).argmax(axis=-1)
    if ( np.all(for_conf < 0) ):
        lb_i = 0
        ub_i = len(for_conf) - 1
        center_i = ub_i // 2
    
    
    return dg_fold[center_i], dg_fold[lb_i], dg_fold[ub_i]
    


if ( not args.dont_estimate_dg_fold ):

    print("Fitting dG_fold for each of your designs")
    print("  lower_bound     center upper_bound  (kcal/mol)")

    records = []
    for name, subdf in df.groupby('ssm_parent'):
        
        # Only fit:
        #  On parts not involved with the interface
        #  On mutations not involving P C or G
        #  On not loops
        #  On positions where neither the parent nor child KD clipped the experimental bounds
        #  On positions where the KD fitting went well
        #  And we actually have a pdb score
        subdf = subdf[ ((subdf['is_monomer_core'] | subdf['is_monomer_boundary'] | subdf['is_monomer_surface']))
                    &  ((subdf['ssm_letter'] != 'P') & (subdf['ssm_letter'] != 'C') & (subdf['ssm_letter'] != 'G'))
                    &  ((subdf['parent_letter'] != 'P') & (subdf['parent_letter'] != 'C') & (subdf['parent_letter'] != 'G'))
                    &  ((~subdf['is_loop']))
                    &  (~subdf['delta_exp_ddg_ub'].isnull() & ~subdf['delta_exp_ddg_lb'].isnull())
                    &  (np.isfinite(subdf['delta_exp_ddg_ub']) & np.isfinite(subdf['delta_exp_ddg_lb']))
                    &  (~subdf['low_conf'] & ~subdf['avid_doesnt_agree'])
                    &  (~subdf['delta_total_score_monomer'].isnull())
                    ]

        delta_energies = subdf['delta_total_score_monomer'].clip(dg_fold_lower_clip, dg_fold_upper_clip).values


        dg_fold, dg_fold_lb, dg_fold_ub = my_dg_fold_fit(
                                                        delta_energies,
                                                        subdf['delta_exp_ddg_lb'].values, 
                                                        subdf['delta_exp_ddg_ub'].values,
                                                        args.dg_fold_ci_width_avg_dev
                                                        )
                            
        records.append({"ssm_parent":name, "dg_fold":dg_fold, "dg_fold_lb":dg_fold_lb, "dg_fold_ub":dg_fold_ub})
        
        if ( not name.startswith("rand") or args.print_shuffle_copy_info ):
            print("    %9.1f  %9.1f  %9.1f -- %s"%(dg_fold_lb, dg_fold, dg_fold_ub, name))


    fold_df = pd.DataFrame(records)

        
    df = df.merge(fold_df, 'inner', 'ssm_parent')
            



else:
    print("Skipping dG_fold fit")



print("Calculating sequence entropy")


entropy_df = df[['chosen_counts',  'ssm_parent', 'ssm_letter', 'ssm_seqpos']]

entropy_ds = pd.pivot_table(entropy_df, index=['ssm_parent', 'ssm_letter', 'ssm_seqpos']).to_xarray()


p_of_this_letter = entropy_ds['chosen_counts'] / entropy_ds['chosen_counts'].sum(dim="ssm_letter")

# This is the shannon entropy equation
with np.errstate(divide='ignore'):
    entropy = - (p_of_this_letter * np.log( p_of_this_letter )).sum(dim='ssm_letter')

# Like doing entropy_ds['entropy'] = entropy
entropy_ds = entropy_ds.assign(entropy=entropy)

# Entropy here is a per sequence position property
entropy_df = entropy_ds.to_dataframe().reset_index()[['entropy', 'ssm_parent', 'ssm_letter', 'ssm_seqpos']]


df = df.merge(entropy_df, 'inner', ['ssm_parent', 'ssm_letter', 'ssm_seqpos'])



cats = ['is_interface_core', 'is_interface_boundary', 'is_monomer_core', 
                                 'is_monomer_boundary', 'is_monomer_surface']


print("   Int Core - Int Bound -  Mon Core - Mon Bound -  Mon Surf")

records = []
for name, subdf in df.groupby('ssm_parent'):
    
    string = ""
    dat = {}
    for term in cats:
        
        part = subdf[subdf[term]]
        seqposs = part['ssm_seqpos'].unique()
        use = part[part['native']]
        assert(len(seqposs) == len(use))
        
        entropy = np.mean(use['entropy'])
        
        string += "%11.1f "%entropy
        dat['entropy_' + term] = entropy
    
    dat['ssm_parent'] = name
    records.append(dat)
        
    string += " -- " + name
    if ( not name.startswith("rand") or args.print_shuffle_copy_info ):
        print(string)

entropy_df = pd.DataFrame(records)

df = df.merge(entropy_df, 'inner', 'ssm_parent')
        

print("")
print("Calculating Rosetta Accuracy")

print("   Int Core - Int Bound -  Mon Core - Mon Bound -  Mon Surf")

use_df = df
use_df = use_df[(True
    & (use_df['ssm_letter'] != 'C')
    & (use_df['ssm_letter'] != 'P')
    & (use_df['parent_letter'] != 'C')
    & (~use_df['low_conf'] & ~use_df['avid_doesnt_agree'])
    & ~(use_df['delta_exp_ddg_lb'].isnull() & use_df['delta_exp_ddg_ub'].isnull())
    &  (np.isfinite(use_df['delta_exp_ddg_ub']) | np.isfinite(use_df['delta_exp_ddg_lb']))
    & (~use_df['is_loop'])
    )]

use_df = use_df[['ssm_parent', 'dg_fold', 'dg_fold_lb', 'dg_fold_ub', 'delta_total_score_monomer', 'delta_total_score_target',
                'delta_ddg_no_repack', 'delta_exp_ddg_lb', 'delta_exp_ddg_ub', 'ssm_seqpos', 'ssm_letter'] 
                + cats].copy()



records = []
records2 = {"index":[],
            "delta_pfold_ddg_lb":[],
            "delta_pfold_ddg_ub":[],
            "delta_rosetta_lb":[],
            "delta_rosetta_ub":[],
            "rosetta_agrees":[]
            }
for name, subdf in use_df.groupby('ssm_parent'):
    


    if ( args.dont_estimate_dg_fold ):
        delta_pfold_ddg_lb = subdf['delta_total_score_monomer'].values
        delta_pfold_ddg_ub = subdf['delta_total_score_monomer'].values
    else:

        my_dg_fold_lin = np.linspace(subdf['dg_fold_lb'].iloc[0], subdf['dg_fold_ub'].iloc[0], 100)

        monomer_effect = subdf['delta_total_score_monomer'].values
        
        dg_bind_monomer = ddg_bind_from_dg_fold_and_delta_e( my_dg_fold_lin[None,:], monomer_effect[:,None] )
        
        # We take the smallest and largest effect that could be generated from any point within the bounds
        delta_pfold_ddg_lb = np.min( dg_bind_monomer, axis=-1)
        delta_pfold_ddg_ub = np.max( dg_bind_monomer, axis=-1)
        
        
    interface_effect = subdf['delta_total_score_target'].values + subdf['delta_ddg_no_repack'].values
    delta_rosetta_lb = delta_pfold_ddg_lb + interface_effect
    delta_rosetta_ub = delta_pfold_ddg_ub + interface_effect
                

    # nan's come from positions where the data clipped. We'll just call them accurate
    within_bounds  = ((delta_rosetta_ub + args.rosetta_kcal_buffer > subdf['delta_exp_ddg_lb'].values) 
                          | np.isnan(subdf['delta_exp_ddg_lb'].values))
    within_bounds &= ((delta_rosetta_lb - args.rosetta_kcal_buffer < subdf['delta_exp_ddg_ub'].values) 
                          | np.isnan(subdf['delta_exp_ddg_ub'].values))

    records2['index'].extend(subdf.index)
    records2['delta_pfold_ddg_lb'].extend(delta_pfold_ddg_lb)
    records2['delta_pfold_ddg_ub'].extend(delta_pfold_ddg_ub)
    records2['delta_rosetta_lb'].extend(delta_rosetta_lb)
    records2['delta_rosetta_ub'].extend(delta_rosetta_ub)
    records2['rosetta_agrees'].extend(within_bounds)

    # Calculate the average accuracy within each of the regions
    string = ""
    dat = {}
    for term in cats:
        
        mask = subdf[term]

        accuracy = np.mean(within_bounds[mask])

        string += "%10.1f%% "%(accuracy*100)
        
        dat['accuracy_' + term] = accuracy
    
    dat['ssm_parent'] = name
    records.append(dat)
        
    string += "-- " + name
    if ( not name.startswith("rand") or args.print_shuffle_copy_info):
        print(string)

acc_df1 = pd.DataFrame(records)
acc_df2 = pd.DataFrame(records2, index=records2['index'])

df = pd.concat((df, acc_df2), axis=1 )
df = df.merge(acc_df1, 'outer', 'ssm_parent')






# We take accuracy of only core regions because Rosetta understands these the best
df['final_rosetta_accuracy_score'] = df['accuracy_is_interface_core'] + df['accuracy_is_monomer_core']

# This is basically the difference in entropy between the core and the surface
#  Don't allow this to go negative because things could validate then for entirely the wrong reason
df['final_entropy_score'] = (-df['entropy_is_interface_core'] - df['entropy_is_monomer_core'] 
                                        + 2*df['entropy_is_monomer_surface']).clip(0, None)


df['final_combined_score'] = 2*df['final_rosetta_accuracy_score'] + df['final_entropy_score']



print("Calculating final p-values")
print(" Rosetta  Entropy")


warnings.filterwarnings("ignore", ".*This pattern has match groups.*")


def get_p_score(value, values):
    std = np.std(values)
    mean = np.mean(values)
    
    zscore = (value - mean) / std
    
    return scipy.stats.norm.cdf(-zscore).clip(1e-100, None)

df = df.reset_index(drop=True)
one_of_each_mask = np.zeros(len(df), bool)
for parent, subdf in df.groupby("ssm_parent"):
    one_of_each_mask[subdf.index[0]] = True

one_of_each_df = df[one_of_each_mask]

records = []

for design in list(sequences):
    
    subdf = one_of_each_df[one_of_each_df['ssm_parent'].str.contains("^(rand[0-9]+_)?" + design + "$")]
    if ( len(subdf) == 0 ):
        continue
    assert(len(subdf) == 1+args.random_copies)
    
    us = subdf[~subdf['ssm_parent'].str.startswith("rand")].iloc[0]
    them = subdf[subdf['ssm_parent'].str.startswith("rand")]
    
    p_entropy = get_p_score(us['final_entropy_score'], them['final_entropy_score'])
    p_rosetta = get_p_score(us['final_rosetta_accuracy_score'], them['final_rosetta_accuracy_score'])
    p_total = get_p_score(us['final_combined_score'], them['final_combined_score'])

    print("%8.0e %8.0e --" % (p_rosetta, p_entropy), design)
    
    
    records.append({'ssm_parent':design, 'p_total':p_total,
                   'p_rosetta':p_rosetta, 'p_entropy':p_entropy})
    

    
    

p_df = pd.DataFrame(records)
    

df = df.merge(p_df, 'inner', 'ssm_parent')
df['target'] = target

if ( args.dump_correlation_data ):
    if ( not args.dump_shuffled_data ):
        df = df[~df['description'].str.startswith("rand")]

    df.to_csv(args.output_file + "_correlation.sc", sep=" ", index=None, na_rep="NaN")



df = df[~df['description'].str.startswith("rand")]

df = df.reset_index(drop=True)
one_of_each_mask = np.zeros(len(df), bool)
for parent, subdf in df.groupby("ssm_parent"):
    one_of_each_mask[subdf.index[0]] = True

final_df = df[one_of_each_mask]

dg_fold_terms = ['dg_fold', 'dg_fold_lb', 'dg_fold_ub']
if ( args.dont_estimate_dg_fold ):
    dg_fold_terms = []

final_df = final_df[['p_total', 'p_rosetta', 'p_entropy'] + dg_fold_terms 
                    + ["entropy_" + x for x in cats]
                    + ["accuracy_" + x for x in cats]
                    + ['target', 'ssm_parent']
                    ]

final_df.to_csv(args.output_file, sep=" ", index=None, na_rep="NaN")
