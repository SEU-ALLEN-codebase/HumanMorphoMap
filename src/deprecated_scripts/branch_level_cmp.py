##########################################################
#Author:          Yufeng Liu
#Create time:     2025-03-18
#Description:               
##########################################################
import os
import glob
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
import umap

from morph_topo.morphology import Morphology, Topology
from ml.feature_processing import standardize_features
from plotters.customized_plotters import sns_jointplot

from config import REG2LOBE


# Function to calculate y-limits based on IQR
def get_ylim(feature_data, multiplier=1.5):
    Q1 = np.percentile(feature_data, 25)  # First quartile
    Q3 = np.percentile(feature_data, 75)  # Third quartile
    IQR = Q3 - Q1
    lower = Q1 - multiplier * IQR
    upper = Q3 + multiplier * IQR
    # Clamp limits to the data’s actual min/max to avoid empty plots
    lower = max(lower, feature_data.min())
    upper = min(upper, feature_data.max())
    return lower, upper

# Function to add significance annotations
def add_significance(ax, data1, data2, x1, x2, y_max, y_min, y_offset):
    _, p_value = mannwhitneyu(data1, data2)
    if p_value < 0.0005:
        sig_marker = '***'
    elif p_value < 0.005:
        sig_marker = '**'
    elif p_value < 0.05:
        sig_marker = '*'
    else:
        sig_marker = 'ns'
    
    ax.plot([x1+0.1, x1+0.1, x2-0.1, x2-0.1], [y_max + y_offset * 0.5, y_max + y_offset, y_max + y_offset, 
            y_max + y_offset * 0.5], lw=1.5, c='red')
    ax.text((x1 + x2) / 2, y_max - y_offset*2, sig_marker, ha='center', va='bottom', fontsize=22, color='red')
    # estimate the difference
    p_diff = (data2.mean() - data1.mean()) / data1.mean() * 100.
    ax.text((x1 + x2) / 2, y_min - y_offset, f'{p_diff:.1f}%', ha='center', va='bottom', fontsize=22,
            color='red')


def get_branch_features(swcfile, rdict, ldict, cdict):

    #####------ Helper function --------####
    def _update_dict(D, v, level):
        try:
            D[level].append(v)
        except KeyError:
            D[level] = [v]

    ####------ End of helper function --------####


    tree = pd.read_csv(swcfile, comment='#', sep=' ', index_col=False,
                       names=('#id', 'type', 'x', 'y', 'z', 'r', 'p'))
    ntree = tree.set_index('#id')

    try:
        morph = Morphology([list(row.values()) for row in tree.to_dict('records')])
    except KeyError:
        return 

    topo_tree, seg_dict = morph.convert_to_topology_tree()
    # get the topology
    topo = Topology(topo_tree)

    # get the branch lengths
    frag_lengths, frag_lengths_dict = morph.calc_frag_lengths()
    path_lengths_dict = morph.calc_seg_path_lengths(seg_dict, frag_lengths_dict)
    euc_lengths, euc_lengths_dict = topo.calc_frag_lengths()
    
    for tip in topo.tips:
        pid = topo.pos_dict[tip][-1]
        cid = tip
        while pid != -1:
            # estimate the current branch
            seg = seg_dict[cid]
            level = topo.order_dict[cid]
            # radius
            if len(seg) > 0:
                r_median = np.median(ntree.loc[seg, 'r'])
                _update_dict(rdict, r_median, level)
            else:
                r_median = ntree.loc[pid, 'r']

            # branch lengths
            _update_dict(ldict, path_lengths_dict[cid], level)
            # contraction
            contraction = euc_lengths_dict[cid] / path_lengths_dict[cid]
            _update_dict(cdict, contraction, level)
            
            # update the cid and pid
            cid = pid
            pid = topo.pos_dict[cid][-1]
        


def comparing_levelwise(swc_dir, cell_type_file, meta_file_neuron, ihc=0, cache_file='./caches/levelwise_cell_comparision.pkl'):

    ####------- Helper function --------####
    def _visualize(dict_p, dict_np, figname):
        # remove keys with less than 10 cases for robustness
        common_keys = []
        nthr = 20
        for k in set(dict_p.keys()).intersection(dict_np.keys()):
            num_p, num_np = len(dict_p[k]), len(dict_np[k])
            print(k, num_p, num_np)
            if (num_p > nthr) and (num_np > nthr):
                common_keys.append(k)
        print("Common keys:", common_keys)
        
        # Convert dictionaries to a long-format DataFrame for common keys
        data = []
        for key in common_keys:
            # Add values from dict1
            for value in dict_p[key]:
                data.append({'Key': key, 'Value': value, 'Source': 'pyramidal'})
            # Add values from dict2
            for value in dict_np[key]:
                data.append({'Key': key, 'Value': value, 'Source': 'nonpyramidal'})

        # Create DataFrame
        df = pd.DataFrame(data)

        # Convert 'Key' to string for better labeling in plots (optional)
        df['Key'] = df['Key'].astype(str)

        # Set up the figure size and style
        plt.figure(figsize=(8, 5))

        # Create a faceted boxplot
        g = sns.catplot(x='Source', y='Value', col='Key', col_wrap=3, 
                        kind='violin', data=df, sharey=False, height=4, aspect=1, cut=True)

         # Add significance annotations
        for ax, feature in zip(g.axes.flat, g.col_names):
            # Extract data for pyramidal and non-pyramidal
            data1 = df[(df['Key'] == feature) & (df['Source'] == 'pyramidal')]
            data2 = df[(df['Key'] == feature) & (df['Source'] == 'nonpyramidal')]
            if len(data1) > 0 and len(data2) > 0:  # Ensure there’s data to compare
                y_max = max(data1['Value'].max(), data2['Value'].max())
                y_min = min(data1['Value'].min(), data2['Value'].min())
                y_offset = (y_max - y_min) * 0.12
                npv = data2['Value'].mean()
                pv = data1['Value'].mean()
                p_diff = (npv - pv) / pv * 100.
                ax.text(0.5, y_min + y_offset, f'{p_diff:.1f}%', ha='center', va='bottom', fontsize=25, color='red')

        # Customize titles and labels
        g.set_titles('Level {col_name}')
        g.set_axis_labels('Dictionary', 'Value')
        plt.suptitle('Comparison of Distributions Across Common Keys (Boxplot)', y=1.05)
        plt.savefig(figname, dpi=300)
        plt.close()


    ####------ End of helper function ------####


    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as fp:
            rdict_p, ldict_p, cdict_p, rdict_np, ldict_np, cdict_np = pickle.load(fp)
        # visualize
        sns.set_theme(style='ticks', font_scale=1.6)
        _visualize(rdict_p, rdict_np, figname='radius.png')
        _visualize(ldict_p, ldict_np, figname='length.png')
        _visualize(cdict_p, cdict_np, figname='contraction.png')
        return


    CELL_DICT = {
        'pyramidal': 0,
        'nonpyramidal': 1
    }

    # load all neurons
    meta_n = pd.read_csv(meta_file_neuron, index_col='cell_id', low_memory=False, encoding='gbk')
    # extract neurons

    ctypes = pd.read_csv(cell_type_file, index_col=0)
    # pyramidal cells
    ctypes_pyramidal = ctypes[(ctypes.CLS2 == CELL_DICT['pyramidal']) & (ctypes.num_annotator > 1)]
    ctypes_nonpyramidal = ctypes[(ctypes.CLS2 == CELL_DICT['nonpyramidal']) & (ctypes.num_annotator > 1)]
    # extract the features
    pyramidal_ids = [int(idx.split('_')[0]) for idx in ctypes_pyramidal.index]
    nonpyramidal_ids = [int(idx.split('_')[0]) for idx in ctypes_nonpyramidal.index]
    # 
    if ihc != 2:
        meta_p = meta_n.loc[meta_n.index.isin(pyramidal_ids) & (meta_n.immunohistochemistry == ihc)]
        meta_np = meta_n.loc[meta_n.index.isin(nonpyramidal_ids) & (meta_n.immunohistochemistry == ihc)]
    else:
        meta_p = meta_n.loc[meta_n.index.isin(pyramidal_ids)]
        meta_np = meta_n.loc[meta_n.index.isin(nonpyramidal_ids)]

    # load the swcs
    rdict_p, ldict_p, cdict_p = {}, {}, {}
    rdict_np, ldict_np, cdict_np = {}, {}, {}
    nfile = 0
    t0 = time.time()
    for swcfile in glob.glob(os.path.join(swc_dir, '*.swc')):
        swcname = os.path.split(swcfile)[-1]
        neuron_id = int(swcname.split('_')[0])
        is_pyramidal = neuron_id in meta_p.index
        is_nonpyramidal = neuron_id in meta_np.index
        if is_pyramidal:
            get_branch_features(swcfile, rdict_p, ldict_p, cdict_p)
        elif is_nonpyramidal:
            get_branch_features(swcfile, rdict_np, ldict_np, cdict_np)
        nfile += 1
        if nfile % 100 == 0:
            print(f'[{nfile}]: {time.time() - t0:.2f} seconds')
            #break

    # save the data
    with open(cache_file, 'wb') as fp:
        pickle.dump([rdict_p, ldict_p, cdict_p, rdict_np, ldict_np, cdict_np], fp)

    print()
    

if __name__ == '__main__':
    #cell_type_file = '../meta/cell_type_annot_rating_cls2_yufeng_unique.csv'
    cell_type_file = '../meta/cell_type_annotation_8.4K_all_CLS2_unique.csv'
    meta_file_neuron = '/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta_filter/meta.csv'
    gf_file = '/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta_filter/l_measure_result.csv'


    if 0:
        pruned_swc_dir = './prune_swc/terminal_branches_pruned'
        comparing_levelwise(pruned_swc_dir, cell_type_file, meta_file_neuron, ihc=ihc)

