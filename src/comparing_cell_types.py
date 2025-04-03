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
    
    ax.plot([x1, x1, x2, x2], [y_max + y_offset * 0.5, y_max + y_offset, y_max + y_offset, 
            y_max + y_offset * 0.5], lw=1.5, c='k')
    ax.text((x1 + x2) / 2, y_max - y_offset, sig_marker, ha='center', va='bottom', fontsize=25)
    # estimate the difference
    p_diff = (data2.mean() - data1.mean()) / data1.mean() * 100.
    ax.text((x1 + x2) / 2, y_min - y_offset, f'{p_diff:.1f}%', ha='center', va='bottom', fontsize=25,
            color='red')


def comparing_cell_types(cell_type_file, gf_file, meta_file_neuron, ihc=0):
    CELL_DICT = {
        'pyramidal': '0',
        'nonpyramidal': '1'
    }
    show_features = [
        'N_stem', 'Number of Branches', 'Number of Tips',
        'Average Diameter', 'Total Length', 'Max Branch Order', 'Average Contraction',
        'Average Fragmentation', 'Average Parent-daughter Ratio', 'Average Bifurcation Angle Local',
        'Average Bifurcation Angle Remote', 'Hausdorff Dimension'
    ]
    show_features_umap = [
        'N_stem, Number of Branches', 'Number of Tips',
        'Total Length', 'Average Diameter', 'Max Branch Order',
        'Average Fragmentation', #'Average Parent-daughter Ratio', 
        'Average Bifurcation Angle Remote'
    ]

    # load all neurons
    gfs = pd.read_csv(gf_file, index_col=0, low_memory=False)[show_features]
    meta_n = pd.read_csv(meta_file_neuron, index_col='cell_id', low_memory=False, encoding='gbk')
    meta_n['lobe'] = [REG2LOBE[r] for r in meta_n['brain_region']]
    # extract neurons

    ctypes = pd.read_csv(cell_type_file, index_col=0)
    # pyramidal cells
    ctypes_pyramidal = ctypes[(ctypes.CLS2 == CELL_DICT['pyramidal']) & (ctypes.num_annotator > 1)]
    ctypes_nonpyramidal = ctypes[(ctypes.CLS2 == CELL_DICT['nonpyramidal']) & (ctypes.num_annotator > 1)]
    # extract the features
    pyramidal_ids = [int(idx.split('_')[0]) for idx in ctypes_pyramidal.index]
    nonpyramidal_ids = [int(idx.split('_')[0]) for idx in ctypes_nonpyramidal.index]

    msel = 1
    if ihc != 2:
        msel = meta_n.immunohistochemistry == ihc
    if lobe != 'all':
        msel = msel & (meta_n['lobe'] == lobe)

    gfs_p = gfs[gfs.index.isin(pyramidal_ids) & msel]
    gfs_np = gfs[gfs.index.isin(nonpyramidal_ids) & msel]
    
    # for different lobes
    print(f'Number of pyramidal and nonpyramidal neurons: {gfs_p.shape[0]}, {gfs_np.shape[0]}')

    # comparison
    gfs_pnp = pd.concat((gfs_p, gfs_np))
    gfs_pnp['cell_type'] = 'nonpyramidal'
    gfs_pnp.loc[gfs_p.index, 'cell_type'] = 'pyramidal'

    if 1:
        # separability on UMAP space
        gfs_pnp_s = gfs_pnp.drop([sf for sf in show_features if sf not in show_features_umap], axis=1)
        standardize_features(gfs_pnp_s, gfs_pnp_s.columns[:-1], inplace=True)
        reducer = umap.UMAP(random_state=1024)
        emb2d = reducer.fit_transform(gfs_pnp_s.iloc[:,:-1])
        gfs_pnp_s[['UMAP1', 'UMAP2']] = emb2d
        # visualize
        gfs_pnp_s['lobe'] = meta_n.loc[gfs_pnp_s.index, 'lobe']
        sns_jointplot(gfs_pnp_s, 'UMAP1', 'UMAP2', None, None, 'cell_type', 'cell_types_on_umap.png', markersize=10, hue_order=None)
        sns_jointplot(gfs_pnp_s, 'UMAP1', 'UMAP2', None, None, 'lobe', 'lobes_on_umap.png', markersize=10, hue_order=None)
    
    if 1:
        # comparison feature-by-feature between cell types
        xname = 'lobe' #'cell_type'
        xtypes = ['FL', 'TL', 'OL'] #['pyramidal', 'nonpyramidal']
        label_dict = {
            'lobe': 'Brain lobe',
            'cell_type': 'Cell type'
        }

        #gfs_pnp[]
        feature_columns = gfs_pnp.columns[:-1]
        if xname == 'lobe':
            gfs_pnp['lobe'] = meta_n.loc[gfs_pnp.index, 'lobe']
            #gfs.pnp = gfs_pnp[gfs_pnp['lobe'].isin(xtypes[:2])]
            gfs_pnp = gfs_pnp[gfs_pnp['cell_type'] == 'pyramidal']
        gfs_pnp_long = gfs_pnp.melt(id_vars=[xname], value_vars=feature_columns, 
                      var_name='Feature', value_name='Value')
        
        # Set up the figure size and style
        sns.set_theme(style='ticks', font_scale=1.8)
        plt.figure(figsize=(10, 8))

        # Create a faceted boxplot
        g = sns.catplot(x=xname, y='Value', col='Feature', col_wrap=4, hue=xname, order=xtypes,
                        kind='violin', data=gfs_pnp_long, sharey=False, height=3, aspect=1.5)

        # Add significance annotations
        for ax, feature in zip(g.axes.flat, g.col_names):
            # Extract data for pyramidal and non-pyramidal
            data1 = gfs_pnp_long[(gfs_pnp_long['Feature'] == feature) & (gfs_pnp_long[xname] == xtypes[0])]['Value'].dropna()
            data2 = gfs_pnp_long[(gfs_pnp_long['Feature'] == feature) & (gfs_pnp_long[xname] == xtypes[1])]['Value'].dropna()
            if len(data1) > 0 and len(data2) > 0:  # Ensure there’s data to compare
                y_max = max(data1.max(), data2.max())
                y_min = min(data1.min(), data2.min())
                y_offset = (y_max - y_min) * 0.12
                add_significance(ax, data1, data2, 0, 1, y_max, y_min, y_offset)


        # Adjust layout and titles
        g.set_titles('{col_name}')
        g.set_axis_labels(label_dict[xname], 'Feature Value')
        plt.suptitle('Comparison of Features Between Pyramidal and Non-Pyramidal Cells (Boxplot)', y=1.05)
        #plt.tight_layout()
        plt.savefig(f'l_measure_among_{xname}_{lobe}.png', dpi=300)
        plt.close()



if __name__ == '__main__':
    #cell_type_file = '../meta/cell_type_annot_rating_cls2_yufeng_unique.csv'
    cell_type_file = '../meta/cell_type_annotation_8.4K_all_CLS2_unique.csv'
    meta_file_neuron = '/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta_filter/meta.csv'
    gf_file = '/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta_filter/swc_1um_cropped_150um_l_measure.csv'
    ihc = 0

    if 0:
        comparing_cell_types(cell_type_file, gf_file, meta_file_neuron, ihc=0)

