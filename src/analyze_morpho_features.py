##########################################################
#Author:          Yufeng Liu
#Create time:     2024-09-27
#Description:               
##########################################################
import os
import sys
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.decomposition import PCA

from config import standardize_features

LOCAL_FEATS = [
    'N_stem',
    'Soma_surface',
    'Average Contraction',
    'Average Bifurcation Angle Local',
    'Average Bifurcation Angle Remote',
    'Average Parent-daughter Ratio',
]

def load_features(gf_file, meta_file, min_neurons=5, standardize=False, remove_na=True):
    # Loading the data
    df = pd.read_csv(gf_file, index_col=0)[LOCAL_FEATS]
    meta = pd.read_csv(meta_file, index_col=0)
    col_reg = '脑区'
    col_xy = 'xy拍摄分辨率(*10e-3μm/px)'
    # 
    sns.set_theme(style='ticks', font_scale=1)
    df['region'] = meta[col_reg]
    df['patient'] = meta['病人编号']
    # filter brain regions with number of neurons smaller than `min_neurons`
    rs, rcnts = np.unique(df.region, return_counts=True)
    rs_filtered = rs[rcnts >= min_neurons]
    dff = df[df.region.isin(rs_filtered)]
    if remove_na:
        dff = dff[dff.isna().sum(axis=1) == 0]
    
    # standardize column-wise
    if standardize:
        standardize_features(dff, LOCAL_FEATS, inplace=True)
    return dff


def feature_distributions(gf_file, meta_file, boxplot=True, min_neurons=5):
    df = load_features(gf_file, meta_file, min_neurons=min_neurons)
    sregions = sorted(np.unique(df.region))
    for feat in LOCAL_FEATS:
        dfi = df[[feat, 'region']]
        if boxplot:
            sns.boxplot(data=dfi, x='region', y=feat, hue='region', order=sregions)
            prefix = 'boxplot'
        else:
            sns.stripplot(data=dfi, x='region', y=feat, s=3, alpha=0.5, hue='region', order=sregions)
            prefix = 'stripplot'
        plt.xticks(rotation=90, rotation_mode='anchor', ha='right', va='center')
        if feat.startswith('Average Bifurcation Angle'):
            plt.ylim(30, 110)
        elif feat.startswith('Average Parent'):
            plt.ylim(0.5, 1.2)
        elif feat.startswith('Average Contraction'):
            plt.ylim(0.9, 1.0)


        plt.subplots_adjust(bottom=0.28)
        plt.savefig(f'{prefix}_{feat.replace(" ", "")}.png', dpi=300)
        plt.close()

def joint_distributions(gf_file, meta_file, min_neurons=5, feature_reducer='UMAP'):
    sns.set_theme(style='ticks', font_scale=1.5)

    df = load_features(gf_file, meta_file, min_neurons=min_neurons, standardize=True)

    cache_file = f'cache_{feature_reducer.lower()}.pkl'
    # map to the UMAP space
    if os.path.exists(cache_file):
        print(f'--> Loading existing {feature_reducer} file')
        with open(cache_file, 'rb') as fp:
            emb = pickle.load(fp)
    else:
        if feature_reducer == 'UMAP':
            reducer = umap.UMAP(random_state=1024)
        elif feature_reducer == 'PCA':
            reducer = PCA(n_components=2)

        emb = reducer.fit_transform(df[LOCAL_FEATS])
        print(reducer.explained_variance_ratio_)
        with open(cache_file, 'wb') as fp:
            pickle.dump(emb, fp)
    
    key1 = f'{feature_reducer}1'
    key2 = f'{feature_reducer}2'
    df[[key1, key2]] = emb
    sregions = sorted(np.unique(df.region))
    indices = [0, 4, 6, 8, 11, 14, 16, 20, 23, 25, 29]

    if feature_reducer == 'UMAP':
        xlim, ylim = (-2,13), (0, 9)
    elif feature_reducer == 'PCA':
        xlim, ylim = (-6,6), (-6,6)

    ii = 0
    for i1, i2 in zip(indices[:-1], indices[1:]):
        cur_regions = sregions[i1:i2]
        cur_df = df[df.region.isin(cur_regions)]
        print(i1, i2, cur_regions)
        sns.jointplot(data=cur_df, x=key1, y=key2, kind='scatter', xlim=xlim, ylim=ylim, 
                      hue='region', marginal_kws={'common_norm': False, 'fill': False},
                      joint_kws={'alpha':1.},
                      )
        plt.legend(frameon=False, ncol=1, handletextpad=0, markerscale=1.5)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(f'{feature_reducer.lower()}_{ii:02d}.png', dpi=300)
        plt.close()

        # visualize the variances across different person
        for region_i in cur_regions:
            df_i = cur_df[cur_df.region == region_i]
            patients_i = df_i.patient
            ps = np.unique(patients_i)
            if ps.shape[0] > 1:
                # plot the comparison of different samples of the same region
                sns.jointplot(data=df_i, x=key1, y=key2, kind='scatter', xlim=xlim, ylim=ylim,
                    hue='patient', marginal_kws={'common_norm':False, 'fill': False},
                    joint_kws={'alpha':1.}
                )
                plt.legend(frameon=False, ncol=1, handletextpad=0, markerscale=1.5)
                plt.xticks([]); plt.yticks([])
                plt.savefig(f'{feature_reducer.lower()}_{region_i}.png', dpi=300)
                plt.close()
                
            
    
        ii += 1

    # also for all neurons
    sns.jointplot(
        data=df, x=key1, y=key2, kind='scatter', xlim=xlim, ylim=ylim,
        joint_kws={'s': 1, 'alpha': 0.5},
        marginal_kws={'common_norm':False, 'fill': False, }
    )
    plt.xticks([]); plt.yticks([])
    plt.savefig(f'{feature_reducer.lower()}_all.png', dpi=300)
    plt.close()

    print()


if __name__ == '__main__':

    if 0:   # temporary
        from global_features import calc_global_features_from_folder
        swc_dir = '/data/kfchen/trace_ws/paper_auto_human_neuron_recon/unified_recon_1um/source500'
        outfile = 'gf_temp.csv'
        calc_global_features_from_folder(swc_dir, outfile)

    if 1:
        gf_file = '/data/kfchen/trace_ws/paper_auto_human_neuron_recon/unified_recon_1um/ptls10.csv'
        meta_file = '../meta/neuron_info_9060_utf8_curated0929.csv'
        #feature_distributions(gf_file, meta_file)
        joint_distributions(gf_file, meta_file, feature_reducer='UMAP')

