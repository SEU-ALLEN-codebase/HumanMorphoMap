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

from spatial import spatial_utils   # pylib

LOCAL_FEATS = [
    'N_stem',
    'Soma_surface',
    #'Average Contraction',
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
    #if surface_sqrt:
    #    dff['Soma_surface'] = np.sqrt(dff.Soma_surface)

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

def joint_distributions(gf_file, meta_file, layer_file=None, min_neurons=5, feature_reducer='UMAP'):

    #----------------- helper functions --------------------#
    def sns_jointplot(data, x, y, xlim, ylim, hue, out_fig, markersize=10):
        g = sns.jointplot(
            data=data, x=x, y=y, kind='scatter', xlim=xlim, ylim=ylim,
            hue=hue,
            joint_kws={'s': markersize, 'alpha': 0.85},
            marginal_kws={'common_norm':False, 'fill': False, }
        )
        # customize the legend for better visiblity
        g.ax_joint.legend(markerscale=15/markersize, labelspacing=0.2, handletextpad=0, frameon=False)
        
        plt.xticks([]); plt.yticks([])
        plt.savefig(out_fig, dpi=300)
        plt.close()
    #--------------- End of helper functions ---------------#


    sns.set_theme(style='ticks', font_scale=1.5)

    df = load_features(gf_file, meta_file, min_neurons=min_neurons, standardize=True)

    cache_file = f'cache_{feature_reducer.lower()}.pkl'
    # map to the UMAP space
    use_precomputed_model = True
    if use_precomputed_model and os.path.exists(cache_file):
        print(f'--> Loading existing {feature_reducer} file')
        with open(cache_file, 'rb') as fp:
            emb = pickle.load(fp)
    else:
        if feature_reducer == 'UMAP':
            reducer = umap.UMAP(random_state=1024)
        elif feature_reducer == 'PCA':
            reducer = PCA(n_components=2)

        emb = reducer.fit_transform(df[LOCAL_FEATS])
        with open(cache_file, 'wb') as fp:
            pickle.dump(emb, fp)
    
    key1 = f'{feature_reducer}1'
    key2 = f'{feature_reducer}2'
    df[[key1, key2]] = emb
    sregions = sorted(np.unique(df.region))
    indices = [0, 4, 6, 8, 11, 14, 16, 20, 23, 25, 29]

    if feature_reducer == 'UMAP':
        xlim, ylim = (-4,14), (0, 10)
    elif feature_reducer == 'PCA':
        xlim, ylim = (-6,6), (-6,6)

    if 0:
        ii = 0
        for i1, i2 in zip(indices[:-1], indices[1:]):
            cur_regions = sregions[i1:i2]
            cur_df = df[df.region.isin(cur_regions)]
            print(i1, i2, cur_regions)
            print(f'--> Comparing feature distribution across brain regions')
            out_fig = f'{feature_reducer.lower()}_{ii:02d}.png'
            sns_jointplot(cur_df, key1, key2, xlim, ylim, 'region', out_fig, markersize=10)

            # visualize the variances across different person
            print(f'==> Comparing feature distributions across patients')
            for region_i in cur_regions:
                df_i = cur_df[cur_df.region == region_i]
                patients_i = df_i.patient
                ps = np.unique(patients_i)
                if ps.shape[0] > 1:
                    # plot the comparison of different samples of the same region
                    out_fig = f'{feature_reducer.lower()}_{region_i}.png'
                    sns_jointplot(df_i, key1, key2, xlim, ylim, 'patient', out_fig, markersize=10)
        
            ii += 1

    if layer_file is None:
        hue = None
    else:
        hue = 'layer'

    if 1:
        # For debug only
        dekock_file = '../resources/public_data/DeKock/gf_150um.csv_standardized.csv'
        df_dekock = pd.read_csv(dekock_file, index_col=0)
        # get the embedding for dekock data
        dekock_emb = reducer.fit_transform(df_dekock.iloc[:,:5])
        df_dekock[[key1, key2]] = dekock_emb
        out_fig = 'dekock_umap.png'
        sns_jointplot(df_dekock, key1, key2, xlim, ylim, 'region', out_fig, markersize=10)
        sys.exit()

    # left-right of the region in brain
    lr = [region[-1] for region in df.region]
    df['lr'] = lr
    
    layers = pd.read_csv(layer_file, index_col=0)
    df['layer'] = layers.loc[df.index]
    # Estimate the spatial auto-correlation
    layers_int = df.layer.map(dict((item, ind) for ind, item in enumerate(np.unique(df.layer))))
    moran_score = spatial_utils.moranI_score(df[[key1,key2]], feats=layers_int.values, weight_type='knn')
    print(f"The Moran's Index score for layer prediction is {moran_score:.3f}")
    
    # also for all neurons
    out_fig = f'{feature_reducer.lower()}_all.png'
    sns_jointplot(df, key1, key2, xlim, ylim, hue, out_fig, markersize=5)


if __name__ == '__main__':

    gf_file = '/data/kfchen/trace_ws/paper_auto_human_neuron_recon/unified_recon_1um/ptls10.csv'
    if 0:   # temporary
        from global_features import calc_global_features_from_folder
        swc_dir = '/data/kfchen/trace_ws/paper_auto_human_neuron_recon/unified_recon_1um/source500'
        outfile = 'gf_temp.csv'
        calc_global_features_from_folder(swc_dir, outfile)

    if 1:
        meta_file = '../meta/neuron_info_9060_utf8_curated0929.csv'
        layer_file = '../resources/public_data/DeKock/predicted_layers_thresholding_outliners.csv'
        #feature_distributions(gf_file, meta_file, min_neurons=5)
        joint_distributions(gf_file, meta_file, layer_file, feature_reducer='UMAP', min_neurons=0)


