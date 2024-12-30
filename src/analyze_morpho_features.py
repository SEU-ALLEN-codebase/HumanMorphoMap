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
import scipy.cluster.hierarchy as sch

from config import standardize_features, LOCAL_FEATS

from spatial import spatial_utils   # pylib
from ml.feature_processing import clip_outliners
from plotters.customized_plotters import sns_jointplot


def load_features(gf_file, meta_file, min_neurons=5, standardize=False, remove_na=True, use_local_features=True):
    # Loading the data
    df = pd.read_csv(gf_file, index_col=0)
    
    if use_local_features:
        fnames = LOCAL_FEATS
    else:
        fnames = df.columns

    df = df[fnames]
    
    meta = pd.read_csv(meta_file, index_col=0)
    col_reg = 'brain_region'
    # 
    sns.set_theme(style='ticks', font_scale=1)
    df['region'] = meta[col_reg]
    df['patient'] = meta['patient_id']
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
        standardize_features(dff, fnames, inplace=True)
    return dff


def feature_distributions(gf_file, meta_file, boxplot=True, min_neurons=5, immuno_id=None):
    df = load_features(gf_file, meta_file, min_neurons=min_neurons)
    sregions = sorted(np.unique(df.region))
    ly_mask = df.index <= immuno_id+1
    for stain in ['ly', 'immuno', 'all']:
        for feat in LOCAL_FEATS:
            if stain == 'ly':
                dfi = df[[feat, 'region']][ly_mask]
            elif stain == 'immuno':
                dfi = df[[feat, 'region']][~ly_mask]
            elif stain == 'all':
                dfi = df[[feat, 'region']]
            else:
                raise ValueError

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
            plt.savefig(f'{prefix}_{feat.replace(" ", "")}_{stain}.png', dpi=300)
            plt.close()


def joint_distributions(gf_file, meta_file, layer_file=None, min_neurons=5, feature_reducer='UMAP', immuno_id=None):
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
        # visualize the effect of patients and regions
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


    # visualize the layer distribution
    if layer_file is None:
        hue = None
    else:
        hue = 'layer'

    # left-right of the region in brain
    lr = [region[-1] for region in df.region]
    df['lr'] = lr
    
    layers = pd.read_csv(layer_file, index_col=0)
    df['layer'] = layers.loc[df.index]
    # Estimate the spatial auto-correlation
    layers_int = df.layer.map(dict((item, ind) for ind, item in enumerate(np.unique(df.layer))))
    moran_score = spatial_utils.moranI_score(df[[key1,key2]], feats=layers_int.values, weight_type='knn')
    print(f"The Moran's Index score for layer prediction is {moran_score:.3f}")
    
    df_ly = df[df.index <= immuno_id+1]
    df_im = df[df.index > immuno_id+1]
    # plot LY and immuno separately
    out_fig = f'{feature_reducer.lower()}_LY.png'
    sns_jointplot(df_ly, key1, key2, xlim, ylim, hue, out_fig, markersize=5)
    out_fig = f'{feature_reducer.lower()}_immuno.png'
    sns_jointplot(df_im, key1, key2, xlim, ylim, hue, out_fig, markersize=5)
    out_fig = f'{feature_reducer.lower()}_all.png'
    sns_jointplot(df, key1, key2, xlim, ylim, hue, out_fig, markersize=5)



def coembedding_dekock_seu(gf_seu_file, meta_seu_file, layer_seu_file, gf_dekock_file):
    df_seu = load_features(gf_seu_file, meta_file, min_neurons=0, standardize=True)
    # clip outliners
    clip_outliners(df_seu, col_ids=np.arange(df_seu.shape[1]-2))
    df_dekock = pd.read_csv(gf_dekock_file, index_col=0)

    # concatenate two datasets
    tmp_cache = 'tmp_reducer.pkl'
    if os.path.exists(tmp_cache):
        with open(tmp_cache, 'rb') as fp:
            reducer = pickle.load(fp)
    else:
        data = np.vstack((df_seu.iloc[:, :5], df_dekock.iloc[:,:5]))
        reducer = umap.UMAP(random_state=1024)
        reducer.fit(data)

        with open(tmp_cache, 'wb') as fp:
            pickle.dump(reducer, fp)

    emb_seu = reducer.transform(df_seu.iloc[:, :5])
    emb_dekock = reducer.transform(df_dekock.iloc[:, :5])

    key1, key2 = 'UMAP1', 'UMAP2'
    xlim, ylim = (-4,14), (0, 10)
    
    # get the embedding for dekock data
    df_dekock[[key1, key2]] = emb_dekock
    out_fig = 'umap_coembedding_dekock.png'
    sns_jointplot(df_dekock, key1, key2, xlim, ylim, 'region', out_fig, markersize=40)
    # for ourdata
    df_seu[[key1, key2]] = emb_seu
    layers = pd.read_csv(layer_seu_file, index_col=0)
    df_seu['layer'] = layers.loc[df_seu.index]
    out_fig = 'umap_coembedding_seu.png'
    sns_jointplot(df_seu, key1, key2, xlim, ylim, 'layer', out_fig, markersize=5)


def clustering(gf_file, meta_file, layer_file=None):
    sns.set_theme(style='ticks', font_scale=1.5)

    # use all features. This is because we have standardized the neurons
    df = load_features(gf_file, meta_file, min_neurons=0, standardize=True, use_local_features=False)
    layers = pd.read_csv(layer_file, index_col=0)


    df22 = df.iloc[:,:22].copy()    # features
    # clip the data for extreme values, as this is mostly deficiency in reconstruction
    dfc22 = df22.clip(-3, 3)
    # Do clustering
    clustmap = sns.clustermap(dfc22, cmap='bwr')
    # get the clusters
    row_linkage = clustmap.dendrogram_row.linkage
    row_clusters = sch.fcluster(row_linkage, t=100, criterion='maxclust')
    # the the clustermaping
    id2map = pd.DataFrame(np.transpose([dfc22.index, row_clusters]), columns=('idx', 'cluster')).set_index('idx')
    # index after clsuter map
    reordered_ind = clustmap.dendrogram_row.reordered_ind
    # dict(zip(*np.unique(row_clusters, return_counts=True)))
    # print original status
    orig_layer_distr = np.unique(layers, return_counts=True)
    print(f'Layer distribution of all neurons: {orig_layer_distr}')

    # major clusters
    c65 = id2map.cluster == 65
    c65_ids = c65.index[c65]
    print('c65: ', np.unique(layers.loc[c65_ids], return_counts=True))
    c79 = id2map.cluster == 79
    c79_ids = c79.index[c79]
    print('c79: ', np.unique(layers.loc[c79_ids], return_counts=True))
    c99 = id2map.cluster == 99
    c99_ids = c99.index[c99]
     print('c99: ', np.unique(layers.loc[c99_ids], return_counts=True))
    import ipdb; ipdb.set_trace()
    print()
    
    


if __name__ == '__main__':

    gf_file = '/data/kfchen/trace_ws/cropped_swc/proposed_1um_l_measure_total.csv'
    meta_file = '/data/kfchen/trace_ws/paper_trace_result/csv_copy/final_neuron_info.csv'
    layer_file = '../resources/public_data/DeKock/predicted_layers_thresholding_outliners.csv'
    immuno_id = 6208
    if 0:   # temporary
        from global_features import calc_global_features_from_folder
        swc_dir = '/data/kfchen/trace_ws/paper_auto_human_neuron_recon/unified_recon_1um/source500'
        outfile = 'gf_temp.csv'
        calc_global_features_from_folder(swc_dir, outfile)

    if 0:
        #feature_distributions(gf_file, meta_file, min_neurons=5, immuno_id=immuno_id)
        joint_distributions(gf_file, meta_file, layer_file, feature_reducer='UMAP', min_neurons=0, immuno_id=immuno_id)
    
    if 0:
        gf_dekock_file = '../resources/public_data/DeKock/gf_150um.csv_standardized.csv'
        coembedding_dekock_seu(gf_file, meta_file, layer_file, gf_dekock_file)

    if 1:
        clustering(gf_file, meta_file, layer_file)


