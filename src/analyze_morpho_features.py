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

from config import standardize_features, LOCAL_FEATS, LOCAL_FEATS2, REG2LOBE, mRMR_FEATS

from spatial import spatial_utils   # pylib
from ml.feature_processing import clip_outliners
from plotters.customized_plotters import sns_jointplot


def load_features(gf_crop_file, meta_file, min_neurons=5, standardize=False, remove_na=True, use_local_features=True, merge_lr=False):
    # Loading the data
    df = pd.read_csv(gf_crop_file, index_col=0)
    
    if use_local_features:
        fnames = LOCAL_FEATS
    else:
        fnames = df.columns

    df = df[fnames]
    
    meta = pd.read_csv(meta_file, index_col='cell_id', encoding='gbk', low_memory=False)
    assert((meta.index != df.index).sum() == 0)
    col_reg = 'brain_region'
    # 
    #print(np.unique(meta[col_reg])); sys.exit()
    if merge_lr:
        #regions = [r.replace('.L', '').replace('.R', '') for r in meta[col_reg]]
        regions = [REG2LOBE[r] for r in meta[col_reg]]
        df['region'] = regions
    else:
        df['region'] = meta[col_reg]

    df['patient'] = meta['patient_number']
    df['ihc'] = meta['immunohistochemistry']
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


def feature_distributions(gf_crop_file, meta_file, boxplot=True, min_neurons=5, immuno_id=None):
    df = load_features(gf_crop_file, meta_file, min_neurons=min_neurons)
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


def joint_distributions(gf_crop_file, meta_file, layer_file=None, min_neurons=5, feature_reducer='UMAP', immuno_id=None):
    sns.set_theme(style='ticks', font_scale=1.5)

    df = load_features(gf_crop_file, meta_file, min_neurons=min_neurons, standardize=True)

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


def clustering(gf_crop_file, meta_file, layer_file=None):

    # --------------- Helper functions ---------------- #
    def get_cluster_distr(df, cluster_id, pregs, layers, players=('L2/3', 'L4', 'L5/6')):
        cc = id2map.cluster == cluster_id
        cc_ids = cc.index[cc]
        #print(cluster_id, cc.sum())

        # region distribution
        rdict = dict(zip(*np.unique(df.region.loc[cc_ids], return_counts=True)))
        rdistr = np.zeros(len(pregs))
        for ir, r in enumerate(pregs):
            if r in rdict:
                rdistr[ir] = rdict[r]
            else:
                rdistr[ir] = 0

        # layer distribution
        ldistr = np.zeros(len(players))
        ldict = dict(zip(*np.unique(layers.loc[cc_ids], return_counts=True)))
        for il, l in enumerate(players):
            if l in ldict:
                ldistr[il] = ldict[l]
            else:
                ldistr[il] = 0
        # 
        iuniq, idistr = np.unique(df[cc].ihc, return_counts=True)

        return cc, cc_ids, rdistr, ldistr, idistr


    def plot_distributions(clusters, classes, distributions, outfig):
        proportions = np.array(distributions).astype(float)
        proportions /= proportions.sum(axis=1, keepdims=True)

        # Cumulative proportions for stacking
        cumulative = np.cumsum(proportions, axis=1)

        # Plot
        sns.set_theme(style='ticks', font_scale=2.0)
        fig, ax = plt.subplots(figsize=(8, 6))
        #colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#33ffff']
        colors = {ie: plt.cm.rainbow(each, bytes=False) for ie, each in enumerate(np.linspace(0, 1, len(classes)))}
        for i, cls in enumerate(classes):
            ax.bar(clusters, proportions[:, i], label=cls, color=colors[i], width=0.5,
                   bottom=(cumulative[:, i - 1] if i > 0 else 0))

        # Formatting
        tname = os.path.split(outfig)[-1].split('_')[0].capitalize()
        ax.set_ylabel('Proportion')
        if tname == 'Ihc':
            tname = 'IHC'
        ax.set_title(f'{tname} proportion across clusters')
        ax.legend(title=None, ncol=len(classes), alignment='center', frameon=False,
                  labelspacing=0.1, handletextpad=0.05, borderpad=0.05)
        ax.set_xlabel('Clusters')
        ax.spines['left'].set_linewidth(2)
        ax.spines['right'].set_linewidth(2)
        ax.spines['top'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        ax.xaxis.set_tick_params(width=2)
        ax.yaxis.set_tick_params(width=2)
        plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
        plt.tight_layout()
        plt.savefig(outfig, dpi=300)
        plt.close()
        print()
        

    # ---------- End of helper functions -------------- #


    sns.set_theme(style='ticks', font_scale=1.6)

    # use all features. This is because we have standardized the neurons
    df = load_features(gf_crop_file, meta_file, min_neurons=0, standardize=True, use_local_features=False, merge_lr=True)
    layers = pd.read_csv(layer_file, index_col=0)


    #df22 = df.iloc[:,:22].copy()    # features
    df22 = df[LOCAL_FEATS2].copy()

    # clip the data for extreme values, as this is mostly deficiency in reconstruction
    dfc22 = df22.clip(-3, 3)
    # do mRMR feature selection
    #import pymrmr
    #rdict = dict(zip(np.unique(df.region), range(len(np.unique(df.region)))))
    #rindices = [rdict[rname] for rname in df.region]
    #dfc22.loc[:, 'region'] = rindices
    #feats = pymrmr.mRMR(dfc22, 'MIQ', 10)
    #print(feats)

    # Do clustering
    #clustmap = sns.clustermap(dfc22[
    #        ['Number of Tips', 'Average Fragmentation', 'Max Euclidean Distance', 
    #         'Total Length', 'N_stem', 'Average Bifurcation Angle Remote']], cmap='RdBu')
    clustmap = sns.clustermap(dfc22, cmap='RdBu_r')
    
    # get the clusters
    row_linkage = clustmap.dendrogram_row.linkage
    row_clusters = sch.fcluster(row_linkage, t=60, criterion='maxclust')
    # the the clustermaping
    id2map = pd.DataFrame(np.transpose([dfc22.index, row_clusters]), columns=('idx', 'cluster')).set_index('idx')
    # index after clsuter map
    reordered_ind = clustmap.dendrogram_row.reordered_ind

    # extract the major clusters
    min_count = 200
    cids, ccnts = np.unique(row_clusters, return_counts=True)

    # assign colors
    salient_cluster_mask = ccnts > min_count
    salient_cluster_ids = cids[salient_cluster_mask]
    
    COLORS = 'rgbcmyk'
    row_cmap = {ie: COLORS[each]
                    for ie, each in zip(salient_cluster_ids, range(len(salient_cluster_ids)))}
    
    row_colors = []
    for idx in row_clusters:
        if idx in row_cmap:
            row_colors.append(row_cmap[idx])
        else:
            row_colors.append('w')
    
    plt.close() # close the previous figure
    clustmap = sns.clustermap(dfc22, cmap='RdBu_r', row_colors=row_colors, xticklabels=1,
                              cbar_pos=(0.3,0.05,0.02,0.05),
                              #cbar_kws=dict(orientation='horizontal')
                              )
    plt.setp(clustmap.ax_heatmap.get_xticklabels(), rotation=45, rotation_mode='anchor', ha='right')
    clustmap.ax_heatmap.set_ylabel('Neuron')
    clustmap.ax_heatmap.tick_params(left=False, right=False, labelleft=False, labelright=False)
    clustmap.ax_col_dendrogram.set_visible(False)
    #configuring the colorbar
    #clustmap.cax.set_xlabel('Standardized\nfeature', fontsize=12)
    clustmap.cax.tick_params(direction='in')

    # save image to  file
    plt.subplots_adjust(left=0.08, right=0.95)
    plt.savefig('clustermap.png', dpi=300); plt.close()

    # dict(zip(*np.unique(row_clusters, return_counts=True)))
    # print original status
    orig_layer_distr = np.unique(layers, return_counts=True)
    print(f'Layer distribution of all neurons: {orig_layer_distr}')


    # Overall layer distribution
    region_dict = dict(zip(*np.unique(df.region, return_counts=True)))
    pregs, pcount = [], 200
    # keep only non-composite areas
    rdistr = []
    for k, v in region_dict.items():
        if ('(' in k) or ('_' in k) or (v < pcount):
            continue
        else:
            pregs.append(k)
            rdistr.append(v)
    rdistr = np.array(rdistr)

    # major layers
    luniq, ldistr = np.unique(layers, return_counts=True)
    iuniq, idistr = np.unique(df.ihc, return_counts=True)

    show_overall = False
    if show_overall:
        ccs, cc_ids, cc_rdistrs, cc_ldistrs, cc_idistrs = [], [], [rdistr], [ldistr], [idistr]
        clusters = ['All']
    else:
        ccs, cc_ids, cc_rdistrs, cc_ldistrs, cc_idistrs = [], [], [], [], []
        clusters = []

    icluster = 0
    for cid in salient_cluster_ids:
        cc, cc_id, cc_rdistr, cc_ldistr, cc_idistr = get_cluster_distr(df, cid, pregs, layers)
        clusters.append(f'C{icluster+1}')
        ccs.append(cc)
        cc_ids.append(cc_id)
        cc_rdistrs.append(cc_rdistr)
        cc_ldistrs.append(cc_ldistr)
        cc_idistrs.append(cc_idistr)    # immuno

        icluster += 1


    plot_distributions(clusters, luniq, cc_ldistrs, 'layer_across_clusters.png')
    plot_distributions(clusters, pregs, cc_rdistrs, 'region_across_clusters.png')
    plot_distributions(clusters, ['w/ IHC', 'w/o IHC'], cc_idistrs, 'ihc_across_clusters.png')
    
    print()

       

    


if __name__ == '__main__':

    gf_crop_file = '/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta_filter/swc_1um_cropped_150um_l_measure.csv'
    gf_no_crop_file = '/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta_filter/l_measure_result.csv'
    meta_file = '/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta_filter/meta.csv'
    layer_file = '../resources/public_data/DeKock/predicted_layers_thresholding_outliners.csv'
    if 0:   # temporary
        from global_features import calc_global_features_from_folder
        swc_dir = '/data/kfchen/trace_ws/paper_auto_human_neuron_recon/unified_recon_1um/source500'
        outfile = 'gf_temp.csv'
        calc_global_features_from_folder(swc_dir, outfile)

    if 0:
        #feature_distributions(gf_crop_file, meta_file, min_neurons=5, immuno_id=immuno_id)
        joint_distributions(gf_crop_file, meta_file, layer_file, feature_reducer='UMAP', min_neurons=0, immuno_id=immuno_id)
    
    if 0:
        gf_dekock_file = '../resources/public_data/DeKock/gf_150um.csv_standardized.csv'
        coembedding_dekock_seu(gf_crop_file, meta_file, layer_file, gf_dekock_file)

    if 1:
        clustering(gf_crop_file, meta_file, layer_file)


