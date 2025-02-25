##########################################################
#Author:          Yufeng Liu
#Create time:     2025-02-24
#Description:               
##########################################################
import os
import random
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist

import matplotlib.pyplot as plt
import seaborn as sns

def process_merfish(cgm):
    # remove low-expression cells
    gcounts = cgm.sum(axis=1)
    gbcounts = (cgm > 0).sum(axis=1)
    gthr = 45 #np.percentile(gcounts, 5)
    gbthr = 15 #np.percentile(gbcounts, 5)
    gmask = (gcounts >= gthr) & (gbcounts >= gbthr)
    # remove low-expression genes
    cbcounts = (cgm > 0).sum(axis=0)
    cbthr = 60 #np.percentile(cbcounts, 5)
    cmask = cbcounts >= cbthr
    
    filtered = cgm.loc[gmask, cgm.columns[cmask]]
    del cgm # release memory
    # For usual cases, mitochondria genes (gene name starts with "mt" in mouse and "MT" in human)
    # should be dicard, but there are no such genes in this dataset. So this step is skiped.

    # Then I would like to normalize each row
    gcounts_f = filtered.sum(axis=1)
    v_norm = 500 # int(((gcounts_f.mean() + gcounts_f.std()) / 100.)) * 100
    filtered = np.log1p(filtered / filtered.values.sum(axis=1, keepdims=True) * v_norm)
    # If multiple-batches or slices, do batch-wise correction. Not applicable here.
    # Extract high-variable genes
    gvar = filtered.std(axis=0)
    gvthr = np.partition(gvar, 50)[50] #np.percentile(gvar, 25)
    gvmask = gvar >= gvthr
    filtered = filtered.loc[:, filtered.columns[gvmask]]
    # Get the top-k components
    pca = PCA(50)
    fpca = pca.fit_transform(filtered)
    print(f'Remained variance ratio after PCA: {pca.explained_variance_ratio_.sum():.3f}')
    
    return fpca, filtered
    

def merfish_vs_distance(merfish_file, gene_file, feat_file):

    # helper functions
    def _plot(dff, num=50, zoom=False, figname='temp', nsample=10000):
        df = dff.copy()
        random.seed(1024)
        if df.shape[0] > nsample:
            df_sample = df.iloc[random.sample(range(df.shape[0]), nsample)]
        else:
            df_sample = df
        sns.scatterplot(df_sample, x='euclidean_distance', y='feature_distance', s=5, alpha=0.75)
        df['A_bin'] = pd.cut(df['euclidean_distance'], bins=np.linspace(0, 5.001, num), right=False)
        median_data = df.groupby('A_bin')['feature_distance'].mean().reset_index()
        median_data['A_bin_start'] = median_data['A_bin'].apply(lambda x: (x.left+x.right)/2.)
        sns.lineplot(x='A_bin_start', y='feature_distance', data=median_data, marker='o', color='r')

        if zoom:
            plt.xlim(0, 2.); plt.ylim(2, 8)

        plt.xlabel('Euclidean distance (mm)')
        plt.ylabel('Feature distance')
        plt.subplots_adjust(bottom=0.15, left=0.15)
        plt.savefig(f'{figname}.png', dpi=300); plt.close()
        print()


    #### end of helper functions

    df_g = pd.read_csv(gene_file)
    df_f = pd.read_csv(feat_file)

    # initialize the cell-by-gene matrix
    # For larger matrix, use sparse matrix
    cgm = np.zeros((df_f.shape[0], df_g.shape[0]))
    # load the data
    salients = pd.read_csv(merfish_file, index_col=0)
    cgm[salients.col.values-1, salients.index.values-1] = salients['val'].values
    cgm = pd.DataFrame(cgm, columns=df_g.name)
    fpca, df_pca = process_merfish(cgm)
    # get the coordinates
    xy = df_f.loc[df_pca.index, ['adjusted.x', 'adjusted.y']]
    # estimate the relationship for pyramidal cells and non-pyramidal cells
    ctypes = df_f.loc[df_pca.index, 'cluster_L1'].apply(lambda x: 'EXC' if x=='EXC' else 'INH')
    #ctypes = df_f.loc[df_pca.index, 'cluster_L1']
    
    sns.set_theme(style='ticks', font_scale=1.7)
    for ctype in np.unique(ctypes):
        print(ctype)
        ct_mask = ctypes == ctype
        xy_cur = xy[ct_mask]
        fpca_cur = fpca[ct_mask]
        
        # pairwise distance and similarity
        fdists = pdist(fpca_cur)
        cdists = pdist(xy_cur) / 1000.0 # to mm
        dff = pd.DataFrame(np.array([cdists, fdists]).transpose(), 
                           columns=('euclidean_distance', 'feature_distance'))

        figname = f'{ctype}_merfish'
        _plot(dff, num=25, zoom=False, figname=figname)

        #import ipdb; ipdb.set_trace()
        print()


if __name__ == '__main__':
    #merfish_file = '../resources/human_merfish/H19/H19.30.001.STG.250.expand.rep1.matrix.csv'
    #gene_file = '../resources/human_merfish/H19/H19.30.001.STG.250.expand.rep1.genes.csv'
    #feat_file = '../resources/human_merfish/H19/H19.30.001.STG.250.expand.rep1.features.csv'
    merfish_file = '../resources/human_merfish/H18/H18.06.006.MTG.250.expand.rep1.matrix.csv'
    gene_file = '../resources/human_merfish/H18/H18.06.006.MTG.250.expand.rep1.genes.csv'
    feat_file = '../resources/human_merfish/H18/H18.06.006.MTG.250.expand.rep1.features.csv'
    merfish_vs_distance(merfish_file, gene_file, feat_file)

