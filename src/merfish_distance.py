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
from scipy.stats import spearmanr, pearsonr, linregress
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
    pca = PCA(50, random_state=1024)
    fpca = pca.fit_transform(filtered)
    print(f'Remained variance ratio after PCA: {pca.explained_variance_ratio_.sum():.3f}')
    
    return fpca, filtered
    

def merfish_vs_distance(merfish_file, gene_file, feat_file, region):

    # helper functions
    def _plot(dff, num=50, zoom=False, figname='temp', nsample=10000):
        df = dff.copy()
        if df.shape[0] > nsample:
            df_sample = df.iloc[random.sample(range(df.shape[0]), nsample)]
        else:
            df_sample = df
        
        plt.figure(figsize=(8,8))
        #sns.scatterplot(df_sample, x='euclidean_distance', y='feature_distance', s=2, alpha=0.75, color='gray')
        df['A_bin'] = pd.cut(df['euclidean_distance'], bins=np.linspace(0, 5.001, num), right=False)
        median_data = df.groupby('A_bin')['feature_distance'].mean().reset_index()
        median_data['A_bin_start'] = median_data['A_bin'].apply(lambda x: (x.left+x.right)/2.)
        median_data['count'] = df.groupby('A_bin').count()['euclidean_distance'].values
        # save for subsequent analysis
        median_data.to_csv(f'{figname}_mean.csv', float_format='%.3f')
        median_data = median_data[~median_data.feature_distance.isna()]
        median_data = median_data[median_data['count'] > 30]

        #sns.lineplot(x='A_bin_start', y='feature_distance', data=median_data, marker='o', color='r')
        g = sns.regplot(x='A_bin_start', y='feature_distance', data=median_data,
                    scatter_kws={'s':100, 'alpha':0.75, 'color':'black'},
                    line_kws={'color':'red', 'alpha':0.5, 'linewidth':5},
                    lowess=True)

        p_spearman = spearmanr(median_data['A_bin_start'], median_data['feature_distance'], alternative='greater')
        p_pearson = pearsonr(median_data['A_bin_start'], median_data['feature_distance'])
        print(f'Spearman and pearson: {p_spearman.statistic:.3f}, {p_pearson.statistic:.3f}')
        slope, intercept, r_value, p_value, std_err = linregress(median_data['A_bin_start'], median_data['feature_distance'])
        print(f'Slope: {slope:.4f}')

        plt.xlim(0, 5.)
        delta = 2.5
        ym = (median_data['feature_distance'].min() + median_data['feature_distance'].max())/2.
        plt.ylim(ym-delta/2, ym+delta/2)

        plt.xlabel('Soma-soma distance (mm)')
        plt.ylabel('Transcriptomic distance')
        ax = plt.gca()
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['right'].set_linewidth(2)
        ax.spines['top'].set_linewidth(2)
        ax.tick_params(width=2)
        plt.subplots_adjust(bottom=0.15, left=0.15)
        plt.savefig(f'{figname}.png', dpi=300); plt.close()
        print()


    #### end of helper functions

    df_g = pd.read_csv(gene_file)
    df_f = pd.read_csv(feat_file)
    
    random.seed(1024)
    np.random.seed(1024)

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
    
    sns.set_theme(style='ticks', font_scale=2.2)
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

        figname = f'{ctype}_merfish_{region}'
        _plot(dff, num=30, zoom=False, figname=figname)

        #import ipdb; ipdb.set_trace()
        print()


if __name__ == '__main__':
    region = 'MTG'
    #merfish_file = f'../resources/human_merfish/H19/H19.30.001.{region}.250.expand.rep1.matrix.csv'
    #gene_file = f'../resources/human_merfish/H19/H19.30.001.{region}.250.expand.rep1.genes.csv'
    #feat_file = f'../resources/human_merfish/H19/H19.30.001.{region}.250.expand.rep1.features.csv'
    merfish_file = f'../resources/human_merfish/H18/H18.06.006.{region}.250.expand.rep1.matrix.csv'
    gene_file = f'../resources/human_merfish/H18/H18.06.006.{region}.250.expand.rep1.genes.csv'
    feat_file = f'../resources/human_merfish/H18/H18.06.006.{region}.250.expand.rep1.features.csv'
    merfish_vs_distance(merfish_file, gene_file, feat_file, region=region)

