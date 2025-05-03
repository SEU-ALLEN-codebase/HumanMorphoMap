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

from file_io import load_image

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
    

def merfish_vs_distance(merfish_file, gene_file, feat_file, region, layer=None):

    # helper functions
    def _plot(dff, num=50, zoom=False, figname='temp', nsample=10000, restrict_range=True):
        df = dff.copy()
        if df.shape[0] > nsample:
            df_sample = df.iloc[random.sample(range(df.shape[0]), nsample)]
        else:
            df_sample = df
        
        plt.figure(figsize=(8,8))
        
        # 创建分箱并计算统计量
        num_bins = num  # 假设num是之前定义的bins数量
        df['A_bin'] = pd.cut(df['euclidean_distance'], bins=np.linspace(0, 5.001, num_bins), right=False)

        # 计算每个bin的统计量（包括区间中点）
        bin_stats = df.groupby('A_bin')['feature_distance'].agg(['median', 'sem', 'count'])
        bin_stats['bin_center'] = [(interval.left + interval.right)/2 for interval in bin_stats.index]
        bin_stats = bin_stats[bin_stats['count'] > 50]  # 过滤低计数区间
        bin_stats.to_csv(f'{figname}_mean.csv', float_format='%.3f')

        # 绘图：点图+误差条（使用实际数值坐标）
        plt.errorbar(x=bin_stats['bin_center'],
                     y=bin_stats['median'],
                     yerr=bin_stats['sem'],  # 95% CI (改用sem则不需要*1.96)
                     fmt='o',
                     markersize=12,
                     color='black',
                     ecolor='gray',
                     elinewidth=3,
                     capsize=7,
                     capthick=3)

        # 添加趋势线（与统计分析一致）
        sns.regplot(x='bin_center', y='median', data=bin_stats,
                    scatter=False,
                    line_kws={'color':'red', 'linewidth':3, 'alpha':0.7},
                    lowess=True)

        # 统计分析（使用与实际坐标一致的数据）
        p_spearman = spearmanr(bin_stats['bin_center'], bin_stats['median'], alternative='greater')
        p_pearson = pearsonr(bin_stats['bin_center'], bin_stats['median'])
        print(f'Spearman: {p_spearman.statistic:.3f}, Pearson: {p_pearson.statistic:.3f}')

        slope, intercept, r_value, p_value, std_err = linregress(bin_stats['bin_center'], bin_stats['median'])
        print(f'Slope: {slope:.4f}, p-value: {p_value:.4g}')

        # 设置坐标轴范围
        plt.xlim(0, 5)
        # Adjust plot limits
        bin_centers = np.linspace(0, 5, num)[:-1] + (5/(num-1))/2

        if restrict_range:
            delta = 2.5
            ym = (bin_stats['median'].min() + bin_stats['median'].max())/2.
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
    #ctypes = df_f.loc[df_pca.index, 'cluster_L1'].apply(lambda x: 'EXC' if x=='EXC' else 'INH')

    if not layer:
        ctypes = df_f.loc[df_pca.index, 'cluster_L1']
        show_ctypes = np.unique(ctypes)
        restrict_range = True
    else:
        ctypes = df_f.loc[df_pca.index, 'cluster_L2']
        show_ctypes = ['eL2/3.IT', 'eL4/5.IT', 'eL5.IT', 'eL6.IT', 'eL6.CT', 'eL6.CAR3', 'eL6.b']
        restrict_range = False

    sns.set_theme(style='ticks', font_scale=2.2)
    for ctype in show_ctypes:
        ct_mask = ctypes == ctype
        xy_cur = xy[ct_mask]
        fpca_cur = fpca[ct_mask]
        print(ctype, ct_mask.sum())
        
        # pairwise distance and similarity
        fdists = pdist(fpca_cur)
        cdists = pdist(xy_cur) / 1000.0 # to mm
        dff = pd.DataFrame(np.array([cdists, fdists]).transpose(), 
                           columns=('euclidean_distance', 'feature_distance'))

        tname = ctype.replace("/", "").replace(".", "")
        figname = f'{tname}_merfish_{region}'
        _plot(dff, num=25, zoom=False, figname=figname, restrict_range=restrict_range)

        #import ipdb; ipdb.set_trace()
        print()

def calculate_volume_and_dimension(atlas_file, id_region):
    # load the image
    atlas = load_image(atlas_file)
    import ipdb; ipdb.set_trace()
    print()


if __name__ == '__main__':
    region = 'STG'
    if region == 'STG':
        merfish_file = f'../resources/human_merfish/H19/H19.30.001.{region}.250.expand.rep1.matrix.csv'
        gene_file = f'../resources/human_merfish/H19/H19.30.001.{region}.250.expand.rep1.genes.csv'
        feat_file = f'../resources/human_merfish/H19/H19.30.001.{region}.250.expand.rep1.features.csv'
    elif region == 'MTG':
        merfish_file = f'../resources/human_merfish/H18/H18.06.006.{region}.250.expand.rep1.matrix.csv'
        gene_file = f'../resources/human_merfish/H18/H18.06.006.{region}.250.expand.rep1.genes.csv'
        feat_file = f'../resources/human_merfish/H18/H18.06.006.{region}.250.expand.rep1.features.csv'

    layer = True
    merfish_vs_distance(merfish_file, gene_file, feat_file, region=region, layer=layer)

    if 0:
        atlas_file = '../resources/mni_icbm152_CerebrA_tal_nlin_sym_09c_u8.nii'
        id_mapper = {
            'MTG': 28,
            'STG': 45,
        }
        id_region = id_mapper[region]
        calculate_volume_and_dimension(atlas_file, id_region)
