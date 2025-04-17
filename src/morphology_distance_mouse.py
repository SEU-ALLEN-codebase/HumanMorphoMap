##########################################################
#Author:          Yufeng Liu
#Create time:     2025-02-26
#Description:               
##########################################################
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.spatial.distance import pdist
from scipy.stats import spearmanr, linregress
from ml.feature_processing import standardize_features

# helper functions
def _plot(dff, num=50, figname='temp', nsample=10000, max_dist=5):
    df = dff.copy()
    df = df[df.euclidean_distance <= max_dist]
    random.seed(1024)
    if df.shape[0] > nsample:
        df_sample = df.iloc[random.sample(range(df.shape[0]), nsample)]
    else:
        df_sample = df
 
    plt.figure(figsize=(8,8))
    #g = sns.regplot(x='euclidean_distance', y='feature_distance', data=df_sample,
    #                scatter_kws={'s':4, 'alpha':0.5, 'color':'black'},
    #                line_kws={'color':'red', 'alpha':0.75, 'linewidth':3},
    #                lowess=True)

    '''   
    ax_scatter = sns.scatterplot(
        df_sample, 
        x='euclidean_distance', 
        y='feature_distance', 
        s=2, 
        alpha=0.3,
        color='black',
        edgecolor='none',
        rasterized=True
    )
    '''

    df['A_bin'] = pd.cut(df['euclidean_distance'], bins=np.linspace(0, max_dist+0.001, num), right=False)
    median_data = df.groupby('A_bin')['feature_distance'].median().reset_index()
    median_data['A_bin_start'] = median_data['A_bin'].apply(lambda x: (x.left+x.right)/2.)
    median_data['count'] = df.groupby('A_bin').count()['euclidean_distance'].values
    # remove low-count bins, to avoid randomness
    median_data = median_data[median_data['count'] > 30]

    #sns.lineplot(x='A_bin_start', y='feature_distance', data=median_data, marker='o', color='r')
    g = sns.regplot(x='A_bin_start', y='feature_distance', data=median_data,
                    scatter_kws={'s':100, 'alpha':0.75, 'color':'black'},
                    line_kws={'color':'red', 'alpha':0.5, 'linewidth':5},
                    lowess=True)

    p_spearman = spearmanr(median_data['A_bin_start'], median_data['feature_distance'], alternative='greater')
    print(f'Spearman coefficient: {p_spearman.statistic:.3f}')
    slope, intercept, r_value, p_value, std_err = linregress(median_data['A_bin_start'], median_data['feature_distance'])
    print(f'Slope, R_pearson, and p-value: {slope:.3f}, {r_value:.3f}, {p_value}')
    

    plt.xlim(0, max_dist)#; plt.ylim(4,6)
    delta = 2.5
    ym = (median_data['feature_distance'].min() + median_data['feature_distance'].max())/2.
    plt.ylim(ym-delta/2, ym+delta/2)

    plt.xlabel('Soma-soma distance (mm)')
    plt.ylabel('Morphological distance')
    ax = plt.gca()
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.tick_params(width=2)
    plt.subplots_adjust(bottom=0.15, left=0.15)
    plt.savefig(f'{figname}.png', dpi=300); plt.close()
    print()

def estimate_cortical_relations(feat_file, spos_file, sreg_file, figname='temp', subset=False):
    cregs = ('ACAd', 'ACAv', 'AId', 'AIp', 'AIv',
       'AUDd', 'AUDp', 'AUDpo', 'AUDv', 'ECT', 
       'FRP', 'MOp', 'MOs', 'ORBl', 'ORBm', 'ORBvl', 
       'RSPagl', 'RSPd', 'RSPv', 'SSp',
       'SSp-bfd', 'SSp-ll', 'SSp-m', 'SSp-n', 'SSp-tr', 'SSp-ul',
       'SSp-un', 'SSs', 'TEa', 'VISC', 'VISa',
       'VISal', 'VISam', 'VISl', 'VISli', 'VISp', 'VISpm', 'VISpor', 'VISrl')
    df = pd.read_csv(feat_file, index_col=0)
    spos = pd.read_csv(spos_file, index_col=0)
    sreg = pd.read_csv(sreg_file, index_col=0)

    df = df[df.index.isin(sreg.index)]
    # keep only isocortical neurons
    isoc = df[sreg.loc[df.index].isin(cregs).values[:,0]]
    # normalize the features
    standardize_features(isoc, isoc.columns, inplace=True)
    
    if subset:
        # use subset of features
        isoc = isoc[['Stems', 'AverageContraction', 'AverageFragmentation', 
                    'AverageParent-daughterRatio', 'AverageBifurcationAngleRemote', 
                    'AverageBifurcationAngleLocal']]
    # distance
    fdists = pdist(isoc)
    # 
    tmp_spos = spos.loc[isoc.index] / 1000. # to mm
    edists = pdist(tmp_spos)

    dff = pd.DataFrame(np.array([edists, fdists]).transpose(), columns=('euclidean_distance', 'feature_distance'))
    sns.set_theme(style='ticks', font_scale=2.2)
    _plot(dff, num=50, figname=figname)
    
    


if __name__ == '__main__':
    ntype = 'final'
    feat_file = f'/home/lyf/Research/publication/parcellation/BrainParcellation/microenviron/data/gf_S3_2um_{ntype}.csv'
    spos_file = '/home/lyf/Research/publication/parcellation/BrainParcellation/evaluation/data/1891_soma_pos.csv'
    sreg_file = '/home/lyf/Research/publication/parcellation/BrainParcellation/evaluation/data/1876_soma_region.csv'
    estimate_cortical_relations(feat_file, spos_file, sreg_file, figname=f'euc_feat_mouse_{ntype}', subset=False)

