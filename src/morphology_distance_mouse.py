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
    sns.kdeplot(
        x='euclidean_distance', 
        y='feature_distance',
        data=df,
        cmap='viridis', 
        levels=5,     # 增加等高线层级
        ax=ax_scatter,
        color='red',
        linewdith=1,
    )
    '''


    df['A_bin'] = pd.cut(df['euclidean_distance'], bins=np.linspace(0, max_dist+0.001, num), right=False)
    median_data = df.groupby('A_bin')['feature_distance'].mean().reset_index()
    median_data['A_bin_start'] = median_data['A_bin'].apply(lambda x: (x.left+x.right)/2.)
    median_data['count'] = df.groupby('A_bin').count()['euclidean_distance'].values
    # remove low-count bins, to avoid randomness
    median_data = median_data[median_data['count'] > 30]

    sns.lineplot(x='A_bin_start', y='feature_distance', data=median_data, marker='o', color='r')
    p_spearman = spearmanr(median_data['A_bin_start'], median_data['feature_distance'], alternative='greater')
    print(f'Spearman coefficient: {p_spearman.statistic:.3f}')
    slope, intercept, r_value, p_value, std_err = linregress(median_data['A_bin_start'], median_data['feature_distance'])
    print(f'Slope: {slope:.4f}')

    plt.xlim(0, max_dist); plt.ylim(2, 10)

    plt.xlabel('Euclidean distance (mm)')
    plt.ylabel('Feature distance')
    plt.subplots_adjust(bottom=0.15, left=0.15)
    plt.savefig(f'{figname}.png', dpi=300); plt.close()
    print()

def estimate_cortical_relations(feat_file, spos_file, sreg_file, figname='temp'):
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
    # distance
    edists = pdist(isoc)
    # 
    tmp_spos = spos.loc[isoc.index] / 1000. # to mm
    fdists = pdist(tmp_spos)

    dff = pd.DataFrame(np.array([edists, fdists]).transpose(), columns=('euclidean_distance', 'feature_distance'))
    sns.set_theme(style='ticks', font_scale=1.7)
    _plot(dff, num=50, figname=figname)
    
    


if __name__ == '__main__':
    ntype = 'dendrite'
    feat_file = f'/home/lyf/Research/publication/parcellation/BrainParcellation/microenviron/data/gf_S3_2um_{ntype}.csv'
    spos_file = '/home/lyf/Research/publication/parcellation/BrainParcellation/evaluation/data/1891_soma_pos.csv'
    sreg_file = '/home/lyf/Research/publication/parcellation/BrainParcellation/evaluation/data/1876_soma_region.csv'
    estimate_cortical_relations(feat_file, spos_file, sreg_file, figname=f'euc_feat_mouse_{ntype}')

