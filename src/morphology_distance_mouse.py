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
from scipy.stats import spearmanr, pearsonr, linregress
from ml.feature_processing import standardize_features


__CREGS__ = ('ACAd', 'ACAv', 'AId', 'AIp', 'AIv',
       'AUDd', 'AUDp', 'AUDpo', 'AUDv', 'ECT', 
       'FRP', 'MOp', 'MOs', 'ORBl', 'ORBm', 'ORBvl', 
       'RSPagl', 'RSPd', 'RSPv', 'SSp',
       'SSp-bfd', 'SSp-ll', 'SSp-m', 'SSp-n', 'SSp-tr', 'SSp-ul',
       'SSp-un', 'SSs', 'TEa', 'VISC', 'VISa',
       'VISal', 'VISam', 'VISl', 'VISli', 'VISp', 'VISpm', 'VISpor', 'VISrl')

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

    # 创建分箱并计算统计量
    num_bins = num  # 假设num是之前定义的bins数量
    df['A_bin'] = pd.cut(df['euclidean_distance'], bins=np.linspace(0, max_dist + 0.001, num_bins), right=False)

    # 计算每个bin的统计量（包括区间中点）
    bin_stats = df.groupby('A_bin')['feature_distance'].agg(['median', 'sem', 'count'])
    bin_stats['bin_center'] = [(interval.left + interval.right)/2 for interval in bin_stats.index]
    bin_stats = bin_stats[bin_stats['count'] > 30]  # 过滤低计数区间
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
    delta = 2.5
    ym = (bin_stats['median'].min() + bin_stats['median'].max())/2.
    plt.ylim(ym-delta/2, ym+delta/2)

    # 优化坐标轴标签
    plt.xlabel('Euclidean Distance')
    plt.ylabel('Feature Distance (median ± 95% CI)')

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

def estimate_cortical_relations(feat_file, meta_file, py_file, region=None, layer=None, figname='temp', subset=False):
    df = pd.read_csv(feat_file, index_col=0)
    meta = pd.read_csv(meta_file, index_col=0)

    df = df[df.index.isin(meta.index)]
    # keep only isocortical neurons
    if (layer is None) and (region is not None):
        cregs = (region,)
        figname = f'{figname}_{region}'
    else:
        cregs = __CREGS__
        figname = f'{figname}_allRegions'

    isoc = df[meta.loc[df.index, 'Soma region'].isin(cregs).values]
    # normalize the features
    standardize_features(isoc, isoc.columns, inplace=True)
    # keep only pyramidal file
    with open(py_file) as fp:
        py_names = [line.strip() for line in fp.readlines()]

    isoc = isoc[isoc.index.isin(py_names)]
    # get the layers
    if layer is not None:
        layers = meta.loc[isoc.index, 'Cortical Lamination of soma']
        isoc = isoc[layers == layer]
        figname = f'{figname}_{layer.replace("/", "")}'
    else:
        figname = f'{figname}_allLayers'
    
    if subset:
        # use subset of features
        isoc = isoc[['Stems', 'AverageContraction', 'AverageFragmentation', 
                    'AverageParent-daughterRatio', 'AverageBifurcationAngleRemote', 
                    'AverageBifurcationAngleLocal']]
    # distance
    fdists = pdist(isoc)
    # 
    tmp_spos = meta.loc[isoc.index, ['Soma_X(CCFv3_1𝜇𝑚)', 'Soma_Y(CCFv3_1𝜇𝑚)', 'Soma_Z(CCFv3_1𝜇𝑚)']] / 1000. # to mm
    #tmp_spos = spos.loc[isoc.index] / 1000. # to mm
    edists = pdist(tmp_spos)

    dff = pd.DataFrame(np.array([edists, fdists]).transpose(), columns=('euclidean_distance', 'feature_distance'))
    sns.set_theme(style='ticks', font_scale=2.2)
    _plot(dff, num=50, figname=figname)
    
    


if __name__ == '__main__':
    ntype = 'dendrite'
    feat_file = f'/home/lyf/Research/publication/parcellation/BrainParcellation/microenviron/data/gf_S3_2um_{ntype}.csv'
    meta_file = './data/TableS6_Full_morphometry_1222.csv'
    py_file = './data/apical_1886_v20231211.txt'
    region = None #'SSp-bfd'
    layer = 'L2/3'
    estimate_cortical_relations(feat_file, meta_file, py_file, region=region, layer=layer, figname=f'euc_feat_mouse_{ntype}', subset=False)

