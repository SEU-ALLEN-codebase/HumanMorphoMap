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


def estimate_principal_axes(feat_file, cell_name='eL4/5.IT', visualize=True):
    # 1. 加载数据并提取目标层细胞
    if type(feat_file) is str:
        df = pd.read_csv(feat_file)
    else:
        df = feat_file
        
    target_cells = df[df['cluster_L2'] == cell_name]  # 或 'eL2/3.IT'
    points = target_cells[['adjusted.x', 'adjusted.y']].values / 1000  # 转换为毫米单位

    # 2. 使用PCA计算主轴
    pca = PCA(n_components=2)
    pca.fit(points)
    center = pca.mean_  # 中心点（毫米单位）
    primary_axis = pca.components_[0]  # 第一主轴方向（单位向量）
    secondary_axis = pca.components_[1]  # 第二主轴方向（单位向量）

    # 3. 动态计算主轴长度（覆盖数据范围）
    def get_axis_line(center, direction, points):
        # 将点投影到主轴上，计算投影范围
        projections = (points - center) @ direction
        min_proj, max_proj = np.min(projections), np.max(projections)
        # 生成主轴线段
        line = np.vstack([
            center + min_proj * direction,
            center + max_proj * direction
        ])
        return line

    primary_line = get_axis_line(center, primary_axis, points)
    secondary_line = get_axis_line(center, secondary_axis, points)

    if visualize:
        # 4. 可视化
        sns.set_theme(style='ticks', font_scale=1.7)
        plt.figure(figsize=(8, 6))
        plt.scatter(points[:, 0], points[:, 1], s=4, alpha=0.3, label='Cells (mm)')
        plt.plot(primary_line[:, 0], primary_line[:, 1], 'r-', linewidth=2, label='Primary Axis')
        plt.plot(secondary_line[:, 0], secondary_line[:, 1], 'b-', linewidth=2, label='Secondary Axis')
        plt.scatter(center[0], center[1], c='black', s=50, marker='^', label='Center')
        plt.legend(markerscale=1.5, frameon=False)
        plt.title('PCA-based Linear Axes (mm scale)')
        plt.xlabel('adjusted.x (mm)')
        plt.ylabel('adjusted.y (mm)')
        plt.axis('equal')  # 保持坐标轴比例一致
        tname = cell_name.replace("/", "").replace(".", "")
        plt.savefig(f'principal_axes_{tname}.png', dpi=300)
        plt.close()

    # 输出主轴信息
    print(f"Primary Axis Direction (unit vector): {primary_axis}")
    print(f"Secondary Axis Direction (unit vector): {secondary_axis}")
    print(f"Center Point (mm): {center}")
    return primary_axis, secondary_axis, center


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

def merfish_vs_distance(merfish_file, gene_file, feat_file, region, layer=None):
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
        tname = ctype.replace("/", "").replace(".", "")
        figname = f'{tname}_merfish_{region}'
        xy_cur = xy[ct_mask]
        
        fpca_cur = fpca[ct_mask]
        print(ctype, ct_mask.sum())
        
        # pairwise distance and similarity
        fdists = pdist(fpca_cur)
        cdists = pdist(xy_cur) / 1000.0 # to mm
        dff = pd.DataFrame(np.array([cdists, fdists]).transpose(), 
                           columns=('euclidean_distance', 'feature_distance'))

        _plot(dff, num=25, zoom=False, figname=figname, restrict_range=restrict_range)


def split_by_pc2_quantiles(xy_cur, fpca_cur, pcs, pc_id, center, quantiles=[0.25, 0.5, 0.75], visualize=True):
    """
    按pc2方向的分位数将点分为4组
    输入:
        xy_cur: 细胞坐标数组 (Nx2)
        pc2: 第二主轴方向（单位向量）
        center: 中心点坐标
        quantiles: 分位数列表（默认[0.25, 0.5, 0.75]）
    返回:
        groups: 字典，key为分位区间名，value为对应坐标数组
    """
    # 计算每个点在pc2方向上的投影值（相对于中心点
    pc = pcs[pc_id]
    proj = (xy_cur - center) @ pc

    # 计算分位点
    q = np.quantile(proj, quantiles)

    # 分组
    groups = {
        '0-25%': (xy_cur[proj <= q[0]], fpca_cur[proj <= q[0]]),
        '25-50%': (xy_cur[(proj > q[0]) & (proj <= q[1])], fpca_cur[(proj > q[0]) & (proj <= q[1])]),
        '50-75%': (xy_cur[(proj > q[1]) & (proj <= q[2])], fpca_cur[(proj > q[1]) & (proj <= q[2])]),
        '75-100%': (xy_cur[proj > q[2]], fpca_cur[proj > q[2]])
    }

    sns.set_theme(style="ticks", font_scale=1.5)

    if visualize:
        plt.figure(figsize=(8, 6))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        for i, (name, pts) in enumerate(groups.items()):
            pts_v = pts[0].values
            plt.scatter(pts_v[:,0], pts_v[:,1], s=5, alpha=0.6, label=name, color=colors[i])

        plt.legend(frameon=False, markerscale=3)
        plt.xlabel('adjusted.x (mm)')
        plt.ylabel('adjusted.y (mm)')
        sns.despine()
        plt.savefig(f'sublayers_pc{pc_id}.png', dpi=300)
        plt.close()

    return groups

def merfish_vs_distance_sublayers(merfish_file, gene_file, feat_file, region):
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
    

    ctypes = df_f.loc[df_pca.index, 'cluster_L2']
    ctype = 'eL2/3.IT'
    restrict_range = False
    pc1, pc2, center = estimate_principal_axes(df_f, cell_name='eL2/3.IT')
        
    sns.set_theme(style='ticks', font_scale=2.2)
    ct_mask = ctypes == ctype
    tname = ctype.replace("/", "").replace(".", "")
    figname = f'{tname}_merfish_{region}'
    xy_cur = xy[ct_mask]
    fpca_cur = fpca[ct_mask]
    
    # get sublayers by percentiles
    pc_id = 0
    groups = split_by_pc2_quantiles(xy_cur, fpca_cur, (pc1, pc2), pc_id, center)
    # 检查每组点数
    for name, pts in groups.items():
        print(f"{name}: {len(pts[0])} points")
    
    for name, (xy_cur_i,fpca_cur_i) in groups.items():
        # pairwise distance and similarity
        fdists = pdist(fpca_cur_i)
        cdists = pdist(xy_cur_i) / 1000.0 # to mm
        dff = pd.DataFrame(np.array([cdists, fdists]).transpose(), 
                           columns=('euclidean_distance', 'feature_distance'))

        figname_cur = f'{figname}_pc{pc_id}_{name.replace("%", "percentile")}'
        _plot(dff, num=25, zoom=False, figname=figname_cur, restrict_range=restrict_range)

        #import ipdb; ipdb.set_trace()
        print()



if __name__ == '__main__':
    region = 'MTG'
    if region == 'STG':
        merfish_file = f'../resources/human_merfish/H19/H19.30.001.{region}.250.expand.rep1.matrix.csv'
        gene_file = f'../resources/human_merfish/H19/H19.30.001.{region}.250.expand.rep1.genes.csv'
        feat_file = f'../resources/human_merfish/H19/H19.30.001.{region}.250.expand.rep1.features.csv'
    elif region == 'MTG':
        merfish_file = f'../resources/human_merfish/H18/H18.06.006.{region}.250.expand.rep1.matrix.csv'
        gene_file = f'../resources/human_merfish/H18/H18.06.006.{region}.250.expand.rep1.genes.csv'
        feat_file = f'../resources/human_merfish/H18/H18.06.006.{region}.250.expand.rep1.features.csv'

    layer = True
    #merfish_vs_distance(merfish_file, gene_file, feat_file, region=region, layer=layer) 
    merfish_vs_distance_sublayers(merfish_file, gene_file, feat_file, region)

    if 0:
        atlas_file = '../resources/mni_icbm152_CerebrA_tal_nlin_sym_09c_u8.nii'
        id_mapper = {
            'MTG': 28,
            'STG': 45,
        }
        id_region = id_mapper[region]
        calculate_volume_and_dimension(atlas_file, id_region)
