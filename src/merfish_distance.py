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
from matplotlib.ticker import MultipleLocator
import seaborn as sns

from file_io import load_image

__COLORS4__ = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']


def hex_to_rgba(hex_color, alpha):
    hex_color = hex_color.lstrip('#')  # 移除开头的 '#'
    return tuple([int(hex_color[i:i+2], 16)/255. for i in (0, 2, 4)] + [alpha])  # 每两位解析为十进制

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
        plt.figure(figsize=(8, 8))
        plt.scatter(points[:, 0], points[:, 1], s=4, alpha=0.5, label='Cells (mm)')
        plt.plot(primary_line[:, 0], primary_line[:, 1], 'r-', linewidth=2, label='Primary Axis')
        plt.plot(secondary_line[:, 0], secondary_line[:, 1], 'b-', linewidth=2, label='Secondary Axis')
        plt.scatter(center[0], center[1], c='black', s=50, marker='^', label='Center')
        plt.legend(markerscale=1.5, frameon=False)
        plt.title('PCA-based Linear Axes (mm scale)')
        plt.xlabel('X coordinates (mm)')
        plt.ylabel('Y coordinates (mm)')
        plt.axis('equal')  # 保持坐标轴比例一致
        tname = cell_name.replace("/", "").replace(".", "")
        plt.savefig(f'principal_axes_{tname}.png', dpi=300)
        plt.close()

    # 输出主轴信息
    print(f"Primary Axis Direction (unit vector): {primary_axis}")
    print(f"Secondary Axis Direction (unit vector): {secondary_axis}")
    print(f"Center Point (mm): {center}")
    return primary_axis, secondary_axis, center


def _plot(dff, num=50, zoom=False, figname='temp', nsample=10000, restrict_range=True, color='black', overall_distribution=False):
    df = dff.copy()
    if df.shape[0] > nsample:
        df_sample = df.iloc[random.sample(range(df.shape[0]), nsample)]
    else:
        df_sample = df

    xn, yn = 'euclidean_distance', 'feature_distance'
    xlim0, xlim1 = 0, 3
    if overall_distribution:
        sns.set_theme(style='ticks', font_scale=1.8)
        plt.figure(figsize=(8,8))
        #sns.scatterplot(df, x='euclidean_distance', y='feature_distance', s=5,
        #                alpha=0.3, edgecolor='none', rasterized=True, color='black')
        sns.displot(df, x=xn, y=yn, cmap='Reds',
                    #cbar=True,
                    #cbar_kws={"label": "Count", 'aspect': 5})
        figname = figname + '_overall'

        plt.xlim(xlim0, xlim1)
    else:
        sns.set_theme(style='ticks', font_scale=2.4)
        plt.figure(figsize=(8,8))
        
        # 创建分箱并计算统计量
        num_bins = num  # 假设num是之前定义的bins数量
        df['A_bin'] = pd.cut(df[xn], bins=np.linspace(0, 5.001, num_bins), right=False)

        # 计算每个bin的统计量（包括区间中点）
        bin_stats = df.groupby('A_bin')[yn].agg(['median', 'mean', 'sem', 'count'])
        bin_stats['bin_center'] = [(interval.left + interval.right)/2 for interval in bin_stats.index]
        bin_stats = bin_stats[bin_stats['count'] > 50]  # 过滤低计数区间
        bin_stats.to_csv(f'{figname}_mean.csv', float_format='%.3f')

        # 绘图：点图+误差条（使用实际数值坐标）
        plt.errorbar(x=bin_stats['bin_center'],
                     y=bin_stats['mean'],
                     yerr=bin_stats['sem'],  # 95% CI (改用sem则不需要*1.96)
                     fmt='o',
                     markersize=12,
                     color='black',
                     ecolor='gray',
                     elinewidth=3,
                     capsize=7,
                     capthick=3)

        # 添加趋势线（与统计分析一致）
        sns.regplot(x='bin_center', y='mean', data=bin_stats, color=color,
                    scatter=False,
                    line_kws={'color':'red', 'linewidth':3, 'alpha':0.7},
                    lowess=True)

        # 统计分析（使用与实际坐标一致的数据）
        p_spearman = spearmanr(bin_stats['bin_center'], bin_stats['mean'], alternative='greater')
        p_pearson = pearsonr(bin_stats['bin_center'], bin_stats['mean'])
        print(f'Spearman: {p_spearman.statistic:.3f}, Pearson: {p_pearson.statistic:.3f}')

        slope, intercept, r_value, p_value, std_err = linregress(bin_stats['bin_center'], bin_stats['mean'])
        print(f'Slope: {slope:.4f}, p-value: {p_value:.4g}')

        # 设置坐标轴范围
        plt.xlim(xlim0, xlim1)
        # Adjust plot limits
        bin_centers = np.linspace(0, 5, num)[:-1] + (5/(num-1))/2

        if restrict_range:
            delta = 2.5
            ym = (bin_stats['mean'].min() + bin_stats['mean'].max())/2.
            plt.ylim(ym-delta/2, ym+delta/2)

    plt.xticks(np.arange(xlim0, xlim1+1))
    plt.xlabel('Soma-soma distance (mm)')
    plt.ylabel('Transcriptomic dissimilarity')
    ax = plt.gca()
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.tick_params(width=2)

    if overall_distribution:
        ax.yaxis.set_major_locator(MultipleLocator(2))
    else:
        ax.yaxis.set_major_locator(MultipleLocator(0.5))

    plt.subplots_adjust(bottom=0.16, left=0.16)
    plt.savefig(f'{figname}.png', dpi=300); plt.close()


def plot_combined(dff_groups, num=25, zoom=False, figname='combined', restrict_range=True):
    """
    将所有分组的子图合并到同一个图中
    输入:
        dff_groups: 字典，key为分组名，value为对应的DataFrame
        num: 分箱数量
        figname: 输出文件名前缀
        restrict_range: 是否限制y轴范围
    """
    sns.set_theme(style='ticks', font_scale=2.4)
    plt.figure(figsize=(8, 8))

    xn, yn = 'euclidean_distance', 'feature_distance'
    
    # 为每个分组绘制图形
    for icur, (name, dff) in enumerate(dff_groups.items()):
        df = dff.copy()
        
        # 创建分箱并计算统计量
        df['A_bin'] = pd.cut(df[xn], bins=np.linspace(0, 5.001, num), right=False)
        
        # 计算每个bin的统计量
        bin_stats = df.groupby('A_bin')[yn].agg(['median', 'mean', 'sem', 'count'])
        bin_stats['bin_center'] = [(interval.left + interval.right)/2 for interval in bin_stats.index]
        bin_stats = bin_stats[bin_stats['count'] > 40]  # 过滤低计数区间
        
        # 为当前分组选择颜色
        color = __COLORS4__[icur] if '__COLORS4__' in globals() else f'C{icur}'
        
        # 绘制当前分组的点图和误差条
        errorbar_alpha = 0.5
        errorbar = plt.errorbar(x=bin_stats['bin_center'],
                     y=bin_stats['mean'],
                     yerr=bin_stats['sem'],
                     fmt='o',
                     markersize=10,
                     color=color,
                     ecolor=hex_to_rgba(color, errorbar_alpha),
                     elinewidth=2,
                     capsize=5,
                     capthick=2,
                     label=name)
        # 单独设置caps的透明度
        for cap in errorbar[1]:  # errorbar[1]对应caps的Line2D对象
            cap.set_alpha(errorbar_alpha)   # 设置alpha值
        
        # 添加趋势线
        sns.regplot(x='bin_center', y='mean', data=bin_stats, color=color,
                    scatter=False,
                    line_kws={'linewidth':2, 'alpha':0.7},
                    lowess=True)
        
        # 打印统计信息
        p_spearman = spearmanr(bin_stats['bin_center'], bin_stats['mean'], alternative='greater')
        p_pearson = pearsonr(bin_stats['bin_center'], bin_stats['mean'])
        print(f'{name} - Spearman: {p_spearman.statistic:.3f}, Pearson: {p_pearson.statistic:.3f}')
        
        slope, intercept, r_value, p_value, std_err = linregress(bin_stats['bin_center'], bin_stats['mean'])
        print(f'{name} - Slope: {slope:.4f}, p-value: {p_value:.4g}\n')
    
    # 设置图形样式
    plt.xlim(0, 5)
    plt.xticks([0,1,2,4,5])
    if restrict_range:
        # 计算所有分组的共同y轴范围
        all_medians = []
        for dff in dff_groups.values():
            df = dff.copy()
            df['A_bin'] = pd.cut(df[xn], bins=np.linspace(0, 5.001, num), right=False)
            bin_stats = df.groupby('A_bin')[yn].agg(['mean'])
            all_medians.extend(bin_stats['mean'].values)
        
        if all_medians:
            delta = 2.5
            ym = (min(all_medians) + max(all_medians))/2.
            plt.ylim(ym-delta/2, ym+delta/2)
    
    # 设置坐标轴和标签
    plt.xlabel('Soma-soma distance (mm)')
    plt.ylabel('Transcriptomic dissimilarity')
    plt.legend(frameon=False, markerscale=1.6, handletextpad=0.10)
    
    # 设置边框和刻度
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.tick_params(width=2)

    ymin, ymax = ax.get_ylim()
    if (ymax - ymin) < 4:
        ytick_step = 0.5
    elif (ymax - ymin) > 8:
        ytick_step = 2
    else:
        ytick_step = 1

    ax.yaxis.set_major_locator(MultipleLocator(ytick_step))  # 每隔1单位显示一个刻度
    ax.xaxis.set_major_locator(MultipleLocator(1))  # 每隔2单位显示一个刻度
    
    plt.subplots_adjust(bottom=0.15, left=0.15)
    plt.savefig(f'{figname}.png', dpi=300)
    plt.close()


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

        overall_distribution = True
        _plot(dff, num=25, zoom=False, figname=figname, restrict_range=restrict_range, 
              overall_distribution=overall_distribution)


def split_by_pc2_quantiles(xy_cur, fpca_cur, pc, pc_id, center, quantiles=[0.25, 0.5, 0.75], visualize=True, cell_name='eL2/3.IT'):
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

    sns.set_theme(style="ticks", font_scale=2.1)

    if visualize:
        plt.figure(figsize=(8, 8))
        colors = __COLORS4__
        for i, (name, pts) in enumerate(groups.items()):
            pts_v = pts[0].values / 1000.
            plt.scatter(pts_v[:,0], pts_v[:,1], s=12, alpha=0.6, label=name, color=colors[i])

        plt.legend(frameon=False, markerscale=3, handletextpad=0.10)
        ax = plt.gca()
        ax.yaxis.set_major_locator(MultipleLocator(1))  # 每隔1单位显示一个刻度
        ax.xaxis.set_major_locator(MultipleLocator(1))  # 每隔2单位显示一个刻度
        plt.xlabel('X coordinates (mm)')
        plt.ylabel('Y coordinates (mm)')
        #sns.despine()
        plt.axis('off')
        tname = cell_name.replace("/", "").replace(".", "")
        plt.savefig(f'sublayers_pc{pc_id}_{tname}.png', dpi=300)
        plt.close()

    return groups

def merfish_vs_distance_sublayers(merfish_file, gene_file, feat_file, region):

    ##################### Helper functions #######################
    def get_subpairs(xy_cur_i, fpca_cur_i, pci, dist_th):
        points = xy_cur_i[['adjusted.x', 'adjusted.y']].values  # 形状 (n, 2)
        indices = xy_cur_i.index.values  # 点对应的原始索引

        # --- 向量化计算 ---
        n = len(points)
        diff = points[:, None, :] - points[None, :, :]  # 形状 (n, n, 2)

        # 欧几里得距离矩阵
        dist_matrix = np.linalg.norm(diff, axis=2)  # 形状 (n, n)

        # 沿 pci 方向的投影距离矩阵
        proj_dist = np.abs(np.dot(diff, pci))  # 形状 (n, n)

        # 筛选条件：投影距离 <= 阈值，且 i < j（避免重复和自身）
        # 生成上三角矩阵的掩码 (i < j)
        triu_mask = np.triu_indices(n, k=1)  # 返回 (row_indices, col_indices)

        # 筛选条件：投影距离 <= 阈值，且 i < j
        mask = np.zeros_like(proj_dist, dtype=bool)
        mask[triu_mask] = proj_dist[triu_mask] <= dist_th

        # 获取符合条件的点对索引和距离
        rows, cols = np.where(mask)
        valid_indices = list(zip(indices[rows], indices[cols]))
        valid_distances = dist_matrix[rows, cols]

        # distance in feature space
        dist_feats = np.linalg.norm(fpca_cur_i[:,None,:] - fpca_cur_i[None,:,:], axis=2)
        valid_dist_feats = dist_feats[rows, cols]

        # 转换为DataFrame
        result_df = pd.DataFrame({
            'Index1': [pair[0] for pair in valid_indices],
            'Index2': [pair[1] for pair in valid_indices],
            'Distance': valid_distances,
            'Distance_feature': valid_dist_feats,
        })

        #print(result_df)
        return result_df
        
    ################## End of helper functions ###################    


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
    pc_id = 1
    dist_th = 0.3


    restrict_range = False
    pc1, pc2, center = estimate_principal_axes(df_f, cell_name='eL2/3.IT')

    if pc_id == 0:
        pci, pcj = pc1, pc2
    elif pc_id == 1:
        pci, pcj = pc2, pc1
    else:
        raise ValueError('Incorrect `pc_id` value!')
        
    sns.set_theme(style='ticks', font_scale=2.2)
    ct_mask = ctypes == ctype
    tname = ctype.replace("/", "").replace(".", "")
    figname = f'{tname}_merfish_{region}'
    xy_cur = xy[ct_mask]
    fpca_cur = fpca[ct_mask]
    
    # get sublayers by percentiles
    groups = split_by_pc2_quantiles(xy_cur, fpca_cur, pci, pc_id, center, cell_name=ctype)
    # 检查每组点数
    for name, pts in groups.items():
        print(f"{name}: {len(pts[0])} points")

    # 使用示例
    dff_groups = {}
    for name, (xy_cur_i, fpca_cur_i) in groups.items():
        subpairs = get_subpairs(xy_cur_i/1000., fpca_cur_i, pci, dist_th)
        cdists = subpairs['Distance']
        fdists = subpairs['Distance_feature']
        dff_groups[name] = pd.DataFrame(np.array([cdists, fdists]).transpose(),
                                       columns=('euclidean_distance', 'feature_distance'))

    figname = f'{figname}_pc{pc_id}'
    plot_combined(dff_groups, num=25, figname=figname, restrict_range=False)

   

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

    layer = False
    merfish_vs_distance(merfish_file, gene_file, feat_file, region=region, layer=layer) 
    #merfish_vs_distance_sublayers(merfish_file, gene_file, feat_file, region)

    if 0:
        atlas_file = '../resources/mni_icbm152_CerebrA_tal_nlin_sym_09c_u8.nii'
        id_mapper = {
            'MTG': 28,
            'STG': 45,
        }
        id_region = id_mapper[region]
        calculate_volume_and_dimension(atlas_file, id_region)
