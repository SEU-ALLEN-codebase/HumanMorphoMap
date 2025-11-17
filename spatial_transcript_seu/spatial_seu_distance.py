##########################################################
#Author:          Yufeng Liu
#Create time:     2025-02-24
#Description:               
##########################################################
import os
import cv2
import cc3d
import random
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, cdist
from scipy import ndimage
from scipy.spatial import cKDTree, KDTree
from skimage.morphology import skeletonize
from skimage.measure import label
from scipy.ndimage import convolve
from scipy.ndimage import distance_transform_edt
from skimage.graph import route_through_array

from scipy.stats import spearmanr, pearsonr, linregress
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import seaborn as sns

import scanpy as sc

from image_utils import get_longest_skeleton
from anatomy.anatomy_vis import detect_edges2d

from config import LAYER_CODES, LAYER_ANCHORS


__COLORS4__ = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']


def hex_to_rgba(hex_color, alpha):
    hex_color = hex_color.lstrip('#')  # 移除开头的 '#'
    return tuple([int(hex_color[i:i+2], 16)/255. for i in (0, 2, 4)] + [alpha])  # 每两位解析为十进制

def process_spatial(st_dir):
    st_file = os.path.join(st_dir, 'SpatialModel/st.h5ad')
    layer_ad_file = os.path.join(st_dir, f'../../data/layers/spatial_adata_{sample_name}_withLaminar.h5ad')
    ctype_file = os.path.join(st_dir, 'predicted_cell_types.csv')

    # Parsing the data
    adata_in = sc.read(st_file)
    ctypes = pd.read_csv(ctype_file, index_col=0)
    adata_layer = sc.read(layer_ad_file, backed=True)

    # merge the data
    adata = adata_in.copy()
    non_sample_indices = [idx.split('_')[-1] for idx in adata.obs.index]
    adata.obs['laminar'] = adata_layer.obs.loc[non_sample_indices, 'laminar'].values
    adata.obs['cell_code'] = ctypes.loc[adata.obs.index, 'cell_code']

    # The data is pre-processed during the previous cell_type prediction
    # (1) 将每一行按照总和固定大小归一化（例如 1e4）
    sc.pp.normalize_total(adata, target_sum=1e4)

    # (2) 转化为 log1p
    sc.pp.log1p(adata)

    # (3) 计算PCA并求前50个PC
    #sc.pp.highly_variable_genes(adata)  # 通常先选择高变基因
    sc.pp.pca(adata, n_comps=50)  # 计算前50个主成分

    return adata


def get_centerline(mask, mask_val, start_anchor, end_anchor):
    """
    :param mask: uint8 图像，其中值为2的区域是目标长条
    :param mask_val: the value for the forground mask
    :param start_anchor: 起始点 (width, height)
    :param end_anchor: 终止点 (width, height)

    :return: 
        - 中心线坐标 (M, 2)
    """
    # 1. 提取值为2的区域并二值化
    binary_mask = (mask == mask_val).astype(bool)
    
    # 2. 找到最大连通区域（去除小的孤立部分）
    ccs, nccs = cc3d.largest_k(binary_mask, k=1, connectivity=8, return_N=True)
    if nccs != 1:
        raise ValueError("No connected components found in mask")

    
    max_label = nccs
    cleaned_mask = ccs == max_label

    # find out start and end points on boundary 
    edges = detect_edges2d(cleaned_mask)
    ecoords = np.array(edges.nonzero()).transpose()

    # 2. 找到mask内离标记点最近的有效点
    def find_nearest_point(foreground_points, point):
        # 计算给定点到所有前景点的距离
        distances = cdist([point[::-1]], foreground_points)  # 输入需为(y,x)
        nearest_idx = np.argmin(distances)
        nearest_y, nearest_x = foreground_points[nearest_idx]
        return (nearest_x, nearest_y)  # 返回(x,y)

    
    adjusted_start = find_nearest_point(ecoords, start_anchor)
    adjusted_end = find_nearest_point(ecoords, end_anchor)
    
    # 3. 计算距离变换（权重矩阵）
    dist_transform = distance_transform_edt(cleaned_mask)
    weights = 1.0 / (dist_transform + 1e-6)  # 中心区域权重小
    
    # 4. 计算最优路径（Dijkstra算法）
    start_yx = (adjusted_start[1], adjusted_start[0])  # 转为(y,x)
    end_yx = (adjusted_end[1], adjusted_end[0])
    path, _ = route_through_array(weights, start_yx, end_yx, fully_connected=True)
    path = np.array(path)
    
    # initialize a new mask
    skel = np.zeros_like(cleaned_mask)
    skel[path[:,0], path[:,1]] = True
    # manually separated mask
    separated_mask = cleaned_mask & ~skel
    # Must use `connectivity=1`
    labeled_mask = cc3d.connected_components(separated_mask, connectivity=4)
    # keep only the two non-zero mask

    # in case of exceptions:
    if labeled_mask.max() > 2:
        tmp_mask = cc3d.largest_k(separated_mask, connectivity=4, k=2)
        # nearest neighbor interpolation
        exceptions = (labeled_mask > 0) & (tmp_mask == 0)
        ex_coords = np.array(np.nonzero(exceptions)).transpose()
        bases = (labeled_mask == 1) | (labeled_mask == 2)
        base_coords = np.array(np.nonzero(bases)).transpose()
        b_kdt = cKDTree(base_coords)

        ex_m =b_kdt.query(ex_coords, k=1)[1]
        labeled_mask[exceptions.nonzero()] = labeled_mask[bases][ex_m.reshape(-1)]

    # make sure only two components exist
    assert(labeled_mask.max() == 2)
    part1 = labeled_mask == 1
    part2 = labeled_mask == 2
    # dorsal and ventral assignment based on their distance to the ccf atlas center
    c1 = np.array(np.nonzero(part1)).mean(axis=1)
    c2 = np.array(np.nonzero(part2)).mean(axis=1)

    # use the white matter as reference
    wm_code = LAYER_CODES['WM']
    wm_mask = mask == wm_code
    cwm = np.array(np.nonzero(wm_mask)).mean(axis=1)
    if np.linalg.norm(cwm - c2) > np.linalg.norm(cwm - c1):
        print('Revert the mask code!')
        labeled_mask[labeled_mask == 1] = 3
        labeled_mask[labeled_mask == 2] = 1
        labeled_mask[labeled_mask == 1] = 2
    
    # 转为(x,y)坐标
    centerline = path[:, ::-1]
    return centerline, labeled_mask, skel


def process_mask_and_compute_distances(mask, mask_val, start_anchor, end_anchor, adata, visualize=False):
    """
    
    :param points: (N, 2) 数组，表示mask内部的点坐标
    :return: 
        - 沿长轴的距离 (N,)
        - 沿短轴的距离 (N,)
    """
    # skeleton_points in format of (w, h)
    skeleton_points, labeled_mask, skel = get_centerline(mask, mask_val, start_anchor, end_anchor)

    #tmp_mask = (labeled_mask * 127).astype(np.uint8)
    #cv2.imwrite('temp.png', tmp_mask)
    
    if visualize:
        plt.figure(figsize=(10, 10))
        plt.imshow(mask, cmap='gray')
        plt.scatter([start_anchor[0], end_anchor[0]], [start_anchor[1], end_anchor[1]], 
                    c='red', s=50, label='Anchors')
        plt.plot(skeleton_points[:, 0], skeleton_points[:, 1], 'b-', lw=2, label='Centerline')
        plt.legend()
        plt.title("P00083 L2-L3 Centerline Extraction")
        plt.savefig(f'centerline_layer{mask_val}.png', dpi=300)
        plt.close()

    points = adata.obsm['xy_pxl'].copy()

    # 4. 构建中心线的KD树用于最近邻搜索
    kdtree = cKDTree(skeleton_points)

    # filter out points does not in the current mask
    fg_mask = (labeled_mask > 0) | skel
    points_int = points.astype(int) # points also in format (w, h)

    in_layer_mask = fg_mask[points_int[:,1], points_int[:,0]] > 0
    points_l = points[in_layer_mask]
    print(f'Number of neurons in target layer: {len(points_l)}')
    
    # 5. 对于每个点，找到中心线上最近的点
    distances, indices = kdtree.query(points_l)
    
    # 计算累积距离
    diffs = np.diff(skeleton_points, axis=0)
    segment_lengths = np.hypot(diffs[:,0], diffs[:,1])
    cumulative_lengths = np.concatenate([[0], np.cumsum(segment_lengths)])
    
    # 沿长轴的距离就是累积距离
    axial_distances = cumulative_lengths[indices]

    # 沿短轴的距离就是到中心线的欧氏距离
    lateral_distances = distances
    # reassign sign for distance
    points_l_int = points_l.astype(int)
    neg_mask = labeled_mask[points_l_int[:,1], points_l_int[:,0]] == 1
    lateral_distances[neg_mask] *= -1
    
    
    if visualize:
        rand_ids = random.sample(range(len(points_l)), 10)
        sel_points = points_l[rand_ids]
        plt.imshow(mask, cmap='gray')
        plt.scatter(skeleton_points[:,0], skeleton_points[:,1], c='b', s=5, label='Centerline')
        plt.scatter(sel_points[:, 0], sel_points[:,1], c='g', s=20, label='Test points')
        for i, (x, y) in enumerate(sel_points):
            #plt.text(x, y, f'{axial_distances[rand_ids][i]:.1f}', color='red')
            plt.text(x, y, f'{lateral_distances[rand_ids][i]:.1f}', color='red')
            
        plt.legend()
        plt.savefig('visualize_skeleton.png', dpi=300)
        plt.close()
    
    # assign values to adata
    adata.obs['in_layer_mask'] = in_layer_mask
    scalef = adata.uns['spatial'][sample_name]['scalefactors']['tissue_hires_scalef']
    adata.obs['axial_distances'] = np.nan
    adata.obs.loc[adata.obs.index[in_layer_mask], 'axial_distances'] = axial_distances / scalef
    adata.obs['lateral_distances'] = np.nan 
    adata.obs.loc[adata.obs.index[in_layer_mask], 'lateral_distances'] = lateral_distances / scalef

    return (
        adata,
        skeleton_points,
    )


def _plot(dff, num=50, figname='temp', nsample=10000, color='black', overall_distribution=False):
    df = dff.copy()
    if df.shape[0] > nsample:
        random.seed(1024)
        df_sample = df.iloc[random.sample(range(df.shape[0]), nsample)]
    else:
        df_sample = df

    xn, yn = 'euclidean_distance', 'feature_distance'
    xlim0, xlim1 = 0, 3
    xmax = 5#dff[xn].max()
    if overall_distribution:
        sns.set_theme(style='ticks', font_scale=1.8)
        plt.figure(figsize=(8,8))
        #sns.scatterplot(df, x='euclidean_distance', y='feature_distance', s=5,
        #                alpha=0.3, edgecolor='none', rasterized=True, color='black')
        sns.displot(df, x=xn, y=yn, cmap='Reds',
                    )
                    #cbar=True,
                    #cbar_kws={"label": "Count", 'aspect': 5})
        figname = figname + '_overall'

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

        # 绘图：点图+误差条（使用实际数值坐标
        show_mask = bin_stats['bin_center'] < xlim1
        show_mask[0] = False    # to manually match to morphology. update 20251114
        plt.errorbar(x=bin_stats['bin_center'][show_mask],
                     y=bin_stats['mean'][show_mask],
                     yerr=bin_stats['sem'][show_mask],  # 95% CI (改用sem则不需要*1.96)
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


    plt.xlabel('Soma-soma distance (mm)')
    plt.ylabel('Transcriptomic dissimilarity')
    ax = plt.gca()
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.tick_params(width=2)

    if overall_distribution:
        #ax.yaxis.set_major_locator(MultipleLocator(2))
        plt.xlim(0, 3)
        plt.ylim(0, 30)
    else:
        #ax.yaxis.set_major_locator(MultipleLocator(0.5))
        plt.xlim(0, 3)

    plt.subplots_adjust(bottom=0.16, left=0.17)
    plt.savefig(f'{figname}.png', dpi=300); plt.close()


def plot_combined(dff_groups, num=25, figname='combined', restrict_range=True):
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

    # estimate the maximal value along x-axis
    max_v = 0
    for name, dff in dff_groups.items():
        max_v_i = dff['euclidean_distance'].max().round(2)
        max_v = max(max_v, max_v_i)
    max_v = max_v.round(2)
        
    
    # 为每个分组绘制图形
    for icur, (name, dff) in enumerate(dff_groups.items()):
        df = dff.copy()
        
        # 创建分箱并计算统计量
        df['A_bin'] = pd.cut(df[xn], bins=np.linspace(0, max_v, num), right=False)
        
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
    plt.xlim(0, max_v)
    #plt.xticks([0,1,2,4,5])
    if restrict_range:
        # 计算所有分组的共同y轴范围
        all_medians = []
        for dff in dff_groups.values():
            df = dff.copy()
            df['A_bin'] = pd.cut(df[xn], bins=np.linspace(0, max_v, num), right=False)
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
        ytick_step = 1
    elif (ymax - ymin) > 8:
        ytick_step = 2
    else:
        ytick_step = 1

    ax.yaxis.set_major_locator(MultipleLocator(ytick_step))  # 每隔1单位显示一个刻度
    #ax.xaxis.set_major_locator(MultipleLocator(1))  # 每隔2单位显示一个刻度
    
    #plt.ylim(7, 18)
    
    
    plt.subplots_adjust(bottom=0.15, left=0.15)
    plt.savefig(f'{figname}.png', dpi=300)
    plt.close()



def split_by_quantiles(xys, xys_l, dist_col, fpca, quantiles=[0.25, 0.5, 0.75], sample_name='',
                      cell_name='l2-l3_lateral', visualize=True):
    """
    按axial或lateral坐标的分位数将点分为4组
    输入:
        distances: coordinates along axial or lateral direction
        quantiles: 分位数列表（默认[0.25, 0.5, 0.75]）
    返回:
        groups: 字典，key为分位区间名，value为对应坐标数组
    """
    # 计算分位点
    distances = xys_l[dist_col]
    qs = np.quantile(distances, quantiles)

    # 分组
    qm1 = distances <= qs[0]
    qm2 = (distances > qs[0]) & (distances <= qs[1])
    qm3 = (distances > qs[1]) & (distances <= qs[2])
    qm4 = distances > qs[2]
    groups = {
        '0-25%': (xys_l[qm1], fpca[qm1], xys[qm1]),
        '25-50%': (xys_l[qm2], fpca[qm2], xys[qm2]),
        '50-75%': (xys_l[qm3], fpca[qm3], xys[qm3]),
        '75-100%': (xys_l[qm4], fpca[qm4], xys[qm4])
    }

    sns.set_theme(style="ticks", font_scale=2.1)

    if visualize:
        plt.figure(figsize=(8, 8))
        colors = __COLORS4__
        for i, (name, pts) in enumerate(groups.items()):
            pts_v = pts[2] / 1000.   # to millimeter
            plt.scatter(pts_v[:,0], pts_v[:,1], s=12, alpha=1., label=name, color=colors[i])

        plt.legend(frameon=False, markerscale=3, handletextpad=0.10)
        ax = plt.gca()
        ax.yaxis.set_major_locator(MultipleLocator(1))  # 每隔1单位显示一个刻度
        ax.xaxis.set_major_locator(MultipleLocator(1))  # 每隔2单位显示一个刻度
        plt.xlabel('X coordinates (mm)')
        plt.ylabel('Y coordinates (mm)')
        #sns.despine()
        plt.axis('off')
        tname = cell_name.replace("/", "").replace(".", "")
        plt.savefig(f'sublayers_{sample_name}_{tname}.png', dpi=300)
        plt.close()

    return groups

def norm_and_pca(adata, N=50):
    # 标准化和归一化
    sc.pp.normalize_total(adata, target_sum=1e4)  # CPM标准化
    sc.pp.log1p(adata)                           # 对数转换
    # High-variable genes
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)  # 选2000个高变基因
    adata= adata[:, adata.var.highly_variable]          # 保留高变基因
    # 
    sc.pp.scale(adata, max_value=10)       # Z-score标准化
    sc.tl.pca(adata, n_comps=N)           # 计算50个主成分

    print(f"Total variance ratio of top-{N} PCs are: {adata.uns['pca']['variance_ratio'][:N].sum().round(4)}")

    return adata

def get_subpairs(xys_i, dist_col, fpca_i):
    points = xys_i[dist_col].values  # (n)
    indices = xys_i.index.values  # 点对应的原始索引

    # --- 向量化计算 ---
    nn = len(points)
    diff = points[:, None] - points[None, :]  # 形状 (n, n)

    # 欧几里得距离矩阵
    dist_matrix = np.fabs(diff)  # 形状 (n, n)

    # 筛选条件：投影距离 <= 阈值，且 i < j（避免重复和自身）
    # 生成上三角矩阵的掩码 (i < j)
    rows, cols = np.triu_indices(nn, k=1)  # 返回 (row_indices, col_indices)

    # 获取符合条件的点对索引和距离
    valid_indices = list(zip(indices[rows], indices[cols]))
    valid_distances = dist_matrix[rows, cols]

    # distance in feature space
    dist_feats = np.linalg.norm(fpca_i[:,None,:] - fpca_i[None,:,:], axis=2)
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
 

def transcript_vs_distance_sublayers(st_file, lay_img_file, celltype_file, 
                                     layer='', sample_name='', direction='lateral'):
    # load the coordinates of cells
    adata_st = sc.read(st_file)
    adata_st = norm_and_pca(adata_st, N=50)

    adata_st.obsm['xy_pxl'] = adata_st.obsm['spatial'] * \
               adata_st.uns['spatial'][sample_name]['scalefactors']['tissue_hires_scalef']

    # get the prymidal cells
    df_ct = pd.read_csv(celltype_file, index_col=0)
    # extract and merge the cell type information into the data
    adata_st.obs['cell_code'] = df_ct.loc[adata_st.obs.index, 'cell_code']


    if direction is None:
        # extract the target spots
        adata_l = adata_st[adata_st.obs.cell_code == 1]
        print(f'Number of excitatory cells: {adata_l.shape[0]}')

        # pure distance-vs-similarity
        cdists = pdist(adata_l.obsm['spatial'] / 1000.)
        fdists = pdist(adata_l.obsm['X_pca'])
        df_dists = pd.DataFrame(np.array([cdists, fdists]).transpose(),
                                columns=('euclidean_distance', 'feature_distance'))

        overall_distribution = False
        figname = f'{sample_name}'
        _plot(df_dists, num=25, figname=figname, overall_distribution=overall_distribution)

    else:
        # Estimate the centerline
        mask = cv2.imread(layer_img_file, cv2.IMREAD_UNCHANGED)
        mask_val = LAYER_CODES[layer]
        start_anchor, end_anchor = LAYER_ANCHORS[sample_name][layer]
        
        # Estimate the lateral and axial coordinates for the points
        adata_st, centerline_points = process_mask_and_compute_distances(
                mask, mask_val, start_anchor, end_anchor, adata_st, visualize=False
        )

        # extract the target spots
        adata_l = adata_st[(adata_st.obs.cell_code == 1) & (adata_st.obs['in_layer_mask'] == True)]
        # Coordinates in layer-corrected space
        xys_l = adata_l.obs[['axial_distances', 'lateral_distances']]
        
        # split to quantiles
        if direction == 'lateral':
            qdirection = 'axial'
        elif direction == 'axial':
            qdirection = 'lateral'
        else:
            raise ValueError(f'Error value for @args [direction]: {direction}')

        dist_col = f'{direction}_distances'
        qdist_col = f'{qdirection}_distances'
        vis_title = f'{layer}_{direction}'

        groups = split_by_quantiles(
                    adata_l.obsm['spatial'], xys_l, qdist_col, adata_l.obsm['X_pca'], 
                    sample_name=sample_name, cell_name=vis_title
        )

        # 检查每组点数
        for name, pts in groups.items():
            print(f"{name}: {len(pts[0])} points")

        # 使用示例
        dff_groups = {}
        for name, (xys_l_i, fpca_i, xys_i) in groups.items():
            subpairs = get_subpairs(xys_l_i/1000., dist_col, fpca_i)
            cdists = subpairs['Distance']
            fdists = subpairs['Distance_feature']
            dff_groups[name] = pd.DataFrame(np.array([cdists, fdists]).transpose(),
                                           columns=('euclidean_distance', 'feature_distance'))

        figname = f'{sample_name}_{layer}_{direction}'
        plot_combined(dff_groups, num=25, figname=figname, restrict_range=False)

   

if __name__ == '__main__':
    sample_name = 'P00083'
    layer = 'L5-L6'
    direction = None   # options: 'lateral', 'axial', and None

    #adata_file = f'data/spatial_data_{sample_name}_withLaminar.h5ad'
    st_dir = f'cell2loc/{sample_name}'

    layer_img_file = f'data/ST-raw/{sample_name}/layers/layer_mask.png'
    st_file = f'{st_dir}/SpatialModel/st.h5ad'
    celltype_file = f'{st_dir}/predicted_cell_types.csv'

    transcript_vs_distance_sublayers(st_file, layer_img_file, celltype_file, layer=layer, sample_name=sample_name, direction=direction)
