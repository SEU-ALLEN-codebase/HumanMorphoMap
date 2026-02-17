##########################################################
#Author:          Yufeng Liu
#Create time:     2025-01-22
#Description:               
##########################################################
import os
import glob
import random
import umap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import SimpleITK as sitk
from scipy.ndimage import center_of_mass
from scipy.spatial.distance import cdist
import scipy.stats as stats
from scipy.stats import linregress, mannwhitneyu

from config import to_PID5, to_TID3
from file_io import load_image
from ml.feature_processing import standardize_features


def get_stars(p):
    if p < 0.001: return '***(<0.001)'
    elif p < 0.01: return '**(<0.01)'
    elif p < 0.05: return '*(<0.05)'
    elif p < 0.1: return '†(<0.1)'  # 趋势性标记
    else: return 'n.s.(≥0.1)'


def _plot_separate_features(gfs_cur, ctype, hue_name='tissue_type', ylim_scale=2.0, pt_code_n=None):
    gfs_cur = gfs_cur.copy()
    sns.set_theme(style='ticks', font_scale=1.8)
    display_features = {
        'Soma_surface': r'Soma Surface ($μm^2$)',
        #'N_stem': 'Number of Stems',
        #'Number of Branches': 'Number of Branches',
        #'Number of Tips': 'Number of Tips',
        'Average Diameter': 'Avg. Branch\nDiameter (μm)',
        'Total Length': 'Total Length (μm)',
        #'Max Branch Order': 'Max Branch Order',
        #a'Average Contraction': 'Avg. Straightness',
        #'Average Fragmentation': 'Avg. Branch Length',
        #'Average Parent-daughter Ratio': 'Avg. Parent-daughter Ratio',
        #'Average Bifurcation Angle Local': 'Avg. Bif. Angle Local',
        #'Average Bifurcation Angle Remote': 'Avg. Bif. Angle Remote', 
        #'Hausdorff Dimension': 'Hausdorff Dimension',
    }

    # 数据准备
    # rename the features
    gfs_cur.rename(columns=display_features, inplace=True)
    features = display_features.values()   #gfs_cur.columns[:-1]  # 所有特征列（排除 distance）
    
    ttypes, ttype_counts = np.unique(gfs_cur[hue_name], return_counts=True)
    gfs_cur = gfs_cur[gfs_cur[hue_name].isin(ttypes[ttype_counts >= 10])]
    print(gfs_cur.groupby(hue_name, observed=False)['pt_code'].nunique())
    print(np.unique(gfs_cur[hue_name], return_counts=True))
     
    
    # 计算全局 y 轴范围（排除异常值）
    y_limits = {}
    for feature in features:
        q1 = gfs_cur[feature].quantile(0.25)
        q3 = gfs_cur[feature].quantile(0.75)
        iqr = q3 - q1
        y_min = max(0, q1 - ylim_scale * iqr)
        y_max = q3 + ylim_scale * iqr #, gfs_cur[feature].max()
        y_limits[feature] = (y_min, y_max)

    # 设置图形（4 列子图）
    n_features = len(features)
    n_cols = 3
    n_rows = int(np.ceil(n_features / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), sharex=True)
    axes = axes.flatten()

    if hue_name == 'tissue_type':
        #if pt_code_n is None:
        #    colors_pal = {
        #        'normal': 'lightcoral', 
        #        'infiltration': 'gold'
        #    }
        #else:
            colors_pal = {
                'normal': '#66c2a5', 
                'infiltration': '#fc8d62'
            }
    else:
        colors_pal = None
    
    # 为每个特征绘制箱线图和回归线
    for i, feature in enumerate(features):
        ax = axes[i]
        
        # 箱线图（调整宽度为 0.5）
        sns.boxplot(
            data=gfs_cur,
            x=hue_name,
            y=feature,
            ax=ax,
            width=0.4,  # 更窄的箱体
            hue=hue_name,
            palette=colors_pal,
            showmeans=False,
            legend=False,
            linewidth=3,
            meanprops={'marker': 'o', 'markerfacecolor': 'red', 'linewidth': 3}
        )
        
        
        # 计算每个分箱的均值并绘制回归线
        bin_means = gfs_cur.groupby(hue_name, observed=False)[feature].median().reset_index()
        bin_means['bin_mid'] = bin_means[hue_name]
        bin_means = bin_means[~bin_means[feature].isna()]

        '''
        # 在每组箱体的中位数位置添加红色圆点
        x_positions = np.arange(len(bin_means))  # 箱线图的x轴位置（0, 1, 2,...）
        #x_positions = [ax.get_xticks()[i] for i in range(len(bin_means))]
        #print(x_positions, bin_means[feature])
        ax.scatter(
            x_positions, 
            bin_means[feature], 
            color="red", 
            marker="o", 
            zorder=50,  # 确保圆点显示在最上层
            s=50,
        )

        # 绘制基线：第一个串口均值为准
        baseline = bin_means[feature].iloc[0]
        ax.axhline(y=baseline, color='orange', linestyle='--', linewidth=2)
        '''
        
        # do statistical test
        group1 = gfs_cur[gfs_cur[hue_name] == bin_means[hue_name][0]][feature]
        group2 = gfs_cur[gfs_cur[hue_name] == bin_means[hue_name][1]][feature]
        u_stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
        
        # 绘制横线和星号
        x1, x2 = 0.12, 0.88
        y_min, y_max = y_limits[feature]  # 标注的y轴位置
        y_delta = (y_max - y_min)
        
        y1, y2 = y_max-0.14*y_delta, y_max-0.12*y_delta
        y3 = y_max-0.11*y_delta
        ax.plot([x1, x1, x2, x2], [y1, y2, y2, y1], lw=2, color='red')
        stars = get_stars(p_value)
        y_text = y1 if stars.startswith('*') else y3
        #ax.text((x1+x2)*0.5, y_text, stars, 
        #       ha='center', va='bottom', color='red')
        print(feature, stars, f'{p_value:.3e}')
        

        # 设置 y 轴范围（排除异常值）
        ax.set_ylim(y_limits[feature])
        ax.set_xlim(-0.5, 1.5)
        
        # 标签和标题
        #ax.set_title(feature)
        ax.set_ylabel(feature)
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=0)
        ax.tick_params(axis='y', direction='in')
        ax.set_xticks([0, 1])
        #ax.set_xticklabels(('<= 5', '> 5'), ha="center")
        if pt_code_n is not None:
            ax.set_xticklabels([pt_code_n.split('-')[0].replace('000', ''), 'infiltration'])
        
        # bold
        ax.spines['left'].set_linewidth(2)
        ax.spines['right'].set_visible(False) #set_linewidth(2)
        ax.spines['top'].set_visible(False) #set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        #ax.legend()

    # 隐藏多余子图
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    if pt_code_n is None:
        figname = f'morph_vs_{hue_name}_{ctype}.png'
    else:
        figname = f'morph_vs_{hue_name}_{ctype}_{pt_code_n}.png'

    plt.savefig(figname, dpi=300)
    plt.close()


def jointfplot(data, x, y, xlim, ylim, hue, out_fig, markersize=10, 
                     hue_order=None, palette=None, bins=25, density=True):
    """
    Parameters:
    -----------
    density : bool
        If True, show proportions in each bin (sum to 1).
        If False, show raw counts.
    """
    sns.set_theme(style='ticks', font_scale=1.8)

    # Statistics
    print(data.groupby(hue, observed=False)['pt_code'].nunique())
    print(np.unique(data[hue], return_counts=True))

    # Set defaults
    if hue_order is None:
        hue_order = sorted(np.unique(data[hue]))
    if palette is None:
        palette = sns.color_palette()

    # Calculate axis limits
    if xlim is None:
        xmin, xmax = data[x].min(), data[x].max()
        xlim = [1.05*xmin - 0.05*xmax, 1.05*xmax - 0.05*xmin]
    if ylim is None:
        ymin, ymax = data[y].min(), data[y].max()
        ylim = [1.05*ymin - 0.05*ymax, 1.05*ymax - 0.05*ymin]

    # Create figure
    fig = plt.figure(figsize=(8, 8))
    gs = GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], 
                 hspace=0.05, wspace=0.05)

    # ======================
    # 1. Main scatter plot
    # ======================
    ax_scatter = fig.add_subplot(gs[1, 0])
    sns.scatterplot(data=data, x=x, y=y, hue=hue, hue_order=hue_order,
                    palette=palette, s=markersize, alpha=0.7, ax=ax_scatter,
                    legend=False)
    ax_scatter.set(xlim=xlim, ylim=ylim)#, xticks=[], yticks=[])
    ax_scatter.set_xlabel('Soma Surface ($μm^2$)')
    ax_scatter.set_ylabel('Avg. Branch\nDiameter ($μm$)')
    
    # Custom legend
    handles, labels = [], []
    for hue_val in hue_order:
        color = sns.color_palette(palette)[hue_order.index(hue_val)]
        handles.append(plt.Line2D([], [], marker='o', color=color, linestyle='None', markersize=8))
        labels.append(hue_val)
    ax_scatter.legend(handles, labels, markerscale=1, labelspacing=0.2, handletextpad=0, frameon=True)

    # ======================
    # 2. Top histogram (x-axis)
    # ======================
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax_scatter)

    # 新增参数
    min_samples_per_bin = 5  # 每个bin至少需要的样本数

    bin_edges = np.linspace(xlim[0], xlim[1], bins+1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = np.diff(bin_edges)

    # 计算总样本数（用于判断稀疏性）
    total_counts = np.histogram(data[x], bins=bin_edges)[0]
    valid_bins = total_counts >= min_samples_per_bin  # 有效bin的mask

    # 只处理有效bins
    bottom = np.zeros(np.sum(valid_bins))
    for i, hue_val in enumerate(hue_order):
        subset = data[data[hue] == hue_val]
        counts = np.histogram(subset[x], bins=bin_edges)[0][valid_bins]
        
        if density:
            bin_totals = total_counts[valid_bins]
            counts = counts / (bin_totals + 1e-10)
        
        ax_histx.bar(bin_centers[valid_bins], counts, width=bin_width[0], 
                    bottom=bottom, color=palette[i], alpha=0.7)
        bottom += counts

    # 调整x轴范围以匹配有效bins
    xlim1 = bin_edges[np.where(valid_bins)[0][0]]
    xlim2 = bin_edges[np.where(valid_bins)[0][-1]]+bin_width[0]
    ax_histx.set_xlim(xlim1, xlim2)
    
    ax_histx.set_ylim(0,1)
    ax_histx.tick_params(axis='x', which='both', bottom=False, labelbottom=False)  # 隐藏x轴刻度
    ax_histx.set(yticks=[0,0.5,1], ylabel='Ratio')

    # ======================
    # 3. Right histogram (y-axis)
    # ======================
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax_scatter)

    # Calculate bins
    bin_edges = np.linspace(ylim[0], ylim[1], bins+1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_height = np.diff(bin_edges)

    total_counts = np.histogram(data[y], bins=bin_edges)[0]
    valid_bins = total_counts >= min_samples_per_bin

    left = np.zeros(np.sum(valid_bins))
    for i, hue_val in enumerate(hue_order):
        subset = data[data[hue] == hue_val]
        counts = np.histogram(subset[y], bins=bin_edges)[0][valid_bins]
        
        if density:
            bin_totals = total_counts[valid_bins]
            counts = counts / (bin_totals + 1e-10)
        
        ax_histy.barh(bin_centers[valid_bins], counts, height=bin_height[0], 
                     left=left, color=palette[i], alpha=0.7)
        left += counts

    ax_histy.set_ylim(bin_edges[np.where(valid_bins)[0][0]], 
                     bin_edges[np.where(valid_bins)[0][-1]]+bin_height[0])
    ax_histy.set_xlim(0,1)
    ax_histy.tick_params(axis='y', which='both', left=False, labelleft=False)  # 隐藏y轴刻度
    ax_histy.set(xticks=[0,0.5,1], xlabel='Ratio')

    # Save figure
    plt.savefig(out_fig, dpi=300, bbox_inches='tight')
    plt.close()


def plot_joint_distribution(gfs_cur, hue_name):
    sns.set_theme(style='ticks', font_scale=1.8)

    display_features = [
        'Soma_surface',
        'Average Diameter',
        'Total Length',
        #'N_stem': 'Number of Stems',
        #'Number of Branches': 'Number of Branches',
        #'Number of Tips': 'Number of Tips',
        #'Max Branch Order': 'Max Branch Order',
        #'Average Contraction': 'Avg. Straightness',
        #'Average Fragmentation': 'Avg. Branch Length',
        #'Average Parent-daughter Ratio': 'Avg. Parent-daughter Ratio',
        #'Average Bifurcation Angle Local': 'Avg. Bif. Angle Local',
        #'Average Bifurcation Angle Remote': 'Avg. Bif. Angle Remote', 
        #'Hausdorff Dimension': 'Hausdorff Dimension',
    ]

    # 数据准备
    ttypes, ttype_counts = np.unique(gfs_cur[hue_name], return_counts=True)
    gfs_cur = gfs_cur[gfs_cur[hue_name].isin(ttypes[ttype_counts >= 10])]
    print(gfs_cur.groupby(hue_name, observed=False)['pt_code'].nunique())
    print(np.unique(gfs_cur[hue_name], return_counts=True))

    # standardization
    features = gfs_cur.columns[:22].values
    dfc = gfs_cur.copy()
    standardize_features(dfc, features, epsilon=1e-8, inplace=True)

    # umap projection
    reducer = umap.UMAP(random_state=1024)
    x, y = 'UMAP1', 'UMAP2'
    dfc[[x,y]] = reducer.fit_transform(dfc[display_features])
    # visualization
    from plotters.customized_plotters import sns_jointplot
    
    out_fig = 'temp.png'
    sns_jointplot(dfc, x, y, None, None, hue_name, out_fig, markersize=15, hue_order=None,
                  palette=None)
    


def plot_nonpyr_ratios(gfs_c):  # may suffer from bias.
    sns.set_theme(style='ticks', font_scale=2.4)

    np_ratios = {
        'normal': [],
        'infiltration': []
    }
    random.seed(1024)   # for duplication
    for t_type in np_ratios.keys():
        gfs_cur = gfs_c[gfs_c['tissue_type'] == t_type]
        tissues = np.unique(gfs_cur.pt_code)
        print(tissues)
        for it in range(5):
            sel_tissues = random.sample(tissues.tolist(), len(tissues)//2+1)
            gfs_cur_sel = gfs_cur[gfs_cur.pt_code.isin(sel_tissues)]
            # calculate the p:np ratio
            num_np, num_p = np.unique(gfs_cur_sel.cell_type, return_counts=True)[1]
            np_ratio = 100.0 * num_np / (num_np + num_p)
            np_ratios[t_type].append(np_ratio)

    # convert the np.array
    for t_type, ratios in np_ratios.items():
        np_ratios[t_type] = np.array(ratios)

    # plot
    means = {key: np.mean(values) for key, values in np_ratios.items()}
    sems = {
        key: np.std(values, ddof=1) / np.sqrt(len(values))  # SEM = 标准差 / sqrt(n)
        for key, values in np_ratios.items()
    }
    # 绘图设置
    groups = list(np_ratios.keys())
    x_pos = np.arange(len(groups))  # 组的位置
    colors = ['#66c2a5', '#fc8d62']  # 每组颜色

    plt.figure(figsize=(6, 6))  # 正方形图像
    # 绘制柱状图
    plt.bar(
        x_pos,
        [means[group] for group in groups],
        yerr=[sems[group] for group in groups],  # 误差线=SEM
        width=0.35,
        linewidth=3,
        capsize=7,  # 误差线端盖长度
        color=colors,
        alpha=0.9,
        edgecolor='black',
        error_kw = {
            'lw': 3,
            'capthick': 3,
            'capsize': 9,
            'ecolor': 'black', 
        },
    )

    # statistical test
    u_stat, p_value = mannwhitneyu(*np_ratios.values())
    print(f'p_value for the percentages: {p_value:.4e}')

    # 添加标签和标题
    gbar = plt.gca()
    gbar.spines['left'].set_linewidth(3)
    gbar.spines['right'].set_linewidth(3)
    gbar.spines['top'].set_linewidth(3)
    gbar.spines['bottom'].set_linewidth(3)

    # 设置刻度线宽度和样式
    plt.tick_params(
        axis='both',      # 同时调整x轴和y轴
        width=3,          # 刻度线宽度
        length=7,         # 刻度线长度
        #labelsize=12,     # 刻度标签字体大小
        bottom=True,      # 显示底部刻度
        top=False,        # 不显示顶部刻度
        left=True,        # 显示左侧刻度
        right=False       # 不显示右侧刻度
    )
    
    plt.xlim(-0.5, 1.5)
    plt.ylim(0, 30)
    plt.xticks(x_pos, groups)
    plt.xlabel('')
    plt.ylabel('% nonpyramidal cells')
    plt.subplots_adjust(left=0.17, bottom=0.17)
    #plt.title('Comparison of NP Ratios (Mean ± SEM)')
    plt.savefig('nonpyramidal_percentage_across_tissue_types.png', dpi=600)
    plt.close()


def compare_variance_components(gfs_cur):
    from scipy.spatial.distance import jensenshannon

    # 为每个特征分别进行分析
    features = ['Soma_surface', 'Average Diameter', 'Total Length']
    results = {}

    for feature in features:
        print(f"\n=== 分析特征: {feature} ===")
        
        # 提取数据
        normal_data = gfs_cur[gfs_cur['tissue_type'] == 'normal'][feature].dropna()
        infiltr_data = gfs_cur[gfs_cur['tissue_type'] == 'infiltration'][feature].dropna()
        
        # 1. 计算组内方差（衡量各组内部的变异程度）
        var_normal = np.var(normal_data, ddof=1)  # 无偏估计
        var_infiltr = np.var(infiltr_data, ddof=1)
        within_group_var = (var_normal + var_infiltr) / 2  # 平均组内方差
        
        print(f"正常组织方差: {var_normal:.2f}")
        print(f"浸润组织方差: {var_infiltr:.2f}")
        print(f"平均组内方差: {within_group_var:.2f}")
        
        # 2. 计算组间差异（效应量）
        #cohens_d = (np.mean(normal_data) - np.mean(infiltr_data)) / np.sqrt(within_group_var)
        #print(f"Cohen's d (效应量): {cohens_d:.3f}")
        
        # 3. 方差比：组间差异 vs 组内差异
        # F统计量本身就包含这个比值的思想
        f_stat, p_value = stats.f_oneway(normal_data, infiltr_data)
        between_group_var = f_stat * within_group_var  # 估算组间方差
        
        variance_ratio = between_group_var / within_group_var
        print(f"方差比 (组间/组内): {variance_ratio:.3f}")
        print(f"ANOVA p-value: {p_value:.2e}")
        

def morphology_difference_between_infiltration_normal(
        meta_file_neuron, meta_file_tissue, gf_file, ctype_file, ihc=0
):
    meta_n = pd.read_csv(meta_file_neuron, index_col=0, low_memory=False, encoding='gbk')
    gfs = pd.read_csv(gf_file, index_col=0, low_memory=False)
    # rename the index to purly cell index, FOR RENAME
    if gfs.index.dtype != 'int64':
        gfs.index = [int(idx.split('_')[0]) for idx in gfs.index]

    # also rename the columns
    column_mapping = {
        'Nodes': 'N_node',
        'SomaSurface': 'Soma_surface',
        'Stems': 'N_stem',
        'Bifurcations': 'Number of Bifurcations',
        'Branches': 'Number of Branches',
        'Tips': 'Number of Tips',
        'OverallWidth': 'Overall Width',
        'OverallHeight': 'Overall Height',
        'OverallDepth': 'Overall Depth',
        'AverageDiameter': 'Average Diameter',
        'Length': 'Total Length',
        'Surface': 'Total Surface',
        'Volume': 'Total Volume',
        'MaxEuclideanDistance': 'Max Euclidean Distance',
        'MaxPathDistance': 'Max Path Distance',
        'MaxBranchOrder': 'Max Branch Order',
        'AverageContraction': 'Average Contraction',
        'AverageFragmentation': 'Average Fragmentation',
        'AverageParent-daughterRatio': 'Average Parent-daughter Ratio',
        'AverageBifurcationAngleLocal': 'Average Bifurcation Angle Local',
        'AverageBifurcationAngleRemote': 'Average Bifurcation Angle Remote',
        'HausdorffDimension': 'Hausdorff Dimension'
    }

    # 假设 df 是你的 DataFrame
    gfs = gfs.rename(columns=column_mapping)
    gfs = gfs.loc[meta_n.cell_id]

    # Tissue types from JSP, which is more consistently annotated
    meta_t = pd.read_csv(meta_file_tissue, index_col=0)
    meta_t.set_index('idx')
    ctypes = pd.read_csv(ctype_file, index_col=0)

    # extract neurons
    # 1. ihc extraction
    ihc_mask = meta_n.immunohistochemistry == ihc
    # 2. tissue extraction
    meta_n['pt_code'] = meta_n['patient_number'] + '-' + meta_n['tissue_block_number']
    meta_t['pt_code'] = meta_t['patient_number'] + '-' + meta_t['tissue_id']
    tissue_mask = (meta_n['pt_code']).isin(meta_t['pt_code'])
    # 3. cell type extraction: to be added
    ctypes_idxs = [int(name.split('_')[0]) for name in ctypes.index]
    ctypes = ctypes.reset_index()
    ctypes.index = ctypes_idxs
    # get the cell types
    ctypes_ = ctypes.loc[gfs.index]
    py_mask = (ctypes_.num_annotator >= 2) & (ctypes_.CLS2 == '0')
    nonpy_mask = (ctypes_.num_annotator >= 2) & (ctypes_.CLS2 == '1')

    # merge with the tissue-type
    c_mask = (ihc_mask & tissue_mask).values & (py_mask | nonpy_mask).values
    gfs_c = gfs[c_mask].copy()
    meta_n_c = meta_n[c_mask]
    
    meta_t_c = meta_t.set_index('pt_code')
    tissue_types = meta_t_c.loc[meta_n_c['pt_code'], 'tissue_type']
    gfs_c['tissue_type'] = tissue_types.values
    gfs_c['pt_code'] = tissue_types.index.values
    gfs_c['region'] = meta_t_c.loc[meta_n_c['pt_code'], 'english_abbr_nj'].values


    ctype_dict = {
        '0':'pyramidal',
        '1':'nonpyramidal',
    }
    gfs_c['cell_type'] = ctypes.loc[gfs_c.index, 'CLS2'].map(ctype_dict)

    # rename the Chinese to English
    gfs_c['tissue_type'] = gfs_c['tissue_type'].replace({
        '正常': 'normal',
        '浸润': 'infiltration'
    })

    # save the meta-information
    gfs_c_meta = gfs_c[['tissue_type', 'pt_code', 'region', 'cell_type']]
    gfs_c_meta.to_csv('tissue_cell_meta_jsp.csv', index=True)

    ### Plotting
    if 1:
        # Morphological feature difference between normal and infiltrated pyr and non-pyr cells
        use_only_glioblastoma = False   # glioblastoma is also a subtype of glioma
        strict_comp = False
        use_only_one_tissue = False
        tissue_pt = 'P00061-T001'

        for ctype_id, ctype in ctype_dict.items():
            gfs_cur = gfs_c[gfs_c['cell_type'] == ctype]
            if (ctype == 'pyramidal') and use_only_glioblastoma:
                # 'P00064-T001', 'P00077-T001' are glioma
                gfs_cur = gfs_cur[~gfs_cur.pt_code.isin(['P00064-T001', 'P00077-T001'])]

            if (ctype == 'pyramidal') and use_only_one_tissue:
                mask_normal = (gfs_cur.tissue_type == 'normal') & (gfs_cur.pt_code == tissue_pt)
                mask_infil = gfs_cur.tissue_type == 'infiltration'
                gfs_cur = gfs_cur[mask_normal | mask_infil]

            print(f'Number of {ctype} cells: {gfs_cur.shape[0]}')

            ############ sample-level statistics
            sample_statis = gfs_cur.groupby('pt_code').agg({
                'region': 'first',
                'tissue_type': 'first',
                'pt_code': 'count'
            }).rename(columns={'pt_code': 'total_neurons'})
            print(sample_statis)

            if strict_comp:
                gfs_cur = gfs_cur[gfs_cur.pt_code.isin(['P00065-T001', 'P00066-T001'])]

            ##### do variance estimation #######
            #import ipdb; ipdb.set_trace()
            compare_variance_components(gfs_cur)

        
            ############ Overall statistic tests for each feature
            if strict_comp:
                _plot_separate_features(gfs_cur, ctype=f'{ctype}_p65p66', ylim_scale=2.5)
            else:
                _plot_separate_features(gfs_cur, ctype=ctype, ylim_scale=2.5)


            if 0:
                ############ statistical test for largest two samples
                if ctype == 'pyramidal':
                    pt_n, pt_c = np.unique(gfs_cur[gfs_cur.tissue_type == 'normal'].pt_code, 
                                             return_counts=True)
                    pt_argsort_ids = np.argsort(pt_c)[::-1]
                    print(dict(zip(pt_n, pt_c)))
                    infil_tissues = np.unique(gfs_cur[gfs_cur.tissue_type == 'infiltration'].pt_code).tolist()
                    
                    nsamples = 2
                    for iptn in pt_argsort_ids[:nsamples]:
                        pt_code = pt_n[iptn]
                        pt_count = pt_c[iptn]
                        gfs_subset = gfs_cur[gfs_cur.pt_code.isin(infil_tissues + [pt_code])].copy()

                        if pt_code == 'P00041-T001':
                            ylim_scale = 3
                        else:
                            ylim_scale = 2.5
                        _plot_separate_features(gfs_subset, ctype=ctype, ylim_scale=ylim_scale, pt_code_n=pt_code)

            if 0:
                ######### statistical test for random subsets
                if ctype == 'pyramidal':
                    np.random.seed(1024)
                    random.seed(1024)
                    infil_tissues = np.unique(gfs_cur[gfs_cur.tissue_type == 'infiltration'].pt_code).tolist()
                    # random selection of pyramidal neurons
                    ntrials = 10
                    pct_sel = 0.5
      
                    normal_neurons = gfs_cur[gfs_cur.tissue_type == 'normal']
                    nsel = int(len(normal_neurons) * pct_sel)
                    
                    for itrial in range(ntrials):
                        sel_ids = random.sample(np.arange(len(normal_neurons)).tolist(), nsel)
                        sel_normal_neurons = normal_neurons.iloc[sel_ids]

                        # concat to form a new dataframe
                        df_rand = pd.concat((sel_normal_neurons, gfs_cur[gfs_cur.tissue_type == 'infiltration']))
                        print(f'\n--> Trial: {itrial}:')
                        _plot_separate_features(
                            df_rand, ctype=f'{ctype}_rand{pct_sel}', 
                            ylim_scale=2.5
                        )
                    

            #plot_joint_distribution(gfs_cur, 'tissue_type')


    if 0:
        # estimate the pyramidal/nonpyramidal cell ratio
        plot_nonpyr_ratios(gfs_c)



if __name__ == '__main__':
    indir = '/PBshare/SEU-ALLEN/Users/ZhixiYun/data/human_regi'
    meta_file_tissue_JSP = '../meta/meta_samples_JSP_0330.xlsx.csv'
    ihc = 1
    
    if 1:
        meta_file_neuron = '/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta_filter/meta.csv'
        gf_file = '../h01-guided-reconstruction/auto8.4k_0510_resample1um_mergedBranches0712_crop100_renamed_noSomaDiameter.csv'
        ctype_file = '../meta/cell_type_annotation_8.4K_all_CLS2_unique.csv'
        morphology_difference_between_infiltration_normal(meta_file_neuron, meta_file_tissue_JSP, gf_file, ctype_file, ihc=1)


