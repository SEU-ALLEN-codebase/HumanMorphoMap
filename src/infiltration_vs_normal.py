##########################################################
#Author:          Yufeng Liu
#Create time:     2025-01-22
#Description:               
##########################################################
import os
import glob
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import SimpleITK as sitk
from scipy.ndimage import center_of_mass
from scipy.spatial.distance import cdist
from scipy.stats import linregress, mannwhitneyu

from config import to_PID5, to_TID3
from file_io import load_image


def morphology_difference_between_infiltration_normal(meta_file_neuron, meta_file_tissue, gf_file, ctype_file, ihc=0):

    ############### Helper functions ################
    # 标注显著性
    def get_stars(p):
        if p < 0.001: return '***(<0.001)'
        elif p < 0.01: return '**(<0.01)'
        elif p < 0.05: return '*(<0.05)'
        elif p < 0.1: return '†(<0.1)'  # 趋势性标记
        else: return 'n.s.(≥0.1)'

    def _plot(gfs_cur, figname):
        sns.set_theme(style='ticks', font_scale=1.8)
        display_features = {
            'Soma_surface': r'Soma surface ($μm^2$)',
            #'N_stem': 'Number of Stems',
            #'Number of Branches': 'Number of Branches',
            #'Number of Tips': 'Number of Tips',
            'Average Diameter': 'Avg. Diameter (μm)',
            'Total Length': 'Total Length (μm)',
            #'Max Branch Order': 'Max Branch Order',
            #'Average Contraction': 'Avg. Straightness',
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
        
        ttypes, ttype_counts = np.unique(gfs_cur['tissue_type'], return_counts=True)
        gfs_cur = gfs_cur[gfs_cur['tissue_type'].isin(ttypes[ttype_counts >= 10])]
        print(gfs_cur.groupby('tissue_type', observed=False)['pt_code'].nunique())
        print(np.unique(gfs_cur['tissue_type'], return_counts=True))
         
        
        # 计算全局 y 轴范围（排除异常值）
        y_limits = {}
        for feature in features:
            q1 = gfs_cur[feature].quantile(0.25)
            q3 = gfs_cur[feature].quantile(0.75)
            iqr = q3 - q1
            y_min = q1 - 2.0 * iqr
            y_max = q3 + 2.0 * iqr
            y_limits[feature] = (y_min, y_max)

        # 设置图形（4 列子图）
        n_features = len(features)
        n_cols = 3
        n_rows = int(np.ceil(n_features / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), sharex=True)
        axes = axes.flatten()

        colors_pal = {
            'normal': 'lightcoral', 
            'infiltration': 'gold'
        }
        
        # 为每个特征绘制箱线图和回归线
        for i, feature in enumerate(features):
            ax = axes[i]
            
            # 箱线图（调整宽度为 0.5）
            sns.boxplot(
                data=gfs_cur,
                x='tissue_type',
                y=feature,
                ax=ax,
                width=0.4,  # 更窄的箱体
                hue='tissue_type',
                palette=colors_pal,
                showmeans=False,
                legend=False,
                linewidth=3,
                meanprops={'marker': 'o', 'markerfacecolor': 'red', 'linewidth': 3}
            )
            
            
            # 计算每个分箱的均值并绘制回归线
            bin_means = gfs_cur.groupby('tissue_type', observed=False)[feature].median().reset_index()
            bin_means['bin_mid'] = bin_means['tissue_type']
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
            group1 = gfs_cur[gfs_cur['tissue_type'] == bin_means['tissue_type'][0]][feature]
            group2 = gfs_cur[gfs_cur['tissue_type'] == bin_means['tissue_type'][1]][feature]
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
            print(stars)
            

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
        plt.savefig(figname, dpi=300)
        plt.close()

    ############# End of Helper functions ###########


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

    if 1:
        # morphological analysis
        for ctype_id, ctype in ctype_dict.items():
            gfs_cur = gfs_c[gfs_c['cell_type'] == ctype]
            print(f'Number of {ctype} cells: {gfs_cur.shape[0]}')
        
            # visualization
            _plot(gfs_cur, f'morph_vs_tissue-types_{ctype}.png')


    if 1:
        sns.set_theme(style='ticks', font_scale=2.4)

        # estimate the pyramidal/nonpyramidal cell ratio
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
        colors = ['lightcoral', 'gold']  # 每组颜色

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

        print()


if __name__ == '__main__':
    indir = '/PBshare/SEU-ALLEN/Users/ZhixiYun/data/human_regi'
    meta_file_tissue_JSP = '../meta/meta_samples_JSP_0330.xlsx.csv'
    ihc = 1
    
    if 1:
        meta_file_neuron = '/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta_filter/meta.csv'
        #gf_file = '/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta_filter/l_measure_result.csv'
        gf_file = '../h01-guided-reconstruction/auto8.4k_0510_resample1um_mergedBranches0712_renamed.csv'
        #gf_file = '../h01-guided-reconstruction/old0708/auto8.4k_0510_pruned_resample1um.csv'
        ctype_file = '../meta/cell_type_annotation_8.4K_all_CLS2_unique.csv'
        morphology_difference_between_infiltration_normal(meta_file_neuron, meta_file_tissue_JSP, gf_file, ctype_file, ihc=1)


