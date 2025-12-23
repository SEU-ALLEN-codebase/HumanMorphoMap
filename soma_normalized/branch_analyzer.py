##########################################################
#Author:          Yufeng Liu
#Create time:     2025-11-15
#Description:               
##########################################################
import os, glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy.stats import mannwhitneyu
from scipy.stats import combine_pvalues
        

from swc_handler import parse_swc
from morph_topo.morphology import Morphology, Topology

class BranchAnalyzer:
    def __init__(self, in_swc, exclude_terminal=True):
        self.exclude_terminal = exclude_terminal
        self._load_data(in_swc)

    def _load_data(self, in_swc):
        # load the data and initialize the topological tree
        #print(f'--> Processing {os.path.split(in_swc)[-1]}')
        tree = parse_swc(in_swc)
        self.morph = Morphology(tree)
        topo_tree, self.seg_dict = self.morph.convert_to_topology_tree()
        self.topo = Topology(topo_tree)

        return True

    def levelwise_features(self):
        ###### level frequency
        if self.exclude_terminal:
            tmp_levels = [level for idx, level in self.topo.order_dict.items() if idx not in self.topo.tips]
            order_counter = Counter(tmp_levels)
        else:
            order_counter = Counter(self.topo.order_dict.values())
        # remove level zero
        order_counter.pop(0)
        
        ###### branch length
        bl_dict = {}
        for term_id, branch in self.seg_dict.items():
            if self.exclude_terminal and term_id in self.topo.tips:
                continue

            current = self.topo.pos_dict[term_id]
            pid = current[6]
            if pid == -1:
                continue

            parent = self.topo.pos_dict[pid]
            branch_ids_all = [term_id] + branch + [pid]
            coords = np.array([self.morph.pos_dict[nid][2:5] for nid in branch_ids_all])
            # calculate the path_distance
            branch_vec = coords[1:] - coords[:-1]
            branch_lengths = np.linalg.norm(branch_vec, axis=1)
            # total length
            total_length = branch_lengths.sum()
            # get the level
            level = self.topo.order_dict[term_id]
            
            try:
                bl_dict[level].append(total_length)
            except KeyError:
                bl_dict[level] = [total_length]

        # calculate the average branch length
        for level, bl in bl_dict.items():
            bl_dict[level] = np.mean(bl)

        return order_counter, bl_dict

def process_single_swc(in_swc):
    swc_name = os.path.split(in_swc)[-1]
    prefix = swc_name[6:-4]
    idx = int(swc_name[:5])
    ba = BranchAnalyzer(in_swc)
    order_counter, bl_dict = ba.levelwise_features()
    
    # 获取所有可能的level
    all_levels = set(order_counter.keys()) & set(bl_dict.keys())
    
    # 为每个level创建数据
    file_results = []
    for level in sorted(all_levels):
        order_count = order_counter.get(level, 0)
        branch_length = bl_dict.get(level, 0.0)
        
        file_results.append({
            'idx': idx,
            'level': level,
            'order_count': order_count,
            'branch_length': branch_length
        })
    
    return file_results
            
def calc_features(in_swc_dir, out_feat_file):
    results = []
    swc_files = glob.glob(os.path.join(in_swc_dir, '*.swc'))#[:20]

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(tqdm(
            executor.map(process_single_swc, swc_files), total=len(swc_files), 
            desc="Processing SWC files"))

    # 创建DataFrame
    df = pd.DataFrame([item for sublist in results for item in sublist])
    
    # 保存到文件
    df.to_csv(out_feat_file, index=False)
    
    return df

def calculate_mannwhitney_pvalues(df, feature_columns, group_col='tissue_type', group1='normal', group2='infiltration'):
    """
    计算两组间各特征的Mann-Whitney U检验p值
    
    Parameters:
    -----------
    df : DataFrame
        包含特征和分组列的数据框
    feature_columns : list
        要检验的特征列名列表
    group_col : str, default='tissue_type'
        分组列名
    group1 : str, default='normal'
        第一组名称
    group2 : str, default='infiltration'
        第二组名称
        
    Returns:
    --------
    DataFrame: 包含各特征检验结果的表格
    """
    results = []
    
    for feature in feature_columns:
        # 提取两组数据
        group1_data = df[df[group_col] == group1][feature].dropna()
        group2_data = df[df[group_col] == group2][feature].dropna()
        
        # 确保两组都有足够数据
        if len(group1_data) < 3 or len(group2_data) < 3:
            print(f"警告: 特征 '{feature}' 的数据量不足，跳过检验")
            results.append({
                'feature': feature,
                'p_value': None,
                'statistic': None,
                'group1_size': len(group1_data),
                'group2_size': len(group2_data),
                'group1_mean': group1_data.mean() if len(group1_data) > 0 else None,
                'group2_mean': group2_data.mean() if len(group2_data) > 0 else None
            })
            continue
        
        # 执行Mann-Whitney U检验
        # alternative参数可选：'two-sided'（双侧）, 'greater'（group1>group2）, 'less'（group1<group2）
        statistic, p_value = mannwhitneyu(
            group1_data, 
            group2_data,
            alternative='two-sided'  # 双侧检验
        )
        
        # 计算效应量（Cohen's d）
        n1, n2 = len(group1_data), len(group2_data)
        pooled_std = np.sqrt(((n1-1)*group1_data.std()**2 + (n2-1)*group2_data.std()**2) / (n1+n2-2))
        cohens_d = (group1_data.mean() - group2_data.mean()) / pooled_std if pooled_std != 0 else 0
        
        # 收集结果
        results.append({
            'feature': feature,
            'p_value': p_value,
            'statistic': statistic,
            'cohens_d': cohens_d,
            'group1_size': n1,
            'group2_size': n2,
            'group1_mean': group1_data.mean(),
            'group2_mean': group2_data.mean(),
            'group1_median': group1_data.median(),
            'group2_median': group2_data.median(),
            'significance': '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
        })
    
    # 创建结果DataFrame
    results_df = pd.DataFrame(results)
    
    # 按p值排序（可选）
    results_df = results_df.sort_values('p_value')
    
    return results_df


class BranchLevelVisualization:
    def __init__(self, feats_orig, feats_scaled):
        """
        初始化可视化类 - 每行一种细胞类型，每列一种level，每个单元格内子图上下排列
        
        Args:
            feats_orig: 原始特征DataFrame
            feats_scaled: 标准化特征DataFrame
        """
        self.feats_orig = feats_orig.copy()
        self.feats_scaled = feats_scaled.copy()
        
        # 确保数据有相同的结构
        required_cols = ['level', 'order_count', 'branch_length', 'tissue_type', 'cell_type']
        for df in [self.feats_orig, self.feats_scaled]:
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"Column '{col}' not found in DataFrame")
        
        # 设置绘图风格
        sns.set_theme(style='ticks', font_scale=2)
        
    def create_level_comparison_plot(self, cell_types=None, save_path=None):
        """
        创建2x3布局，每行一种细胞类型，每列一种level，每个单元格内3个子图上下排列
        
        Args:
            cell_types: 要显示的细胞类型列表，如果为None则显示前2种
            save_path: 图片保存路径
        """
        # 获取细胞类型
        if cell_types is None:
            all_cell_types = sorted(self.feats_orig['cell_type'].unique())
            cell_types = all_cell_types[:2]
            print(f"显示前2个细胞类型: {cell_types}")
        
        if len(cell_types) > 2:
            print(f"警告: 2x3布局只支持2行，将只显示前2个细胞类型: {cell_types[:2]}")
            cell_types = cell_types[:2]
        
        # 定义要显示的levels
        levels = [1, 2, 3]
        
        # 创建图形 - 调整高度以容纳上下排列的子图
        fig = plt.figure(figsize=(20, 16))
        
        # 创建主网格：行=细胞类型，列=levels (2行 x 3列)
        main_gs = GridSpec(len(cell_types), len(levels), figure=fig, 
                          hspace=0.25, wspace=0.15,
                          top=0.92, bottom=0.08,
                          left=0.08, right=0.95)
        
        # 颜色定义
        colors = {'normal': '#66c2a5', 'infiltration': '#fc8d62'}
        
        # 遍历每个单元格：行=细胞类型，列=level
        for row_idx, cell_type in enumerate(cell_types):
            for col_idx, level in enumerate(levels):
                print(f'\n{cell_type}/level{level}:')
                # 在当前单元格内创建3个子图的子网格（上下排列）
                cell_gs = GridSpecFromSubplotSpec(3, 1, 
                                                 subplot_spec=main_gs[row_idx, col_idx],
                                                 hspace=0.5)  # 上下子图间距
                
                # 子图1: orig中branch_length分布（顶部）
                ax1 = fig.add_subplot(cell_gs[0, 0])
                self._plot_vertical_kde(self.feats_orig, level, cell_type, 
                                       'branch_length', ax1, colors, 
                                       plot_type='orig_branch',
                                       show_legend=False, #(row_idx==0 and col_idx==0),
                                       show_y_label=(col_idx==0))
                orig_bl_mask = (self.feats_orig.level==level) & (self.feats_orig.cell_type==cell_type)
                orig_bl_means = self.feats_orig[orig_bl_mask][['branch_length', 'tissue_type']].groupby('tissue_type').mean()
                orig_bl_diff = orig_bl_means.loc["infiltration"] - orig_bl_means.loc["normal"]
                orig_bl_ratio = orig_bl_diff / orig_bl_means.loc["normal"]
                print(f'  Original branch length diff: {orig_bl_diff.item():.2f}, {orig_bl_ratio.item():.4f}')
                orig_bl_p = calculate_mannwhitney_pvalues(self.feats_orig[orig_bl_mask], ['branch_length'])
                print(f'      p-value: {orig_bl_p["p_value"]}')
                
                
                # 子图2: scaled中branch_length分布（中间）
                ax2 = fig.add_subplot(cell_gs[1, 0])
                self._plot_vertical_kde(self.feats_scaled, level, cell_type,
                                       'branch_length', ax2, colors,
                                       plot_type='scaled_branch',
                                       show_legend=False,
                                       show_y_label=(col_idx==0))
                norm_bl_mask = (self.feats_scaled.level==level) & (self.feats_scaled.cell_type==cell_type)
                norm_bl_means = self.feats_scaled[norm_bl_mask][['branch_length', 'tissue_type']].groupby('tissue_type').mean()
                norm_bl_diff = norm_bl_means.loc["infiltration"] - norm_bl_means.loc["normal"]
                norm_bl_ratio = norm_bl_diff / norm_bl_means.loc["normal"]
                print(f'  Normalized branch length diff: {norm_bl_diff.item():.2f}, {norm_bl_ratio.item():.4f}')
                norm_bl_p = calculate_mannwhitney_pvalues(self.feats_scaled[norm_bl_mask], ['branch_length'])
                print(f'      p-value: {norm_bl_p["p_value"]}')
                
                # 子图3: orig中order_count分布（底部）
                ax3 = fig.add_subplot(cell_gs[2, 0])
                self._plot_vertical_kde(self.feats_orig, level, cell_type,
                                       'order_count', ax3, colors,
                                       plot_type='orig_order',
                                       show_legend=False,
                                       show_y_label=(col_idx==0))
                orig_bf_means = self.feats_orig[orig_bl_mask][['order_count', 'tissue_type']].groupby('tissue_type').mean()
                orig_bf_diff = orig_bf_means.loc["infiltration"] - orig_bf_means.loc["normal"]
                orig_bf_ratio = orig_bf_diff / orig_bf_means.loc["normal"]
                print(f'  Original branch length diff: {orig_bf_diff.item():.2f}, {orig_bf_ratio.item():.4f}')
                orig_bf_p = calculate_mannwhitney_pvalues(self.feats_orig[orig_bl_mask], ['branch_length'])
                print(f'      p-value: {orig_bf_p["p_value"]}')
        
        
        # 调整布局
        plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图片已保存至: {save_path}")
        
        plt.show()
    
    def _plot_vertical_kde(self, df, level, cell_type, column, ax, colors, 
                          plot_type='orig_branch', show_legend=False, show_y_label=True):
        """
        绘制垂直排列的KDE子图
        
        Args:
            df: 数据DataFrame
            level: 层级
            cell_type: 细胞类型
            column: 要绘制的列名
            ax: 坐标轴
            colors: 颜色字典
            plot_type: 子图类型标识
            show_legend: 是否显示图例
            show_y_label: 是否显示y轴标签
        """
        # 过滤数据
        data = df[(df['level'] == level) & (df['cell_type'] == cell_type)]
        
        if len(data) == 0:
            ax.text(0.5, 0.5, 'No data', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=9, style='italic', alpha=0.7)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xticks([])
            ax.set_yticks([])
            return
        
        # 绘制KDE
        for tissue, color in colors.items():
            tissue_data = data[data['tissue_type'] == tissue]
            if len(tissue_data) > 0:
                sns.kdeplot(data=tissue_data, x=column,
                           color=color, alpha=0.1, fill=True,
                           label=tissue.capitalize() if show_legend else '',
                           ax=ax, linewidth=4, bw_adjust=0.8)
        
        # 根据子图类型设置标题
        title_map = {
            'orig_branch': 'Original Branch Length',
            'scaled_branch': 'Scaled Branch Length',
            'orig_order': 'Original Order Count'
        }
        
        #ax.set_title(title_map.get(plot_type, ''), pad=8)
        
        # 简化坐标轴标签
        if column == 'branch_length':
            xlabel = r'Length (μm)'
        else:
            xlabel = 'Number'
        
        ax.set_xlabel('')
        
        if show_y_label:
            ax.set_ylabel('Probility\nDensity')
        else:
            ax.set_ylabel('')
        
        # 添加网格
        #ax.grid(True, alpha=0.15, linestyle='--', linewidth=0.5)
        
        # 添加样本数量信息
        n_normal = len(data[data['tissue_type'] == 'normal'])
        n_infil = len(data[data['tissue_type'] == 'infiltration'])
        
        if n_normal + n_infil > 0 and False:
            # 在右上角添加样本数量
            ax.text(0.98, 0.95, f'n={n_normal}/{n_infil}',
                   transform=ax.transAxes,
                   ha='right', va='top',
                   bbox=dict(boxstyle='round', facecolor='white', 
                            alpha=0.8, edgecolor='gray', pad=0.1))
        
        # 调整y轴从0开始
        y_lim = ax.get_ylim()
        ax.set_ylim(0, y_lim[1] * 1.15)

        if column == 'order_count':
            ax.set_xlim(0, 15)
        else:
            ax.set_xlim(2, 55)
        
        # 如果是order_count，设置整数刻度
        if column == 'order_count':
            xlim = ax.get_xlim()
            if xlim[1] - xlim[0] < 15:
                ticks = np.arange(int(xlim[0]), int(xlim[1]) + 1, 2)
                ax.set_xticks(ticks)
        else:
            ax.set_xticks([2, 20, 40])
        
        # 显示图例
        if show_legend:
            ax.legend(loc='upper right', framealpha=0.9, frameon=False)
        
        # 简化边框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(3)
        ax.spines['left'].set_linewidth(3)
    

def levelwise_comparison(feat_file_orig, feat_file_scaled, neuron_meta_file):

    ########## Helper functions ###########
    def _load_features(feat_file, meta):
        # Parsing the data
        feats = pd.read_csv(feat_file, index_col=0, low_memory=False)
        feats[meta.columns] = meta.loc[feats.index]

        return feats

    ########## End of helper functions ###########


    # loading the neurons and their meta-informations
    meta = pd.read_csv(neuron_meta_file, index_col=0)
    feats_orig = _load_features(feat_file_orig, meta)
    feats_scaled = _load_features(feat_file_scaled, meta)
    
    ## Visualization
    visualizer = BranchLevelVisualization(feats_orig, feats_scaled)
    
    # 方法1：创建2x3布局（每行一个细胞类型）
    print("创建2x3布局可视化...")
    visualizer.create_level_comparison_plot(
        cell_types=['pyramidal', 'nonpyramidal'],
        save_path='branch_level_comparison_2x3.png'
    )

       
    
if __name__ == '__main__':

    swc_dir_scaled = f'./data/scale_cropped'
    feat_file_scaled = f'./data/scale_cropped_levelwise_features.csv'
    swc_dir_orig = f'./data/orig_morph_cropped'
    feat_file_orig = f'./data/orig_morph_cropped_levelwise_features.csv'
    

    neuron_meta_file = '../src/tissue_cell_meta_jsp.csv'

    #calc_features(swc_dir_scaled, feat_file_scaled)

    levelwise_comparison(feat_file_orig, feat_file_scaled, neuron_meta_file)

