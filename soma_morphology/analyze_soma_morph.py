##########################################################
#Author:          Yufeng Liu
#Create time:     2025-12-02
#Description:               
##########################################################
import pickle
import numpy as np
import pandas as pd

from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')

def extract_features_to_dataframe(dict_s: Dict[int, Dict[str, Any]]) -> pd.DataFrame:
    """
    更健壮的版本，处理各种嵌套结构
    
    Args:
        dict_s: 字典，key为cell_id，value为每个细胞的features字典
        
    Returns:
        pd.DataFrame: 包含关键特征的DataFrame
    """
    
    # 定义特征提取函数
    def extract_feature(features_dict, path):
        """递归提取嵌套特征"""
        try:
            if isinstance(path, str):
                return features_dict.get(path, np.nan)
            
            elif isinstance(path, list):
                current = features_dict
                for key in path:
                    if isinstance(current, dict) and key in current:
                        current = current[key]
                    else:
                        return np.nan
                return current if current is not None else np.nan
            
            return np.nan
        except:
            return np.nan
    
    # 特征定义
    feature_definitions = [
        ('Volume', ['soma_volume', 'volume_voxel_based']),
        #('max_min_radius_ratio', ['soma_anisotropy', 'max_min_ratio']),
        ('Anisotropy', ['soma_anisotropy', 'anisotropy_index']),
        #('fg_el_ratio', 'fg_el_ratio'),
        ('Smoothness', ['soma_smoothness', 'smoothness_score']),
        ('Roughness', ['soma_smoothness', 'roughness_index']),
        ('Surface   \nComplexity', ['soma_smoothness', 'surface_complexity']),
        ('Curvature\nVariance ', ['soma_smoothness', 'curvature_variance']),
        ('Sphericity', 'soma_sphericity')
    ]
    
    # 收集数据
    records = []
    
    for cell_id, features in dict_s.items():
        if not isinstance(features, dict):
            print(f"Warning: features for cell_id {cell_id} is not a dict")
            continue
        
        record = {'cell_id': cell_id}
        
        for feature_name, path in feature_definitions:
            value = extract_feature(features, path)
            record[feature_name] = value
        
        records.append(record)
    
    # 创建DataFrame
    if not records:
        return pd.DataFrame()
    
    df = pd.DataFrame.from_records(records)
    df = df.set_index('cell_id')
    
    # 确保数据类型正确
    numeric_columns = [
        'soma_volume',
        'max_min_radius_ratio',
        'anisotropy_index',
        'fg_el_ratio',
        'smoothness_score',
        'roughness_index',
        'surface_complexity',
        'curvature_variance',
        'soma_sphericity'
    ]
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


# 批量处理版本
def batch_extract_features_to_dataframes(dict_s: Dict[int, Dict[str, Any]], 
                                        batch_size: int = 1000) -> pd.DataFrame:
    """
    分批提取特征，适用于大型数据集
    
    Args:
        dict_s: 输入字典
        batch_size: 每批处理的数量
        
    Returns:
        pd.DataFrame: 合并后的DataFrame
    """
    from tqdm import tqdm
    
    all_dfs = []
    cell_ids = list(dict_s.keys())
    
    # 分批处理
    for i in tqdm(range(0, len(cell_ids), batch_size), desc="Extracting features"):
        batch_cell_ids = cell_ids[i:i+batch_size]
        batch_dict = {cell_id: dict_s[cell_id] for cell_id in batch_cell_ids}
        
        df_batch = extract_features_to_dataframe(batch_dict)
        all_dfs.append(df_batch)
    
    # 合并所有批次
    if all_dfs:
        result_df = pd.concat(all_dfs)
    else:
        result_df = pd.DataFrame()
    
    return result_df


# 验证函数
def validate_feature_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    验证提取的DataFrame
    
    Args:
        df: 提取的特征DataFrame
        
    Returns:
        包含验证结果的字典
    """
    validation = {}
    
    if df.empty:
        validation['status'] = 'empty'
        return validation
    
    validation['total_cells'] = len(df)
    validation['missing_values'] = df.isna().sum().to_dict()
    validation['feature_stats'] = {}
    
    # 计算基本统计
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            validation['feature_stats'][column] = {
                'mean': float(df[column].mean()),
                'std': float(df[column].std()),
                'min': float(df[column].min()),
                'max': float(df[column].max()),
                'median': float(df[column].median())
            }
    
    return validation

class CellTypeTissueComparison:
    def __init__(self, df, cell_type_col='cell_type', tissue_col='tissue_type'):
        """
        初始化分析器
        
        Args:
            df: 包含所有特征的DataFrame
            cell_type_col: cell_type列名
            tissue_col: tissue_type列名
        """
        self.df = df.copy()
        self.cell_type_col = cell_type_col
        self.tissue_col = tissue_col
        
        # 特征列（排除分类列）
        self.categorical_cols = [cell_type_col, tissue_col, 'region', 'pt_code']
        self.feature_cols = [col for col in df.columns if col not in self.categorical_cols]
        
    
    def statistical_test(self):
        """
        对不同cell_type进行正常vs浸润的统计检验
        """
        results = {}
        
        for cell_type in self.df[self.cell_type_col].unique():
            cell_type_data = self.df[self.df[self.cell_type_col] == cell_type]
            
            # 检查是否有足够的数据
            if len(cell_type_data) < 10:
                print(f"Warning: {cell_type} has only {len(cell_type_data)} samples, skipping...")
                continue
            
            cell_type_results = {}
            
            for feature in self.feature_cols:
                normal_data = cell_type_data[cell_type_data[self.tissue_col] == 'normal'][feature].dropna()
                infiltration_data = cell_type_data[cell_type_data[self.tissue_col] == 'infiltration'][feature].dropna()
                
                if len(normal_data) > 3 and len(infiltration_data) > 3:
                    # Mann-Whitney U检验
                    u_stat, p_value = stats.mannwhitneyu(normal_data, infiltration_data, alternative='two-sided')
                    test_used = 'mannwhitneyu'
                    is_normal = False
                    
                    # 计算效应量
                    cohen_d = self._calculate_effect_size(normal_data, infiltration_data)
                    
                    median_diff = normal_data.median() - infiltration_data.median()
                    percent_diff = (median_diff / normal_data.median()) * 100 if normal_data.median() != 0 else np.nan
                    
                    cell_type_results[feature] = {
                        'p_value': p_value,
                        'test_used': test_used,
                        'cohen_d': cohen_d,
                        'normal_mean': normal_data.mean() if is_normal else normal_data.median(),
                        'infiltration_mean': infiltration_data.mean() if is_normal else infiltration_data.median(),
                        'normal_std': normal_data.std() if is_normal else normal_data.quantile(0.75) - normal_data.quantile(0.25),
                        'infiltration_std': infiltration_data.std() if is_normal else infiltration_data.quantile(0.75) - infiltration_data.quantile(0.25),
                        'normal_n': len(normal_data),
                        'infiltration_n': len(infiltration_data),
                        'percent_diff': percent_diff,
                        'is_significant': p_value < 0.05
                    }
            
            results[cell_type] = cell_type_results
        
        # FDR校正
        self.results = self._apply_fdr_correction(results)
        return self.results
    
    def _calculate_effect_size(self, group1, group2):
        """计算Cohen's d效应量"""
        n1, n2 = len(group1), len(group2)
        mean1, mean2 = np.mean(group1), np.mean(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        
        # 合并标准差
        pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0
        return (mean2 - mean1) / pooled_std
    
    def _apply_fdr_correction(self, results):
        """对p值进行FDR校正"""
        corrected_results = {}
        
        for cell_type, feature_results in results.items():
            p_values = [info['p_value'] for info in feature_results.values()]
            rejected, corrected_p, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
            
            corrected_feature_results = {}
            for (feature, info), is_rej, adj_p in zip(feature_results.items(), rejected, corrected_p):
                info['adj_p_value'] = adj_p
                info['is_significant_adj'] = is_rej
                corrected_feature_results[feature] = info
            
            corrected_results[cell_type] = corrected_feature_results
        
        return corrected_results
    
    def create_comprehensive_visualization(self, save_path=None):
        """
        创建综合可视化
        """
        n_cell_types = len(self.df[self.cell_type_col].unique())
        
        # 创建大图
        fig = plt.figure(figsize=(15, 6 * n_cell_types))
        
        gs = fig.add_gridspec(n_cell_types, 2, hspace=0.45, wspace=0.4,
                              width_ratios=[0.6, 0.4])
        
        for idx, cell_type in enumerate(sorted(self.df[self.cell_type_col].unique(), reverse=True)):
            cell_type_data = self.df[self.df[self.cell_type_col] == cell_type]
            print(f'Data for {cell_type}: {np.unique(cell_type_data.tissue_type, return_counts=True)}')
            
            if len(cell_type_data) < 10:
                continue
            
            # 1. 小提琴图+箱线图
            ax1 = fig.add_subplot(gs[idx, 0])
            self._plot_violin_boxplot(ax1, cell_type_data, cell_type)
            
            # 3. 特征差异热图
            #ax3 = fig.add_subplot(gs[idx, 1])
            #self._plot_feature_heatmap(ax3, cell_type_data, cell_type)
            
            # 4. 效应量条形图
            ax4 = fig.add_subplot(gs[idx, 1])
            self._plot_effect_size_bar(ax4, cell_type, idx)
        
        #plt.suptitle('Cell Type Feature Comparison: Normal vs Infiltration', 
        #             fontsize=16, y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()
    
    def _plot_violin_boxplot(self, ax, data, cell_type):
        """绘制小提琴图+箱线图"""
        melted_data = pd.melt(data, 
                             id_vars=[self.tissue_col], 
                             value_vars=self.feature_cols,
                             var_name='Feature', 
                             value_name='Value')
        
        # 标准化数据以便比较
        for feature in self.feature_cols:
            feature_data = melted_data[melted_data['Feature'] == feature]['Value']
            if feature_data.std() > 0:
                melted_data.loc[melted_data['Feature'] == feature, 'Value'] = (
                    (feature_data - feature_data.mean()) / feature_data.std()
                )
        
        sns.violinplot(x='Feature', y='Value', hue=self.tissue_col,
                       data=melted_data, ax=ax, split=True, inner='box',
                       #palette={'normal': 'lightcoral', 'infiltration': 'gold'},
                       palette={'normal': '#66c2a5', 'infiltration': '#fc8d62'}
                       )
        # 然后去掉所有violin的轮廓线
        for path in ax.collections:
            # 设置轮廓线宽度为0
            path.set_linewidth(0)
            # 或者设置边缘颜色为透明
            path.set_edgecolor('none')

        
        ax.set_title(f'Soma Morphology ({cell_type})')
        ax.set_ylabel('Standardized Value')
        if cell_type == 'pyramidal':
            ax.set_xlabel('')
        #    ax.set_xticks([])
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        ax.legend(title='', ncols=2, loc='upper center', frameon=False)
   
    def _plot_effect_size_bar(self, ax, cell_type, idx):
        """绘制效应量条形图"""
        if cell_type not in self.results:
            ax.text(0.5, 0.5, 'No significant results', 
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Effect Size ({cell_type})')
            return
        
        effect_sizes = []
        features = []
        sig_status = []
        
        for feature, info in self.results[cell_type].items():
            effect_sizes.append(info['cohen_d'])
            features.append(feature)
            sig_status.append(info['is_significant_adj'])
        
        # 创建DataFrame并排序
        effect_df = pd.DataFrame({
            'Feature': features,
            'Cohen_d': effect_sizes,
            'Significant': sig_status
        }).sort_values('Cohen_d', ascending=False)
        
        # 颜色
        colors = ['red' if sig else 'gray' for sig in effect_df['Significant']]
        
        # 绘制条形图
        bars = ax.barh(effect_df['Feature'], effect_df['Cohen_d'], color=colors)
        
        # 添加数值标签
        for bar, d in zip(bars, effect_df['Cohen_d']):
            width = bar.get_width()
            ax.text(width + (0.02 if width >= 0 else -0.35), 
                   bar.get_y() + bar.get_height()/2,
                   f'{d:.2f}', va='center' 
                   )
        
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax.axvline(x=0.2, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.axvline(x=-0.2, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        ax.set_title(f'Cohen\'s d Effect Size ({cell_type})')
        
        #if cell_type == 'pyramidal':
        #    ax.set_xlabel('')
        #    ax.set_xticks([])
        #    ax.set_xticklabels('')
        #else:
        ax.set_xlabel('Cohen\'s d')
        ax.set_xlim(-0.9, 0.9)
    
    def create_summary_table(self):
        """
        创建统计摘要表格
        """
        summary_data = []
        
        for cell_type in sorted(self.df[self.cell_type_col].unique()):
            cell_type_data = self.df[self.df[self.cell_type_col] == cell_type]
            
            if cell_type not in self.results:
                continue
            
            for feature in self.feature_cols:
                if feature in self.results[cell_type]:
                    info = self.results[cell_type][feature]
                    
                    summary_data.append({
                        'Cell_Type': cell_type,
                        'Feature': feature,
                        'Normal_Mean': info['normal_mean'],
                        'Infiltration_Mean': info['infiltration_mean'],
                        'Percent_Diff': info['percent_diff'],
                        'P_Value': info['p_value'],
                        'Adj_P_Value': info['adj_p_value'],
                        'Cohen_d': info['cohen_d'],
                        'Significant_Raw': info['is_significant'],
                        'Significant_Adj': info['is_significant_adj'],
                        'Test_Used': info['test_used'],
                        'Normal_n': info['normal_n'],
                        'Infiltration_n': info['infiltration_n']
                    })
        
        summary_df = pd.DataFrame(summary_data)
        
        # 保存到CSV
        summary_df.to_csv('./data/cell_type_comparison_summary.csv', index=False)
        print("Summary table saved to cell_type_comparison_summary.csv")
        
        return summary_df
    
    def run_full_analysis(self, save_path='cell_type_comparison.png'):
        """
        运行完整的分析流程
        """
        print("Starting cell type comparison analysis...")
        print(f"Total cells: {len(self.df)}")
        print(f"Cell types: {self.df[self.cell_type_col].unique()}")
        print(f"Tissue types: {self.df[self.tissue_col].unique()}")
        
        # 1. 统计检验
        print("\n1. Performing statistical tests...")
        results = self.statistical_test()
        
        # 2. 创建可视化
        print("\n2. Creating visualizations...")
        sns.set_theme(style='ticks', font_scale=1.5)
        self.create_comprehensive_visualization(save_path)
        
        # 3. 创建摘要表格
        print("\n3. Creating summary table...")
        summary_df = self.create_summary_table()
        
        print("\nAnalysis completed!")
        
        return {
            'results': results,
            'summary_df': summary_df,
            'df': self.df
        }


def soma_divergence_among_infiltration_and_normal(
        soma_morph_file, meta_file_neuron, meta_file_tissue, ctype_file, ihc=1
):
    meta_n = pd.read_csv(meta_file_neuron, index_col=0, low_memory=False, encoding='gbk')
    
    # get features
    with open(soma_morph_file, 'rb') as fp:
        dict_s = pickle.load(fp)
    
    gfs = extract_features_to_dataframe(dict_s)
    
    #gfs = gfs.loc[meta_n[meta_n.cell_id.isin(gfs.index)].cell_id]
    meta_n = meta_n[meta_n.cell_id.isin(gfs.index)]
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

    # Estimate the mean volume/size for each type
    for tissue_type in np.unique(gfs_c.tissue_type):
        for cell_type in ctype_dict.values():
            volumes = gfs_c[(gfs_c.tissue_type == tissue_type) & (gfs_c.cell_type == cell_type)].Volume
            vol_mean = volumes.mean()
            rad_mean = np.power(volumes*3/(4*np.pi), 1/3).mean()
            print(f"Size information for {tissue_type}-{cell_type}:\n"
                  f"    Total neurons: {len(volumes)}\n"
                  f"    Mean volume: {vol_mean}\n"
                  f"    Mean size: {rad_mean}\n")
        """
        #Size information for infiltration-pyramidal:
        #    Total neurons: 64
        #    Mean volume: 1005.3320499381251
        #    Mean size: 6.013118062406376

        #Size information for infiltration-nonpyramidal:
        #    Total neurons: 15
        #    Mean volume: 1203.0395677662666
        #    Mean size: 6.277403685704873

        #Size information for normal-pyramidal:
        #    Total neurons: 600
        #    Mean volume: 1437.7391514297483
        #    Mean size: 6.859441023231381

        #Size information for normal-nonpyramidal:
        #    Total neurons: 114
        #    Mean volume: 1340.9893020830525
        #    Mean size: 6.711429953871266

        Size information for infiltration-pyramidal:
            Total neurons: 77
            Mean volume: 1217.8111526048442
            Mean size: 6.317660558334493

        Size information for infiltration-nonpyramidal:
            Total neurons: 18
            Mean volume: 1374.6429797517224
            Mean size: 6.541582599275184

        Size information for normal-pyramidal:
            Total neurons: 728
            Mean volume: 1558.0285303933983
            Mean size: 7.018362170992384

        Size information for normal-nonpyramidal:
            Total neurons: 147
            Mean volume: 1444.7150612580476
            Mean size: 6.8742583770965355
        """

    # Do customized comparison
    analyzer = CellTypeTissueComparison(
        df=gfs_c,
        cell_type_col='cell_type',
        tissue_col='tissue_type'
    )
    
    # 运行完整分析
    analysis_results = analyzer.run_full_analysis(
        save_path='cell_type_comparison_results.png'
    )
    
    # 单独查看每个cell_type的统计结果
    for cell_type, results in analysis_results['results'].items():
        print(f"\n=== {cell_type} ===")
        sig_features = [feat for feat, info in results.items() if info['is_significant_adj']]
        print(f"Significant features (FDR corrected): {len(sig_features)}")
        for feat in sig_features[:5]:  # 只显示前5个显著特征
            info = results[feat]
            print(f"  {feat}: p={info['adj_p_value']:.2e}, d={info['cohen_d']:.2f}")


if __name__ == '__main__':
    meta_file_neuron = '/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta_filter/meta.csv'
    meta_file_tissue_JSP = '../meta/meta_samples_JSP_0330.xlsx.csv'
    ctype_file = '../meta/cell_type_annotation_8.4K_all_CLS2_unique.csv'
    soma_morph_file = './data/soma_features.pkl'

    soma_divergence_among_infiltration_and_normal(soma_morph_file, meta_file_neuron, meta_file_tissue_JSP, ctype_file)


