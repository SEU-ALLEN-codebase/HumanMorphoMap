##########################################################
#Author:          Yufeng Liu
#Create time:     2025-12-03
#Description:               
##########################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')

class MorphologyFeatureAnalyzer:
    def __init__(self, feature_file='data/lmfeatures_scale_cropped_renamed.csv', 
                 meta_file='../src/tissue_cell_meta_jsp.csv'):
        """
        初始化形态学特征分析器
        
        Args:
            feature_file: 特征文件路径
            meta_file: 元数据文件路径
        """
        # 加载特征数据
        self.features_df = pd.read_csv(feature_file, index_col=0)
        
        # 加载元数据
        self.meta_df = pd.read_csv(meta_file, index_col=0)
        
        # 合并数据
        self.df = pd.merge(self.features_df, self.meta_df, 
                          left_index=True, right_index=True, how='inner')
        self.df = self.df.reset_index()
        diameter_name = 'Avg. Branch\n Diameter  '
        length_name = 'Proximal\nDendrite\nLength '
        self.df.rename(columns={
                        'index': 'cell_id',
                        'Average Diameter': diameter_name,
                        'Total Length': length_name
        }, inplace=True)
        
        
        # 删除多余的ID列
        if 'ID' in self.df.columns and 'cell_id' in self.df.columns:
            self.df = self.df.drop('ID', axis=1)

        self.cmp_features = [
            diameter_name,
            length_name,
        ]
        
        # 获取特征列（排除ID和元数据列）
        self.meta_cols = ['cell_id', 'tissue_type', 'pt_code', 'region', 'cell_type']
        self.feature_cols = self.cmp_features #[col for col in self.df.columns if col not in self.meta_cols]
        
        print(f"数据加载完成:")
        print(f"  总细胞数: {len(self.df)}")
        print(f"  特征数: {len(self.feature_cols)}")
        print(f"  细胞类型: {self.df['cell_type'].unique().tolist()}")
        print(f"  组织类型: {self.df['tissue_type'].unique().tolist()}")
        
        # 设置绘图风格
        sns.set_theme(style='ticks', font_scale=1.8)
    
    def statistical_analysis(self):
        """
        对不同cell_type进行正常vs浸润的统计检验
        """
        results = {}
        
        for cell_type in self.df['cell_type'].unique():
            cell_type_data = self.df[self.df['cell_type'] == cell_type]
            
            # 检查是否有足够的数据
            if len(cell_type_data) < 10:
                print(f"Warning: {cell_type} has only {len(cell_type_data)} samples, skipping...")
                continue
            
            cell_type_results = {}
            
            for feature in self.feature_cols:
                normal_data = cell_type_data[cell_type_data['tissue_type'] == 'normal'][feature].dropna()
                infiltration_data = cell_type_data[cell_type_data['tissue_type'] == 'infiltration'][feature].dropna()
                
                if len(normal_data) > 3 and len(infiltration_data) > 3:
                    # Mann-Whitney U检验
                    u_stat, p_value = stats.mannwhitneyu(normal_data, infiltration_data)
                    test_used = 'mannwhitneyu'
                    
                    # 计算效应量
                    cohen_d = self._calculate_effect_size(normal_data, infiltration_data)
                    
                    median_diff = normal_data.median() - infiltration_data.median()
                    percent_diff = (median_diff / normal_data.median()) * 100 if normal_data.median() != 0 else np.nan
                    normal_val = normal_data.median()
                    infiltration_val = infiltration_data.median()
                    normal_std = normal_data.quantile(0.75) - normal_data.quantile(0.25)
                    infiltration_std = infiltration_data.quantile(0.75) - infiltration_data.quantile(0.25)
                    
                    cell_type_results[feature] = {
                        'p_value': p_value,
                        'test_used': test_used,
                        'cohen_d': cohen_d,
                        'normal_mean': normal_val,
                        'infiltration_mean': infiltration_val,
                        'normal_std': normal_std,
                        'infiltration_std': infiltration_std,
                        'normal_n': len(normal_data),
                        'infiltration_n': len(infiltration_data),
                        'percent_diff': percent_diff,
                        'is_significant': p_value < 0.05
                    }
                else:
                    cell_type_results[feature] = {
                        'p_value': 1.0,
                        'test_used': 'insufficient_data',
                        'cohen_d': 0,
                        'normal_mean': np.nan,
                        'infiltration_mean': np.nan,
                        'normal_std': np.nan,
                        'infiltration_std': np.nan,
                        'normal_n': len(normal_data),
                        'infiltration_n': len(infiltration_data),
                        'percent_diff': np.nan,
                        'is_significant': False
                    }
            
            results[cell_type] = cell_type_results
        
        # FDR校正
        self.results = self._apply_fdr_correction(results)
        return self.results
    
    def _calculate_effect_size(self, group1, group2):
        """计算Cohen's d效应量"""
        n1, n2 = len(group1), len(group2)
        if n1 == 0 or n2 == 0:
            return 0
        
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
            if p_values:
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
        
        Args:
            save_path: 保存路径
        """
        n_cell_types = len(self.df['cell_type'].unique())
        
        # 创建大图
        fig = plt.figure(figsize=(12, 5 * n_cell_types))
        
        # 创建2列的网格规格，宽度比例为0.6:0.4
        gs = fig.add_gridspec(n_cell_types, 2, 
                             hspace=0.4, wspace=0.5,
                             width_ratios=[0.5, 0.45])
        
        for idx, cell_type in enumerate(sorted(self.df['cell_type'].unique(), reverse=True)):
            cell_type_data = self.df[self.df['cell_type'] == cell_type]
            
            if len(cell_type_data) < 10:
                continue
            
            # 获取最显著的特征
            significant_features = self.cmp_features
            
            if len(significant_features) == 0:
                continue
            
            # 左侧面板（占60%宽度）
            # 1. 小提琴图+log2_FC折线图（顶部）
            ax1 = fig.add_subplot(gs[idx, 0])
            #self._plot_violin_boxplot(ax1, cell_type_data, cell_type, significant_features)
            self._plot_boxplot(ax1, cell_type_data, cell_type, significant_features)
            
            # 4. 效应量条形图（底部）
            ax4 = fig.add_subplot(gs[idx, 1])
            self._plot_effect_size_bar(ax4, cell_type, idx, significant_features)
        
        #plt.suptitle('Morphology Feature Comparison: Normal vs Infiltration', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
   
    def _plot_violin_boxplot(self, ax, data, cell_type, features):
        """绘制小提琴图"""
        # 准备数据
        plot_data = data[['tissue_type'] + features].melt(
            id_vars=['tissue_type'], 
            var_name='Feature', 
            value_name='Value'
        )
        
        # 标准化每个特征的数据
        for feature in features:
            feature_data = plot_data[plot_data['Feature'] == feature]['Value']
            if feature_data.std() > 0:
                plot_data.loc[plot_data['Feature'] == feature, 'Value'] = (
                    (feature_data - feature_data.mean()) / feature_data.std()
                )
        
        # 绘制小提琴图
        sns.violinplot(x='Feature', y='Value', hue='tissue_type',
                      data=plot_data, ax=ax, split=True, inner='box',
                      #palette={'normal': 'lightcoral', 'infiltration': 'gold'},
                      palette={'normal': '#66c2a5', 'infiltration': '#fc8d62'},
                      )
        
        # 然后去掉所有violin的轮廓线
        for path in ax.collections:
            # 设置轮廓线宽度为0
            path.set_linewidth(0)
            # 或者设置边缘颜色为透明
            path.set_edgecolor('none')
       
        # 设置左侧y轴
        ax.set_ylabel('Standardized Value')
        ax.set_ylim(-3.5, 5.5)
        if cell_type == 'pyramidal':
            ax.set_xlabel('')
            #ax.set_xticks([])
        #else:
            #plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_xticks([0, 1], self.cmp_features)

        spine_wd = 2
        ax.spines['left'].set_linewidth(spine_wd)
        ax.spines['right'].set_linewidth(spine_wd)
        ax.spines['bottom'].set_linewidth(spine_wd)
        ax.spines['top'].set_linewidth(spine_wd)

        ax.legend(title='', ncols=1, loc='upper center', frameon=False)

    def _plot_boxplot(self, ax, data, cell_type, features):
        """绘制箱线图"""
        # 准备数据（与原代码相同）
        plot_data = data[['tissue_type'] + features].melt(
            id_vars=['tissue_type'],
            var_name='Feature',
            value_name='Value'
        )

        # 标准化每个特征的数据（与原代码相同）
        for feature in features:
            feature_data = plot_data[plot_data['Feature'] == feature]['Value']
            if feature_data.std() > 0:
                plot_data.loc[plot_data['Feature'] == feature, 'Value'] = (
                    (feature_data - feature_data.mean()) / feature_data.std()
                )

        # *** 核心改动1：将 sns.violinplot 替换为 sns.boxplot ***
        # 使用 boxplot，设置 inner='box' 已不再需要
        print(plot_data.groupby('tissue_type').count())
        sns.boxplot(x='Feature', y='Value', hue='tissue_type',
                    data=plot_data, ax=ax,
                    # palette={'normal': '#66c2a5', 'infiltration': '#fc8d62'},
                    palette={'normal': '#66c2a5', 'infiltration': '#fc8d62'},
                    # 可根据需要调整箱线图外观，例如：
                    linewidth=2,      # 箱线轮廓线宽
                    fliersize=2,        # 异常点大小
                    width=0.4,          # 箱体宽度
                    meanprops={'marker': 'o', 'markerfacecolor': 'red', 'linewidth': 3},
                    dodge=True,
                    gap=0.25,
                    )

        # *** 核心改动2：删除专门处理小提琴图轮廓的循环（共6行）***
        # for path in ax.collections:
        #     path.set_linewidth(0)
        #     path.set_edgecolor('none')

        # 设置左侧y轴（与原代码相同）
        ax.set_ylabel('Standardized Value')
        ax.set_ylim(-3.5, 5.5)
        if cell_type == 'pyramidal':
            ax.set_xlabel('')
        else:
            ax.set_xlabel('Morphological Features')

        customized_xticks = [*self.cmp_features]
        customized_xticks[1] = 'Proximal Branch\nLength    '
        ax.set_xticks([0, 1], customized_xticks)

        # 设置坐标轴线宽（与原代码相同）
        spine_wd = 2
        ax.spines['left'].set_linewidth(spine_wd)
        ax.spines['right'].set_linewidth(spine_wd)
        ax.spines['bottom'].set_linewidth(spine_wd)
        ax.spines['top'].set_linewidth(spine_wd)

        # legend customize
        handles, labels = ax.get_legend_handles_labels()
        # 2. 找到目标标签的索引并修改（这里假设是第一个）
        target_index = 1  # 根据实际情况调整索引
        labels[target_index] = 'infiltrated'
        ax.legend(handles, labels, title='', ncols=2, loc='upper center', frameon=False,
                  handletextpad=0.2, labelspacing=0, columnspacing=0.75,
        )
        
        
    
    def _plot_effect_size_bar(self, ax, cell_type, idx, features):
        """绘制效应量条形图"""
        if cell_type not in self.results:
            ax.text(0.5, 0.5, 'No analysis results', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # 收集效应量数据
        effect_sizes = []
        sig_status = []
        
        for feature in features:
            if feature in self.results[cell_type]:
                info = self.results[cell_type][feature]
                effect_sizes.append(info['cohen_d'])
                sig_status.append(info['is_significant_adj'])
        
        if not effect_sizes:
            return
        
        # 创建条形图
        x_pos = np.arange(len(effect_sizes))
        colors = ['red' if sig else 'gray' for sig in sig_status]
        
        bars = ax.barh(x_pos, effect_sizes, color=colors, edgecolor='black', height=0.25)
        
        # 添加数值标签
        for i, (bar, d) in enumerate(zip(bars, effect_sizes)):
            width = bar.get_width()
            ax.text(width + (0.02 if width >= 0 else -0.35), 
                   bar.get_y() + bar.get_height()/2,
                   f'{d:.2f}', va='center')
        
        # 添加参考线
        ax.axvline(x=0, color='black', linestyle='-', linewidth=2.)
        ax.axvline(x=0.2, color='gray', linestyle='--', linewidth=1., alpha=0.5)
        ax.axvline(x=-0.2, color='gray', linestyle='--', linewidth=1., alpha=0.5)
        
        # 设置y轴标签
        ax.set_yticks(x_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()  # 使特征从上到下显示
        
        ax.set_xlabel(r"Cohen's $d$")
        #if cell_type == 'pyramidal':
        #    ax.set_xlabel('')
        #    ax.set_xticks([])
        #    ax.set_xticklabels('')
        #else:
        ax.set_xlim(-0.75, 0.95)
        ax.set_ylim(1.5, -0.5)
        
        spine_wd = 2
        ax.spines['left'].set_linewidth(spine_wd)
        ax.spines['right'].set_linewidth(spine_wd)
        ax.spines['bottom'].set_linewidth(spine_wd)
        ax.spines['top'].set_linewidth(spine_wd)
        
        # 添加效应量解释
        #ax.text(0.02, 0.98, 'Small: |d|<0.5\nMedium: 0.5≤|d|<0.8\nLarge: |d|≥0.8',
        #        transform=ax.transAxes,
        #        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        #        verticalalignment='top')
    
    def create_summary_table(self, output_file='morphology_comparison_summary.csv'):
        """创建统计摘要表格"""
        summary_data = []
        
        for cell_type in sorted(self.df['cell_type'].unique()):
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
        
        if not summary_df.empty:
            summary_df.to_csv(output_file, index=False)
            print(f"Summary table saved to {output_file}")
        
        return summary_df
    
    def run_full_analysis(self, save_path='morphology_comparison.png'):
        """运行完整的分析流程"""
        print("Starting morphology feature comparison analysis...")
        print(f"Total cells: {len(self.df)}")
        print(f"Cell types: {self.df['cell_type'].unique().tolist()}")
        
        # 1. 统计检验
        print("\n1. Performing statistical tests...")
        results = self.statistical_analysis()
        
        # 2. 创建可视化
        print("\n2. Creating visualizations...")
        self.create_comprehensive_visualization(save_path=save_path)
        
        # 3. 创建摘要表格
        print("\n3. Creating summary table...")
        summary_df = self.create_summary_table()
        
        print("\nAnalysis completed!")
        
        return {
            'results': results,
            'summary_df': summary_df,
            'data': self.df
        }


# 使用示例
if __name__ == "__main__":
    # 创建分析器实例
    analyzer = MorphologyFeatureAnalyzer(
        feature_file='data/lmfeatures_scale_cropped_renamed_noSomaDiameter.csv',
        #feature_file = '../spatial-enhanced/data/auto8.4k_0510_resample1um_mergedBranches0712_crop100_renamed_noSomaDiameter_ME_notIncludeSelf.csv',
        #feature_file = '../h01-guided-reconstruction/auto8.4k_0510_resample1um_mergedBranches0712_crop100_renamed_noSomaDiameter.csv',
        meta_file='../src/tissue_cell_meta_jsp.csv'
    )
    
    # 运行完整分析
    results = analyzer.run_full_analysis(save_path='morphology_comparison.png')
    
    # 查看显著特征
    for cell_type, cell_results in results['results'].items():
        sig_features = [feat for feat, info in cell_results.items() 
                       if info.get('is_significant_adj', False)]
        print(f"\n{cell_type}: {len(sig_features)} significantly different features")
        for feat in sig_features[:5]:
            info = cell_results[feat]
            print(f"  {feat}: Cohen's d = {info['cohen_d']:.2f}, adj p = {info['adj_p_value']:.4f}")



