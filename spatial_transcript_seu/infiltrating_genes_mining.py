##########################################################
#Author:          Yufeng Liu
#Create time:     2025-12-15
#Description:               
##########################################################
import os
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from scipy import stats
import statsmodels.stats.multitest as multi
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


__known_gbm_genes__ = ['EGFR', 'VEGFA', 'PTEN', 'TP53', 'IDH1', #'MGMT', #MGMT is ambiguous for tumor
                       'GFAP', 'OLIG2', 'NES', 'SOX2', 'CDK4', 'MDM2',
                       'ATRX', 'CD274', 'CDKN2A', 'TERT']


def load_data_single(st_file):
    adata_st = sc.read(st_file)
    
    # remove possible duplicate genes
    keep_genes = ~adata_st.var['SYMBOL'].duplicated(keep='first')
    adata_st = adata_st[:, keep_genes]
    return adata_st


class SpatialTranscriptomicsAnalyzer:
    def __init__(self, cell_type_specific=False, abundant_thresh=0.4):
        self.de_results = None
        self.abd_thresh = abundant_thresh
        self.cell_type_specific = cell_type_specific

    def load_data(self, st_p65_file, st_p66_file):
        """加载空间转录组数据"""
        self.adata_p65 = load_data_single(st_p65_file)
        self.adata_p66 = load_data_single(st_p66_file)
       
        print(f"正常组织 (p65): {self.adata_p65.shape}")
        print(f"浸润组织 (p66): {self.adata_p66.shape}")
        
        # 确保基因名一致
        if 'SYMBOL' in self.adata_p65.var.columns:
            self.adata_p65.var_names = self.adata_p65.var['SYMBOL'].values
        if 'SYMBOL' in self.adata_p66.var.columns:
            self.adata_p66.var_names = self.adata_p66.var['SYMBOL'].values
            
        return self
    
    def preprocess_data(self, adata_p65=None, adata_p66=None, min_cells=10, min_genes=500):
        """数据预处理"""
        print("数据预处理...")

        update_self_p65 = False
        if adata_p65 is None:
            adata_p65 = self.adata_p65
            update_self_p65 = True

        update_self_p66 = False
        if adata_p66 is None:
            adata_p66 = self.adata_p66
            update_self_p66 = True

        common_genes = list(set(adata_p65.var_names) & set(adata_p66.var_names))
    
        if len(common_genes) < min(len(adata_p65.var_names), len(adata_p66.var_names)):
            print(f"使用共同基因集: {len(common_genes)} 个基因")
            adata_p65 = adata_p65[:, common_genes].copy()
            adata_p66 = adata_p66[:, common_genes].copy()
        
        # 合并两个数据集用于统一预处理
        adata_combined = ad.concat(
            {'p65': adata_p65, 'p66': adata_p66},
            label='sample',
            #keys=['p65', 'p66']
        )
        
        # 基本过滤
        sc.pp.filter_cells(adata_combined, min_genes=min_genes)
        sc.pp.filter_genes(adata_combined, min_cells=min_cells)
        
        # 归一化
        sc.pp.normalize_total(adata_combined, target_sum=1e4)
        sc.pp.log1p(adata_combined)
        
        # 分离回原始数据
        adata_p65 = adata_combined[adata_combined.obs['sample'] == 'p65'].copy()
        adata_p66 = adata_combined[adata_combined.obs['sample'] == 'p66'].copy()
        
        # 移除样本列
        del adata_p65.obs['sample']
        del adata_p66.obs['sample']

        if update_self_p65:
            self.adata_p65 = adata_p65
        if update_self_p66:
            self.adata_p66 = adata_p66
        
        print(f"预处理后 - p65: {adata_p65.shape}, p66: {adata_p66.shape}")
        
        return adata_p65, adata_p66

    def get_cell_type_specific(self, adata_st, abundant_thresh=0.4, min_cells=50):
        # 
        q05 = adata_st.obsm['q05_cell_abundance_w_sf']
        q05 = q05 / q05.values.sum(axis=1).reshape(-1,1)
        abd = (q05 > abundant_thresh).sum(axis=1) > 0
        adata_f = adata_st[abd]

        argmax_ids = np.argmax(adata_f.obsm['q05_cell_abundance_w_sf'], axis=1)
        cell_types = np.array([cname[23:] for cname in q05.columns])[argmax_ids]
        adata_f.obs['predicted_cell_type'] = cell_types
        adata_f.obs['prediction_confidence'] = adata_f.obsm['q05_cell_abundance_w_sf'].max(axis=1)

        # filter by cell numbers
        cts, cnts = np.unique(adata_f.obs['predicted_cell_type'], return_counts=True)
        filtered_cts = cts[cnts > min_cells]
        adata_f = adata_f[adata_f.obs['predicted_cell_type'].isin(filtered_cts)].copy()

        return adata_f

    def perform_differential_expression(self, adata_p65=None, adata_p66=None, method='t-test'):
        """整体差异表达分析"""
        if adata_p65 is None:
            adata_p65 = self.adata_p65
        if adata_p66 is None:
            adata_p66 = self.adata_p66

        # 获取共同基因
        common_genes = list(set(adata_p65.var_names) & set(adata_p66.var_names))
        print(f"共同基因数: {len(common_genes)}")
        
        de_results = []
        
        for gene in common_genes:
            # 获取表达值
            expr_p65 = adata_p65[:, gene].X
            expr_p66 = adata_p66[:, gene].X
            
            if hasattr(expr_p65, 'toarray'):
                expr_p65 = expr_p65.toarray().flatten()
                expr_p66 = expr_p66.toarray().flatten()
            
            # 计算基本统计
            mean_p65 = np.mean(expr_p65)
            mean_p66 = np.mean(expr_p66)
            
            # 避免除零错误
            if mean_p65 == 0:
                mean_p65 = 0.1
            if mean_p66 == 0:
                mean_p66 = 0.1
            
            log2fc = np.log2(mean_p66 / mean_p65)
            
            # 统计检验
            if method == 'wilcoxon':
                try:
                    _, p_value = stats.ranksums(expr_p65, expr_p66)
                except:
                    p_value = 1.0
            elif method == 't-test':
                try:
                    _, p_value = stats.ttest_ind(expr_p65, expr_p66, equal_var=False)
                except:
                    p_value = 1.0
            else:
                p_value = 1.0
            
            de_results.append({
                'gene': gene,
                'log2fc': log2fc,
                'mean_p65': mean_p65,
                'mean_p66': mean_p66,
                'p_value': p_value,
                'abs_log2fc': abs(log2fc)
            })
                
        
        # 创建DataFrame
        self.de_results = pd.DataFrame(de_results)
        
        # 多重检验校正
        if len(self.de_results) > 0:
            _, pvals_fdr, _, _ = multi.multipletests(
                self.de_results['p_value'], method='fdr_bh'
            )
            self.de_results['adj_p_value'] = pvals_fdr
            self.de_results['neg_log10_p'] = -np.log10(self.de_results['adj_p_value'])
        
        # 排序
        self.de_results = self.de_results.sort_values(['abs_log2fc', 'adj_p_value'], 
                                                     ascending=[False, True])
        
        print(f"分析完成，得到 {len(self.de_results)} 个基因的差异表达结果")
        
        return self
    
    def identify_key_genes(self, log2fc_threshold=1.0, adj_p_threshold=0.05, 
                          min_cell_types=2):
        """鉴定关键性改变的基因"""
        print(f"\n鉴定关键性改变的基因...")
        print(f"阈值: |log2FC| > {log2fc_threshold}, adj.p < {adj_p_threshold}")
        
        return self._identify_key_genes_overall(log2fc_threshold, adj_p_threshold)
    
    def _identify_key_genes_overall(self, log2fc_threshold, adj_p_threshold):
        """从整体结果中鉴定关键基因"""
        if self.de_results is None:
            print("没有差异表达结果")
            return pd.DataFrame()
        
        # 筛选显著基因
        significant_genes = self.de_results[
            (abs(self.de_results['log2fc']) > log2fc_threshold) &
            (self.de_results['adj_p_value'] < adj_p_threshold)
        ].copy()
        
        print(f"显著差异表达基因数: {len(significant_genes)}")
        
        # 分类
        up_genes = significant_genes[significant_genes['log2fc'] > 0]
        down_genes = significant_genes[significant_genes['log2fc'] < 0]
        
        print(f"上调基因: {len(up_genes)}个")
        print(f"下调基因: {len(down_genes)}个")
        
        # 保存结果
        self.key_up_genes = up_genes
        self.key_down_genes = down_genes
        
        return significant_genes
    
    def create_volcano_plot(self, output_path='./results_infiltration/volcano_plot.png', font_scale=1.8):
        """创建火山图"""
        plt.figure(figsize=(10, 8))
        sns.set_theme(style='ticks', font_scale=font_scale)

        # 设置颜色
        colors = {'up-regulated': 'red',
                 'down-regulated': 'blue',
                 'non-significant': 'gray'}

        # 标记显著性
        self.de_results['significant'] = (
            (abs(self.de_results['log2fc']) > 1) & 
            (self.de_results['adj_p_value'] < 0.05)
        )
        
        # 绘制散点图
        for regulation, color in colors.items():
            if regulation == 'up-regulated':
                subset_mask = self.de_results['significant'] & (self.de_results['log2fc'] > 0)
            elif regulation == 'down-regulated':
                subset_mask = self.de_results['significant'] & (self.de_results['log2fc'] < 0)
            else:
                subset_mask = ~self.de_results['significant']

            subset = self.de_results[subset_mask]
            r_ratio = subset.shape[0] / self.de_results.shape[0]
            print(f'{regulation}: {r_ratio*100:.2f}%')

            plt.scatter(subset['log2fc'], subset['neg_log10_p'],
                       c=color, s=20, alpha=0.6, label=f'{regulation} ({r_ratio*100:.2f}%)')
        
        # 添加阈值线
        plt.axvline(x=1, color='black', linestyle='--', alpha=0.5, linewidth=2)
        plt.axvline(x=-1, color='black', linestyle='--', alpha=0.5, linewidth=2)
        plt.axhline(y=-np.log10(0.05), color='black', linestyle='--', alpha=0.5, linewidth=2)
        
        # 标记top基因
        #top_genes = self.de_results.nlargest(20, 'abs_log2fc')
        #for _, row in top_genes.iterrows():
        #    plt.annotate(row['gene'], (row['log2fc'], row['neg_log10_p']),
        #                fontsize=8, alpha=0.85)
        
        #plt.xlabel('log2 Fold Change (Infiltrated/Normal)')
        plt.xlabel(r'${log_2}$ Fold Change (Infiltrated/Normal)', fontsize=24)
        plt.ylabel(f'-{r"$log_{10}$"}(Adjusted $p$-value)')
        #plt.title('Infiltrated vs. Normal')
        
        # 保存
        plt.legend(frameon=False, markerscale=2.5, labelspacing=0.1,
                   handletextpad=0.02, borderpad=0.05
        )
        Path(output_path).parent.mkdir(exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=600)
        plt.close()
        print(f"火山图已保存到: {output_path}")

    def plot_gbm_gene_barplot(self, log2fc_threshold=1.0, adj_p_threshold=0.05,
                         output_path='./results_infiltration/infil_gene_barplot.png'):
        """
        绘制已知GBM相关基因的log2FC条形图
        红色表示显著差异：|log2FC| > threshold && adj.p < threshold
        """

        # 已知GBM相关基因列表及其描述
        gbm_genes_info = {
            'EGFR': '表皮生长因子受体（常扩增/突变）',
            'VEGFA': '血管内皮生长因子A（血管生成）',
            'PTEN': '肿瘤抑制基因（常缺失）',
            'TP53': '肿瘤抑制基因p53',
            'IDH1': '异柠檬酸脱氢酶1（突变型预后较好）',
            #'MGMT': 'O6-甲基鸟嘌呤-DNA甲基转移酶',
            'GFAP': '胶质纤维酸性蛋白（星形胶质细胞标志）',
            'OLIG2': '少突胶质细胞转录因子',
            'NES': '巢蛋白（神经干细胞标志）',
            'SOX2': '干细胞转录因子',
            'CDK4': '细胞周期蛋白依赖性激酶4',
            'MDM2': 'p53负调控因子',
            'ATRX': '染色质重塑蛋白',
            'CD274': '程序性死亡配体1',
            'CDKN2A': '细胞周期抑制剂（常缺失）',
            'TERT': '端粒酶逆转录酶（常突变）'
        }

        # 检查de_results是否存在
        if self.de_results is None or len(self.de_results) == 0:
            print("错误：没有差异表达结果数据")
            return

        # 设置基因显示顺序（按照您的输出顺序）
        gene_order = list(gbm_genes_info.keys())

        # 收集数据
        effect_sizes = []
        sig_status = []
        p_values = []
        gene_labels = []
        found_genes = []

        results_df = self.de_results.set_index('gene')

        for gene in gene_order:
            if gene in results_df.index:
                row = results_df.loc[gene]
                effect_sizes.append(row['log2fc'])
                p_values.append(row['adj_p_value'])

                # 判断显著性：|log2FC| > threshold && adj.p < threshold
                is_significant = (abs(row['log2fc']) > log2fc_threshold) and (row['adj_p_value'] < adj_p_threshold)
                sig_status.append(is_significant)

                # 创建包含描述的基因标签
                gene_label = f"{gene}"
                gene_labels.append(gene_label)
                found_genes.append(gene)
            else:
                print(f"警告：基因 {gene} 未在差异表达结果中找到")

        if not effect_sizes:
            print("错误：没有找到任何已知GBM基因的差异表达数据")
            return

        # 创建图形
        fig, ax = plt.subplots(figsize=(12, 9))
        xlim0, xlim1 = -1.4, 3.3

        # 创建条形图
        x_pos = np.arange(len(effect_sizes))

        # 设置颜色：显著为红色，不显著为灰色
        colors = ['#FF6B6B' if sig else '#B0B0B0' for sig in sig_status]  # 红色 vs 浅灰色

        bars = ax.barh(x_pos, effect_sizes, color=colors, edgecolor='black',
                      height=0.6, alpha=0.8)

        # 添加数值标签和p值标记
        n_sig = 0
        for i, (bar, d, p_val, sig) in enumerate(zip(bars, effect_sizes, p_values, sig_status)):
            width = bar.get_width()

            # 确定文本位置（在条形的右侧或左侧）
            if width >= 0:
                text_x = width + 0.02  # 右侧稍微偏移
                ha = 'left'
                color = 'darkred' if sig else 'black'
            else:
                text_x = width - 0.02  # 左侧稍微偏移
                ha = 'right'
                color = 'darkred' if sig else 'black'


            # 在条形内部添加p值标记
            if p_val < 0.001:
                p_text = '***'
            elif p_val < 0.01:
                p_text = '**'
            elif p_val < 0.05:
                p_text = '*'
            else:
                p_text = 'n.s.'

            # 添加log2FC值
            if d > xlim1 - 0.5:
                ax.text(text_x - 0.5, bar.get_y() + bar.get_height()/2,
                       #f'{d:.2f} ({p_text})', va='center', ha=ha,
                        f'{d:.2f}', va='center', ha=ha,
                       color=color, fontsize=18)
            else:
                ax.text(text_x, bar.get_y() + bar.get_height()/2,
                       #f'{d:.2f} ({p_text})', va='center', ha=ha,
                        f'{d:.2f}', va='center', ha=ha,
                       color=color, fontsize=18)

            if sig:
                n_sig += 1

        # 添加参考线
        ax.axvline(x=0, color='black', linestyle='-', linewidth=2.5, alpha=0.8)
        ax.axvline(x=log2fc_threshold, color='red', linestyle='--',
                  linewidth=2, alpha=0.75, label=f'FC threshold (±{log2fc_threshold})')
        ax.axvline(x=-log2fc_threshold, color='red', linestyle='--',
                  linewidth=2, alpha=0.75)

        # 添加显著性区域阴影
        ax.axvspan(log2fc_threshold, xlim1, alpha=0.05,
                  color='red', label='Significant up-regulation')
        ax.axvspan(xlim0, -log2fc_threshold, alpha=0.05,
                  color='blue', label='Significant down-regulation')

        # 设置y轴标签
        ax.set_yticks(x_pos)
        ax.set_yticklabels(gene_labels, fontsize=22)
        ax.invert_yaxis()  # 使特征从上到下显示

        # 设置x轴标签和范围
        ax.tick_params(axis='x', labelsize=22)
        ax.set_xlabel(r'${log_2}$ Fold Change (Infiltrated/Normal)', fontsize=24)

        # 自动调整x轴范围，给标签留出空间
        x_min = min(effect_sizes) - 0.5
        x_max = max(effect_sizes) + 0.5
        ax.set_xlim(x_min, x_max)

        # 设置y轴范围
        ax.set_ylim(len(effect_sizes)-0.5, -0.5)

        # 添加网格
        #ax.grid(True, alpha=0.3, linestyle='--', axis='x')

        # 设置边框粗细
        spine_wd = 2.5
        ax.spines['left'].set_linewidth(spine_wd)
        ax.spines['right'].set_linewidth(spine_wd)
        ax.spines['bottom'].set_linewidth(spine_wd)
        ax.spines['top'].set_linewidth(spine_wd)

        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#FF6B6B', edgecolor='black', alpha=0.8,
                  label=f'Significant (|{r"$log_2$"}FC|>{int(log2fc_threshold)}, {r"$p$"}<{adj_p_threshold})'),
            Patch(facecolor='#B0B0B0', edgecolor='black', alpha=0.8,
                  label='Not significant'),
            Patch(facecolor='none', edgecolor='red', linestyle='--', linewidth=1.5,
                  label=f'{r"$log_2$"}FC threshold (±{log2fc_threshold})')
        ]

        ax.legend(handles=legend_elements, loc='best', fontsize=20,
                  framealpha=0.9, edgecolor='black')

        # 添加标题
        #title = f'Differential Expression of Possible GBM-related Genes\n(Infiltrated vs. Normal)'
        #ax.set_title(title, fontsize=26, pad=8)
        ax.set_xlim(xlim0, xlim1)

        # 调整布局
        plt.tight_layout()

        # 保存图形
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=600, bbox_inches='tight')

        print(f"✓ GBM基因条形图已保存到: {output_path}")

        # 打印详细信息
        n_total = len(gbm_genes_info)
        print(f"\n详细统计信息:")
        print(f"="*60)
        print(f"阈值设置: |log2FC| > {log2fc_threshold} & adj.p < {adj_p_threshold}")
        print(f"分析基因数: {n_total}")
        print(f"显著差异基因数: {n_sig} ({n_sig/n_total*100:.1f}%)")

        # 打印显著基因列表
        print(f"\n显著差异基因列表:")
        for gene, d, p_val, sig in zip(found_genes, effect_sizes, p_values, sig_status):
            if sig:
                direction = "↑上调" if d > 0 else "↓下调"
                print(f"  {gene:8s}: {direction} (log2FC={d:6.2f}, adj.p={p_val:.2e})")

        return fig, ax
    
    def create_spatial_plots(self, genes_of_interest, output_dir='./results_infiltration'):
        """创建基因空间表达图"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        for gene in genes_of_interest:
            if gene not in self.adata_p65.var_names:
                continue
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # 使用seaborn风格
            sns.set_style("white")
            
            for ax, (adata, title) in zip(axes, 
                                         [(self.adata_p65, 'Normal Tissue (p65)'), 
                                          (self.adata_p66, 'Infiltrated Tissue (p66)')]):
                
                # 获取坐标
                if 'spatial' in adata.obsm:
                    x = adata.obsm['spatial'][:, 0]
                    y = adata.obsm['spatial'][:, 1]
                elif 'array_row' in adata.obs and 'array_col' in adata.obs:
                    x = adata.obs['array_col']
                    y = adata.obs['array_row']
                else:
                    print(f"没有空间坐标信息，跳过{gene}")
                    continue
                
                # 获取表达值
                expr = adata[:, gene].X
                if hasattr(expr, 'toarray'):
                    expr = expr.toarray().flatten()
                
                # 创建散点图
                scatter = ax.scatter(
                    x, y, 
                    c=expr, 
                    cmap='magma',  # 10x常用配色
                    s=100,  # 固定大小
                    edgecolor='black',
                    linewidth=0.5,
                    alpha=0.9
                )
                
                # 添加颜色条
                cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('Expression', fontsize=12)
                
                # 设置标题和标签
                ax.set_title(f'{title}\n{gene}', fontsize=14, fontweight='bold')
                ax.set_xlabel('X coordinate', fontsize=12)
                ax.set_ylabel('Y coordinate', fontsize=12)
                ax.set_aspect('equal')
                
                # 移除边框
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_linewidth(1.5)
                ax.spines['bottom'].set_linewidth(1.5)
            
            plt.tight_layout()
            plt.savefig(output_dir / f'visium_style_{gene}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"创建了 {gene} 的Visium风格图")
        
    
    def save_results(self, output_dir='./results_infiltration'):
        """保存分析结果"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        print(f"保存结果到 {output_dir}...")
        
        # 保存整体差异表达结果
        if self.de_results is not None:
            self.de_results.to_csv(output_dir / 'all_differential_expression.csv', index=False)
        
        # 保存关键基因
        if hasattr(self, 'key_genes') and self.key_genes is not None:
            self.key_genes.to_csv(output_dir / 'key_changed_genes.csv', index=False)
        
        # 保存上调/下调基因
        if hasattr(self, 'key_up_genes') and self.key_up_genes is not None:
            self.key_up_genes.to_csv(output_dir / 'upregulated_genes.csv', index=False)
        
        if hasattr(self, 'key_down_genes') and self.key_down_genes is not None:
            self.key_down_genes.to_csv(output_dir / 'downregulated_genes.csv', index=False)
        
        print(f"所有结果已保存")

def analyze_spatial_transcriptomics_celltype(st_p65_file, st_p66_file, plot_only=True):
    """主分析函数"""
    print("="*70)
    print("空间转录组分析: 浸润组织 vs 正常组织")
    print("="*70)

    # 初始化分析器
    analyzer = SpatialTranscriptomicsAnalyzer(cell_type_specific=True)

    # 1. 加载数据
    analyzer.load_data(st_p65_file, st_p66_file)
    
    adata_p65 = analyzer.get_cell_type_specific(analyzer.adata_p65, abundant_thresh=0.4)
    adata_p66 = analyzer.get_cell_type_specific(analyzer.adata_p66, abundant_thresh=0.4)

    celltypes = set(adata_p65.obs['predicted_cell_type']) & set(adata_p66.obs['predicted_cell_type'])
    
    for celltype in celltypes:
        p65ct = adata_p65[adata_p65.obs['predicted_cell_type'] == celltype]
        p66ct = adata_p66[adata_p66.obs['predicted_cell_type'] == celltype]
        
        # 4. 数据预处理
        p65ct, p66ct = analyzer.preprocess_data(p65ct, p66ct, min_cells=10, min_genes=500)
        print(f'[#{celltype}] p65={p65ct.shape[0]}, p66={p66ct.shape[0]}')
     
        cell_name = ''.join(word.capitalize() for word in celltype.split())
        result_path = f'./results_infiltration_{cell_name}'
        os.makedirs(result_path, exist_ok=True)
        result_file = os.path.join(result_path, 'all_differential_expression.csv')
   
        if not (plot_only and os.path.exists(result_file)):
            # 5. 执行差异表达分析
            analyzer.perform_differential_expression(
                p65ct,
                p66ct,
                method='t-test',
            )
            
            # 6. 鉴定关键性改变的基因
            key_genes = analyzer.identify_key_genes(
                log2fc_threshold=1.0,
                adj_p_threshold=0.05,
                min_cell_types=1  # 至少在2个细胞类型中一致改变
            )

            # 8. 创建关键基因的空间表达图
            if len(key_genes) > 0 and False:
                top_genes = key_genes.head(10)['gene'].tolist()
                analyzer.create_spatial_plots(top_genes, os.path.join(result_path, 'spatial_plots'))
            
            # 9. 保存结果
            analyzer.save_results(result_path)
    
        else:
            analyzer.de_results = pd.read_csv(result_file)
     
        # 7. 创建可视化
        analyzer.create_volcano_plot(os.path.join(result_path, 'infil_volcano_plot.png'))
        analyzer.plot_gbm_gene_barplot(output_path=os.path.join(result_path, 'infil_gene_barplot.png'))
        
        # 10. 与GBM相关基因比较
        print("\n" + "="*70)
        print("与已知GBM相关基因比较")
        print("="*70)
        
        if analyzer.de_results is not None:
            results_df = analyzer.de_results.set_index('gene')
            
            print("已知GBM基因在空间转录组中的表达变化:")
            for gene in __known_gbm_genes__:
                if gene in results_df.index:
                    row = results_df.loc[gene]
                    direction = "↑上调" if row['log2fc'] > 0 else "↓下调"
                    sig = "显著" if row['adj_p_value'] < 0.05 else "不显著"
                    print(f"  {gene}: {direction} (log2FC={row['log2fc']:.2f}, "
                          f"adj.p={row['adj_p_value']:.2e}) - {sig}")
                else:
                    print(f"  {gene}: 未在差异表达结果中")
        
    return analyzer

def analyze_spatial_transcriptomics_overall(st_p65_file, st_p66_file, plot_only=True):
    """主分析函数"""
    print("="*70)
    print("空间转录组分析: 浸润组织 vs 正常组织")
    print("="*70)

    result_path = './results_infiltration'

    os.makedirs(result_path, exist_ok=True)
    result_file = os.path.join(result_path, 'all_differential_expression.csv')
    # 初始化分析器
    analyzer = SpatialTranscriptomicsAnalyzer(cell_type_specific=False)

    if not (plot_only and os.path.exists(result_file)):
        # 1. 加载数据
        analyzer.load_data(st_p65_file, st_p66_file)
        
        # 4. 数据预处理
        analyzer.preprocess_data(min_cells=40, min_genes=500)
        
        # 5. 执行差异表达分析
        analyzer.perform_differential_expression(
            method='t-test',
        )
        
        # 6. 鉴定关键性改变的基因
        key_genes = analyzer.identify_key_genes(
            log2fc_threshold=1.0,
            adj_p_threshold=0.05,
            min_cell_types=1  # 至少在2个细胞类型中一致改变
        )

        # 8. 创建关键基因的空间表达图
        if len(key_genes) > 0 and False:
            top_genes = key_genes.head(10)['gene'].tolist()
            analyzer.create_spatial_plots(top_genes, os.path.join(result_path, 'spatial_plots'))
        
        # 9. 保存结果
        analyzer.save_results(result_path)
    
    else:
        analyzer.de_results = pd.read_csv(result_file)
 
    # 7. 创建可视化
    analyzer.create_volcano_plot(os.path.join(result_path, 'infil_volcano_plot.png'))
    analyzer.plot_gbm_gene_barplot(output_path=os.path.join(result_path, 'infil_gene_barplot.png'))
    
    # 10. 与GBM相关基因比较
    print("\n" + "="*70)
    print("与已知GBM相关基因比较")
    print("="*70)
    
    if analyzer.de_results is not None:
        results_df = analyzer.de_results.set_index('gene')
        
        print("已知GBM基因在空间转录组中的表达变化:")
        for gene in __known_gbm_genes__:
            if gene in results_df.index:
                row = results_df.loc[gene]
                direction = "↑上调" if row['log2fc'] > 0 else "↓下调"
                sig = "显著" if row['adj_p_value'] < 0.05 else "不显著"
                print(f"  {gene}: {direction} (log2FC={row['log2fc']:.2f}, "
                      f"adj.p={row['adj_p_value']:.2e}) - {sig}")
            else:
                print(f"  {gene}: 未在差异表达结果中")
    
    return analyzer


if __name__ == '__main__':
    st_p65_file = './cell2loc/P00065_500/SpatialModel/st_allgenes.h5ad'
    # The full cell gene matrix is only in the original data
    st_p66_file = './cell2loc/P00066/SpatialModel/st_allgenes.h5ad'
    cell_type_specific = False

    # 运行分析
    if cell_type_specific:
        analyzer = analyze_spatial_transcriptomics_celltype(st_p65_file, st_p66_file)
    else:
        analyzer = analyze_spatial_transcriptomics_overall(st_p65_file, st_p66_file)

