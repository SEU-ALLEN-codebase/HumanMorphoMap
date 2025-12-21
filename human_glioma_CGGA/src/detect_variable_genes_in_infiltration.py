##########################################################
#Author:          Yufeng Liu
#Create time:     2025-12-08
#Description:               
##########################################################
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.stats.multitest as multi
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

class GBM_DifferentialExpression_FPKM:
    def __init__(self):
        self.gbm_samples = []
        self.normal_samples = []
        self.gbm_fpkm = None
        self.normal_fpkm = None
        self.de_results = None
        self.volcano_data = None
        
    def load_metadata(self, meta_file_path, batch_num):
        """加载临床元数据并筛选GBM样本"""
        meta_data = pd.read_csv(meta_file_path, sep='\t', low_memory=False)
        
        # 筛选GBM样本
        gbm_meta = meta_data[meta_data['Histology'] == 'GBM'].copy()
        
        print(f"Batch {batch_num}: 总共{len(meta_data)}个样本，其中{len(gbm_meta)}个GBM样本")
        
        # 提取GBM样本ID
        gbm_sample_ids = gbm_meta['CGGA_ID'].tolist()
        
        return gbm_sample_ids, gbm_meta
    
    def load_fpkm_data(self, fpkm_file_path, sample_ids=None, is_normal=False):
        """加载FPKM表达数据"""
        try:
            # 读取FPKM数据
            if is_normal:
                # 正常组织数据可能格式略有不同
                fpkm_data = pd.read_csv(fpkm_file_path, sep='\t', low_memory=False, index_col=0)
            else:
                fpkm_data = pd.read_csv(fpkm_file_path, sep='\t', low_memory=False, index_col=0)
         
            # 如果提供了样本ID列表，则筛选特定样本
            if sample_ids is not None:
                # 确保sample_ids在数据中存在的列
                available_samples = [s for s in sample_ids if s in fpkm_data.columns]
                if len(available_samples) < len(sample_ids):
                    missing = set(sample_ids) - set(available_samples)
                    print(f"警告: {len(missing)}个样本在FPKM数据中不存在")
                    if len(missing) > 0 and len(missing) <= 10:
                        print(f"缺失样本: {list(missing)[:10]}")
                fpkm_data = fpkm_data[available_samples]
            
            print(f"加载的FPKM数据形状: {fpkm_data.shape}")
            
            # 基本数据检查
            print(f"FPKM值范围: [{fpkm_data.min().min():.2f}, {fpkm_data.max().max():.2f}]")
            print(f"FPKM平均值: {fpkm_data.mean().mean():.2f}")
            
            return fpkm_data
            
        except Exception as e:
            print(f"加载FPKM数据时出错: {e}")
            # 尝试其他可能的列名
            try:
                fpkm_data = pd.read_csv(fpkm_file_path, sep='\t', low_memory=False)
                print("数据列名:", fpkm_data.columns[:10].tolist())
            except:
                pass
            raise
    
    def filter_low_expressed_genes(self, fpkm_data, min_fpkm=0.1, min_samples=0.1):
        """
        过滤低表达基因
        min_fpkm: 最小FPKM值阈值
        min_samples: 最小表达样本比例
        """
        if isinstance(min_samples, float):
            min_samples = int(len(fpkm_data.columns) * min_samples)
        
        # 过滤：在至少min_samples个样本中FPKM > min_fpkm
        keep_genes = ((fpkm_data > min_fpkm).sum(axis=1) >= min_samples)
        filtered_data = fpkm_data.loc[keep_genes]
        
        print(f"过滤低表达基因: {len(filtered_data)} / {len(fpkm_data)} 个基因保留")
        print(f"过滤阈值: FPKM > {min_fpkm}, 至少 {min_samples} 个样本")
        
        return filtered_data
    
    def log_transform_fpkm(self, fpkm_data, pseudo_count=1):
        """对FPKM数据进行log2转换（加伪计数）"""
        log_fpkm = np.log2(fpkm_data + pseudo_count)
        return log_fpkm
    
    def normalize_data(self, gbm_fpkm, normal_fpkm):
        """数据标准化（如果需要）"""
        # FPKM已经是标准化数据，这里可以添加额外的标准化步骤
        # 例如：分位数标准化或中位数标准化
        
        # 简单的中位数标准化
        gbm_median = gbm_fpkm.median().median()
        normal_median = normal_fpkm.median().median()
        overall_median = (gbm_median + normal_median) / 2
        
        gbm_normalized = gbm_fpkm / gbm_median * overall_median
        normal_normalized = normal_fpkm / normal_median * overall_median
        
        return gbm_normalized, normal_normalized
    
    def perform_de_analysis(self, gbm_data, normal_data, method='t-test', use_log=True):
        """
        执行差异表达分析
        method: 't-test', 'mannwhitney', or 'foldchange'
        use_log: 是否使用log转换后的数据
        """
        print(f"\n执行差异表达分析 (方法: {method})...")
        print(f"GBM样本数: {gbm_data.shape[1]}, 正常样本数: {normal_data.shape[1]}")
        
        de_results = []
        
        # 获取共同的基因
        common_genes = set(gbm_data.index) & set(normal_data.index)
        print(f"共同基因数: {len(common_genes)}")
        
        for gene in common_genes:
            gbm_expr = gbm_data.loc[gene].values.astype(float)
            normal_expr = normal_data.loc[gene].values.astype(float)
            
            # 计算log2 fold change
            gbm_mean = np.mean(gbm_expr)
            normal_mean = np.mean(normal_expr)
            
            # 避免除零错误
            if normal_mean == 0:
                normal_mean = 0.1
            if gbm_mean == 0:
                gbm_mean = 0.1
            
            log2fc = np.log2(gbm_mean / normal_mean)
            
            # 根据选择的方法计算p值
            if method == 't-test':
                # 学生t检验
                t_stat, p_value = stats.ttest_ind(gbm_expr, normal_expr, 
                                                  equal_var=False)  # Welch's t-test
                
            elif method == 'mannwhitney':
                # Mann-Whitney U检验（非参数）
                u_stat, p_value = stats.mannwhitneyu(gbm_expr, normal_expr, 
                                                     alternative='two-sided')
                
            elif method == 'foldchange':
                # 仅基于fold change，无统计检验
                p_value = 1.0
                
            else:
                raise ValueError(f"未知方法: {method}")
            
            de_results.append({
                'gene': gene,
                'log2fc': log2fc,
                'gbm_mean': gbm_mean,
                'normal_mean': normal_mean,
                'p_value': p_value,
                'abs_log2fc': abs(log2fc),
                'gbm_std': np.std(gbm_expr),
                'normal_std': np.std(normal_expr)
            })
                
        
        # 创建结果DataFrame
        results_df = pd.DataFrame(de_results)
        # remove nan values
        results_df = results_df[results_df.isna().sum(axis=1) == 0]
        
        if len(results_df) == 0:
            raise ValueError("没有找到可分析的共同基因")
        
        # 校正p值（多重检验校正）
        if method != 'foldchange':
            _, results_df['adj_p_value'], _, _ = multi.multipletests(
                results_df['p_value'], method='fdr_bh'
            )
        else:
            results_df['adj_p_value'] = results_df['p_value']
        
        # 添加-log10(p值)用于火山图
        results_df['neg_log10_p'] = -np.log10(results_df['adj_p_value'])
        
        # 排序：首先按绝对log2FC，然后按p值
        results_df = results_df.sort_values(['abs_log2fc', 'adj_p_value'], 
                                           ascending=[False, True])
        
        return results_df
    
    def get_significant_genes(self, results_df, log2fc_threshold=1, adj_p_threshold=0.05):
        """获取显著差异表达基因"""
        # 上调基因
        up_genes = results_df[
            (results_df['log2fc'] > log2fc_threshold) & 
            (results_df['adj_p_value'] < adj_p_threshold)
        ].copy()
        
        # 下调基因
        down_genes = results_df[
            (results_df['log2fc'] < -log2fc_threshold) & 
            (results_df['adj_p_value'] < adj_p_threshold)
        ].copy()
        
        print(f"显著上调基因数: {len(up_genes)}")
        print(f"显著下调基因数: {len(down_genes)}")
        
        # 为基因分类添加标签
        results_df['regulation'] = 'non-significant'
        results_df.loc[up_genes.index, 'regulation'] = 'up-regulated'
        results_df.loc[down_genes.index, 'regulation'] = 'down-regulated'
        
        return up_genes, down_genes, results_df
    
    def create_volcano_plot(self, results_df, output_path='./figures/gbm_volcano_plot.png', font_scale=1.8):
        """创建火山图
            The volcano plot exhibit a bimodal distribution rather than a unimodal one. This may be attributed to 
            the highly divergent expression patterns between normal and tumor tissues. To verify, I checked the 
            log2FC of several housekeeper genes, and they indeed show values near 0.

            The tested genes are: 'GAPDH', 'ACTB', 'TUBB', 'HPRT1', 'TBP', 'PPIA', 'RPL13A', 'UBC', 'PGK1', 'B2M'
        """
        plt.figure(figsize=(10, 8))
        sns.set_theme(style='ticks', font_scale=font_scale)
        
        # 设置颜色
        colors = {'up-regulated': 'red', 
                 'down-regulated': 'blue', 
                 'non-significant': 'gray'}
        
        # 绘制散点图
        for regulation, color in colors.items():
            subset = results_df[results_df['regulation'] == regulation]
            r_ratio = subset.shape[0] / results_df.shape[0]
            print(f'{regulation}: {r_ratio*100:.2f}%')
            plt.scatter(subset['log2fc'], subset['neg_log10_p'], 
                       c=color, s=20, alpha=0.6, label=f'{regulation} ({r_ratio*100:.2f}%)')
        
        # 添加阈值线
        plt.axvline(x=1, color='black', linestyle='--', alpha=0.5, linewidth=2)
        plt.axvline(x=-1, color='black', linestyle='--', alpha=0.5, linewidth=2)
        plt.axhline(y=-np.log10(0.05), color='black', linestyle='--', alpha=0.5, linewidth=2)
        
        # 标记top基因
        top_genes = results_df.nlargest(20, 'abs_log2fc')
        for idx, row in top_genes.iterrows():
            plt.annotate(row['gene'], (row['log2fc'], row['neg_log10_p']),
                        fontsize=8, alpha=0.85)
        
        plt.xlabel(r'${log_2}$ Fold Change (Infiltrated/Normal)')
        plt.ylabel(f'-{r"$log_{10}$"}(Adjusted $p$-value)')
        #plt.title('GBM vs. Normal')
        plt.legend(frameon=False, markerscale=2.5, labelspacing=0.1, 
                   handletextpad=0.02, borderpad=0.05
        )
        #plt.grid(True, alpha=0.3)
        
        # 保存图片
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        
        print(f"火山图已保存到: {output_path}")
    

    def create_heatmap(self, gbm_data, normal_data, top_n=50, output_path='./figures/heatmap.png'):
        """创建热图显示top差异表达基因（简化版）"""
        sns.set_theme(style='ticks', font_scale=1.2)
        # 获取top差异表达基因
        top_genes = self.de_results.head(top_n)['gene'].tolist()

        # 提取数据
        heatmap_data = pd.concat([
            gbm_data.loc[top_genes],
            normal_data.loc[top_genes]
        ], axis=1)

        # log2转换
        heatmap_data_log = np.log2(heatmap_data + 1)

        # 标准化（按行）
        heatmap_data_zscore = heatmap_data_log.apply(
            lambda x: (x - x.mean()) / x.std(), axis=1
        )

        # 创建热图
        plt.figure(figsize=(14, 10))
        
        # 绘制热图，调整colorbar尺寸
        ax = sns.heatmap(heatmap_data_zscore, cmap='RdBu_r', center=0,
                        xticklabels=False, yticklabels=True,
                        cbar_kws={
                            'shrink': 0.2,  # 缩小到原来的20%
                            'aspect': 5,     # 高宽比：高度是宽度的5倍
                            'label': 'Z-score',
                            'pad': 0.02      # colorbar与热图的间距
                        })
        
        # 获取colorbar对象并调整标签字体
        cbar = ax.collections[0].colorbar
        cbar.ax.set_ylabel('Z-score', fontsize=18, rotation=270, labelpad=20)
        cbar.ax.tick_params(labelsize=16)
        
        plt.title(f'Top {top_n} Differentially Expressed Genes (GBM vs Normal)', 
                 fontsize=24, pad=30)
        plt.xlabel('Samples', fontsize=22)
        plt.ylabel('Genes', fontsize=22)
        
        # 添加样本类型分隔线
        gbm_samples_count = gbm_data.shape[1]
        plt.axvline(x=gbm_samples_count, color='white', linewidth=3)
        plt.text(gbm_samples_count/2, -0.5, 'GBM Samples', 
                ha='center', fontsize=16)
        plt.text(gbm_samples_count + normal_data.shape[1]/2, -0.5, 'Normal Samples',
                ha='center', fontsize=16)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    def plot_gbm_gene_barplot(self, log2fc_threshold=1.0, adj_p_threshold=0.05, 
                         output_path='./figures/gbm_gene_barplot.png'):
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
        
        xlim0, xlim1 = -1.4, 3.3
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
        ax.set_xlabel(r'${log_2}$ Fold Change (GBM/Normal)', fontsize=24)
        
        # 自动调整x轴范围，给标签留出空间
        x_min = min(effect_sizes) - 0.5
        x_max = max(effect_sizes) + 0.5
        ax.set_xlim(xlim0, xlim1)
        
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
        #title = f'Differential Expression of Possible GBM-related Genes\n(GBM vs. Normal)'
        #ax.set_title(title, fontsize=26, pad=8)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图形
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
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
    
    def run_analysis_pipeline(self, meta_files, fpkm_files, normal_fpkm_file):
        """运行完整分析流程"""
        print("=" * 70)
        print("开始GBM vs 正常组织差异表达分析 (使用FPKM数据)")
        print("=" * 70)
        
        # 1. 加载GBM元数据并获取样本ID
        all_gbm_samples = []
        for i, meta_file in enumerate(meta_files, 1):
            print(f"\n处理元数据文件 {i}: {meta_file}")
            gbm_samples, _ = self.load_metadata(meta_file, i)
            all_gbm_samples.extend(gbm_samples)
        
        print(f"\n总共获取到 {len(all_gbm_samples)} 个GBM样本")
        
        # 2. 加载GBM FPKM数据
        gbm_fpkm_list = []
        for i, fpkm_file in enumerate(fpkm_files, 1):
            print(f"\n加载FPKM文件 {i}: {fpkm_file}")
            fpkm_data = self.load_fpkm_data(fpkm_file, all_gbm_samples)
            gbm_fpkm_list.append(fpkm_data)
        
        # 合并两个batch的FPKM数据
        self.gbm_fpkm = pd.concat(gbm_fpkm_list, axis=1)
        print(f"\n合并后的GBM FPKM数据形状: {self.gbm_fpkm.shape}")
        
        # 3. 加载正常组织FPKM数据
        print(f"\n加载正常组织FPKM文件: {normal_fpkm_file}")
        self.normal_fpkm = self.load_fpkm_data(normal_fpkm_file, is_normal=True)
        
        # 4. 数据预处理
        print("\n进行数据预处理...")
        # 过滤低表达基因
        self.gbm_fpkm = self.filter_low_expressed_genes(
            self.gbm_fpkm, min_fpkm=0.1, min_samples=0.1
        )
        self.normal_fpkm = self.filter_low_expressed_genes(
            self.normal_fpkm, min_fpkm=0.1, min_samples=0.1
        )
        
        # 5. 标准化（可选）
        print("\n数据标准化...")
        self.gbm_fpkm, self.normal_fpkm = self.normalize_data(
            self.gbm_fpkm, self.normal_fpkm
        )
        
        # 6. log2转换（通常用于差异表达分析）
        print("log2转换...")
        gbm_log = self.log_transform_fpkm(self.gbm_fpkm)
        normal_log = self.log_transform_fpkm(self.normal_fpkm)
        
        # 7. 执行差异表达分析
        self.de_results = self.perform_de_analysis(
            gbm_log, normal_log, method='t-test'
        )

        return

    def save_results(self, output_dir='./fpkm_de_results'):
        """保存分析结果"""
        Path(output_dir).mkdir(exist_ok=True)
        
        # 保存完整的差异表达结果
        self.de_results.to_csv(f'{output_dir}/all_differential_expression_results.csv', 
                               index=False)
        
        # 保存显著上调基因
        up_genes = self.de_results[self.de_results['regulation'] == 'up-regulated']
        up_genes.to_csv(f'{output_dir}/significantly_upregulated_genes.csv', 
                        index=False)
        
        # 保存显著下调基因
        down_genes = self.de_results[self.de_results['regulation'] == 'down-regulated']
        down_genes.to_csv(f'{output_dir}/significantly_downregulated_genes.csv', 
                          index=False)
        
        # 保存top基因
        top_genes = self.de_results.head(100)
        top_genes.to_csv(f'{output_dir}/top_100_differential_genes.csv', 
                         index=False)
        
        # 保存基因表达矩阵（用于其他分析）
        gbm_common = self.gbm_fpkm.loc[self.de_results['gene']]
        normal_common = self.normal_fpkm.loc[self.de_results['gene']]
        
        combined_expression = pd.concat([gbm_common, normal_common], axis=1)
        combined_expression.to_csv(f'{output_dir}/expression_matrix_common_genes.csv')
        
        print(f"\n所有结果已保存到 {output_dir} 目录")
        
    def generate_summary_report(self):
        """生成分析总结报告"""
        if self.de_results is None:
            print("请先运行分析流程")
            return
        
        total_genes = len(self.de_results)
        up_genes = self.de_results[self.de_results['regulation'] == 'up-regulated']
        down_genes = self.de_results[self.de_results['regulation'] == 'down-regulated']
        
        print("\n" + "=" * 70)
        print("差异表达分析总结报告")
        print("=" * 70)
        print(f"分析的总基因数: {total_genes}")
        print(f"显著上调基因数 (log2FC > 1, adj.p < 0.05): {len(up_genes)}")
        print(f"显著下调基因数 (log2FC < -1, adj.p < 0.05): {len(down_genes)}")
        print(f"GBM样本数: {self.gbm_fpkm.shape[1]}")
        print(f"正常组织样本数: {self.normal_fpkm.shape[1]}")
        
        if len(up_genes) > 0:
            print(f"\nTop 10 上调基因:")
            for i, row in up_genes.head(10).iterrows():
                print(f"  {row['gene']}: log2FC = {row['log2fc']:.2f}, "
                      f"adj.p = {row['adj_p_value']:.2e}, "
                      f"GBM均值 = {row['gbm_mean']:.2f}, "
                      f"Normal均值 = {row['normal_mean']:.2f}")
        
        if len(down_genes) > 0:
            print(f"\nTop 10 下调基因:")
            for i, row in down_genes.head(10).iterrows():
                print(f"  {row['gene']}: log2FC = {row['log2fc']:.2f}, "
                      f"adj.p = {row['adj_p_value']:.2e}, "
                      f"GBM均值 = {row['gbm_mean']:.2f}, "
                      f"Normal均值 = {row['normal_mean']:.2f}")

# 使用示例
def main():
    # 初始化分析器
    analyzer = GBM_DifferentialExpression_FPKM()
    
    # 设置文件路径
    meta_files = [
        "../data/CGGA.mRNAseq_325_clinical.20200506.txt",
        "../data/CGGA.mRNAseq_693_clinical.20200506.txt"
    ]
    
    fpkm_files = [
        "../data/CGGA.mRNAseq_325.RSEM-genes.20200506.txt",
        "../data/CGGA.mRNAseq_693.RSEM-genes.20200506.txt"
    ]
    
    normal_fpkm_file = "../data/CGGA_RNAseq_Control_20.txt"
    cache_file = 'cached.pkl'    

    import os, pickle

    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as fp:
            gbm_fpkm, normal_fpkm, de_results = pickle.load(fp)
            analyzer.de_results = de_results
            analyzer.gbm_fpkm = gbm_fpkm
            analyzer.normal_fpkm = normal_fpkm

        
        # 8. 获取显著差异表达基因
        print("\n筛选显著差异表达基因...")
        up_genes, down_genes, analyzer.de_results = analyzer.get_significant_genes(
            analyzer.de_results, log2fc_threshold=1, adj_p_threshold=0.05
        )

        # 生成报告
        analyzer.generate_summary_report()

    else:
        # 运行完整分析流程
        print("开始分析流程...")
        analyzer.run_analysis_pipeline(
            meta_files, fpkm_files, normal_fpkm_file
        )

        # 8. 获取显著差异表达基因
        print("\n筛选显著差异表达基因...")
        up_genes, down_genes, analyzer.de_results = analyzer.get_significant_genes(
            analyzer.de_results, log2fc_threshold=1, adj_p_threshold=0.05
        )
        
        # 生成报告
        analyzer.generate_summary_report()

        # save the files
        print("Writing to caching file for speed up")
        with open(cache_file, 'wb') as fp:
            pickle.dump([analyzer.gbm_fpkm, analyzer.normal_fpkm, analyzer.de_results], fp)

    # 创建可视化
    analyzer.create_volcano_plot(analyzer.de_results)
    analyzer.create_heatmap(analyzer.gbm_fpkm, analyzer.normal_fpkm, top_n=50)
    analyzer.plot_gbm_gene_barplot(log2fc_threshold=1.0, adj_p_threshold=0.05)
    
    # 保存结果
    #analyzer.save_results('./gbm_fpkm_de_results')
    
    # 检查已知GBM标志物
    print("\n" + "=" * 70)
    print("已知GBM相关基因的差异表达情况:")
    print("=" * 70)
    
    known_gbm_markers = {
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
    
    results_df = analyzer.de_results.set_index('gene')
    for gene, description in known_gbm_markers.items():
        if gene in results_df.index:
            row = results_df.loc[gene]
            fc_status = "↑上调" if row['log2fc'] > 0 else "↓下调"
            significance = "显著" if row['adj_p_value'] < 0.05 else "不显著"
            print(f"{gene:10s} ({description:30s}): {fc_status} "
                  f"(log2FC={row['log2fc']:6.2f}, adj.p={row['adj_p_value']:.2e}) - {significance}")
        else:
            print(f"{gene:10s} ({description:30s}): 未在分析基因列表中")
            

if __name__ == "__main__":
    main()
