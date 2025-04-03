##########################################################
#Author:          Yufeng Liu
#Create time:     2025-03-18
#Description:               
##########################################################
import os
import glob
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
import umap
import fastcluster
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm  

from morph_topo.morphology import Morphology, Topology
from ml.feature_processing import standardize_features
from plotters.customized_plotters import sns_jointplot

from config import REG2LOBE, region_mapper


def load_and_process_features(gf_file, clip=None):
    # load all neurons
    gfs = pd.read_csv(gf_file, index_col=0, low_memory=False)
    # standardize
    gfs_s = standardize_features(gfs, gfs.columns, inplace=False)
    
    if clip is not None:
        # clipping for outliner data
        return gfs_s.clip(-clip, clip)
    else:
        return gfs_s

def morphological_clustering(gf_file):
    gfs_sc = load_and_process_features(gf_file, clip=3)

    # do clustering
    # 2. 使用fastcluster计算行列聚类
    #print("Calculating row clustering...")
    row_linkage = fastcluster.linkage_vector(gfs_sc, method='ward', metric='euclidean')

    #print("Calculating column clustering...")
    col_linkage = fastcluster.linkage_vector(gfs_sc.T, method='ward', metric='euclidean')

    g_clust = sns.clustermap(gfs_sc, 
                             row_linkage=row_linkage,
                             col_linkage=col_linkage,
                             cmap='viridis',
                            )
    plt.savefig('temp.png', dpi=300)
    plt.close()


def find_min_distance_vectors(df):
    """
    找出每个脑区中距离其他向量平均距离最小的特征向量
    
    参数:
        df: 包含特征向量和'region'列的DataFrame
    
    返回:
        包含每个脑区最小距离向量的DataFrame
    """
    # 按脑区分组
    grouped = df.groupby('region')
    
    min_distance_vectors = []
    
    for region, group in tqdm(grouped, desc="Processing regions"):
        # 提取特征向量（排除非数值列，如'region'）
        features = group.select_dtypes(include=[np.number]).values
        
        # 计算成对距离矩阵
        dist_matrix = squareform(pdist(features, 'euclidean'))
        
        # 计算每个向量到其他向量的平均距离（忽略自身距离0）
        avg_distances = np.mean(dist_matrix, axis=1)
        
        # 找到平均距离最小的索引
        min_idx = np.argmin(avg_distances)
        
        # 获取对应的最小距离向量
        min_vector = group.iloc[min_idx].copy()
        min_vector['avg_distance'] = avg_distances[min_idx]  # 添加平均距离信息
        
        min_distance_vectors.append(min_vector)
    
    # 合并所有结果
    result_df = pd.concat(min_distance_vectors, axis=1).T
    result_df.set_index('region', inplace=True)
    return result_df


def diversity_and_stereotypy(gf_file, meta_file_neuron, ctype_file, ihc=0):
    sns.set_theme(style='ticks', font_scale=1.5)
    # load and process the features
    gfs_sc = load_and_process_features(gf_file, clip=3)
    
    # get the meta file of neuron
    meta_n = pd.read_csv(meta_file_neuron, index_col='cell_id', low_memory=False, encoding='gbk')
    reg_mapper = region_mapper()
    meta_n['region'] = [reg_mapper[r] for r in meta_n['brain_region']]

    if ihc != 2:
        ihc_mask = (meta_n.immunohistochemistry == ihc).values
        meta_n = meta_n[ihc_mask]
        gfs_sc = gfs_sc[ihc_mask]

    # 3. cell type extraction
    ctypes = pd.read_csv(ctype_file, index_col=0)
    ctypes_idxs = [int(name.split('_')[0]) for name in ctypes.index]
    ctypes = ctypes.reset_index()
    ctypes.index = ctypes_idxs
    # get the cell types
    ctypes_ = ctypes.loc[gfs_sc.index]
    py_mask = (ctypes_.num_annotator >= 2) & (ctypes_.CLS2 == '0')
    nonpy_mask = (ctypes_.num_annotator >= 2) & (ctypes_.CLS2 == '1')

    #gfs_sc = gfs_sc[py_mask]
    #meta_n = meta_n[py_mask]

    # get the median features
    gfs_r = gfs_sc.copy()
    gfs_r['region'] = meta_n['region']
    gfs_r = gfs_r.groupby('region').filter(lambda x: len(x) >= 10)

    # get the min-distance vector
    min_dist_regional_vectors = find_min_distance_vectors(gfs_r)
    v_regions = min_dist_regional_vectors.drop('avg_distance', axis=1)
    print(min_dist_regional_vectors['avg_distance'])

    # calculate the pairwise correlations
    corr_matrix = v_regions.T.corr()

    '''
    corr_long = corr_matrix.reset_index().melt(id_vars='region', var_name='Region2', value_name='Correlation')
    corr_long = corr_long.rename(columns={'region': 'Region1'})

    g = sns.relplot(
        data=corr_long,
        x='Region1', y='Region2',
        size=np.abs(corr_long['Correlation']),  # 使用绝对值决定大小
        hue='Correlation',
        palette='coolwarm',
        sizes=(10, 200),  # 调整最小和最大点的大小
        height=10,
        aspect=1
    )

    # 4. 美化图形
    g.set_axis_labels("", "")
    g.ax.set_xticklabels(g.ax.get_xticklabels(), rotation=90)
    g.ax.set_yticklabels(g.ax.get_yticklabels(), rotation=0)
    g.fig.suptitle('Inter-region Feature Vector Correlations', y=1.02)
    '''

    # 绘制聚类热图
    sns.clustermap(
        corr_matrix,
        cmap="coolwarm",  # 红-蓝配色表示正负相关
        center=0,         # 0 表示不相关
        annot=True,       # 显示数值
        fmt=".2f",        # 数值格式
        figsize=(12, 12), # 调整大小
        dendrogram_ratio=0.1,  # 调整树状图大小
        linewidths=0.5,   # 单元格边框宽度
    )

    plt.title("Clustered Correlation Matrix of Brain Regions", pad=20)

    plt.tight_layout()
    plt.savefig('temp.png', dpi=300)
    plt.close()
    
    print()
   

if __name__ == '__main__':
    #cell_type_file = '../meta/cell_type_annot_rating_cls2_yufeng_unique.csv'
    cell_type_file = '../meta/cell_type_annotation_8.4K_all_CLS2_unique.csv'
    meta_file_neuron = '/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta_filter/meta.csv'
    #gf_file = '/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta_filter/swc_1um_cropped_150um_l_measure.csv'
    gf_file = '/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta_filter/l_measure_result.csv'
    ctype_file = '../meta/cell_type_annotation_8.4K_all_CLS2_unique.csv'
    ihc = 0

    if 1:
        diversity_and_stereotypy(gf_file, meta_file_neuron, ctype_file)


