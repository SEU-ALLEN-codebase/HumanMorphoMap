##########################################################
#Author:          Yufeng Liu
#Create time:     2025-05-10
#Description:               
##########################################################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from global_features import calc_global_features_from_folder


def compare_features(manual_file='gf_H01_resample1um_prune25um.csv',
                     auto_file='auto8.4k_0510_pruned_resample1um.csv'):
    # 读取两个CSV文件
    df1 = pd.read_csv(manual_file, index_col=0)
    df2 = pd.read_csv(auto_file, index_col=0)

    # 提取特征列（排除第一列的名称列）
    #features = df1.columns
    features = ['Stems', 'AverageContraction', 'AverageFragmentation', 'AverageParent-daughterRatio', 
                'AverageBifurcationAngleRemote', 'HausdorffDimension']

    # 设置图形大小
    sns.set_theme(style='ticks', font_scale=1.3)
    nrows = int(np.ceil(len(features) / 4))
    plt.figure(figsize=(12, 12 / 4 * nrows))

    # 为每个特征创建一个子图
    for i, feature in enumerate(features, 1):
        plt.subplot(nrows, 4, i)  # 6行4列的子图布局
        
        # 获取两个数据集的值
        values1 = df1[feature].values
        values2 = df2[feature].values
        
        # 确定合适的bins
        combined = np.concatenate([values1, values2])
        bins = np.linspace(min(combined), max(combined), 40)
        
        # 绘制直方图
        plt.hist(values1, bins=bins, alpha=0.5, label='gf_H01', color='blue', density=True)
        plt.hist(values2, bins=bins, alpha=0.5, label='auto8.4k', color='orange', density=True)
        
        plt.title(feature)
        #plt.legend()

    # 调整布局防止重叠
    plt.tight_layout()
    plt.suptitle('Feature Comparison Between gf_H01 and auto8.4k', y=1.02)
    plt.savefig('temp.png', dpi=300)
    plt.close()

if __name__ == '__main__':
    if 1:
        # calculate global features
        #swc_dir = '/home/lyf/Research/publication/humain10k/HumanMorphoMap/h01-guided-reconstruction/data/H01_resample1um_prune25um'
        #outfile = 'gf_H01_resample1um_prune25um.csv'
    
        swc_dir = '/home/lyf/Research/publication/humain10k/HumanMorphoMap/h01-guided-reconstruction/data/auto8.4k_0510_resample1um_mergedBranches0712_branchPruned1029'
        outfile = 'auto8.4k_0510_resample1um_mergedBranches0712_branchPruned.csv'

        #swc_dir = '/home/lyf/Research/publication/humain10k/HumanMorphoMap/h01-guided-reconstruction/data/manual_resampled1um'
        #outfile = 'manual_resampled1um.csv'

        ##### for comparison on test-set
        #method = 'skelrec'
        #swc_dir = f'/data2/kfchen/tracing_ws/14k_raw_img_data/0722_origin_swc/{method}'
        #outfile = f'test232/lmeasure/gf_original_{method}.csv'

        nprocessors = 16
        calc_global_features_from_folder(swc_dir, outfile, nprocessors=nprocessors, timeout=360)

    if 0:
        compare_features()

