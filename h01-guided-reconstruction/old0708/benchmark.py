##########################################################
#Author:          Yufeng Liu
#Create time:     2025-05-17
#Description:               
##########################################################
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if 0:
    # Extract the 232 neurons in test-set
    orig_test242_dir = '/data/kfchen/trace_ws/paper_trace_result/nnunet/newcel_0.1/8_estimated_radius_swc'
    recon_gf_file = 'auto8.4k_0510_pruned_resample1um.csv'

    swcnames = [int(os.path.split(swcfile)[-1][:-4]) for swcfile in glob.glob(os.path.join(orig_test242_dir, '*swc'))]
    gf_rec = pd.read_csv(recon_gf_file, index_col=0)

    test232 = gf_rec.index[gf_rec.index.isin(swcnames)]
    test232.to_frame().to_csv('test232.csv', index=0)


if 1:

    # 设置中文显示（如果需要）
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    sns.set_theme(style='ticks', font_scale=1.7)

    # 1. 读取数据
    auto_df = pd.read_csv('../auto8.4k_0510_resample1um_mergedBranches0712.csv')
    manual_df = pd.read_csv('../manual_resampled1um_renamed.csv')
    test_ids = pd.read_csv('../test232.csv')

    # 2. 提取测试集数据
    auto_test = auto_df[auto_df['ID'].isin(test_ids['ID'])]
    manual_test = manual_df[manual_df['ID'].isin(test_ids['ID'])]

    # 确保ID顺序一致
    auto_test = auto_test.sort_values('ID').reset_index(drop=True)
    manual_test = manual_test.sort_values('ID').reset_index(drop=True)

    # 3. 计算相对值（manual/auto）
    # 排除ID列和非数值列（如果有）
    features = auto_test.columns.difference(['ID'])
    #sel_features = ['N_stem', 'Number of Bifurcations', 'Number of Branches', 'Number of Tips', 
    #                'Width', 'Height', 'Depth', 'Length', 'Max Euclidean Distance', 
    #                'Max Path Distance', 'Max Branch Order']
    sel_features = ['Stems', 'Bifurcations', 'Branches', 'Tips', 
                    'OverallWidth', 'OverallHeight', 'OverallDepth', 'Length', 'MaxEuclideanDistance', 
                    'MaxPathDistance', 'MaxBranchOrder']
    features = [feat for feat in features if feat in sel_features]
    ratio_df = pd.DataFrame()

    for feature in features:
        # 处理可能的零除情况
        mask = (auto_test[feature] != 0)
        ratio_df[feature] = auto_test[feature][mask] / manual_test[feature][mask]

    # 4. 绘制violinplot
    plt.figure(figsize=(16, 8))
    sns.violinplot(data=ratio_df, inner="quartile", cut=0)
    plt.axhline(y=1, color='r', linestyle='--', linewidth=1)
    plt.title('测试集神经元特征相对值分布 (Auto/Manual)', fontsize=16)
    plt.ylabel('相对值 (Auto/Manual)', fontsize=12)
    plt.xlabel('特征', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0.25, 1.75)

    plt.tight_layout()
    plt.savefig('temp.png', dpi=300)
    plt.close()
