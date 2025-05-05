##########################################################
#Author:          Yufeng Liu
#Create time:     2025-05-03
#Description:               
##########################################################
import numpy as np
import pandas as pd
from sklearn.linear_model import RANSACRegressor

def estimate_principal_axes(df, cell_name='eL2/3.IT'):
    """
    用RANSAC拟合鲁棒主轴，返回pc1方向、pc2方向（垂直）和中心点
    输入:
        df: DataFrame，包含 'adjusted.x', 'adjusted.y', 'cluster_L2' 列
        cell_name: 目标细胞类型（如 'eL2/3.IT'）
    返回:
        pc1: 第一主轴方向（单位向量）
        pc2: 第二主轴方向（单位向量，垂直于pc1）
        center: 中心点坐标（毫米单位）
    """
    # 提取目标细胞并转换单位（毫米）
    target_cells = df[df['cluster_L2'] == cell_name]
    points = target_cells[['adjusted.x', 'adjusted.y']].values / 1000
    
    # RANSAC拟合第一主轴（pc1）
    ransac = RANSACRegressor()
    x, y = points[:, 0].reshape(-1, 1), points[:, 1]
    ransac.fit(x, y)
    slope = ransac.estimator_.coef_[0]
    pc1 = np.array([1, slope])  # 方向向量
    pc1 = pc1 / np.linalg.norm(pc1)  # 单位化
    
    # 计算垂直方向（pc2）
    pc2 = np.array([-slope, 1])
    pc2 = pc2 / np.linalg.norm(pc2)
    
    # 中心点
    center = np.mean(points, axis=0)
    
    return pc1, pc2, center

# 示例调用
cell_name = 'eL4/5.IT'
df_f = pd.read_csv('../resources/human_merfish/H18/H18.06.006.MTG.250.expand.rep1.features.csv')
pc1, pc2, center = estimate_principal_axes(df_f, cell_name=cell_name)

# 提取所有L2/3细胞的坐标（毫米单位）
xy_cur = df_f[df_f['cluster_L2'] == cell_name][['adjusted.x', 'adjusted.y']].values / 1000

def split_by_pc2_quantiles(xy_cur, pc2, center, quantiles=[0.25, 0.5, 0.75]):
    """
    按pc2方向的分位数将点分为4组
    输入:
        xy_cur: 细胞坐标数组 (Nx2)
        pc2: 第二主轴方向（单位向量）
        center: 中心点坐标
        quantiles: 分位数列表（默认[0.25, 0.5, 0.75]）
    返回:
        groups: 字典，key为分位区间名，value为对应坐标数组
    """
    # 计算每个点在pc2方向上的投影值（相对于中心点）
    proj = (xy_cur - center) @ pc2
    
    # 计算分位点
    q = np.quantile(proj, quantiles)
    
    # 分组
    groups = {
        '0-25%': xy_cur[proj <= q[0]],
        '25-50%': xy_cur[(proj > q[0]) & (proj <= q[1])],
        '50-75%': xy_cur[(proj > q[1]) & (proj <= q[2])],
        '75-100%': xy_cur[proj > q[2]]
    }
    return groups

# 调用分组函数
groups = split_by_pc2_quantiles(xy_cur, pc2, center)

# 检查每组点数
for name, pts in groups.items():
    print(f"{name}: {len(pts)} points")

# 可视化（可选）
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="ticks", font_scale=1.5)

plt.figure(figsize=(8, 6))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
for i, (name, pts) in enumerate(groups.items()):
    plt.scatter(pts[:, 0], pts[:, 1], s=5, alpha=0.6, label=name, color=colors[i])

plt.legend(frameon=False, markerscale=3)
plt.xlabel('adjusted.x (mm)')
plt.ylabel('adjusted.y (mm)')
sns.despine()
plt.savefig('temp.png', dpi=300)
plt.close()
