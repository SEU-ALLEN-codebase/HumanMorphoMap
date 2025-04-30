##########################################################
#Author:          Yufeng Liu
#Create time:     2025-04-23
#Description:               
##########################################################
import numpy as np
import seaborn as sns
from scipy.stats import norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib


def plot_partial_sphere_with_gradient(R=1, resolution=200, figname="temp.png"):
    """
    最终优化版：统一所有截面颜色梯度显示
    
    参数:
    R - 球体半径 (默认1)
    resolution - 网格分辨率 (默认100)
    figname - 输出文件名 (默认"temp.png")
    """
    sns.set_theme(style='ticks', font_scale=1.8)

    # 创建球体网格
    phi, theta = np.mgrid[0:np.pi:resolution*1j, 0:2*np.pi:resolution*1j]
    x = R * np.sin(phi) * np.cos(theta)
    y = R * np.sin(phi) * np.sin(theta)
    z = R * np.cos(phi)
    distance = np.sqrt(x**2 + y**2 + z**2) / R

    # 创建掩膜移除第一卦限
    mask = ~((x > 0) & (y > 0) & (z > 0))
    x_masked = np.where(mask, x, np.nan)
    y_masked = np.where(mask, y, np.nan)
    z_masked = np.where(mask, z, np.nan)
    distance_masked = np.where(mask, distance, np.nan)

    # 创建图形
    fig = plt.figure(figsize=(12, 10), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    cmap = plt.cm.Reds#viridis
    colors_subset = cmap(np.linspace(0.1, 1.0, 256))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('subset', colors_subset)


    # ===== 统一截面生成方法 =====
    def create_uniform_section(axis_pair):
        """生成统一标准的截面平面"""
        u = np.linspace(0, np.pi/2, resolution//2)
        r = np.linspace(0, R, resolution//2)
        u_grid, r_grid = np.meshgrid(u, r)
        
        # 生成三维坐标
        coord = np.zeros((*r_grid.shape, 3))
        if axis_pair == 'xy':
            coord[...,0] = r_grid * np.sin(u_grid)  # X
            coord[...,1] = r_grid * np.cos(u_grid)  # Y
        elif axis_pair == 'yz':
            coord[...,1] = r_grid * np.sin(u_grid)  # Y
            coord[...,2] = r_grid * np.cos(u_grid)  # Z
        elif axis_pair == 'xz':
            coord[...,0] = r_grid * np.sin(u_grid)  # X
            coord[...,2] = r_grid * np.cos(u_grid)  # Z
        
        # 计算有效区域和颜色
        norm_distance = np.linalg.norm(coord, axis=-1) / R
        valid = norm_distance <= 1.0
        colors = cmap(1 - norm_distance)
        colors[~valid] = (0,0,0,0)  # 完全透明
        
        # 创建曲面（禁用光照）
        sec_surf = ax.plot_surface(
            coord[...,0], coord[...,1], coord[...,2],
            facecolors=colors,
            rstride=1, cstride=1,
            alpha=0.8,
            antialiased=True,
            zorder=3,
            shade=False,  # 关键修改：禁用光照
            edgecolor='none',
        )
        return sec_surf


    # 创建三个统一截面
    create_uniform_section('xy')
    create_uniform_section('yz')
    create_uniform_section('xz')

    # ===== 绘制主体球体 =====
    
    surf = ax.plot_surface(x_masked, y_masked, z_masked, 
                          facecolors=plt.cm.Greys_r(1-distance_masked),
                          rstride=1, cstride=1, 
                          alpha=0.05, linewidth=0,
                          zorder=1,
                          shade=False,  # 保持一致的渲染方式
                          edgecolor='none',
                          )

    # ===== 3. 添加坐标轴标识 =====
    axis_length = R * 1.2
    ax.quiver(0, 0, 0, axis_length, 0, 0, color='r', arrow_length_ratio=0.1, linewidth=2, zorder=10, alpha=1.)
    ax.quiver(0, 0, 0, 0, axis_length, 0, color='g', arrow_length_ratio=0.1, linewidth=2, zorder=10, alpha=1.)
    ax.quiver(0, 0, 0, 0, 0, axis_length, color='b', arrow_length_ratio=0.1, linewidth=2, zorder=10, alpha=1.)
    ax.text(axis_length*1.1, 0, 0, 'X', color='r', fontsize=16, zorder=4)
    ax.text(0, axis_length*1.02, 0, 'Y', color='g', fontsize=16, zorder=4)
    ax.text(0, 0, axis_length*1.02, 'Z', color='b', fontsize=16, zorder=4)

    # ===== 4. 绘制赤道线 =====
    theta_eq = np.linspace(0, 2*np.pi, 100)
    x_eq = R * np.cos(theta_eq)
    y_eq = R * np.sin(theta_eq)
    ax.plot(x_eq, y_eq, np.zeros_like(x_eq), 
           color='black', linestyle='--', linewidth=2, zorder=4, label='Equator')
    

    # ===== 视觉优化设置 =====
    ax.view_init(elev=25, azim=45)
    #ax.set_title(f'Uniform Gradient Sphere (R={R})', pad=15)
    ax.set_xlim(-R*1.1, R*1.1)
    ax.set_ylim(-R*1.1, R*1.1)
    ax.set_zlim(-R*1.1, R*1.1)
    ax.set_box_aspect([1,1,1])
    ax.grid(False)
    ax.xaxis.pane.fill = False  # 禁用背景面
    ax.yaxis.pane.fill = False  # 禁用背景面
    ax.zaxis.pane.fill = False  # 禁用背景面
    #[ax.w_xaxis.pane.set_visible(False) for axis in ['x','y','z']]
    #ax.text(0, 0, R*1.1, "Removed 1/8", color='red', ha='center', zorder=4)
    plt.axis('off')

    # 颜色条设置（显式设置范围）
    norm = plt.Normalize(vmin=0, vmax=1)
    m = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    m.set_array([])
    cbar = plt.colorbar(m, ax=ax, label='Similarity', pad=0.15, shrink=0.15, aspect=5)

    # 保存输出
    plt.tight_layout()
    plt.savefig(figname, bbox_inches='tight', dpi=300)
    plt.close()


def plot_cell_distribution(R=1.0, total_cells=500, figname="cell_distribution.png"):
    """
    绘制两种细胞随距离的分布变化
    
    参数:
    R - 作用范围半径 (默认1.0)
    total_cells - 总细胞数 (默认500)
    figname - 输出文件名 (默认"cell_distribution.png")
    """
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    
    # 设置圆形区域
    circle = plt.Circle((0, 0), R, color='lightgray', alpha=0.2, label=f'Linear region')
    ax.add_patch(circle)
    
    # 生成随机位置（极坐标）
    np.random.seed(1024)
    r = np.random.uniform(0, R, total_cells)
    theta = np.random.uniform(0, 2*np.pi, total_cells)
    
    # 转换为笛卡尔坐标
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    # 计算两种细胞的概率分布
    # 红色细胞概率：随距离增加而减少（正态分布）
    red_prob = norm.pdf(r, loc=0, scale=R/2)
    # 蓝色细胞概率：随距离增加而增加（互补）
    blue_prob = norm.pdf(r, loc=R, scale=R/2)
    
    # 标准化概率
    prob_sum = red_prob + blue_prob
    red_prob /= prob_sum
    blue_prob /= prob_sum
    
    # 根据概率分配细胞类型
    cell_types = np.random.rand(total_cells)
    red_cells = cell_types < red_prob
    blue_cells = ~red_cells
    
    # 绘制红色圆形细胞（大小随距离减小）
    ax.scatter(x[red_cells], y[red_cells], 
               c='red', marker='o', 
               s=30, #*(1-r[red_cells]/R),  # 大小随距离减小
               alpha=0.7, label='Cell type 1')
    
    # 绘制蓝色三角形细胞（大小随距离增加）
    ax.scatter(x[blue_cells], y[blue_cells], 
               c='blue', marker='^', 
               s=30, #*(1+r[blue_cells]/R),  # 大小随距离增加
               alpha=0.7, label='Cell type 2')
    
    # 添加中心点标记
    ax.scatter(0, 0, c='black', marker='*', s=120, label='Center')
    
    # 设置图形属性
    ax.set_xlim(-R*1.2, R*1.2)
    ax.set_ylim(-R*1.2, R*1.2)
    ax.set_aspect('equal')
    #ax.set_title('Cell Distribution with Distance Dependency', pad=20)
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    
    # 添加距离刻度环
    #for rad in np.linspace(0, R, 5)[1:-1]:
    #    circle = plt.Circle((0, 0), rad, color='gray', fill=False, linestyle='--', alpha=0.5)
    #    ax.add_patch(circle)
    #    ax.text(rad, 0, f'{rad:.1f}R', ha='left', va='center', backgroundcolor='white')
    
    # 添加图例
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1), markerscale=1.5)
    plt.axis('off')
    
    # 保存图像
    plt.savefig(figname, bbox_inches='tight', dpi=300)
    plt.close()


def plot_cell_color_gradient(R=1.0, num_cells=500, figname="cell_gradient.png"):
    """
    绘制细胞颜色随距离渐变的分布图
    
    参数:
    R - 区域半径 (默认1.0)
    num_cells - 细胞数量 (默认500)
    figname - 输出文件名 (默认"cell_gradient.png")
    """

    sns.set_theme(style='ticks', font_scale=1.8)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    
    # 自定义颜色渐变：从深蓝到浅蓝再到红
    colors = [(0, 0, 0.6), (0, 0.7, 1), (1, 0.3, 0)]  # 蓝→浅蓝→红
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("custom_gradient", colors, N=256)
    
    # 设置圆形区域背景
    background = plt.Circle((0, 0), R, color='lightgray', alpha=0.2)
    ax.add_patch(background)
    
    # 生成均匀随机分布的细胞位置（极坐标）
    np.random.seed(1024)
    r = R * np.sqrt(np.random.uniform(0, 1, num_cells))  # 确保均匀分布
    theta = np.random.uniform(0, 2*np.pi, num_cells)
    
    # 转换为笛卡尔坐标
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    # 计算归一化距离 (0到1)
    norm_dist = r / R
    
    # 绘制细胞（大小固定，颜色随距离变化）
    sc = ax.scatter(x, y, c=norm_dist, cmap=cmap,
                   s=50, alpha=0.8, edgecolors='white', linewidth=0.5)
    
    # 添加颜色条
    cbar = plt.colorbar(sc, ax=ax, label='', shrink=0.15, aspect=5)
    cbar.set_ticks([0, 1], )
    cbar.ax.tick_params(length=0)
    cbar.set_ticklabels(['Similar', 'Dissimilar'])
    cbar.solids.set(alpha=1)
    cbar.outline.set_linewidth(0)
    
    
    # 设置图形属性
    ax.set_xlim(-R*1.1, R*1.1)
    ax.set_ylim(-R*1.1, R*1.1)
    ax.set_aspect('equal')
    #ax.set_title('Cell Dissimilarity Gradient', pad=20)
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    ax.grid(False)
    
    # 添加距离刻度环
    #for rad in [R/3, 2*R/3, R]:
    #    circle = plt.Circle((0, 0), rad, color='gray', fill=False, 
    #                      linestyle='--', alpha=0.4, linewidth=1)
    #    ax.add_patch(circle)
    #    ax.text(rad+0.05, 0, f'{rad:.1f}R', ha='left', va='center', 
    #           fontsize=8, backgroundcolor='white')
    
    # 添加中心标记
    ax.scatter(0, 0, c='red', marker='^', s=220, label='')
    #ax.legend(loc='upper right')
    plt.axis('off')
    
    # 保存图像
    plt.savefig(figname, bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == '__main__':
    #plot_cell_distribution(R=1.5, total_cells=800)
    plot_cell_color_gradient(R=1.5, num_cells=800, figname="cell_dissimilarity_gradient.png")
    #plot_partial_sphere_with_gradient(R=1.5, figname="uniform_gradient_sphere.png")
    
