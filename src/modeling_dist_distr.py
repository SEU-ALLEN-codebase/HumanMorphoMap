##########################################################
#Author:          Yufeng Liu
#Create time:     2025-04-23
#Description:               
##########################################################
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors


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
    cmap = colors.LinearSegmentedColormap.from_list('subset', colors_subset)


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

if __name__ == '__main__':
    plot_partial_sphere_with_gradient(R=1.5, figname="uniform_gradient_sphere.png")
