##########################################################
#Author:          Yufeng Liu
#Create time:     2025-12-01
#Description:     Adapted from Kaifeng's implementation at: https://github.com/kechanf/hb_seg/blob/v1.4/simple_swc_tool/soma_detection_from_seg.py
##########################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import measure, morphology
import tifffile
from skimage.transform import resize
import os
import glob
import pandas as pd
import cc3d
from scipy.optimize import curve_fit

from scipy import linalg
from scipy.optimize import least_squares
from scipy.optimize import minimize
from scipy.spatial import ConvexHull, HalfspaceIntersection

from file_io import save_image


def fit_bounding_ellipsoid(points):
    """快速计算包含所有点的最小外接椭球"""
    if len(points) < 10:
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        center = (min_coords + max_coords) / 2
        radii = (max_coords - min_coords) / 2
        rotation = np.eye(3)
        return center, radii, rotation
    
    centered = points - np.mean(points, axis=0)
    cov_matrix = centered.T @ centered / (len(points) - 1)
    
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    points_pca = centered @ eigenvectors
    min_pca = np.min(points_pca, axis=0)
    max_pca = np.max(points_pca, axis=0)
    
    center_pca = (min_pca + max_pca) / 2
    radii = (max_pca - min_pca) / 2
    
    center = np.mean(points, axis=0) + center_pca @ eigenvectors.T
    rotation = eigenvectors
    
    return center, radii, rotation

def get_ellipsoid_voxels(center, radii, rotation, bbox_shape, margin=0.05):
    """
    获取椭球内部的所有体素坐标
    
    参数:
        center: 椭球中心 (3,)
        radii: 椭球半径 (3,)
        rotation: 旋转矩阵 (3, 3)
        bbox_shape: 边界框的形状 (Z, Y, X)
        margin: 边界裕量，稍微放大椭球
    
    返回:
        ellipsoid_coords: 椭球内部体素的坐标数组 (N, 3)
        mask: 布尔掩码，形状为bbox_shape，True表示在椭球内
    """
    # 创建坐标网格
    z_coords, y_coords, x_coords = np.indices(bbox_shape)
    coords = np.stack([z_coords.ravel(), y_coords.ravel(), x_coords.ravel()], axis=-1)
    
    # 将坐标转换到椭球坐标系
    coords_centered = coords - center
    coords_rotated = coords_centered @ rotation
    
    # 归一化坐标（添加裕量）
    radii_with_margin = radii * (1.0 + margin)
    normalized = coords_rotated / radii_with_margin
    
    # 计算距离并创建掩码
    distances = np.sum(normalized**2, axis=1)
    inside_mask_flat = distances <= 1.0
    
    # 重塑为原始形状
    mask = inside_mask_flat.reshape(bbox_shape)
    
    # 获取内部坐标
    ellipsoid_coords = coords[inside_mask_flat]
    
    return ellipsoid_coords, mask

def calculate_foreground_ratio_in_ellipsoid(seg_mask, center, radii, rotation):
    """
    计算椭球内部前景点的比例
    
    返回:
        ratio: 前景点占椭球内部总点数的比例
        ellipsoid_volume: 椭球内部的体素总数
        foreground_count: 椭球内部的前景点数
    """
    bbox_shape = seg_mask.shape
    
    # 获取椭球内部的所有体素坐标（添加微小裕量确保边界点被包含）
    ellipsoid_coords, ellipsoid_mask = get_ellipsoid_voxels(
        center, radii, rotation, bbox_shape, margin=0.01
    )
    
    # 计算椭球体积（体素数）
    ellipsoid_volume = len(ellipsoid_coords)
    
    if ellipsoid_volume == 0:
        return 0.0, 0, 0
    
    # 统计椭球内部的前景点
    # 方法1：使用掩码直接计算（更快）
    foreground_in_ellipsoid = np.logical_and(seg_mask > 0, ellipsoid_mask)
    foreground_count = np.sum(foreground_in_ellipsoid)
    
    # 方法2：使用坐标计算（备用）
    # foreground_count = 0
    # for coord in ellipsoid_coords:
    #     z, y, x = coord.astype(int)
    #     if seg_mask[z, y, x] > 0:
    #         foreground_count += 1
    
    fg_el_ratio = 1.0 * foreground_count / ellipsoid_volume
    fg_crop_ratio = 1.0 * foreground_count / np.prod(bbox_shape)
    
    return fg_el_ratio, ellipsoid_volume, foreground_count, fg_crop_ratio


def get_the_largest_mask(seg):
    # 使用 cc3d 进行连通区域标记
    # connectivity=26 表示使用 26 邻域（包括对角）
    labels_out, N = cc3d.connected_components(seg, connectivity=26, return_N=True)
    
    if N > 1:
        print(f'Found {N} connected components. Keeping only the largest one.')
        
        # 统计每个连通区域的大小
        component_sizes = []
        for label_id in range(1, N + 1):
            component_mask = (labels_out == label_id)
            component_size = np.sum(component_mask)
            component_sizes.append((label_id, component_size))
        
        # 按大小排序，找到最大的区域
        component_sizes.sort(key=lambda x: x[1], reverse=True)
        largest_label_id, largest_size = component_sizes[0]
        
        print(f'Largest component: label {largest_label_id}, size = {largest_size} voxels')
        
        # 仅保留最大的连通区域
        seg = (labels_out == largest_label_id).astype(np.uint8)
        return seg
    else:
        return seg


def fit_soma(seg_file, out_dir, df_meta):
    # get filename and cell id
    filename = os.path.split(seg_file)[-1]
    cell_id = int(filename.split('_')[0])

    # get the resolution
    try:
        z_rez, xy_rez = df_meta.loc[cell_id][['z_resolution', 'xy_resolution']] / 1000.0 # to um
    except KeyError:
        print('----> Could not find resolution...')
        return 

    resolution = (z_rez, xy_rez, xy_rez)

    print(f'Loading image: {filename}')
    seg = tifffile.imread(seg_file).astype(np.uint8)

    print(f'do a pre-crop for acceleration, using a relative small kernel')
    pre_kernel = morphology.ball(3)
    pre_seg = morphology.opening(seg, pre_kernel)
    # check if multiple connected components, if yes, keep only the components whose size is larger
    # 检查是否存在多个连通区域，如果有，仅保留最大的一个
    if np.sum(pre_seg) > 0:
        pre_seg = get_the_largest_mask(pre_seg)
    else:
        print('----> No salient soma found!')
        return 
    

    # 1. 找到opening后mask在三个维度的非零坐标范围
    nonzero_coords = np.where(pre_seg > 0)
    
    if len(nonzero_coords[0]) == 0:
        # 如果opening后没有任何mask，返回整个图像
        print(f"Warning: No mask found after opening for {filename}")
        cropped_seg = seg
        z_min_exp, z_max_exp, y_min_exp, y_max_exp, x_min_exp, x_max_exp = (0, seg.shape[0], 0, seg.shape[1], 0, seg.shape[2])
    else:
        # 获取三个维度的min和max
        z_min, z_max = np.min(nonzero_coords[0]), np.max(nonzero_coords[0])
        y_min, y_max = np.min(nonzero_coords[1]), np.max(nonzero_coords[1]) 
        x_min, x_max = np.min(nonzero_coords[2]), np.max(nonzero_coords[2])
        
        # 2. 每侧扩充1个像素，并确保不超出图像边界
        k = 5
        z_min_exp = max(0, z_min - k)
        z_max_exp = min(seg.shape[0], z_max + k+1)  # +2 因为切片是左闭右开
        y_min_exp = max(0, y_min - k)
        y_max_exp = min(seg.shape[1], y_max + k+1)
        x_min_exp = max(0, x_min - k)
        x_max_exp = min(seg.shape[2], x_max + k+1)
        
        # 3. 根据扩展后的边界裁剪原图
        cropped_seg = seg[z_min_exp:z_max_exp, 
                          y_min_exp:y_max_exp, 
                          x_min_exp:x_max_exp]
        
        # 返回裁剪后的图像和边界坐标（用于后续参考）
        bbox_coords = (z_min_exp, z_max_exp, y_min_exp, y_max_exp, x_min_exp, x_max_exp)
        
        print(f"Original shape: {seg.shape}, Cropped shape: {cropped_seg.shape}")
        print(f"Crop bbox: Z[{z_min_exp}:{z_max_exp}], Y[{y_min_exp}:{y_max_exp}], X[{x_min_exp}:{x_max_exp}]")

    
    #### Operate on the cropped image
    print(f'Resizing to homogenous image in cropped space')
    crop_z, crop_y, crop_x = cropped_seg.shape
    seg_crop = seg[z_min_exp:z_max_exp, y_min_exp:y_max_exp, x_min_exp:x_max_exp]
    scale_z = resolution[0] / resolution[1]
    seg_c_r = resize(seg_crop, (int(round(crop_z*scale_z)), crop_y, crop_x), order=0)
    seg_c_r = (seg_c_r - seg_c_r.min()) / (seg_c_r.max() - seg_c_r.min())
    seg_c_r = np.where(seg_c_r > 0, 1, 0).astype(np.uint8)
    # 填补空洞
    seg_c_r = ndimage.binary_fill_holes(seg_c_r).astype(int)


    # 定义核半径列表
    min_radius, max_radius= 2, 15+1
    kernel_radii = np.arange(min_radius, max_radius)

    # 存储高频能量占比和高频幅值平均值
    fg_ratios = []

    # 存储每个阶段的MIP图像
    mip_images = []

    print('Iterative evaluation')
    final_opened_img = None
    final_radius = None
    for radius in kernel_radii:
        # 创建球形结构元素
        struct = morphology.ball(radius)

        opened_img = morphology.opening(seg_c_r, struct)
        if(np.sum(opened_img) == 0):
            break

        foreground_coords = np.where(opened_img > 0)
        if len(foreground_coords[0]) < 10:  # 需要足够的点来拟合椭球
            print(f"Radius {radius}: Too few points ({len(foreground_coords[0])}) for ellipsoid fitting.")
            continue

        # double check the foreground information
        opened_img = get_the_largest_mask(opened_img)
        
        # 将坐标转换为数组形式 (N, 3)
        points = np.column_stack(foreground_coords)
        #import ipdb; ipdb.set_trace()

        try:
            # 拟合椭球
            center, radii, rotation = fit_bounding_ellipsoid(points)
            print(center, radii, rotation)
            fg_el_ratio, ellipsoid_volume, foreground_count, fg_crop_ratio = calculate_foreground_ratio_in_ellipsoid(opened_img, center, radii, rotation)
            
            print(f"Radius {radius}: foreground ratio of ellipsoid = {fg_el_ratio:.3f} "
                  f"\n    foreground ratio in cropped_image = {fg_crop_ratio:.3f} "
                  f"(points: {len(points)}, radii: {radii})\n")
            
            # 保存当前结果（无论是否终止）
            final_opened_img = opened_img
            final_radius = radius

            fg_ratios.append(fg_el_ratio)
            
            # 判断终止条件
            ratio_thresh = 0.65
            if fg_el_ratio > ratio_thresh:
                print(f"Radius {radius}: Termination condition met (ratio = {fg_el_ratio:.3f} > ratio_thresh)")
                break
                
        except Exception as e:
            print(f"Radius {radius}: Ellipsoid fitting failed - {str(e)}")
            # 保存当前结果并继续
            final_opened_img = opened_img
            final_radius = radius
            break
        
        
    print(f'meta-analyses')
    fig, ax1 = plt.subplots()
    # x label
    ax1.set_xlabel('Kernel radius')
    # 设置大小
    fig.set_size_inches(4, 2.5)
    kernel_radii = kernel_radii[:len(fg_ratios)]
    print(f'kernel_radii and fg_ratio: {kernel_radii}, {fg_ratios}')

    #color = 'tab:blue'
    #ax1.set_ylabel('Forground ratio', color=color)
    #ax1.plot(kernel_radii, fg_ratios, marker='s', color=color)
    #ax1.tick_params(axis='y', labelcolor=color)
    # plot line y

    #fig.tight_layout()
    #plt.savefig(os.path.join(out_dir, filename.replace('.tif', '_ellipse_fg_ratio.png')))
    #plt.close()

    print(f'Get the final results, using best_radius: {final_radius}')
    # high_freq_averages_fit的最低点
    result_open_img = morphology.opening(seg_c_r, morphology.ball(final_radius))
    result_open_img = result_open_img * seg_c_r
    result_img = resize(result_open_img, (crop_z, crop_y, crop_x), order=0)
    result_img = np.where(result_img > 0, 1, 0).astype(np.uint8)

    print(f'Saving MIPs')
    # plt
    mip_list = [
        cropped_seg.max(axis=0),
        cropped_seg.max(axis=1),
        cropped_seg.max(axis=2),
        result_open_img.max(axis=0),
        result_open_img.max(axis=1),
        result_open_img.max(axis=2),
        result_img.max(axis=0),
        result_img.max(axis=1),
        result_img.max(axis=2)
    ]

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        ax.imshow(mip_list[i], cmap='gray')
        # if(i < 3):
        #     ax.imshow(mip_list[i+6], cmap='jet', alpha=0.5)  # 调整alpha以控制透明度
        # if(i >= 3 and i < 6):
        #     ax.imshow(mip_list[i+3], cmap='jet', alpha=0.5)  # 调整alpha以控制透明度

        ax.set_title(f'MIP {i+1}')
        ax.axis('off')
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(out_dir, 'mip', filename.replace('.tif', '_MIP.png')))
    plt.close()

    # save the final image
    save_image(os.path.join(out_dir, 'mask', filename), result_img*255)
    # also save the resized image
    result_open_img = np.where(result_open_img > 0, 1, 0).astype(np.uint8)
    save_image(os.path.join(out_dir, 'mask_isotropy', filename), result_open_img*255)

    return center, radii, rotation, fg_el_ratio, z_min_exp, z_max_exp, y_min_exp, y_max_exp, x_min_exp, x_max_exp, z_rez, xy_rez

def fit_all(seg_dir, out_dir, df_meta, out_file):
    soma_infos = []
    for ifile, seg_file in enumerate(glob.glob(os.path.join(seg_dir, '*.tif'))):
        result = fit_soma(seg_file, out_dir, df_meta)
        if result is not None:
            filename = os.path.split(seg_file)[-1]
            cell_id = int(filename.split('_')[0])
            soma_infos.append([cell_id, filename, *result])
        
        if ifile % 30 == 0:
            print(f'\n====> Processing: {ifile}')
            #if ifile > 0: break
    
    # to dataframe
    column_names = [
        'cell_id', 
        'filename',
        'center',      # 元组或数组: (z, y, x)
        'radii',       # 元组或数组: (r_z, r_y, r_x)  
        'rotation',    # 3x3数组或展平的9个值
        'fg_el_ratio', # 前景体素在椭球中的比例
        'z_min_exp',   # 扩展后的Z轴最小值
        'z_max_exp',   # 扩展后的Z轴最大值
        'y_min_exp',   # 扩展后的Y轴最小值
        'y_max_exp',   # 扩展后的Y轴最大值
        'x_min_exp',   # 扩展后的X轴最小值
        'x_max_exp',   # 扩展后的X轴最大值
        'z_rez', 
        'xy_rez'
    ]

    # 创建DataFrame
    df_soma_info = pd.DataFrame(soma_infos, columns=column_names)

    # 将center展开为三列
    df_soma_info[['center_z', 'center_y', 'center_x']] = pd.DataFrame(
        df_soma_info['center'].tolist(), index=df_soma_info.index
    )

    # 将radii展开为三列
    df_soma_info[['radius_z', 'radius_y', 'radius_x']] = pd.DataFrame(
        df_soma_info['radii'].tolist(), index=df_soma_info.index
    )

    # 将3x3旋转矩阵展平为9个值
    df_soma_info['rotation_flat'] = df_soma_info['rotation'].apply(lambda x: x.flatten())
    # 然后可以展开为9列
    rotation_cols = [f'rot_{i}' for i in range(9)]
    df_soma_info[rotation_cols] = pd.DataFrame(
        df_soma_info['rotation_flat'].tolist(), index=df_soma_info.index
    )

    # 可选：删除原始列
    df_soma_info.drop(['center', 'radii', 'rotation'], axis=1, inplace=True)


    # 设置cell_id为索引
    df_soma_info.set_index('cell_id', inplace=True)
    df_soma_info.to_csv(out_file, index=True)

    return df_soma_info
            

if __name__ == "__main__":

    meta_file = '/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta_filter/meta.csv'
    #seg_dir = '/data/kfchen/trace_ws/paper_trace_result/nnunet/proposed_9k/0_seg'  # original 8k
    seg_dir = '/data2/kfchen/tracing_ws/14k_raw_img_data/long_590_test_data_for_nnunet/0_seg'
    out_dir = '/data2/lyf/data/human10k_tmp/data/0_seg_soma-lyf'
    out_file = 'fitted_soma_info_0.4k.csv'    # 0.4k 
    
    #seg_file = f'{seg_dir}/02764_P025_T01_-S032_LTL_R0613_RJ-20230201_RJ.tif'

    df_meta = pd.read_csv(meta_file, index_col='cell_id', low_memory=False, encoding='gbk')
    fit_all(seg_dir, out_dir, df_meta, out_file)
    
        
    


