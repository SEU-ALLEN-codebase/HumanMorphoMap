##########################################################
#Author:          Yufeng Liu
#Create time:     2025-06-10
#Description:               
##########################################################
import os
import glob
import cv2
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import ndimage

from config import LAYER_CODES, LAYER_CODES_REV


def get_rotation_angles(rotation_file=
        'data/ST-raw/rotate_angles_of_layer_annotation.csv'):
    rotations = pd.read_csv(rotation_file, index_col=0)
    return rotations


def fill_unlabeled_areas(he_img, mask_img):
    # 1. 将3通道mask转换为单通道（假设3个通道值相同）
    if mask_img.ndim == 3:
        mask = mask_img[:,:,0].copy()  # 取第一个通道
    else:
        mask = mask_img.copy()
    
    # 2. 确定前景区域（HE图像中灰度>128的区域）
    gray_he = cv2.cvtColor(he_img, cv2.COLOR_BGR2GRAY) if he_img.ndim == 3 else he_img
    foreground = (gray_he < 230)
    # fill hollows
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    #foreground = cv2.morphologyEx((foreground).astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel)
    #foreground = cv2.dilate((foreground).astype(np.uint8)*255, kernel, iterations=1)
    
    # 3. 找到需要填充的点（在前景中但mask为0的点）
    to_fill = np.logical_and(foreground, mask == 0)
    
    # 4. 创建标记点的二进制掩码（已标记区域）
    labeled_mask = (mask > 0)
    
    # 5. 计算距离变换并获取最近邻索引
    # 注意：我们需要计算从每个未标记点到最近标记点的距离
    # 因此输入应该是标记区域的补集（即未标记区域为True）
    distances, indices = ndimage.distance_transform_edt(
        ~labeled_mask, 
        return_indices=True
    )
    
    # 6. 填充未标记区域
    filled_mask = mask.copy()
    
    # 获取需要填充的点的坐标
    fill_coords = np.where(to_fill)
    
    # 获取这些点的最近邻坐标
    # indices的形状是 (ndim, height, width)
    nearest_coords = (indices[0][to_fill], indices[1][to_fill])
    
    # 获取最近邻的值并填充
    filled_mask[fill_coords] = mask[nearest_coords]
    # zeroing the background area
    filled_mask[~foreground] = 0
    
    return filled_mask

def get_layer_masks(sample_dir, annot_dir, rot_angle):
    '''
    The original image maybe rotated when labeling
    '''
    # load the image
    hires_image_file = os.path.join(sample_dir, 'spatial/tissue_hires_image.png')
    img = cv2.imread(hires_image_file)
    # rotate the original image
    if rot_angle == 180:
        rotateCode = cv2.ROTATE_180
        rotateCode_r = cv2.ROTATE_180
    elif rot_angle == 90:
        rotateCode = cv2.ROTATE_90_COUNTERCLOCKWISE
        rotateCode_r = cv2.ROTATE_90_CLOCKWISE
    elif rot_angle == 270:
        rotateCode = cv2.ROTATE_90_CLOCKWISE
        rotateCode_r = cv2.ROTATE_90_COUNTERCLOCKWISE

    if rot_angle != 0:
        img_rot = cv2.rotate(img, rotateCode=rotateCode)
        img_rot_mask = img_rot.copy()
    else:
        img_rot_mask = img.copy()

    # map the annotations to see if they are correct
    img_rot_mask.fill(0)
    annot_files = sorted(glob.glob(os.path.join(annot_dir, 'tissue_hires_image*csv')))
    for annot_file in annot_files:
        layer = os.path.split(annot_file)[-1][:-4].split('_')[-1]
        print(layer)
        layer_code = LAYER_CODES[layer]
        # load the data
        dfl = pd.read_csv(annot_file)
        xy = dfl[['X', 'Y']]
        img_rot_mask[xy['Y'], xy['X']] = layer_code
    
    if rot_angle != 0:
        # rotate back
        img_mask = cv2.rotate(img_rot_mask, rotateCode=rotateCode_r)
    else:
        img_mask = img_rot_mask.copy()
    
    # interpolate all foreground pixels
    filled_mask_ch1 = fill_unlabeled_areas(img, img_mask)
    filled_mask_ch3 = cv2.cvtColor(filled_mask_ch1, cv2.COLOR_GRAY2BGR)
    img_concat = np.vstack((img, img_mask*25, filled_mask_ch3*25))
    cv2.imwrite(os.path.join(annot_dir, f'concated_mask.png'), img_concat)
    # save the mask_file
    cv2.imwrite(os.path.join(annot_dir, 'layer_mask.png'), filled_mask_ch1)
    cv2.imwrite(os.path.join(annot_dir, 'layer_mask_mul25.png'), filled_mask_ch1*25)

    return filled_mask_ch1

def get_layers(layer_mask, yy, xx, pct_outlier=0.05):
    """
    获取点 (yy, xx) 的前景标签，若落在背景上则用最近邻插值修正。
    
    参数:
        layer_mask: 单通道mask，0=背景，1-7=前景
        yy: 点的行坐标（一维数组）
        xx: 点的列坐标（一维数组）
    
    返回:
        labels: 修正后的前景标签（1-7）
    """
    # 1. 提取原始标签
    labels = layer_mask[yy, xx]
    
    # 2. 标记需要修正的点（落在背景上的点）
    bg_mask = (labels == 0)
    if not np.any(bg_mask):
        return labels  # 无背景点，直接返回

    if 1.0 * bg_mask.sum() / len(labels) > pct_outlier:
        raise ValueError("Too much outliers, please check the system")
    
    # 3. 计算最近前景的坐标
    # 3.1 创建一个前景的二进制掩码（1-7为True，0为False）
    foreground = (layer_mask > 0)
    
    # 3.2 计算距离变换，得到最近前景的坐标
    _, nearest_coords = ndimage.distance_transform_edt(
        ~foreground,  # 输入是背景区域（~foreground）
        return_indices=True
    )
    
    # 4. 修正背景点的标签
    yy_bg, xx_bg = yy[bg_mask], xx[bg_mask]
    nearest_yy = nearest_coords[0, yy_bg, xx_bg]
    nearest_xx = nearest_coords[1, yy_bg, xx_bg]
    labels[bg_mask] = layer_mask[nearest_yy, nearest_xx]
    
    return labels

def assign_layers_to_spots(layer_file, spots_file, visual_check=False, save=True):
    # load the layer mask
    layer_mask = cv2.imread(layer_file, cv2.IMREAD_UNCHANGED)

    # spots information
    adata = sc.read(spots_file, backed='r')
    spots_coords_pxl = np.round(adata.obsm['spatial']).astype(int)
    
    if visual_check:
        # visualize
        layer_mask_vis = layer_mask * 25
        point_size = 2  # 控制点的扩展范围（2 表示 5x5 区域）
        height, width = layer_mask_vis.shape[:2]
        for y, x in spots_coords_pxl:
            y_min, y_max = max(0, y-point_size), min(height, y+point_size+1)
            x_min, x_max = max(0, x-point_size), min(width, x+point_size+1)
            layer_mask_vis[y_min:y_max, x_min:x_max] = 255

        vis_dir = os.path.split(layer_file)[0]
        cv2.imwrite(os.path.join(vis_dir, 'spots_on_mask.png'), layer_mask_vis)

    yy, xx = spots_coords_pxl[:,0], spots_coords_pxl[:,1]
    layer_codes = get_layers(layer_mask, yy, xx)
    layer_names = [LAYER_CODES_REV[k] for k in layer_codes]
    adata.obs['laminar'] = layer_names
    
    if save:
        new_file = f'{spots_file[:-5]}_withLaminar.h5ad'
        adata.write(new_file)


if __name__ == '__main__':
    sample_id = 'P00083'
    sample_dir = f'data/ST-raw/{sample_id}'
    annot_dir = f'data/ST-raw/{sample_id}/layers'
    spots_file = f'data/layers/spatial_adata_{sample_id}.h5ad'
    
    rotations = get_rotation_angles()   # rotation angles counter-clockwise
    rot_angle = rotations.loc[sample_id].values[0]

    get_layer_masks(sample_dir, annot_dir, rot_angle)
    if 0:
        layer_file = os.path.join(annot_dir, 'layer_mask.png')
        assign_layers_to_spots(layer_file, spots_file, visual_check=True)

