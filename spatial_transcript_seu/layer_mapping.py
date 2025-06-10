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

from config import LAYER_CODES


def get_rotation_angles(rotation_file=
        '/data2/lyf/data/transcriptomics/ST_SEU/P00117/rotate_angles_of_layer_annotation.csv'):
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

def assign_spots_layers(sample_dir, annot_dir, rot_angle, spots=None):
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

    img_rot = cv2.rotate(img, rotateCode=rotateCode)

    # map the annotations to see if they are correct
    img_rot_mask = img_rot.copy()
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
    
    # rotate back
    img_mask = cv2.rotate(img_rot_mask, rotateCode=rotateCode_r)
    
    # check the validity
    #img_concat = np.vstack((img, img_mask*20))
    #cv2.imwrite(f'concated_mask.png', img_concat)

    # interpolate all foreground pixels
    filled_mask_ch1 = fill_unlabeled_areas(img, img_mask)
    filled_mask_ch3 = cv2.cvtColor(filled_mask_ch1, cv2.COLOR_GRAY2BGR)
    img_concat = np.vstack((img, img_mask*25, filled_mask_ch3*25))
    cv2.imwrite(f'concated_mask.png', img_concat)


if __name__ == '__main__':
    sample_id = 'P00117'
    sample_dir = f'/PBshare/SEU-ALLEN/Users/WenYe/Human-Brain-ST-data/{sample_id}'
    annot_dir = f'/data2/lyf/data/transcriptomics/ST_SEU/{sample_id}/layers'
    
    rotations = get_rotation_angles()   # rotation angles counter-clockwise
    rot_angle = rotations.loc[sample_id].values[0]

    assign_spots_layers(sample_dir, annot_dir, rot_angle)   

