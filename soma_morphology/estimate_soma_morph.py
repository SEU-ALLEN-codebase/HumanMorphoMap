##########################################################
#Author:          Yufeng Liu
#Create time:     2025-12-01
#Description:               
##########################################################
import os
import glob
import time
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
import tifffile
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from skimage import measure
import pickle


class SomaFeatureCalculator:
    def __init__(self, 
                 mask_dir: str = "/data2/lyf/data/human10k_tmp/data/0_seg_soma-lyf/mask_isotropy",
                 info_file: str = "./fitted_soma_info.csv"):
        """
        初始化胞体特征计算器
        
        Args:
            mask_dir: 分割mask文件目录
            info_file: 包含细胞信息的CSV文件路径
        """
        self.mask_dir = Path(mask_dir)
        self.info_df = pd.read_csv(info_file, index_col=0)
        
        random.seed(1024)
        np.random.seed(1024)
        
    def get_soma_features(self, cell_id: int) -> Dict:
        """
        计算给定cell_id的胞体特征
        
        Args:
            cell_id: 细胞ID
            
        Returns:
            包含所有计算特征的字典
        """
        # 获取细胞信息
        try:
            cell_info = self.info_df.loc[cell_id]
        except KeyError:
            print(f"Cell ID {cell_id} not found in info file")
        
        # 加载mask
        mask = self._load_mask(cell_info['filename'])
        
        # 计算各个特征
        features = {
            'cell_id': cell_id,
            'soma_volume': self._calculate_soma_volume(mask, cell_info['xy_rez']),
            'soma_anisotropy': self._calculate_anisotropy(cell_info),
            'fg_el_ratio': cell_info['fg_el_ratio'],
            'soma_smoothness': self._calculate_surface_smoothness(mask, cell_info['xy_rez']),
            'soma_sphericity': self._calculate_sphericity(mask, cell_info['xy_rez']),
            'soma_elongation': self._calculate_elongation(cell_info),
            'soma_flatness': self._calculate_flatness(cell_info),
            'aspect_ratio_z_xy': self._calculate_aspect_ratio_z_xy(cell_info)
        }
        
        return features
    
    def _load_mask(self, filename: str) -> np.ndarray:
        """加载tif格式的mask文件"""
        mask_path = self.mask_dir / filename
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask file not found: {mask_path}")
        
        mask = tifffile.imread(mask_path)
        return mask.astype(bool)
    
    def _calculate_soma_volume(self, mask: np.ndarray, xy_rez: float) -> float:
        """
        计算胞体体积
        
        体积 = 前景点数目 * (xy_rez)^3
        
        Args:
            mask: 3D二值mask
            xy_rez: xy平面分辨率
            
        Returns:
            体积（立方微米或相应单位）
        """
        print('    Calculate volume...')
        voxel_count = np.sum(mask)
        voxel_volume = xy_rez ** 3
        volume = voxel_count * voxel_volume
        
        return {
            'volume_voxel_based': volume,
            'voxel_count': int(voxel_count),
            'voxel_volume': voxel_volume,
        }
    
    def _calculate_anisotropy(self, cell_info: pd.Series) -> Dict:
        """
        计算胞体各向异性特征
        
        Args:
            cell_info: 包含细胞信息的pandas Series
            
        Returns:
            包含各项异性特征的字典
        """
        # absolute anistropy
        print('    Calculate anisotropy...')
        xnew, ynew, znew = sorted([cell_info['radius_z'], cell_info['radius_y'], cell_info['radius_x']])
        radii = {
            'z': znew,
            'y': ynew,
            'x': xnew
        }
        
        # 计算各向异性比
        anisotropy_ratios = {
            'z_to_x': radii['z'] / radii['x'],
            'z_to_y': radii['z'] / radii['y'],
            'y_to_x': radii['y'] / radii['x']
        }
        
        # 主方向上的各向异性指数
        max_radius = max(radii.values())
        min_radius = min(radii.values())
        
        # 计算几个常用的各向异性指标
        features = {
            'radii': radii,
            'anisotropy_ratios': anisotropy_ratios,
            'max_min_ratio': max_radius / min_radius if min_radius > 0 else float('inf'),
            'anisotropy_index': (max_radius - min_radius) / max_radius if max_radius > 0 else 0,
            'sphericity_index': min_radius / max_radius if max_radius > 0 else 0,
            'aspect_ratio': {
                'z_over_x': radii['z'] / radii['x'],
                'z_over_y': radii['z'] / radii['y']
            }
        }
        
        return features
    
    def _calculate_surface_smoothness(self, mask: np.ndarray, xy_rez: float) -> Dict:
        """
        计算胞体表面平滑度
        
        通过计算表面法向量变化或表面曲率来评估平滑度
        
        Args:
            mask: 3D二值mask
            xy_rez: xy平面分辨率
            
        Returns:
            包含平滑度指标的字典
        """
        # 提取表面体素
        print('    Surface smoothness')
        surface_voxels = self._extract_surface_voxels(mask)
        
        if len(surface_voxels) == 0:
            return {
                'surface_area': 0,
                'smoothness_score': 0,
                'curvature_variance': 0,
                'surface_complexity': 0
            }
        
        t0 = time.time()
        
        # 计算表面面积（近似）
        surface_area = self._calculate_surface_area(mask, xy_rez)
        #print(f'      > t0: {time.time() - t0:4f}')
        
        # 计算表面粗糙度/平滑度
        smoothness_metrics = self._calculate_surface_roughness(mask, surface_voxels)
        #print(f'      > t1: {time.time() - t0:4f}')
        
        # 计算表面曲率变化
        curvature_metrics = self._estimate_curvature(mask)
        #print(f'      > t2: {time.time() - t0:4f}, curvature: {curvature_metrics}')
        
        return {
            'surface_area': surface_area,
            'surface_voxel_count': len(surface_voxels),
            'smoothness_score': smoothness_metrics.get('smoothness_score', 0),
            'curvature_variance': curvature_metrics.get('curvature_variance', 0),
            'mean_curvature': curvature_metrics.get('mean_curvature', 0),
            'surface_complexity': smoothness_metrics.get('complexity', 0),
            'roughness_index': smoothness_metrics.get('roughness_index', 0)
        }
    
    def _extract_surface_voxels(self, mask: np.ndarray) -> np.ndarray:
        """提取表面体素"""
        # 使用形态学操作找到表面
        structure = ndimage.generate_binary_structure(3, 1)
        eroded = ndimage.binary_erosion(mask, structure=structure)
        surface = mask & ~eroded
        
        # 获取表面体素坐标
        surface_coords = np.argwhere(surface)
        return surface_coords
    
    def _calculate_surface_area(self, mask: np.ndarray, xy_rez: float) -> float:
        """计算表面面积"""
        # 使用marching cubes或简单估计
        if False:
            # 尝试使用marching cubes计算更精确的表面积
            verts, faces, _, _ = measure.marching_cubes(mask.astype(float), level=0.5)
            
            # 计算每个面的面积
            triangles = verts[faces]
            edges1 = triangles[:, 1, :] - triangles[:, 0, :]
            edges2 = triangles[:, 2, :] - triangles[:, 0, :]
            cross_prod = np.cross(edges1, edges2)
            areas = 0.5 * np.sqrt(np.sum(cross_prod**2, axis=1))
            total_area = np.sum(areas) * (xy_rez ** 2)
            
        else:
            # 简单的6邻接表面估计
            structure = np.ones((3, 3, 3), dtype=bool)
            eroded = ndimage.binary_erosion(mask, structure=structure)
            surface = mask & ~eroded
            surface_voxel_count = np.sum(surface)
            
            # 每个表面体素贡献的面积（近似）
            voxel_area = 6 * (xy_rez ** 2)  # 立方体6个面
            total_area = surface_voxel_count * voxel_area / 6  # 平均每个体素暴露1个面
        
        return total_area
    
    def _calculate_surface_roughness(self, mask: np.ndarray, surface_coords: np.ndarray) -> Dict:
        """计算表面粗糙度指标"""
        if len(surface_coords) < 10:
            return {'smoothness_score': 0, 'roughness_index': 0, 'complexity': 0}
        
        # 计算表面法向量的变化
        normals = []
        for coord in surface_coords[:100]:  # 采样部分点以减少计算量
            normal = self._estimate_normal(mask, coord)
            if normal is not None:
                normals.append(normal)
        
        if len(normals) < 3:
            return {'smoothness_score': 0, 'roughness_index': 0, 'complexity': 0}
        
        normals = np.array(normals)
        
        # 计算法向量之间的角度差异
        dot_products = []
        for i in range(len(normals)):
            for j in range(i+1, min(i+10, len(normals))):  # 比较邻近点
                dot = np.dot(normals[i], normals[j])
                dot_products.append(dot)
        
        if dot_products:
            mean_dot = np.mean(dot_products)
            std_dot = np.std(dot_products)
            
            # 平滑度评分（1表示完全平滑）
            smoothness_score = max(0, min(1, mean_dot))
            
            # 粗糙度指数
            roughness_index = std_dot
            
            # 表面复杂度（与理想球面的偏差）
            complexity = 1 - smoothness_score
            
            return {
                'smoothness_score': float(smoothness_score),
                'roughness_index': float(roughness_index),
                'complexity': float(complexity)
            }
        else:
            return {'smoothness_score': 0, 'roughness_index': 0, 'complexity': 0}
    
    def _estimate_normal(self, mask: np.ndarray, coord: np.ndarray) -> Optional[np.ndarray]:
        """估计表面点法向量"""
        z, y, x = coord
        
        # 检查邻近区域
        neighborhood = mask[max(0, z-1):z+2, 
                          max(0, y-1):y+2, 
                          max(0, x-1):x+2]
        
        if neighborhood.size < 27:
            return None
        
        # 使用中心差分计算梯度作为法向量估计
        if z > 0 and z < mask.shape[0]-1:
            dz = int(mask[z+1, y, x]) - int(mask[z-1, y, x])
        else:
            dz = 0
            
        if y > 0 and y < mask.shape[1]-1:
            dy = int(mask[z, y+1, x]) - int(mask[z, y-1, x])
        else:
            dy = 0
            
        if x > 0 and x < mask.shape[2]-1:
            dx = int(mask[z, y, x+1]) - int(mask[z, y, x-1])
        else:
            dx = 0
        
        normal = np.array([dx, dy, dz])
        norm = np.linalg.norm(normal)
        
        if norm > 0:
            return normal / norm
        return None

    def _estimate_curvature(self, mask: np.ndarray) -> Dict:
        """快速曲率近似"""
        try:
            # 提取表面
            surface = self._extract_surface_voxels(mask)
            if len(surface) < 10:
                return self._get_default_curvature()
            
            # 采样
            sample_size = min(200, len(surface))
            sample_indices = np.random.choice(len(surface), sample_size, replace=False)
            sampled_surface = surface[sample_indices]
            
            curvatures = []
            
            for point in sampled_surface:
                z, y, x = point
                
                # 提取7x7x7邻域（更大的邻域获得更稳定的曲率估计）
                if (z >= 3 and z < mask.shape[0]-3 and 
                    y >= 3 and y < mask.shape[1]-3 and 
                    x >= 3 and x < mask.shape[2]-3):
                    
                    neighborhood = mask[z-3:z+4, y-3:y+4, x-3:x+4]
                    
                    # 简单曲率估计：表面点的局部球面拟合
                    curvature = self._simple_sphere_fit(neighborhood)
                    if curvature is not None:
                        curvatures.append(curvature)
            
            if len(curvatures) == 0:
                return self._get_default_curvature()
            
            curvatures = np.array(curvatures)
            
            return {
                'mean_curvature': float(np.mean(curvatures)),
                'curvature_variance': float(np.var(curvatures)),
                'max_curvature': float(np.max(curvatures)),
                'min_curvature': float(np.min(curvatures)),
                'n_points': len(curvatures)
            }
            
        except Exception as e:
            print(f"Fast curvature estimation failed: {e}")
            return self._get_default_curvature()

    def _simple_sphere_fit(self, neighborhood):
        """简单球面拟合曲率估计"""
        # 获取表面点坐标
        surface_points = np.argwhere(neighborhood > 0.5)
        
        if len(surface_points) < 5:
            return None
        
        # 中心化
        center = np.mean(surface_points, axis=0)
        centered = surface_points - center
        
        # 计算到中心的距离
        distances = np.linalg.norm(centered, axis=1)
        
        if np.std(distances) < 1e-6:
            return 0.0  # 完美球面
        
        # 曲率近似为距离方差的倒数
        curvature = 1.0 / (np.var(distances) + 1e-6)
        
        # 归一化到合理范围
        return np.clip(curvature, 0, 10)
    
    def _estimate_curvature_slow(self, mask: np.ndarray) -> Dict:
        """估计表面曲率"""
        try:
            # 使用高斯滤波后的图像计算曲率
            smoothed = gaussian_filter(mask.astype(float), sigma=1.0)
            
            # 计算梯度
            gradient = np.gradient(smoothed)
            
            # 计算二阶导数
            hessian = []
            for g in gradient:
                hessian.append(np.gradient(g))
            
            hessian = np.array(hessian)
            
            # 计算平均曲率（简化版本）
            curvature = np.zeros_like(smoothed)
            for i in range(smoothed.shape[0]):
                for j in range(smoothed.shape[1]):
                    for k in range(smoothed.shape[2]):
                        if mask[i, j, k]:
                            # 计算Hessian矩阵的特征值
                            H = np.array([
                                [hessian[0, 0, i, j, k], hessian[0, 1, i, j, k], hessian[0, 2, i, j, k]],
                                [hessian[1, 0, i, j, k], hessian[1, 1, i, j, k], hessian[1, 2, i, j, k]],
                                [hessian[2, 0, i, j, k], hessian[2, 1, i, j, k], hessian[2, 2, i, j, k]]
                            ])
                            
                            eigenvalues = np.linalg.eigvals(H)
                            # 平均曲率近似
                            curvature[i, j, k] = np.mean(eigenvalues)
            
            # 只考虑表面区域
            surface = self._extract_surface_voxels(mask)
            surface_curvature = []
            
            for coord in surface[:500]:  # 采样以减少计算量
                z, y, x = coord
                surface_curvature.append(curvature[z, y, x])
            
            if surface_curvature:
                return {
                    'mean_curvature': float(np.mean(surface_curvature)),
                    'curvature_variance': float(np.var(surface_curvature)),
                    'max_curvature': float(np.max(surface_curvature)),
                    'min_curvature': float(np.min(surface_curvature))
                }
                
        except Exception as e:
            print(f"Curvature estimation failed: {e}")
        
        return {'mean_curvature': 0, 'curvature_variance': 0, 'max_curvature': 0, 'min_curvature': 0}

    def _get_default_curvature(self) -> Dict:
        """返回默认曲率值"""
        return {
            'mean_curvature': 0.0,
            'curvature_variance': 0.0,
            'max_curvature': 0.0,
            'min_curvature': 0.0,
            'median_curvature': 0.0,
            'n_points': 0
        }
    
    def _calculate_sphericity(self, mask: np.ndarray, xy_rez: float) -> float:
        """
        计算球度（sphericity）
        
        sphericity = (π^(1/3) * (6V)^(2/3)) / A
        其中V是体积，A是表面积
        """
        # 计算体积
        print('    Sphericity')
        volume_dict = self._calculate_soma_volume(mask, xy_rez)
        volume = volume_dict['volume_voxel_based']
        
        # 计算表面积
        area_dict = self._calculate_surface_smoothness(mask, xy_rez)
        area = area_dict['surface_area']
        
        if area > 0:
            sphericity = (np.pi ** (1/3)) * ((6 * volume) ** (2/3)) / area
            return max(0, min(1, sphericity))
        
        return 0.0
    
    def _calculate_elongation(self, cell_info: pd.Series) -> float:
        """计算伸长率"""
        print('    Elongation...')
        radii = [cell_info['radius_z'], cell_info['radius_y'], cell_info['radius_x']]
        radii.sort(reverse=True)
        
        if radii[0] > 0:
            return 1 - (radii[-1] / radii[0])
        return 0.0
    
    def _calculate_flatness(self, cell_info: pd.Series) -> float:
        """计算扁平度"""
        print('    Flatness...')
        radii = [cell_info['radius_z'], cell_info['radius_y'], cell_info['radius_x']]
        radii.sort(reverse=True)
        
        if radii[0] > 0:
            return 1 - (radii[1] / radii[0])
        return 0.0
    
    def _calculate_aspect_ratio_z_xy(self, cell_info: pd.Series) -> Dict:
        """计算z轴与xy平面的长宽比"""
        # z方向半径
        print('    Z-to-XY ratio...')
        radius_z = cell_info['radius_z']
        
        # xy平面平均半径
        radius_xy_avg = (cell_info['radius_x'] + cell_info['radius_y']) / 2
        
        if radius_xy_avg > 0:
            z_to_xy_ratio = radius_z / radius_xy_avg
        else:
            z_to_xy_ratio = float('inf')
        
        return {
            'z_to_xy_ratio': z_to_xy_ratio,
            'is_oblate': z_to_xy_ratio < 0.9,  # 扁圆形
            'is_prolate': z_to_xy_ratio > 1.1,  # 长圆形
            'is_spherical': 0.9 <= z_to_xy_ratio <= 1.1  # 接近球形
        }


def batch_calculate_features(soma_info_file, out_pkl, cell_ids=None):
    # Initialize the calculator
    calculator = SomaFeatureCalculator(info_file=soma_info_file)

    if not (cell_ids and isinstance(cell_ids, list)):
        info_df = pd.read_csv(soma_info_file, index_col='cell_id')
        cell_ids = info_df.index.values
    
    """批量计算特征"""
    all_features = {}
    for cell_id in cell_ids:
        features = calculator.get_soma_features(cell_id)
        all_features[cell_id] = features
        print(f"✓ Calculated features for cell_id {cell_id}")
        #except Exception as e:
        #    print(f"✗ Error for cell_id {cell_id}: {e}")

        #if len(all_features) >= 10:
        #    break   # debug

    # dump to file
    with open(out_pkl, 'wb') as fp:
        pickle.dump(all_features, fp)
    
    return all_features




# 使用示例
if __name__ == "__main__":
    soma_info_file = './data/fitted_soma_info.csv'
    out_pkl = './data/soma_features.pkl'

    batch_calculate_features(soma_info_file, out_pkl)
        
    
    


