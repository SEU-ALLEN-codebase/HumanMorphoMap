##########################################################
#Author:          Yufeng Liu
#Create time:     2025-12-20
#Description:               
##########################################################
import os
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
import tifffile
from typing import Tuple, Optional, Dict, List
import warnings
warnings.filterwarnings('ignore')
from scipy.spatial.transform import Rotation


class SomaDataMatcher:
    def __init__(self):
        # 设置路径
        self.mask_dir = Path("/data2/kfchen/tracing_ws/soma_seg/masks")
        self.crop_mask_dir = Path("/data2/lyf/data/human10k_tmp/data/0_seg_soma-lyf/mask")
        self.csv_path = Path("./data/fitted_soma_info_8.4k.csv")
        self.id_map_path = Path("newID_oldID.csv")
        
        # 加载数据
        self.id_map = None
        self.soma_info = None
        self.matched_data = []
        
    def load_id_map(self) -> pd.DataFrame:
        """加载新旧ID映射关系"""
        print("正在加载ID映射表...")
        try:
            self.id_map = pd.read_csv(self.id_map_path)
            # 确保列名存在
            required_cols = ['cell_id', 'cell_id_backUp']
            for col in required_cols:
                if col not in self.id_map.columns:
                    raise ValueError(f"列 {col} 不存在于映射表中")
            
            print(f"成功加载 {len(self.id_map)} 条映射记录")
            return self.id_map
        except Exception as e:
            print(f"加载ID映射表失败: {e}")
            raise
    
    def load_soma_info(self) -> pd.DataFrame:
        """加载soma信息CSV"""
        print("正在加载soma信息...")
        try:
            self.soma_info = pd.read_csv(self.csv_path)
            print(f"成功加载 {len(self.soma_info)} 条soma记录")
            return self.soma_info
        except Exception as e:
            print(f"加载soma信息失败: {e}")
            raise
    
    def get_mask_files(self) -> Dict[int, Path]:
        """获取所有标注mask文件，返回{新ID: 文件路径}的字典"""
        mask_files = {}
        for mask_file in self.mask_dir.glob("*.tif"):
            try:
                # 从文件名提取新ID
                new_id = int(mask_file.stem[:-5])  # 假设文件名就是ID.tif
                mask_files[new_id] = mask_file
            except ValueError:
                print(f"警告: 无法从文件名 {mask_file.name} 解析ID")
                continue
        
        print(f"找到 {len(mask_files)} 个标注mask文件")
        return mask_files
    
    def match_ids(self) -> Dict[int, int]:
        """建立新ID到老ID的映射关系"""
        id_map_df = self.load_id_map()
        
        # 创建新ID->老ID的映射字典
        new_to_old = {}
        for _, row in id_map_df.iterrows():
            new_id = row['cell_id']
            old_id = row['cell_id_backUp']
            if pd.notna(new_id) and pd.notna(old_id):
                new_to_old[int(new_id)] = int(old_id)
        
        print(f"建立了 {len(new_to_old)} 个ID映射关系")
        return new_to_old
    
    def find_crop_image(self, old_id: int) -> Optional[Path]:
        """根据老ID查找crop图像"""
        # 在soma_info中查找对应的记录
        if old_id in self.soma_info['cell_id'].values:
            record = self.soma_info[self.soma_info['cell_id'] == old_id].iloc[0]
            
            # 查找对应的crop mask文件
            crop_filename = record['filename']
            crop_path = self.crop_mask_dir / crop_filename
            
            if crop_path.exists():
                return crop_path
            else:
                # 尝试其他可能的命名格式
                possible_filenames = [
                    crop_filename,
                    f"{old_id:05d}_{crop_filename}",
                    f"{old_id:05d}_{Path(crop_filename).name}"
                ]
                
                for filename in possible_filenames:
                    test_path = self.crop_mask_dir / filename
                    if test_path.exists():
                        return test_path
        
        return None
    
    def get_bbox_from_record(self, old_id: int) -> Optional[dict]:
        """从CSV记录中获取bbox信息"""
        if old_id in self.soma_info['cell_id'].values:
            record = self.soma_info[self.soma_info['cell_id'] == old_id].iloc[0]
            
            bbox_info = {
                'z_min': int(record['z_min_exp']),
                'z_max': int(record['z_max_exp']),
                'y_min': int(record['y_min_exp']),
                'y_max': int(record['y_max_exp']),
                'x_min': int(record['x_min_exp']),
                'x_max': int(record['x_max_exp']),
                'center_z': float(record['center_z']),
                'center_y': float(record['center_y']),
                'center_x': float(record['center_x']),
                'radius_z': float(record['radius_z']),
                'radius_y': float(record['radius_y']),
                'radius_x': float(record['radius_x']),
                'xy_rez': float(record['xy_rez']),
                'z_rez': float(record['z_rez']),
                'filename': record['filename'],
                'rotation_matrix': self.parse_rotation_matrix(record)
            }
            return bbox_info
        return None
    
    def parse_rotation_matrix(self, record: pd.Series) -> np.ndarray:
        """解析旋转矩阵"""
        try:
            # 尝试从字符串解析
            if isinstance(record['rotation_flat'], str):
                # 清理字符串
                rot_str = record['rotation_flat'].replace('[', '').replace(']', '').replace('\n', ' ')
                rot_values = [float(x) for x in rot_str.split()]
                if len(rot_values) == 9:
                    return np.array(rot_values).reshape(3, 3)
            
            # 尝试从单独的列解析
            rot_cols = [f'rot_{i}' for i in range(9)]
            if all(col in record.index for col in rot_cols):
                rot_matrix = np.zeros((3, 3))
                for i in range(9):
                    rot_matrix[i//3, i%3] = record[f'rot_{i}']
                return rot_matrix
        except Exception as e:
            print(f"解析旋转矩阵失败: {e}")
        
        # 返回单位矩阵作为默认值
        return np.eye(3)
    
    def match_all_data(self) -> pd.DataFrame:
        """匹配所有数据"""
        print("开始匹配数据...")
        
        # 加载数据
        self.load_soma_info()
        id_mapping = self.match_ids()
        mask_files = self.get_mask_files()
        
        matched_records = []
        
        for new_id, mask_path in mask_files.items():
            if new_id not in id_mapping:
                print(f"警告: 新ID {new_id} 在映射表中找不到对应老ID")
                continue
                
            old_id = id_mapping[new_id]
            
            # 查找crop图像
            crop_path = self.find_crop_image(old_id)
            bbox_info = self.get_bbox_from_record(old_id)
            
            if crop_path and bbox_info:
                record = {
                    'new_id': new_id,
                    'old_id': old_id,
                    'mask_path': str(mask_path),
                    'crop_path': str(crop_path),
                    'filename': bbox_info['filename'],
                    **bbox_info
                }
                matched_records.append(record)
                print(f"匹配成功: 新ID {new_id} -> 老ID {old_id}")
            else:
                print(f"警告: 老ID {old_id} 的crop图像或bbox信息未找到")
        
        # 转换为DataFrame
        matched_df = pd.DataFrame(matched_records)
        print(f"\n匹配完成！成功匹配 {len(matched_df)}/{len(mask_files)} 个文件")
        
        return matched_df
    
    def calculate_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """计算两个二值mask的IOU"""
        # 确保是二值mask
        mask1_bin = (mask1 > 0).astype(np.uint8)
        mask2_bin = (mask2 > 0).astype(np.uint8)
        
        # 计算交集和并集
        intersection = np.logical_and(mask1_bin, mask2_bin).sum()
        union = np.logical_or(mask1_bin, mask2_bin).sum()
        
        # 避免除零
        if union == 0:
            return 0.0
        
        return float(intersection) / float(union)
    
    def calculate_dice(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """计算Dice系数"""
        mask1_bin = (mask1 > 0).astype(np.uint8)
        mask2_bin = (mask2 > 0).astype(np.uint8)
        
        intersection = np.logical_and(mask1_bin, mask2_bin).sum()
        sum_masks = mask1_bin.sum() + mask2_bin.sum()
        
        if sum_masks == 0:
            return 0.0
        
        return 2.0 * intersection / sum_masks
    
    def load_and_compare_masks(self, matched_df: pd.DataFrame) -> pd.DataFrame:
        """加载并比较估计mask和标注mask"""
        print("\n开始加载和比较mask...")
        
        results = []
        
        from scipy.ndimage import zoom
        
        for idx, record in matched_df.iterrows():
            new_id = record['new_id']
            old_id = record['old_id']
            
            print(f"处理 ID {new_id} (老ID: {old_id})...")
            
            # 1. 加载估计的crop mask
            crop_mask_path = Path(record['crop_path'])
            if not crop_mask_path.exists():
                print(f"  警告: crop mask不存在: {crop_mask_path}")
                continue
                
            estimated_mask = tifffile.imread(crop_mask_path)
            
            # 2. 加载标注的完整mask
            annotated_mask_path = Path(record['mask_path'])
            if not annotated_mask_path.exists():
                print(f"  警告: 标注mask不存在: {annotated_mask_path}")
                continue
                
            annotated_full_mask = tifffile.imread(annotated_mask_path)
            #annotated_full_mask = np.flip(annotated_full_mask, axis=1)
            
            # 计算新的shape
            original_shape = annotated_full_mask.shape
            #new_shape = (
            #    original_shape[0],  # z轴保持不变
            #    int(original_shape[1] * scale_factor),  # y轴缩放
            #    int(original_shape[2] * scale_factor)   # x轴缩放
            #)
            
            # 使用最近邻插值进行resize（order=0）
            annotated_full_mask = zoom(annotated_full_mask, 
                                     (0.5, 0.5, 0.5), 
                                     order=0)
            
            print(f"  原始标注mask形状: {original_shape}")
            print(f"  缩放后标注mask形状: {annotated_full_mask.shape}")
            
            # 3. 裁剪标注mask到bbox区域
            z_min = record['z_min']
            z_max = record['z_max']
            y_min = record['y_min']
            y_max = record['y_max']
            x_min = record['x_min']
            x_max = record['x_max']
            
            # 检查bbox是否在标注mask范围内
            if (z_min < 0 or z_max > annotated_full_mask.shape[0] or
                y_min < 0 or y_max > annotated_full_mask.shape[1] or
                x_min < 0 or x_max > annotated_full_mask.shape[2]):
                print(f"  警告: bbox超出标注mask范围")
                print(f"  标注mask形状: {annotated_full_mask.shape}")
                print(f"  bbox: z[{z_min}:{z_max}], y[{y_min}:{y_max}], x[{x_min}:{x_max}]")
                
                # 尝试调整bbox到有效范围
                z_min = max(0, z_min)
                z_max = min(annotated_full_mask.shape[0], z_max)
                y_min = max(0, y_min)
                y_max = min(annotated_full_mask.shape[1], y_max)
                x_min = max(0, x_min)
                x_max = min(annotated_full_mask.shape[2], x_max)
                print(f"  调整后bbox: z[{z_min}:{z_max}], y[{y_min}:{y_max}], x[{x_min}:{x_max}]")
            
            # 裁剪标注mask
            annotated_crop_mask = annotated_full_mask[
                z_min:z_max,
                y_min:y_max,
                x_min:x_max
            ]
            
            # 4. 调整mask大小以确保维度匹配
            # 如果维度不匹配，尝试resize或padding
            if estimated_mask.shape != annotated_crop_mask.shape:
                print(f"  警告: mask形状不匹配")
                print(f"  估计mask形状: {estimated_mask.shape}")
                print(f"  标注crop mask形状: {annotated_crop_mask.shape}")
                
                # 使用较小的形状作为基准
                min_shape = (
                    min(estimated_mask.shape[0], annotated_crop_mask.shape[0]),
                    min(estimated_mask.shape[1], annotated_crop_mask.shape[1]),
                    min(estimated_mask.shape[2], annotated_crop_mask.shape[2])
                )
                
                # 裁剪两个mask到相同大小
                estimated_mask_cropped = estimated_mask[
                    :min_shape[0],
                    :min_shape[1],
                    :min_shape[2]
                ]
                annotated_crop_mask_cropped = annotated_crop_mask[
                    :min_shape[0],
                    :min_shape[1],
                    :min_shape[2]
                ]
            else:
                estimated_mask_cropped = estimated_mask
                annotated_crop_mask_cropped = annotated_crop_mask

            # flip
            annotated_crop_mask_cropped = np.flip(annotated_crop_mask_cropped, axis=1)
            
            # resize to 1um x 1um x 1um
            xy_rez = record['xy_rez']
            z_rez = record['z_rez']
            print(xy_rez, z_rez)
            #annotated_crop_mask_cropped = zoom(annotated_crop_mask_cropped, 
            #                         (1/z_rez, 1/xy_rez, 1/xy_rez), 
            #                         order=0)
            #estimated_mask_cropped = zoom(estimated_mask_cropped, 
            #                         (1/z_rez, 1/xy_rez, 1/xy_rez), 
            #                         order=0)
            #print(estimated_mask_cropped.shape, annotated_crop_mask_cropped.shape)
        
            # 5. 计算评估指标
            iou = self.calculate_iou(estimated_mask_cropped, annotated_crop_mask_cropped)
            dice = self.calculate_dice(estimated_mask_cropped, annotated_crop_mask_cropped)

            print(dice)
            cv2.imwrite(f'img0_0_{old_id}_dice{dice:.2f}.png', estimated_mask_cropped.max(axis=0))
            cv2.imwrite(f'img1_0_{old_id}_dice{dice:.2f}.png', annotated_crop_mask_cropped.max(axis=0))
            cv2.imwrite(f'img0_1_{old_id}_dice{dice:.2f}.png', estimated_mask_cropped.max(axis=1))
            cv2.imwrite(f'img1_1_{old_id}_dice{dice:.2f}.png', annotated_crop_mask_cropped.max(axis=1))

            # 6. 计算体积信息
            estimated_volume = (estimated_mask_cropped > 0).sum()
            annotated_volume = (annotated_crop_mask_cropped > 0).sum()
            volume_ratio = estimated_volume / annotated_volume if annotated_volume > 0 else 0
            
            # 7. 保存结果
            result = {
                'new_id': new_id,
                'old_id': old_id,
                'filename': record['filename'],
                'iou': iou,
                'dice': dice,
                'estimated_volume': estimated_volume,
                'annotated_volume': annotated_volume,
                'volume_ratio': volume_ratio,
                'estimated_shape': estimated_mask.shape,
                'annotated_full_shape': annotated_full_mask.shape,
                'annotated_crop_shape': annotated_crop_mask.shape,
                'bbox': f"z[{record['z_min']}:{record['z_max']}], y[{record['y_min']}:{record['y_max']}], x[{record['x_min']}:{record['x_max']}]",
                'crop_mask_path': str(crop_mask_path),
                'annotated_mask_path': str(annotated_mask_path),
                'match_success': True
            }
            
            results.append(result)
            print(f"  完成: IOU={iou:.4f}, Dice={dice:.4f}, 体积比={volume_ratio:.4f}")
                
       
        # 转换为DataFrame
        results_df = pd.DataFrame(results)
        
        if len(results_df) > 0:
            # 计算统计信息
            successful = results_df[results_df['match_success']]
            if len(successful) > 0:
                print(f"\n评估统计信息:")
                print(f"成功处理: {len(successful)}/{len(matched_df)}")
                print(f"平均IOU: {successful['iou'].mean():.4f} (±{successful['iou'].std():.4f})")
                print(f"平均Dice: {successful['dice'].mean():.4f} (±{successful['dice'].std():.4f})")
                print(f"中位数IOU: {successful['iou'].median():.4f}")
                print(f"中位数Dice: {successful['dice'].median():.4f}")
                print(f"最大IOU: {successful['iou'].max():.4f}")
                print(f"最小IOU: {successful['iou'].min():.4f}")
                print(f"平均体积比: {successful['volume_ratio'].mean():.4f}")
        
        return results_df
    
    def visualize_comparison(self, results_df: pd.DataFrame, save_dir: Optional[str] = None):
        """可视化比较结果"""
        if len(results_df) == 0:
            print("没有结果可供可视化")
            return
        
        try:
            import matplotlib.pyplot as plt
            
            # 只选择成功匹配的
            successful_df = results_df[results_df['match_success']]
            if len(successful_df) == 0:
                print("没有成功匹配的结果可供可视化")
                return
            
            # 创建可视化目录
            if save_dir:
                viz_dir = Path(save_dir) / "visualization"
                viz_dir.mkdir(parents=True, exist_ok=True)
            
            # 1. IOU和Dice分布直方图
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # IOU分布
            axes[0, 0].hist(successful_df['iou'], bins=20, edgecolor='black', alpha=0.7)
            axes[0, 0].set_xlabel('IOU')
            axes[0, 0].set_ylabel('频数')
            axes[0, 0].set_title('IOU分布')
            axes[0, 0].axvline(successful_df['iou'].mean(), color='r', linestyle='--', label=f'均值: {successful_df["iou"].mean():.3f}')
            axes[0, 0].legend()
            
            # Dice分布
            axes[0, 1].hist(successful_df['dice'], bins=20, edgecolor='black', alpha=0.7)
            axes[0, 1].set_xlabel('Dice系数')
            axes[0, 1].set_ylabel('频数')
            axes[0, 1].set_title('Dice系数分布')
            axes[0, 1].axvline(successful_df['dice'].mean(), color='r', linestyle='--', label=f'均值: {successful_df["dice"].mean():.3f}')
            axes[0, 1].legend()
            
            # 体积比分布
            axes[0, 2].hist(successful_df['volume_ratio'], bins=20, edgecolor='black', alpha=0.7)
            axes[0, 2].set_xlabel('体积比 (估计/标注)')
            axes[0, 2].set_ylabel('频数')
            axes[0, 2].set_title('体积比分布')
            axes[0, 2].axvline(1.0, color='r', linestyle='--', label='理想值: 1.0')
            axes[0, 2].legend()
            
            # 2. 散点图：IOU vs 体积比
            axes[1, 0].scatter(successful_df['volume_ratio'], successful_df['iou'], alpha=0.6)
            axes[1, 0].set_xlabel('体积比')
            axes[1, 0].set_ylabel('IOU')
            axes[1, 0].set_title('IOU vs 体积比')
            axes[1, 0].axvline(1.0, color='r', linestyle='--', alpha=0.3)
            
            # 3. 散点图：估计体积 vs 标注体积
            axes[1, 1].scatter(successful_df['annotated_volume'], successful_df['estimated_volume'], alpha=0.6)
            axes[1, 1].set_xlabel('标注体积 (voxels)')
            axes[1, 1].set_ylabel('估计体积 (voxels)')
            axes[1, 1].set_title('体积对比')
            max_vol = max(successful_df['annotated_volume'].max(), successful_df['estimated_volume'].max())
            axes[1, 1].plot([0, max_vol], [0, max_vol], 'r--', alpha=0.3, label='y=x')
            axes[1, 1].legend()
            
            # 4. 性能排名
            sorted_df = successful_df.sort_values('iou', ascending=False).head(10)
            axes[1, 2].barh(range(len(sorted_df)), sorted_df['iou'])
            axes[1, 2].set_yticks(range(len(sorted_df)))
            axes[1, 2].set_yticklabels([f"{int(row['new_id'])}" for _, row in sorted_df.iterrows()])
            axes[1, 2].set_xlabel('IOU')
            axes[1, 2].set_title('Top 10 IOU排名')
            
            plt.tight_layout()
            
            if save_dir:
                fig_path = viz_dir / "comparison_summary.png"
                plt.savefig(fig_path, dpi=150, bbox_inches='tight')
                print(f"统计图已保存到: {fig_path}")
            
            plt.show()
            
        except ImportError:
            print("需要matplotlib来进行可视化")
        except Exception as e:
            print(f"可视化时出错: {e}")
    
    def save_evaluation_results(self, results_df: pd.DataFrame, output_path: str = "./evaluation_results"):
        """保存评估结果"""
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存详细结果
        detailed_path = output_dir / "detailed_evaluation_results.csv"
        results_df.to_csv(detailed_path, index=False)
        print(f"详细结果已保存到: {detailed_path}")
        
        # 保存汇总统计
        if len(results_df) > 0:
            successful = results_df[results_df['match_success']]
            if len(successful) > 0:
                summary_stats = {
                    'total_samples': len(results_df),
                    'successful_samples': len(successful),
                    'success_rate': len(successful) / len(results_df) * 100,
                    'mean_iou': successful['iou'].mean(),
                    'std_iou': successful['iou'].std(),
                    'median_iou': successful['iou'].median(),
                    'max_iou': successful['iou'].max(),
                    'min_iou': successful['iou'].min(),
                    'mean_dice': successful['dice'].mean(),
                    'std_dice': successful['dice'].std(),
                    'median_dice': successful['dice'].median(),
                    'max_dice': successful['dice'].max(),
                    'min_dice': successful['dice'].min(),
                    'mean_volume_ratio': successful['volume_ratio'].mean(),
                    'std_volume_ratio': successful['volume_ratio'].std(),
                    'median_volume_ratio': successful['volume_ratio'].median()
                }
                
                summary_df = pd.DataFrame([summary_stats])
                summary_path = output_dir / "evaluation_summary.csv"
                summary_df.to_csv(summary_path, index=False)
                print(f"汇总统计已保存到: {summary_path}")
                
                # 保存top和bottom结果
                top_n = successful.sort_values('iou', ascending=False).head(10)
                bottom_n = successful.sort_values('iou', ascending=True).head(10)
                
                top_path = output_dir / "top_10_iou.csv"
                bottom_path = output_dir / "bottom_10_iou.csv"
                
                top_n.to_csv(top_path, index=False)
                bottom_n.to_csv(bottom_path, index=False)
                
                print(f"Top 10 IOU已保存到: {top_path}")
                print(f"Bottom 10 IOU已保存到: {bottom_path}")
    
    def run_full_evaluation(self):
        """运行完整的评估流程"""
        print("=" * 60)
        print("Soma Mask评估工具")
        print("=" * 60)
        
        try:
            # 1. 匹配数据
            matched_df = self.match_all_data()
            
            if len(matched_df) == 0:
                print("没有匹配到数据，无法进行评估")
                return
            
            # 2. 加载并比较mask，计算IOU
            results_df = self.load_and_compare_masks(matched_df)
            
            if len(results_df) > 0:
                # 3. 保存结果
                self.save_evaluation_results(results_df)
                
                # 4. 可视化（可选）
                if False:
                    self.visualize_comparison(results_df, save_dir="./evaluation_results")
                
                # 5. 打印简要报告
                successful = results_df[results_df['match_success']]
                if len(successful) > 0:
                    print("\n" + "=" * 60)
                    print("评估报告摘要")
                    print("=" * 60)
                    print(f"总样本数: {len(results_df)}")
                    print(f"成功评估数: {len(successful)} ({len(successful)/len(results_df)*100:.1f}%)")
                    print(f"平均IOU: {successful['iou'].mean():.4f}")
                    print(f"平均Dice系数: {successful['dice'].mean():.4f}")
                    print(f"IOU范围: [{successful['iou'].min():.4f}, {successful['iou'].max():.4f}]")
                    print(f"最佳IOU: {successful['iou'].max():.4f} (ID: {successful.loc[successful['iou'].idxmax(), 'new_id']})")
                    print(f"最差IOU: {successful['iou'].min():.4f} (ID: {successful.loc[successful['iou'].idxmin(), 'new_id']})")
                    print("=" * 60)
            
            return results_df
            
        except Exception as e:
            print(f"评估过程出错: {e}")
            import traceback
            traceback.print_exc()
            return None




def evaluate():
    """主函数"""
    matcher = SomaDataMatcher()
    results = matcher.run_full_evaluation()
    
    if results is not None:
        print("\n评估完成！")
        print(f"结果已保存到 ./evaluation_results/ 目录")


if __name__ == "__main__":
    evaluate()
