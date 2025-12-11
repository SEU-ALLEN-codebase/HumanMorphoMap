##########################################################
#Author:          Yufeng Liu
#Create time:     2025-12-11
#Description:               
##########################################################
import pickle
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from ml.feature_processing import standardize_features


def calculate_kth_distances(distance_matrix, k=5):
    """
    k-th distance using np.partition for efficiency
    """
    # 将压缩的距离矩阵转换为方阵
    full_dist_matrix = squareform(distance_matrix)
    n_cells = full_dist_matrix.shape[0]
    
    if n_cells <= k:
        return np.array([])
    
    # 使用partition找到第k+1小的元素（因为索引0是自身）
    # partition会在第k+1个位置放置第k+1小的元素，左边是更小的，右边是更大的
    partitioned = np.partition(full_dist_matrix, k, axis=1)
    
    # 获取每个细胞的第k个最近邻距离（第k+1小的元素）
    kth_distances = partitioned[:, k]
    
    return kth_distances

def estimate_RME(pdist_file='./data/pairwise_distance_in_slice.pkl'):
    """
    Estimate the radius of microenvironment, based on the 75 percentile of k-th nearest points

    Below are the output (20251211):
        总共有 311 个slices

        统计信息:
        有效slices数（细胞数>5）: 193
        符合条件的距离总数: 2682

        第5最近邻距离统计结果：
        25%分位数: 740.348714
        50%分位数（中位数）: 961.825022
        75%分位数: 1359.800067
    """

    # 1. 加载pickle文件
    with open(pdist_file, 'rb') as f:
        data = pickle.load(f)
    
    print(f"总共有 {len(data)} 个slices")
    
    # 2. 使用partition方法向量化计算
    all_kth_distances = []
    valid_slices_count = 0
    
    # 3. 遍历每个slice
    for slice_name, (cell_ids, distances) in data.items():
        n_cells = len(cell_ids)
        
        # 只有当slice中有超过5个细胞时才处理
        if n_cells > 5:
            valid_slices_count += 1
            
            # 使用partition方法计算第5个最近邻距离
            kth_distances = calculate_kth_distances(distances, k=5)
            
            # 添加到总列表中
            if len(kth_distances) > 0:
                all_kth_distances.append(kth_distances)
    
    # 4. 合并结果并计算统计量
    if all_kth_distances:
        all_kth_distances_array = np.concatenate(all_kth_distances)
        
        print(f"\n统计信息:")
        print(f"有效slices数（细胞数>5）: {valid_slices_count}")
        print(f"符合条件的距离总数: {len(all_kth_distances_array)}")
        
        # 使用向量化方式一次性计算所有分位数
        percentiles = np.percentile(all_kth_distances_array, [25, 50, 75])
        q25, q50, q75 = percentiles
        
        print(f"\n第5最近邻距离统计结果：")
        print(f"25%分位数: {q25:.6f}")
        print(f"50%分位数（中位数）: {q50:.6f}")
        print(f"75%分位数: {q75:.6f}")
        
        # 使用向量化计算其他统计量
        stats = {
            'min': np.min(all_kth_distances_array),
            'max': np.max(all_kth_distances_array),
            'mean': np.mean(all_kth_distances_array),
            'std': np.std(all_kth_distances_array)
        }
        
        print(f"\n补充统计信息：")
        for stat_name, stat_value in stats.items():
            print(f"{stat_name}: {stat_value:.6f}")
    else:
        print("没有找到符合条件的距离数据")

    return q75



class MicroEnvironmentFeatureCalculator:
    def __init__(self, distance_file, soma_feature_file, morpho_feature_file, R=1359.8):
        """
        初始化ME特征计算器
        
        参数:
        distance_file: 包含细胞间距离的pickle文件路径
        soma_feature_file: soma形态特征文件路径
        morpho_feature_file: 神经元形态特征文件路径
        R: Microenvironment半径，默认为1359.8
        """
        self.R = R
        self.distance_file = distance_file
        self.soma_feature_file = soma_feature_file
        self.morpho_feature_file = morpho_feature_file
        
        # 特征列名
        self.soma_features = ['fg_el_ratio', 'radius_z', 'radius_y', 'radius_x']
        # rotation_flat不作为特征
        
        # 存储数据
        self.slice_data = {}  # slice_name -> (cell_ids, distance_matrix)
        self.soma_features_df = None
        self.morpho_features_df = None
        self.cell_to_slice = {}  # cell_id -> slice_name
        
    def load_distance_data(self):
        """加载距离数据"""
        print("加载距离数据...")
        with open(self.distance_file, 'rb') as f:
            data = pickle.load(f)
        
        # 处理每个slice的数据
        for slice_name, slice_info in data.items():
            cell_ids, distances = slice_info
            
            # 将压缩的距离矩阵转换为完整的方阵
            n_cells = len(cell_ids)
            if n_cells > 1:
                distance_matrix = squareform(distances)
            else:
                distance_matrix = np.zeros((1, 1))
            
            self.slice_data[slice_name] = {
                'cell_ids': np.array(cell_ids),
                'distance_matrix': distance_matrix
            }
            
            # 建立细胞到slice的映射
            for cell_id in cell_ids:
                self.cell_to_slice[cell_id] = slice_name
        
        print(f"加载了 {len(data)} 个slices的距离数据")
        print(f"总共 {len(self.cell_to_slice)} 个细胞")
        
    def load_soma_features(self):
        """加载soma形态特征"""
        print("加载soma形态特征...")
        self.soma_features_df = pd.read_csv(self.soma_feature_file, index_col='cell_id')
        
        # 确保所有特征都存在
        for feature in self.soma_features:
            if feature not in self.soma_features_df.columns:
                print(f"警告: 特征 '{feature}' 不在soma特征文件中")
        
        print(f"加载了 {len(self.soma_features_df)} 个细胞的soma特征")
        print(f"特征维度: {len(self.soma_features)}")
        
    def load_morphology_features(self):
        """加载神经元形态特征"""
        print("加载神经元形态特征...")
        self.morpho_features_df = pd.read_csv(self.morpho_feature_file, index_col=0)
        self.morpho_features_df.index.name = 'cell_id'
        
        # 所有列都是特征
        self.morpho_features = self.morpho_features_df.columns.tolist()
        
        print(f"加载了 {len(self.morpho_features_df)} 个细胞的形态特征")
        print(f"特征维度: {len(self.morpho_features)}")
        
    def get_cell_features(self, cell_id, feature_type='soma', standardized=False):
        """获取细胞的特征向量"""
        if feature_type == 'soma':
            if standardized:
                df = self.soma_features_df_s
            else:
                df = self.soma_features_df
            fnames = self.soma_features
        else:
            if standardized:
                df = self.morpho_features_df_s
            else:
                df = self.morpho_features_df
            fnames = self.morpho_features

        if cell_id not in df.index:
            return None

        return df.loc[cell_id, fnames].astype(float).values
 
   
    def find_neighbors_within_R(self, cell_id, R):
        """找到距离细胞cell_id在R范围内的邻居（使用预计算的距离矩阵）"""
        if cell_id not in self.cell_to_slice:
            return [], []
        
        slice_name = self.cell_to_slice[cell_id]
        if slice_name not in self.slice_data:
            return [], []
        
        slice_info = self.slice_data[slice_name]
        cell_ids = slice_info['cell_ids']
        distance_matrix = slice_info['distance_matrix']
        
        # 找到目标细胞的索引
        target_idx = np.where(cell_ids == cell_id)[0]
        if len(target_idx) == 0:
            return [], []
        
        target_idx = target_idx[0]
        
        # 获取到所有细胞的距离
        distances = distance_matrix[target_idx]
        
        # 找到在R范围内的邻居（排除自身）
        neighbor_mask = distances <= R
        neighbor_mask[target_idx] = False  # 排除自身
        
        neighbor_indices = np.where(neighbor_mask)[0]
        neighbor_distances = distances[neighbor_indices]
        neighbor_cell_ids = cell_ids[neighbor_indices]
        
        return neighbor_cell_ids, neighbor_distances
    
    def compute_feature_similarity(self, target_f, neighbors_f):
        """
        计算特征相似度（使用欧氏距离的倒数作为相似度）
        返回相似度得分（越高表示越相似）
        """
        if len(neighbors_f) == 0:
            return np.array([])
        
        # 计算欧氏距离
        distances = np.linalg.norm(neighbors_f - target_f, axis=1)
        
        # 将距离转换为相似度（使用高斯核）
        similarities = np.exp(-distances**2 / 2)
        
        return similarities
    
    def compute_ME_feature(self, cell_id, feature_type='soma'):
        """
        计算细胞的ME特征
        
        参数:
        cell_id: 目标细胞ID
        feature_type: 特征类型，'soma' 或 'morphology'
        
        返回:
        ME特征向量
        """
        # 1. 获取目标细胞的特征
        target_feature = self.get_cell_features(cell_id, feature_type)
        target_feature_s = self.get_cell_features(cell_id, feature_type, standardized=True)
        
        # 2. 找到R范围内的邻居
        neighbor_cell_ids, neighbor_distances = self.find_neighbors_within_R(cell_id, self.R)
        
        if len(neighbor_cell_ids) == 0:
            # 没有邻居，返回原始特征
            return target_feature, False
        
        # 3. 获取邻居的特征
        neighbor_features = []
        neighbor_features_s = []
        valid_neighbor_indices = []
        
        for i, neighbor_id in enumerate(neighbor_cell_ids):
            neighbor_feature = self.get_cell_features(neighbor_id, feature_type)
            if neighbor_feature is None:
                continue

            neighbor_features.append(neighbor_feature)
            neighbor_feature_s = self.get_cell_features(neighbor_id, feature_type, standardized=True)
            neighbor_features_s.append(neighbor_feature_s)
            valid_neighbor_indices.append(i)
        
        if len(neighbor_features) == 0:
            return target_feature, False
        
        neighbor_features = np.array(neighbor_features)
        neighbor_features_s = np.array(neighbor_features_s)
        neighbor_distances = neighbor_distances[valid_neighbor_indices]
        neighbor_cell_ids = neighbor_cell_ids[valid_neighbor_indices]
        
        # 4. 根据邻居数量选择要使用的神经元
        N = len(neighbor_cell_ids)
        
        if N > 5:
            # 计算特征相似度
            similarities = self.compute_feature_similarity(target_feature_s, neighbor_features_s)
            
            # 选择最相似的5个邻居
            top_k = min(5, N)
            top_indices = np.argsort(similarities)[-top_k:]  # 相似度最高的k个
            
            selected_indices = top_indices
        else:
            # 使用所有邻居
            selected_indices = np.arange(N)
        
        # 5. 计算权重
        selected_distances = neighbor_distances[selected_indices]
        selected_features = neighbor_features[selected_indices]
        
        # 包括目标细胞自身
        all_distances = np.concatenate([[0], selected_distances])  # 自身距离为0
        all_features = np.vstack([target_feature.reshape(1, -1), selected_features])
        
        # 计算权重：exp(-d/R)
        weights = np.exp(-all_distances / self.R)
        weights = weights / np.sum(weights)  # 归一化
        
        # 6. 计算加权平均特征
        ME_feature = np.sum(all_features * weights.reshape(-1, 1), axis=0)
        
        return ME_feature, True
    
    def compute_all_ME_features(self, feature_type='soma'):
        """
        计算所有细胞的ME特征
        
        参数:
        feature_type: 特征类型，'soma' 或 'morphology'
        
        返回:
        包含cell_id和ME特征的DataFrame
        """
        print(f"\n计算 {feature_type} ME特征...")
        
        # 获取所有细胞ID
        if feature_type == 'soma':
            cell_ids = self.soma_features_df.index.tolist()
        else:  # morphology
            cell_ids = self.morpho_features_df.index.tolist()
        
        ME_features_list = []
        valid_cell_ids = []
        
        # 使用进度条
        n_updated = 0
        for cell_id in tqdm(cell_ids, desc=f"Processing {feature_type} cells"):
            ME_feature, updated = self.compute_ME_feature(cell_id, feature_type)
            n_updated += updated
            
            if ME_feature is not None:
                ME_features_list.append(ME_feature)
                valid_cell_ids.append(cell_id)
        print(f"{n_updated} neurons found neighbors...")
        
        # 创建DataFrame
        if feature_type == 'soma':
            feature_names = self.soma_features
        else:
            feature_names = self.morpho_features
        
        ME_features_array = np.array(ME_features_list)
        ME_features_df = pd.DataFrame(ME_features_array, columns=feature_names, index=valid_cell_ids)
        ME_features_df.index.name = 'cell_id'
        
        print(f"成功计算了 {len(valid_cell_ids)} 个细胞的{feature_type} ME特征")
        
        return ME_features_df
    
    def run(self):
        """运行完整的ME特征计算流程"""
        print("=" * 60)
        print("开始计算Microenvironment特征")
        print(f"ME半径 R = {self.R}")
        print("=" * 60)
        
        # 1. 加载所有数据
        self.load_distance_data()
        self.load_soma_features()
        self.load_morphology_features()

        # standardize the features for similarity estimation
        self.soma_features_df_s = standardize_features(
                self.soma_features_df, self.soma_features, inplace=False
        )
        self.morpho_features_df_s = standardize_features(
                self.morpho_features_df, self.morpho_features, inplace=False
        )
        
        # 2. 计算soma ME特征
        soma_ME_df = self.compute_all_ME_features('soma')
        
        # 3. 计算形态ME特征
        morpho_ME_df = self.compute_all_ME_features('morphology')
        
        # 4. 保存结果
        print("\n保存结果...")
        soma_ME_df.to_csv('soma_ME_features.csv')
        morpho_ME_df.to_csv('morphology_ME_features.csv')
        
        print(f"\n结果已保存:")
        print(f"- soma ME特征: soma_ME_features.csv ({len(soma_ME_df)} 个细胞)")
        print(f"- 形态 ME特征: morphology_ME_features.csv ({len(morpho_ME_df)} 个细胞)")
        
        # 5. 显示统计信息
        print(f"\n特征维度:")
        print(f"- soma特征: {len(self.soma_features)}")
        print(f"- 形态特征: {len(self.morpho_features)}")
        
        # 6. 统计邻居信息
        self.analyze_neighbor_statistics()
        
        return soma_ME_df, morpho_ME_df
    
    def analyze_neighbor_statistics(self):
        """分析邻居统计信息"""
        print("\n邻居统计信息:")
        
        neighbor_counts = []
        for cell_id in self.cell_to_slice.keys():
            neighbor_cell_ids, _ = self.find_neighbors_within_R(cell_id, self.R)
            neighbor_counts.append(len(neighbor_cell_ids))
        
        neighbor_counts = np.array(neighbor_counts)
        
        print(f"平均邻居数: {np.mean(neighbor_counts):.2f}")
        print(f"最小邻居数: {np.min(neighbor_counts)}")
        print(f"最大邻居数: {np.max(neighbor_counts)}")
        print(f"邻居数分布:")
        print(f"  0个邻居: {np.sum(neighbor_counts == 0)} 个细胞")
        print(f"  1-5个邻居: {np.sum((neighbor_counts >= 1) & (neighbor_counts <= 5))} 个细胞")
        print(f"  超过5个邻居: {np.sum(neighbor_counts > 5)} 个细胞")

# 主程序
if __name__ == "__main__":
    # 设置文件路径
    distance_file = "./data/pairwise_distance_in_slice.pkl"
    soma_feature_file = "../soma_morphology/data/fitted_soma_info_8.4k.csv"
    morpho_feature_file = "../h01-guided-reconstruction/auto8.4k_0510_resample1um_mergedBranches0712_crop100_renamed_noSomaDiameter.csv"
    
    # 创建计算器并运行
    calculator = MicroEnvironmentFeatureCalculator(
        distance_file=distance_file,
        soma_feature_file=soma_feature_file,
        morpho_feature_file=morpho_feature_file,
        R=1359.8
    )
    
    soma_ME_df, morpho_ME_df = calculator.run()
    
    # 显示前几行结果
    print("\nSoma ME特征示例 (前5个细胞):")
    print(soma_ME_df.head())
    
    print("\n形态 ME特征示例 (前5个细胞):")
    print(morpho_ME_df.head())
    
    # 验证特征一致性
    print("\n验证特征一致性:")
    print(f"Soma ME特征维度: {soma_ME_df.shape[1]}, 应与原始特征数 {len(calculator.soma_features)} 一致")
    print(f"形态 ME特征维度: {morpho_ME_df.shape[1]}, 应与原始特征数 {len(calculator.morpho_features)} 一致")

    
