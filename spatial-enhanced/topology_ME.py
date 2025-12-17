##########################################################
#Author:          Yufeng Liu
#Create time:     2025-12-16
#Description:               
##########################################################
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler


class FeatureProcessor:
    def __init__(self, soma_feature_file, morpho_feature_file):
        self.soma_feature_columns = ['fg_el_ratio', 'radius_z', 'radius_y', 'radius_x']
        self.load_morphology_features(morpho_feature_file)
        self.load_soma_features(soma_feature_file)
        
        # 检查索引一致性
        self.common_cells = list(set(self.morpho_features_df.index) & 
                                set(self.soma_features_df.index))
        print(f"共同细胞数: {len(self.common_cells)}")

    def load_soma_features(self, soma_feature_file):
        print("加载soma形态特征...")
        self.soma_features_df = pd.read_csv(soma_feature_file, index_col='cell_id')

        print(f"加载了 {len(self.soma_features_df)} 个细胞的形态特征")
        print(f"特征维度: {len(self.soma_feature_columns)}")

    def load_morphology_features(self, morpho_feature_file):
        print("加载神经元形态特征...")
        self.morpho_features_df = pd.read_csv(morpho_feature_file, index_col=0)
        self.morpho_features_df.index.name = 'cell_id'
        self.morpho_features = self.morpho_features_df.columns.tolist()

        print(f"加载了 {len(self.morpho_features_df)} 个细胞的形态特征")
        print(f"特征维度: {len(self.morpho_features)}")

        
    def normalize_morpho_features(self):
        """归一化形态特征"""
        scaler = StandardScaler()
        morpho_features = self.morpho_features_df.values
        morpho_features_normalized = scaler.fit_transform(morpho_features)
        
        self.morpho_features_df_s = pd.DataFrame(
            morpho_features_normalized,
            index=self.morpho_features_df.index,
            columns=self.morpho_features_df.columns
        )
        
        # 保存scaler供后续使用
        self.morpho_scaler = scaler
        
        print(f"形态特征归一化完成，形状: {self.morpho_features_df_s.shape}")
        return self.morpho_features_df_s
    
    def find_k_nearest_neighbors(self, k=5, include_self=False):
        """
        为每个细胞找到k个最近邻细胞
        基于归一化的形态特征
        """
        if not hasattr(self, 'morpho_features_df_s'):
            self.normalize_morpho_features()
        
        # 获取归一化特征矩阵
        X = self.morpho_features_df_s.values
        cell_ids = self.morpho_features_df_s.index.tolist()
        
        # 计算距离矩阵
        print("计算距离矩阵...")
        dist_matrix = distance.squareform(distance.pdist(X, 'euclidean'))
        
        # 为每个细胞找到最近邻
        k_neighbors_dict = {}
        
        for i, cell_id in enumerate(cell_ids):
            # 获取当前细胞到所有其他细胞的距离
            distances = dist_matrix[i]
            
            # 获取最近邻的索引（排除自身）
            if include_self:
                # 包含自身，找k+1个最近邻（包含自身）
                nearest_indices = np.argsort(distances)[:k+1]
            else:
                # 不包含自身，找k个最近邻
                # 第一个总是自身（距离为0），所以从第二个开始
                nearest_indices = np.argsort(distances)[1:k+1]
            
            # 转换为cell_id
            nearest_cells = [cell_ids[idx] for idx in nearest_indices]
            k_neighbors_dict[cell_id] = nearest_cells
        
        self.k_neighbors = k_neighbors_dict
        self.k = k
        self.include_self = include_self
        
        print(f"为 {len(cell_ids)} 个细胞找到了 {k} 个最近邻")
        return k_neighbors_dict
    
    def compute_morpho_mean_features(self):
        """计算形态特征的平均值（基于最近邻）"""
        if not hasattr(self, 'k_neighbors'):
            self.find_k_nearest_neighbors()
        
        print("计算形态特征平均值...")
        
        # 为每个细胞计算最近邻的平均特征
        morpho_mean_features = []
        
        for cell_id, neighbor_ids in self.k_neighbors.items():
            # 获取邻居的特征
            if self.include_self:
                # 包含自身，邻居数 = k+1
                neighbor_features = self.morpho_features_df.loc[neighbor_ids]
            else:
                # 不包含自身，邻居数 = k
                neighbor_features = self.morpho_features_df.loc[neighbor_ids]
            
            # 计算平均值
            mean_features = neighbor_features.mean(axis=0)
            mean_features.name = cell_id
            morpho_mean_features.append(mean_features)
        
        # 创建DataFrame
        self.morpho_me = pd.DataFrame(morpho_mean_features)
        
        print(f"形态特征平均值计算完成，形状: {self.morpho_me.shape}")
        return self.morpho_me
    
    def compute_soma_mean_features(self):
        """计算soma特征的平均值（基于形态特征的最近邻）"""
        if not hasattr(self, 'k_neighbors'):
            self.find_k_nearest_neighbors()
        
        print("计算soma特征平均值...")
        
        # 检查soma特征列是否存在
        missing_columns = [col for col in self.soma_feature_columns 
                          if col not in self.soma_features_df.columns]
        
        if missing_columns:
            raise ValueError(f"以下soma特征列不存在: {missing_columns}")
        
        # 为每个细胞计算最近邻的soma特征平均值
        soma_mean_features = []
        cell_ids_in_soma = []
        
        for cell_id, neighbor_ids in self.k_neighbors.items():
            # 只处理在soma_features_df中存在的细胞
            if cell_id not in self.soma_features_df.index:
                continue
            
            # 找到在soma_features_df中存在的邻居
            valid_neighbor_ids = [nid for nid in neighbor_ids 
                                 if nid in self.soma_features_df.index]
            
            if not valid_neighbor_ids:
                continue
            
            # 获取邻居的soma特征（只取指定的4列）
            neighbor_soma_features = self.soma_features_df.loc[valid_neighbor_ids, 
                                                              self.soma_feature_columns]
            
            # 计算平均值
            mean_soma_features = neighbor_soma_features.mean(axis=0)
            mean_soma_features.name = cell_id
            soma_mean_features.append(mean_soma_features)
            cell_ids_in_soma.append(cell_id)
        
        # 创建DataFrame
        if soma_mean_features:
            self.soma_me = pd.DataFrame(soma_mean_features)
            print(f"Soma特征平均值计算完成，形状: {self.soma_me.shape}")
        else:
            self.soma_me = pd.DataFrame()
            print("警告：没有找到有效的soma特征数据")
        
        return self.soma_me
    
    def run_pipeline(self, k=5, include_self=False):
        """运行完整处理流程"""
        print("="*60)
        print("开始特征处理流程")
        print("="*60)
        
        # 1. 归一化形态特征
        self.normalize_morpho_features()
        
        # 2. 找到最近邻
        self.find_k_nearest_neighbors(k=k, include_self=include_self)
        
        # 3. 计算形态特征平均值
        morpho_me = self.compute_morpho_mean_features()
        
        # 4. 计算soma特征平均值
        soma_me = self.compute_soma_mean_features()
        
        print("="*60)
        print("处理完成！")
        print("="*60)
        print(f"形态特征平均值形状: {morpho_me.shape}")
        print(f"Soma特征平均值形状: {soma_me.shape}")
        
        return morpho_me, soma_me
    
    def visualize_neighbors(self, cell_id, n_examples=3):
        """可视化示例细胞的最近邻"""
        if not hasattr(self, 'k_neighbors'):
            print("请先运行find_k_nearest_neighbors")
            return
        
        if cell_id not in self.k_neighbors:
            print(f"细胞 {cell_id} 不存在")
            return
        
        neighbors = self.k_neighbors[cell_id]
        
        print(f"\n细胞 {cell_id} 的最近邻:")
        print(f"邻居数量: {len(neighbors)}")
        print(f"邻居ID: {neighbors}")
        
        # 显示原始特征对比
        print(f"\n原始特征对比:")
        print(f"{'特征':<30} {'自身':<15} {'邻居平均':<15}")
        
        self_cell_features = self.morpho_features_df.loc[cell_id]
        neighbor_features = self.morpho_features_df.loc[neighbors].mean()
        
        for feature in self.morpho_features_df.columns[:5]:  # 只显示前5个特征
            self_val = self_cell_features[feature]
            neighbor_avg = neighbor_features[feature]
            print(f"{feature:<30} {self_val:<15.4f} {neighbor_avg:<15.4f}")
        
        # 显示soma特征对比（如果存在）
        if cell_id in self.soma_features_df.index:
            print(f"\nSoma特征对比:")
            self_soma = self.soma_features_df.loc[cell_id, self.soma_feature_columns]
            
            # 获取有效的邻居soma特征
            valid_neighbors = [n for n in neighbors if n in self.soma_features_df.index]
            if valid_neighbors:
                neighbor_soma = self.soma_features_df.loc[valid_neighbors, 
                                                         self.soma_feature_columns].mean()
                
                print(f"{'特征':<20} {'自身':<15} {'邻居平均':<15}")
                for feature in self.soma_feature_columns:
                    self_val = self_soma[feature]
                    neighbor_avg = neighbor_soma[feature]
                    print(f"{feature:<20} {self_val:<15.4f} {neighbor_avg:<15.4f}")

# 使用示例
def main(morpho_feature_file, soma_feature_file, soma_feature_file_out, morpho_feature_file_out, include_self=True):
    # 创建处理器
    processor = FeatureProcessor(
        morpho_feature_file,
        soma_feature_file
    )
    
    # 运行处理流程
    morpho_me, soma_me = processor.run_pipeline(k=5, include_self=include_self)
    # keep all other features
    other_columns = [col for col in processor.soma_features_df.columns if col not in soma_me.columns]
    soma_me[other_columns] = processor.soma_features_df[other_columns]
    
    # Save to file
    soma_me.to_csv(soma_feature_file_out)
    morpho_me.to_csv(morpho_feature_file_out)

    # 查看结果
    print("\n形态特征平均值示例（前5行）:")
    print(morpho_me.head())
    
    print("\nSoma特征平均值示例（前5行）:")
    print(soma_me.head())
    
    # 可视化示例
    example_cell = morpho_me.index[0]  # 取第一个细胞
    processor.visualize_neighbors(example_cell)
    
    return processor, morpho_me, soma_me

if __name__ == "__main__":
    # 设置文件路径
    soma_feature_file = "../soma_morphology/data/fitted_soma_info_8.4k.csv"
    morpho_feature_file = "../h01-guided-reconstruction/auto8.4k_0510_resample1um_mergedBranches0712_crop100_renamed_noSomaDiameter.csv"
    #morpho_feature_file = '../soma_normalized/data/lmfeatures_scale_cropped_renamed_noSomaDiameter.csv'
    soma_feature_file_ME = './data/fitted_soma_info_8.4k_ME_notIncludeSelf.csv'
    morpho_feature_file_ME = './data/auto8.4k_0510_resample1um_mergedBranches0712_crop100_renamed_noSomaDiameter_ME_notIncludeSelf.csv'
    include_self = False

    # 运行主函数
    processor, morpho_me, soma_me = main(
                soma_feature_file, morpho_feature_file, 
                soma_feature_file_ME, morpho_feature_file_ME, 
                include_self=include_self
    )

   

