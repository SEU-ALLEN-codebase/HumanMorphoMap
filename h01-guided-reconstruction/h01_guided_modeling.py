##########################################################
#Author:          Yufeng Liu
#Create time:     2025-05-12
#Description:     
##########################################################

import os
import glob
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import seaborn as sns


from swc_handler import parse_swc, write_swc
from morph_topo import morphology


class StemFeatures:
    '''
    Features include:
        - median radius
        - median intensity
        - euclidean length
        - straightness
        - minimal angle between other stems
        - number of stems within 60 degrees
    '''

    def __init__(self, swcfile, max_nstems=12):
        tree = parse_swc(swcfile)
        self._get_basic_info(tree)
        self.max_nstems = max_nstems
        
    def _get_basic_info(self, tree):
        self.morph = morphology.Morphology(tree)
        topo_tree, self.seg_dict = self.morph.convert_to_topology_tree()
        # convert to topo tree
        self.topo = morphology.Topology(topo_tree)
        # get the soma-connecting points
        self.primary_pts = self.morph.child_dict[self.morph.idx_soma]
        # get the primary branches
        self.primary_branches = self.get_primary_branches()

        self.subtrees = self._get_subtrees()

    def _get_subtrees(self):
        """
        获取树结构中每个一级节点的所有子节点（子树）

        参数:
            morph: 树结构对象，包含child_dict和idx_soma属性

        返回:
            dict: 键为一级节点，值为该节点下所有子节点的集合（包括自己）
        """
        morph = self.morph
        if not hasattr(morph, 'child_dict') or not hasattr(morph, 'idx_soma'):
            raise ValueError("morph对象必须包含child_dict和idx_soma属性")

        subtrees = {}
        root = morph.idx_soma

        # 获取一级节点（根的直接子节点）
        first_level_nodes = morph.child_dict.get(root, [])

        # 对每个一级节点，收集其所有子节点
        for node in first_level_nodes:
            subtree_nodes = set()
            stack = [node]  # 使用栈实现DFS

            while stack:
                current = stack.pop()
                subtree_nodes.add(current)
                # 将当前节点的子节点加入栈
                for child in morph.child_dict.get(current, []):
                    stack.append(child)

            subtrees[node] = subtree_nodes

        subtrees = subtrees

        return subtrees

    def get_primary_branches(self):
        primary_branches = {}
        for b_terminal in self.topo.child_dict.get(self.morph.idx_soma):
            branch = self.seg_dict[b_terminal]
            primary_branches[b_terminal] = branch

        return primary_branches

    def _median_radius(self):
        radii_dict = {}
        for bt, bnodes in self.primary_branches.items():
            radii = [self.morph.pos_dict[bt][5]]
            for bn in bnodes:
                radii.append(self.morph.pos_dict[bn][5])
            median_radius = np.median(radii)
            radii_dict[bt] = median_radius

        return radii_dict

    def _median_intensity(self, imgfile):   # The function is intense, as it requires image loading, implement later
        img = load_image(imgfile)
        if imgfile[-3:].upper() == 'TIF':
            pass

        intensity_dict = {}
        for bt, bnodes in self.primary_branches.items():
            intensity = [self.morph.pos_dict[bt][5]]
            for bn in bnodes:
                radii.append(self.morph.pos_dict[bn][5])
            median_radius = np.median(radii)
            radii_dict[bt] = median_radius

        return radii_dict

    def _euclidean_length(self):
        soma_pos = np.array(self.morph.pos_dict[self.morph.idx_soma][2:5])
        euc_dict = {}
        for bt in self.primary_branches.keys():
            eudist = np.linalg.norm(soma_pos - np.array(self.morph.pos_dict[bt][2:5]))
            euc_dict[bt] = eudist

        return euc_dict
    
    def _path_length(self):
        path_dict = {}
        for bt, bnodes in self.primary_branches.items():
            coords = [self.morph.pos_dict[bt][2:5]]
            for bnode in bnodes:
                coords.append((self.morph.pos_dict[bnode][2:5]))
            coords.append((self.morph.pos_dict[self.morph.idx_soma][2:5]))
            # to array and estimate the path distances
            coords = np.array(coords)
            branch_vec = coords[1:] - coords[:-1]
            branch_lengths = np.linalg.norm(branch_vec, axis=1)
            # total length
            total_length = branch_lengths.sum()
            path_dict[bt] = total_length

        return path_dict

    def _straightness(self, euc_dict, path_dict):
        assert(len(euc_dict) == len(path_dict))
        str_dict = {}
        for k, euc_l in euc_dict.items():
            path_l = path_dict[k]
            straight = euc_l / path_l
            str_dict[k] = straight

        return str_dict

    def _angles(self, max_nodes=10):
        dfv = []
        soma_pos = np.array(self.morph.pos_dict[self.morph.idx_soma][2:5])
        for bt, bnodes in self.primary_branches.items():
            if len(bnodes) <= max_nodes:
                vi = np.array(self.morph.pos_dict[bt][2:5]) - soma_pos
            else:
                vi = np.array(self.morph.pos_dict[bnodes[max_nodes-1]][2:5]) - soma_pos
            dfv.append((bt, *vi))
        
        dfv = pd.DataFrame(dfv, columns=('stem_id', 'vx', 'vy', 'vz')).set_index('stem_id')
        # normalize
        dfvn = dfv.div(np.sqrt(np.sum(dfv**2, axis=1)), axis=0)

        # estimate similarity
        cosine_sim = dfvn.dot(dfvn.T)
        # exclude self
        np.fill_diagonal(cosine_sim.values, -1)
        # find out stem with minimal angle
        min_angle_values = cosine_sim.max(axis=1)
        # how many stems with small angles
        count_above_threshold =  (cosine_sim > 0.707).sum(axis=1)   # < 30 deg
        result = pd.DataFrame({
            'min_cos_similarity': min_angle_values,
            'count_above_0.707': count_above_threshold
        })
        
        return result

    
    def calc_features(self):
        rad_dict = self._median_radius()
        euc_dict = self._euclidean_length()
        path_dict = self._path_length()
        str_dict = self._straightness(euc_dict, path_dict)
        dfvn = self._angles()
        # merge to the dataframe
        dfvn['radius'] = pd.Series(rad_dict)  
        dfvn['euc_distance'] = pd.Series(euc_dict)
        dfvn['straightness'] = pd.Series(str_dict)  

        return dfvn


def calc_features_all(swc_dir, out_csv=None, visualize=True):
    if os.path.exists(out_csv):
        merged_df = pd.read_csv(out_csv, index_col=0)
    else:
        dfs = []
        for swcfile in glob.glob(os.path.join(swc_dir, '*.swc')):
            swc_name = os.path.split(swcfile)[-1][:-4]
            print(swc_name)
        
            sf = StemFeatures(swcfile)
            features = sf.calc_features()
            features['name'] = swc_name
            dfs.append(features)

        # merge all dataframes
        merged_df = pd.concat(dfs)

        merged_df.to_csv(out_csv, index=True)

    if visualize:
        """
        绘制DataFrame中所有特征的分布（每个特征一个子图）。
        """
        # 获取需要绘制的特征列
        features = [col for col in merged_df.columns if col not in ['stem_id', 'name']]
        n_features = len(features)
        
        # 计算子图的行数
        n_cols = 5
        n_rows = ((n_features - 1)// n_cols) + 1
        
        # 创建图形和子图网格
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*2, n_rows*2))
        axes = axes.flatten()  # 将axes展平为1D数组
        
        # 遍历每个特征并绘制分布
        xlims = {
            'min_cos_similarity': (-1, 1),
            'count_above_0.707': (0, 5),
            'radius': (0, 15),
            'euc_distance': (0, 200),
            'straightness': (0.5, 1)
        }
        for i, feature in enumerate(features):
            ax = axes[i]
            sns.histplot(merged_df[feature], ax=ax, kde=True, bins=30, 
                         stat='proportion', color='gray',
                         line_kws={'color': 'red'})  # 直方图+KDE曲线
            ax.set_title(f"{feature}", fontsize=12)
            ax.set_xlabel("")
            ax.set_xlim(*xlims[feature])

            if i != 0:
                ax.set_yticks([])
                ax.set_yticklabels([])
                ax.set_ylabel("")
            else:
                ax.set_ylabel("Proportion", fontsize=12)
        
        # 隐藏多余的空子图
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        plt.savefig(f'feature_distribution_{out_csv}.png', dpi=300)

    return merged_df
    
    
def calc





if __name__ == '__main__':

    dataset = 'h01'
    if dataset == 'h01':
        h01_dir = './data/H01_resample1um_prune25um'
        h01_feat_file = 'h01_stem_features.csv'
    else:
        h01_dir = './data/auto8.4k_0510_pruned_resample1um' 
        h01_feat_file = 'auto8.4k_0510_pruned_resamnple1um_stem_features.csv'
    
    calc_features_all(h01_dir, out_csv=h01_feat_file)

    

