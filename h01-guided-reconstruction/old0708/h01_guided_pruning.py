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

from swc_handler import parse_swc, write_swc
from morph_topo import morphology

class SWCPruneByStems:
    def __init__(self, swcfile, swc_out, max_nstems=12):
        # load the file
        tree = parse_swc(swcfile)
        self._get_basic_info(tree)
        self.max_nstems = max_nstems

        self.swc_out = swc_out
    
    def _get_basic_info(self, tree):
        self.morph = morphology.Morphology(tree)
        topo_tree, seg_dict = self.morph.convert_to_topology_tree()
        # convert to topo tree
        self.topo = morphology.Topology(topo_tree)
        # get the soma-connecting points
        self.primary_pts = self.morph.child_dict[self.morph.idx_soma]
        self.subtrees = self._get_subtrees()

    def _merge_and_reset_info(self, merged):
        new_tree = []
        for node in self.morph.tree:
            nid = node[0]
            if nid in merged:
                new_node = list(node)
                new_node[-1] = merged[nid]
                new_node = tuple(new_node)
            else:
                new_node = node
            new_tree.append(new_node)
        # reset the objects
        self._get_basic_info(new_tree)

    def _remove_nodes(self, to_remove_nodes):
        new_tree = []
        for node in self.morph.tree:
            nid = node[0]
            if nid not in to_remove_nodes:
                new_tree.append(node)
        # reset the objects
        self._get_basic_info(new_tree)

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
        
    def find_primary_node_overlaps(self):
        """
        仅检查一级节点之间的半径重叠
        
        参数:
            morph: 树结构对象，需包含:
                   - child_dict: 子节点字典
                   - idx_soma: 根节点ID
                   - pos_dict: 节点位置和属性字典
        
        返回:
            dict: 键为一级节点ID，值为:
                  - None (如果没有重叠的一级节点)
                  - 重叠的一级节点ID (距离根节点最近的)
        """
        morph = self.morph
        # 获取所有一级节点
        first_level_nodes = morph.child_dict.get(morph.idx_soma, [])
        
        # 准备结果字典
        result = {node: None for node in first_level_nodes}
        
        # 提取一级节点的位置和半径信息
        node_info = {}
        root_pos = np.array(morph.pos_dict[morph.idx_soma][2:5])  # 根节点位置
        
        for node_id in first_level_nodes:
            attr = morph.pos_dict[node_id]
            node_pos = np.array([attr[2], attr[3], attr[4]])
            node_info[node_id] = {
                'pos': node_pos,
                'radius': attr[5],
                'dist_to_root': np.linalg.norm(node_pos - root_pos)
            }
        
        # 转换为数组以便向量化计算
        node_ids = np.array(first_level_nodes)
        node_positions = np.array([node_info[n]['pos'] for n in first_level_nodes])
        node_radii = np.array([node_info[n]['radius'] for n in first_level_nodes])
        node_dists = np.array([node_info[n]['dist_to_root'] for n in first_level_nodes])
        
        # 计算所有一级节点之间的距离矩阵
        dist_matrix = cdist(node_positions, node_positions)
        
        # 计算半径和矩阵（R_i + R_j）
        radius_sum_matrix = node_radii.reshape(-1, 1) + node_radii.reshape(1, -1)
        
        # 找出满足条件的节点对（排除自比较）
        for i, node1 in enumerate(first_level_nodes):
            # 找出与当前节点重叠的其他一级节点
            overlapping_mask = (dist_matrix[i] <= radius_sum_matrix[i]) & \
                              (dist_matrix[i] > 0)  # 排除自身
            
            overlapping_indices = np.where(overlapping_mask)[0]
            
            if len(overlapping_indices) > 0:
                # 找出距离根节点最近的
                nearest_idx = overlapping_indices[np.argmin(node_dists[overlapping_indices])]
                result[node1] = node_ids[nearest_idx]
        
        return result

    def get_primary_node_merges(self):
        """
        找出需要合并的一级节点对（将子节点较少节点的父节点改为重叠节点）
        
        返回:
            dict: {
                需要修改的节点ID: 新的父节点ID
            }
        """
        morph = self.morph
        
        # 1. 获取所有一级节点间的重叠关系
        overlaps = self.find_primary_node_overlaps()
        #print(sum([v is None for v in overlaps.values()]))
        
        # 2. 获取每个一级节点的子节点数量
        child_counts = {
            node: len(self.subtrees.get(node, []))
            for node in overlaps.keys()
        }
        
        # 3. 找出需要合并的节点对
        merges = {}
        processed = set()
        
        for node1, node2 in overlaps.items():
            if node2 is None or node1 in processed or node2 in processed:
                continue
            
            # 比较两个节点的子节点数量
            if child_counts[node1] < child_counts[node2]:
                merges[node1] = node2
                processed.update([node1, node2])
            elif child_counts[node1] > child_counts[node2]:
                merges[node2] = node1
                processed.update([node1, node2])
            else:
                # 如果子节点数量相同，选择距离根节点更近的作为父节点
                dist1 = np.linalg.norm(np.array(morph.pos_dict[node1][2:5]) - np.array(morph.pos_dict[morph.idx_soma][2:5]))
                dist2 = np.linalg.norm(np.array(morph.pos_dict[node2][2:5]) - np.array(morph.pos_dict[morph.idx_soma][2:5]))
                if dist1 < dist2:
                    merges[node2] = node1
                else:
                    merges[node1] = node2
                processed.update([node1, node2])

        ''' # I think greedy prune is enough, so I comment out this
        # 4. 处理与多个节点重叠的情况
        for node1, node2 in overlaps.items():
            if node2 is None or node1 in processed:
                continue
            
            # 收集所有重叠节点
            overlapping_nodes = [n for n, m in overlaps.items() 
                               if m == node1]
            overlapping_nodes.append(node2)
            
            # 选择子节点最多的作为父节点
            candidates = [(n, child_counts[n]) for n in overlapping_nodes]
            best_parent = max(candidates, key=lambda x: x[1])[0]
            
            for node in overlapping_nodes:
                if node != best_parent:
                    merges[node] = best_parent
                    processed.add(node)
        '''
        
        return merges


    def _prune_by_radius(self):
        '''
        Merge primary nodes within the radius range of nodes in other non-soma points
        '''
        nstems = len(self.subtrees)
        delta = 1024
        while (nstems > self.max_nstems) and (delta > 0):
            prev_nstems = nstems
            merged = self.get_primary_node_merges()
            self._merge_and_reset_info(merged)
            nstems = len(self.subtrees)
            delta = prev_nstems - nstems
            print(delta)
        
    def _prune_by_subtree_size(self):
        nstems = len(self.subtrees)
        if nstems <= self.max_nstems:
            return
        
        # sort the subtrees by number of nodes
        tree_sizes = {node: len(subtree) for node, subtree in self.subtrees.items()}
        sorted_tree_sizes = sorted(tree_sizes.items(), key=lambda x:x[1])
        to_remove_nodes = []
        for (stree_idx, stree_cnt) in sorted_tree_sizes[:-self.max_nstems]:
            to_remove_nodes.extend(self.subtrees[stree_idx])

        self._remove_nodes(set(to_remove_nodes))
        

    def run(self):
        self._prune_by_radius()
        self._prune_by_subtree_size()
        nstems = len(self.subtrees)
        assert (nstems <= self.max_nstems)
    
        # save to file
        write_swc(self.morph.tree, self.swc_out)


def prune_all(swc_dir, gf_file, output_dir):
    # Parsing the feature file
    gfs = pd.read_csv(gf_file, index_col=0)

    cnter = 0
    for swc_file in glob.glob(os.path.join(swc_dir, '*.swc')):
        cnter += 1
        
        swc_name = os.path.split(swc_file)[-1]
        swc_out = os.path.join(output_dir, swc_name)
        if os.path.exists(swc_out):
            continue

        # check the number of stems
        nstems = gfs.loc[swc_name[:-4], 'Stems']
        if nstems > 12:
            print(f'[{cnter}] #stems={nstems} for neuron: {swc_name}')
            # Prune or merge the excessive subtrees
            pruner = SWCPruneByStems(swc_file, swc_out)
            pruner.run()
        else:
            os.system(f'cp "{swc_file}" {output_dir}')



if __name__ == '__main__':
    swc_dir = '/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta_filter/swc_1um'
    #swcfile = os.path.join(swc_dir, '11620_P027_T01_-_S011_-_STG.R_MTG.R_R0613_HZY_20230301_RJ.swc')
    gf_file = 'auto8.4k_0510_resample1um.csv'
    output_dir = './data/auto8.4k_0510_pruned'
    
    prune_all(swc_dir, gf_file, output_dir=output_dir)

    

