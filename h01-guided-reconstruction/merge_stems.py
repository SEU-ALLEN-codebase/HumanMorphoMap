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
    def __init__(self, tree):
        # load the file
        self._get_basic_info(tree)

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
        
    def get_primary_node_merges(self, itree, itree_partner):
        """
        找出需要合并的一级节点对（将子节点较少节点的父节点改为重叠节点）
        
        返回:
            需要修改的节点ID: 新的父节点ID
        """
        morph = self.morph

        import ipdb; ipdb.set_trace()
        
        # 1. 获取所有一级节点间的重叠关系
        overlaps = self.find_primary_node_overlaps(itree, itree_partner)
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


    def _merge_subtrees(self, itree, itree_partner):
        # find out its nearby subtree
        merged = self.get_primary_node_merges(itree, itree_partner)
        self._merge_and_reset_info(merged)

        return self.morph.tree
        

if __name__ == '__main__':
    swc_dir = '/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta_filter/swc_1um'
    #swcfile = os.path.join(swc_dir, '11620_P027_T01_-_S011_-_STG.R_MTG.R_R0613_HZY_20230301_RJ.swc')
    gf_file = 'auto8.4k_0510_resample1um.csv'
    output_dir = './data/auto8.4k_0510_pruned'
    
    prune_all(swc_dir, gf_file, output_dir=output_dir)

    

