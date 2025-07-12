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


# estimate the in-radius mask of these nodes
def find_first_outside_vec(coords, soma_pos, soma_radius):
    """
    使用向量化计算，找到第一个在胞体外的点
    - NOTE: the points (coords) in from soma to terminal
    """
    
    # 计算每个点到胞体中心的距离 (欧式距离)
    distances = np.linalg.norm(coords - soma_pos, axis=1)
    
    # 找到第一个距离 > soma_radius 的索引
    outside_mask = distances > soma_radius
    if np.any(outside_mask):
        return np.argmax(outside_mask)  # 第一个 True 的位置
    else:
        return len(coords)-1  # 所有点都在胞体内，使用最后的点


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
        self.primary_branches_r = self._get_primary_branches_r()
        self.subtrees = self._get_subtrees()

    def _get_primary_branches_r(self):
        primary_branches_r = {}
        for primary_pt in self.primary_pts:
            pt = primary_pt
            cur_branch = []
            while (pt in self.morph.child_dict) and (len(self.morph.child_dict[pt]) < 2):
                cur_branch.append(pt)
                pt = self.morph.child_dict[pt][0]
            else:
                cur_branch.append(pt)

            primary_branches_r[primary_pt] = cur_branch

        return primary_branches_r

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

    def _merge_and_remove_nodes(self, merged, to_remove_nodes):
        new_tree = []
        for node in self.morph.tree:
            nid = node[0]
            if nid in merged:
                new_node = list(node)
                new_node[-1] = merged[nid]
                new_node = tuple(new_node)
            else:
                if nid not in to_remove_nodes:
                    new_node = node
            new_tree.append(new_node)
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
        
    def get_primary_node_merges(self, itree, itree_partner, use_new_stem=False):
        """
        找出需要合并的一级节点对（将子节点较少节点的父节点改为重叠节点）
        
        返回:
            需要修改的节点ID: 新的父节点ID
        """
        morph = self.morph

        # find out the nodes within soma
        root_pos = np.array(morph.pos_dict[morph.idx_soma][2:5])  # 根节点位置
        root_radius = morph.pos_dict[morph.idx_soma][5]
        
        # get the coordinates of itree1
        pbcoords1 = np.array([morph.pos_dict[idx][2:5] for idx in self.primary_branches_r[itree]])
        pbcoords2 = np.array([morph.pos_dict[idx][2:5] for idx in self.primary_branches_r[itree_partner]])

        
        # 计算两个 branch 的第一个外部点
        idx_tree = find_first_outside_vec(pbcoords1, root_pos, root_radius)
        idx_partner = find_first_outside_vec(pbcoords2, root_pos, root_radius)

        if use_new_stem:
            # we create a virtual primary branch/ stem to better accommandate the points
            raise NotImplementedError
        else:
            tmp_idx = max(idx_partner-2, 0) # back to previous nodes, to alleviate large angle
            donor_id = self.primary_branches_r[itree][idx_tree]
            receptor_id = self.primary_branches_r[itree_partner][tmp_idx]

            to_remove_nodes = set(self.primary_branches_r[itree][:idx_tree])
            merges = {donor_id: receptor_id}

        #print(f"First point outside soma in branch1: index {idx_tree} / {len(pbcoords1)}")
        #print(f"First point outside soma in branch2: index {idx_partner} / {len(pbcoords2)}")

        return merges, to_remove_nodes


    def _merge_subtrees(self, itree, itree_partner):
        #print(f'#stems before and after merging: {len(self.subtrees)}', end=' --> ')
        merged, to_remove_nodes = self.get_primary_node_merges(itree, itree_partner)
        self._merge_and_remove_nodes(merged, to_remove_nodes)
        #print(f'{len(self.subtrees)}')

        return self.morph.tree
        

if __name__ == '__main__':
    swc_dir = '/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta_filter/swc_1um'
    #swcfile = os.path.join(swc_dir, '11620_P027_T01_-_S011_-_STG.R_MTG.R_R0613_HZY_20230301_RJ.swc')
    gf_file = 'auto8.4k_0510_resample1um.csv'
    output_dir = './data/auto8.4k_0510_pruned'
    
    prune_all(swc_dir, gf_file, output_dir=output_dir)

    

