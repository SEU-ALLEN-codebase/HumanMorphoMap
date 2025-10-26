##########################################################
#Author:          Yufeng Liu
#Create time:     2025-10-25
#Description:               
##########################################################

import os
import glob

import numpy as np
import pandas as pd

from swc_handler import parse_swc
from morph_topo.morphology import Morphology, Topology

class BranchFeatures:

    def __init__(self, in_swc, epsilon=1e-8):
        self._load_data(in_swc)
        self.epsilon = epsilon
                
    def _load_data(self, in_swc):
        # load the data and initialize the topological tree
        print(f'--> Processing {os.path.split(in_swc)[-1]}')
        tree = parse_swc(in_swc)
        self.morph = Morphology(tree)
        topo_tree, self.seg_dict = self.morph.convert_to_topology_tree()
        self.topo = Topology(topo_tree)

        return True

    def calc_features(self, scale_r=1.0, length_thres=10):
        # initialize the coordinate dict for later use
        coords_dict = {}
        for node in self.morph.tree:
            coords_dict[node[0]] = np.array(node[2:5])

        # iterate over all features
        fseg_dict = {}
        sxyz = coords_dict[self.topo.idx_soma]
        srad = self.topo.pos_dict[self.topo.idx_soma][5]
        for sid, seg_ids in self.seg_dict.items():
            # Check if the branch is salient
            # level >= 2
            current = self.topo.pos_dict[sid]
            pid = current[6]
            if pid == -1:
                continue
            
            parent = self.topo.pos_dict[pid]
            if parent[6] == -1:
                continue

            pid2 = parent[6]
            parent2 = self.topo.pos_dict[pid2]

            # outside of the soma
            xyz_cur = coords_dict[sid]
            xyz_par = coords_dict[pid]
            dist_s_cur = np.linalg.norm(sxyz - xyz_cur)
            dist_s_par = np.linalg.norm(sxyz - xyz_par)
            if (dist_s_cur < srad*scale_r) or (dist_s_par < srad*scale_r):
                continue

            # length > length_thres
            euc_length = np.linalg.norm(xyz_cur - xyz_par)
            if euc_length < length_thres:
                continue

            # calculate the features
            seg_ids_all = [sid] + seg_ids + [pid]
            vn = xyz_cur - xyz_par  # parent-to-current
            vn /= (np.linalg.norm(vn) + self.epsilon)
            # radius
            radii = [self.morph.pos_dict[nid][5] for nid in seg_ids_all]
            radius = np.median(radii) if len(radii) >= 3 else np.mean(radii)
            max_radius = max(radii)
            
            # soma-parent-current orientation
            vsp = xyz_par - sxyz
            vsp /= (np.linalg.norm(vsp) + self.epsilon)
            cos_ang1 = np.dot(vsp, vn)
            ang_s1 = np.arccos(cos_ang1)
            
            # parent-current-branch angle
            xyz_par2 = coords_dict[pid2]
            vn1 = xyz_par - xyz_par2
            vn1 /= (np.linalg.norm(vn1) + self.epsilon)
            cos_ang2 = np.dot(vn1, vn)
            ang_s2 = np.arccos(cos_ang2)

            fseg_dict[sid] = (radius, max_radius, ang_s1, ang_s2)

        print(f'   {len(fseg_dict)} branches in {len(self.seg_dict)} branches')

        return fseg_dict

def dataset_features(in_swc_dir, out_feat_file):
    data_list = []
    num_processed = 0
    for in_swc in glob.glob(os.path.join(in_swc_dir, '*.swc')):
        swc_name = os.path.split(in_swc)[-1][:-4]
        bf = BranchFeatures(in_swc)
        fseg_dict = bf.calc_features()
        
        # iterate over all sids for a swc file
        for sid, features in fseg_dict.items():
            radius, max_radius, ang_s1, ang_s2 = features
            data_list.append({
                'swc_name': swc_name,
                'sid': sid,
                'radius': radius,
                'max_radius': max_radius,
                'ang_s1': ang_s1,
                'ang_s2': ang_s2
            })

        num_processed += 1
        if num_processed % 10 == 0:
            print(f'---> {num_processed}...')

    # 创建DataFrame
    df = pd.DataFrame(data_list)

    # 设置index为swc_name
    df.set_index('swc_name', inplace=True)

    df.to_csv(out_feat_file, index=True)

if __name__ == '__main__':
    if 0:   # calculate the branch features for datasets
        #in_swc_dir = 'data/H01_resample1um_prune25um'
        #out_feat_file = 'data/H01_resample1um_prune25um_branch_features.csv'
        in_swc_dir = 'data/auto8.4k_0510_resample1um_mergedBranches0712'
        out_feat_file = 'data/auto8.4k_0510_resample1um_mergedBranches0712_branch_features.csv'
        dataset_features(in_swc_dir, out_feat_file)

    if 1:
        # prune
        pass
