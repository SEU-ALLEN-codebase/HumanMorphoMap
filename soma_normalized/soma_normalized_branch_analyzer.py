##########################################################
#Author:          Yufeng Liu
#Create time:     2025-11-15
#Description:               
##########################################################
import os, glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import seaborn as sns

from swc_handler import parse_swc
from morph_topo.morphology import Morphology, Topology

class BranchAnalyzer:
    def __init__(self, in_swc, exclude_terminal=True):
        self.exclude_terminal = exclude_terminal
        self._load_data(in_swc)

    def _load_data(self, in_swc):
        # load the data and initialize the topological tree
        #print(f'--> Processing {os.path.split(in_swc)[-1]}')
        tree = parse_swc(in_swc)
        self.morph = Morphology(tree)
        topo_tree, self.seg_dict = self.morph.convert_to_topology_tree()
        self.topo = Topology(topo_tree)

        return True

    def levelwise_features(self):
        ###### level frequency
        if self.exclude_terminal:
            tmp_levels = [level for idx, level in self.topo.order_dict.items() if idx not in self.topo.tips]
            order_counter = Counter(tmp_levels)
        else:
            order_counter = Counter(self.topo.order_dict.values())
        # remove level zero
        order_counter.pop(0)
        
        ###### branch length
        bl_dict = {}
        for term_id, branch in self.seg_dict.items():
            if self.exclude_terminal and term_id in self.topo.tips:
                continue

            current = self.topo.pos_dict[term_id]
            pid = current[6]
            if pid == -1:
                continue

            parent = self.topo.pos_dict[pid]
            branch_ids_all = [term_id] + branch + [pid]
            coords = np.array([self.morph.pos_dict[nid][2:5] for nid in branch_ids_all])
            # calculate the path_distance
            branch_vec = coords[1:] - coords[:-1]
            branch_lengths = np.linalg.norm(branch_vec, axis=1)
            # total length
            total_length = branch_lengths.sum()
            # get the level
            level = self.topo.order_dict[term_id]
            
            try:
                bl_dict[level].append(total_length)
            except KeyError:
                bl_dict[level] = [total_length]

        # calculate the average branch length
        for level, bl in bl_dict.items():
            bl_dict[level] = np.mean(bl)

        return order_counter, bl_dict

def process_single_swc(in_swc):
    swc_name = os.path.split(in_swc)[-1]
    prefix = swc_name[6:-4]
    idx = int(swc_name[:5])
    ba = BranchAnalyzer(in_swc)
    order_counter, bl_dict = ba.levelwise_features()
    
    # 获取所有可能的level
    all_levels = set(order_counter.keys()) & set(bl_dict.keys())
    
    # 为每个level创建数据
    file_results = []
    for level in sorted(all_levels):
        order_count = order_counter.get(level, 0)
        branch_length = bl_dict.get(level, 0.0)
        
        file_results.append({
            'idx': idx,
            'level': level,
            'order_count': order_count,
            'branch_length': branch_length
        })
    
    return file_results
            
def calc_features(in_swc_dir, out_feat_file):
    results = []
    swc_files = glob.glob(os.path.join(in_swc_dir, '*.swc'))#[:20]

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(tqdm(
            executor.map(process_single_swc, swc_files), total=len(swc_files), 
            desc="Processing SWC files"))

    # 创建DataFrame
    df = pd.DataFrame([item for sublist in results for item in sublist])
    
    # 保存到文件
    df.to_csv(out_feat_file, index=False)
    
    return df

###### Analyze
def levelwise_lobes(feat_file, meta_file, cell_type_file, ihc=0, ctype=0):
    # Parsing the data
    feats = pd.read_csv(feat_file, index_col=0, low_memory=False)
    meta = pd.read_csv(meta_file, index_col=2, low_memory=False, encoding='gbk')

    meta_cols = ['brain_region', 'age', 'gender', 'immunohistochemistry']
    feats[meta_cols] = meta.loc[feats.index][meta_cols]

    # get the cell types and ihc status
    ctypes = pd.read_csv(ctype_file, index_col=0, low_memory=False)
    ctypes.index = [int(ctype[:5]) for ctype in ctypes.index]
    # ihc
    ihc_mask = feats.immunohistochemistry == ihc
    #ctype_mask = ctypes.loc[feats.index, 'CLS2'] == str(ctype)
    ctype_mask = (ctypes.loc[feats.index, 'CLS2'] == str(ctype)) & \
                 (ctypes.loc[feats.index, 'num_annotator'] >= 2)

    feats = feats[ihc_mask & ctype_mask]

    # 
    import sys
    sys.path.append('../src')
    from config import REG2LOBE
    from plotters.customized_plotters import sns_jointplot
    
    feats['lobe'] = feats.brain_region.map(REG2LOBE)
    
    for level in range(10):
        feats_l = feats[feats.level == level].drop(columns='level')
        if len(feats_l) < 10:
            continue
        
        figname = f'lobe_level{level}.png'
        sns_jointplot(feats_l[~feats_l.gender.isna()], x='order_count', 
                      y='branch_length', xlim=None, ylim=None, hue='lobe', 
                      out_fig=figname, markersize=8)


def levelwise_infiltration(feat_file, neuron_meta_file, ctype='pyramidal', figstr=''):
    # Parsing the data
    feats = pd.read_csv(feat_file, index_col=0, low_memory=False)

    # loading the neurons and their meta-informations
    meta = pd.read_csv(neuron_meta_file, index_col=0)

    # 
    import sys
    sys.path.append('../src')
    from config import REG2LOBE
    from plotters.customized_plotters import sns_jointplot
    
    feats[meta.columns] = meta.loc[feats.index]
    feats = feats[feats.cell_type == ctype]
    feats['lobe'] = feats.region.map(REG2LOBE)
    
    for level in range(4):
        feats_l = feats[feats.level == level].drop(columns='level')
        if len(feats_l) < 10:
            continue
        
        figname = f'infiltrationVSnormal_level{level}_{ctype}_{figstr}.png'
        sns_jointplot(feats_l, x='order_count', 
                      y='branch_length', xlim=None, ylim=None, hue='tissue_type', 
                      out_fig=figname, markersize=25, ms_scale=1.)


        
    
if __name__ == '__main__':

    dset = 'scale' # 'scale', 'orig_morph'
    in_swc_dir = f'./data/{dset}_cropped'
    out_feat_file = f'./data/{dset}_cropped_levelwise_features.csv'


    neuron_meta_file = '../src/tissue_cell_meta_jsp.csv'

    #calc_features(in_swc_dir, out_feat_file)

    #levelwise_lobes(out_feat_file, meta_file, ctype_file, ihc=0)

    sns.set_theme(style='ticks', font_scale=1.5)
    for ctype in ['pyramidal', 'nonpyramidal']:
        levelwise_infiltration(out_feat_file, neuron_meta_file, ctype, figstr=dset)

