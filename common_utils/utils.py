##########################################################
#Author:          Yufeng Liu
#Create time:     2025-12-03
#Description:               
##########################################################
import os
import glob

import numpy as np
import pandas as pd

from swc_handler import parse_swc
from morph_topo.morphology import Morphology, Topology

def calc_non_primary_avg_diameter(swc_file):
    tree = parse_swc(swc_file)
    morph = Morphology(tree)
    topo_tree, seg_dict = morph.convert_to_topology_tree()
    topo = Topology(topo_tree)

    # the all nodes on primary stems
    snodes = topo.child_dict[topo.idx_soma]
    dnodes = set([topo.idx_soma])
    for snode in snodes:
        dnodes.update(seg_dict[snode])

    radii = [node[5] for idx,node in morph.pos_dict.items() if idx not in dnodes]
    avg_d = np.mean(radii) * 2
    
    return avg_d

def calc_non_soma_avg_diameter(swc_file, scale=1.0):
    df_tree = pd.read_csv(swc_file, sep=' ', names=('type', 'x', 'y', 'z', 'r', 'pid'),
                     comment='#', index_col=0
    )

    srad = df_tree[df_tree.pid == -1]['r']
    sxyz = df_tree[df_tree.pid == -1][['x', 'y', 'z']].values.reshape(1,3)
    coords = df_tree[['x', 'y', 'z']].values
    distances = np.linalg.norm(coords - sxyz, axis=1)

    mask = distances > srad.iloc[0]*scale
     
    avg_d = df_tree[mask]['r'].mean() * 2
    
    return avg_d

def correct_lmeasures(lmfile, swc_dir, ex_type='Soma'):
    df = pd.read_csv(lmfile, index_col=0)
    dcol = 'AverageDiameter'
    if dcol not in df.columns:
        dcol = 'Average Diameter'

    if ex_type == 'Soma':
        fc = calc_non_soma_avg_diameter
    elif ex_type == 'Primary':
        fc = calc_non_primary_avg_diameter

    for cell_id, lmfeatures in df.iterrows():
        swc_file = glob.glob(os.path.join(swc_dir, f'{cell_id:05d}_*.swc'))[0]
        avg_d = fc(swc_file)
        print(f'--> {cell_id}: {df.loc[cell_id, dcol]:.4f}: {avg_d:.4f}')
        df.loc[cell_id, dcol] = avg_d


    # write to file
    out_file = f'{lmfile[:-4]}_no{ex_type}Diameter.csv'
    df.to_csv(out_file, index=True)
        

if __name__ == '__main__':
    lmfile = '../soma_normalized/data/lmfeatures_scale_cropped.csv'
    swc_dir = '../soma_normalized/data/scale_cropped'
    #lmfile = '../h01-guided-reconstruction/auto8.4k_0510_resample1um_mergedBranches0712.csv'
    #swc_dir = '../h01-guided-reconstruction/data/auto8.4k_0510_resample1um_mergedBranches0712'
    ex_type = 'Soma'
    
    correct_lmeasures(lmfile, swc_dir, ex_type=ex_type)

