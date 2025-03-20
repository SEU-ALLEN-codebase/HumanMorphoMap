##########################################################
#Author:          Yufeng Liu
#Create time:     2025-03-19
#Description:               
##########################################################
import os
import glob
import time
import numpy as np
import pandas as pd

from swc_handler import parse_swc, write_swc
from morph_topo.morphology import Morphology, Topology


def prune_terminal_branches(swcfile, outfile):
    # load the swc
    tree = pd.read_csv(swcfile, comment='#', sep=' ', index_col=False,
                       names=('#id', 'type', 'x', 'y', 'z', 'r', 'p'))
    # construct morphology
    morph = Morphology(tree.values)
    # analyze the topology
    topo_tree, seg_dict = morph.convert_to_topology_tree()
    # remove all nodes in terminal branches
    to_remove_nodes = np.concatenate([seg_dict[tip_id] for tip_id in morph.tips] + [list(morph.tips)]).astype(int)
    keep_tree = tree[~(tree['#id'].isin(to_remove_nodes))]
    # write file
    keep_tree_list = [list(row.values()) for row in keep_tree.to_dict('records')]
    write_swc(keep_tree_list, outfile)

def prune_terminal_branches_for_all(swc_dir, out_dir):
    cnt = 0
    t0 = time.time()
    for swcfile in glob.glob(os.path.join(swc_dir, '*swc')):
        filename = os.path.split(swcfile)[-1]
        outfile = os.path.join(out_dir, filename)
        if os.path.exists(outfile):
            continue

        prune_terminal_branches(swcfile, outfile)
        
        cnt += 1
        if cnt % 5 == 0:
            print(f'[{cnt}]: {time.time() - t0:.3f} seconds')

if __name__ == '__main__':
    swc_dir = '/data/kfchen/trace_ws/paper_trace_result/nnunet/proposed_9k/8_estimated_radius_swc'
    out_dir = './terminal_branches_pruned'
    prune_terminal_branches_for_all(swc_dir, out_dir)

