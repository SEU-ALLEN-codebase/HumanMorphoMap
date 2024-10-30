##########################################################
#Author:          Yufeng Liu
#Create time:     2024-10-08
#Description:               
##########################################################
import os
import sys
import glob
import numpy as np
import pandas as pd

from swc_handler import parse_swc, trim_swc, crop_spheric_from_soma
from global_features import calc_global_features_from_folder

sys.setrecursionlimit(100000)

CROP_RADIUS = 150 # in micron

def get_swc_dims(swcfile):
    df = pd.read_csv(swcfile, comment='#', sep=' ', index_col=0 ,
                     names=('id', 'type', 'x', 'y', 'z', 'r', 'pid'))
    xmin, ymin, zmin = np.min(df[['x', 'y', 'z']], axis=0)
    xmax, ymax, zmax = np.max(df[['x', 'y', 'z']], axis=0)
    return xmin, ymin, zmin, xmax, ymax, zmax

def spheric_cropping(in_dir, out_dir, radius, remove_axon=True):
    ninswc = 0
    for inswc in glob.glob(os.path.join(in_dir, '*.swc')):
        filename = os.path.split(inswc)[-1]
        outswc = os.path.join(out_dir, filename)
        ninswc += 1
        if ninswc % 20 == 0:
            print(filename)
        if os.path.exists(outswc):
            continue

        df_tree = pd.read_csv(inswc, comment='#', sep=' ', index_col=0 ,
                     names=('id', 'type', 'x', 'y', 'z', 'r', 'pid'))
        if remove_axon:
            df_tree = df_tree[df_tree.type != 2]

        # cropping
        tree_out = crop_spheric_from_soma(df_tree, CROP_RADIUS)
        
        tree_out.to_csv(outswc, sep=' ', index=True, header=False)

if __name__ == '__main__':
    if 0:
        in_dir = 'one_point_soma'
        for swcfile in glob.glob(os.path.join(in_dir, '*.swc')):
            xmin, ymin, zmin, xmax, ymax, zmax = get_swc_dims(swcfile)
            # 
            print(f'{os.path.split(swcfile)[-1][:-8]}: x=[{xmin:.1f}:{xmax:.1f}], y=[{ymin:.1f}:{ymax:.1f}], z=[{zmin:.1f}:{zmax:.1f}]')
        
    if 1:
        in_dir = 'merged'
        out_dir = 'merged_crop150um'
        remove_axon = True
        #spheric_cropping(in_dir, out_dir, radius=CROP_RADIUS, remove_axon=remove_axon)
        calc_global_features_from_folder(out_dir, outfile='gf_150um.csv')

        

        
