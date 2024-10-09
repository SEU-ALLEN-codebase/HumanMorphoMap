##########################################################
#Author:          Yufeng Liu
#Create time:     2024-10-08
#Description:               
##########################################################
import os
import glob
import numpy as np
import pandas as pd

from swc_handler import parse_swc, trim_swc, crop_spheric_from_soma

CROP_RADIUS = 150 # in micron

def get_swc_dims(swcfile):
    df = pd.read_csv(swcfile, comment='#', sep=' ', index_col=0 ,
                     names=('id', 'type', 'x', 'y', 'z', 'r', 'pid'))
    xmin, ymin, zmin = np.min(df[['x', 'y', 'z']], axis=0)
    xmax, ymax, zmax = np.max(df[['x', 'y', 'z']], axis=0)
    return xmin, ymin, zmin, xmax, ymax, zmax



if __name__ == '__main__':
    if 0:
        in_dir = 'one_point_soma'
        for swcfile in glob.glob(os.path.join(in_dir, '*.swc')):
            xmin, ymin, zmin, xmax, ymax, zmax = get_swc_dims(swcfile)
            # 
            print(f'{os.path.split(swcfile)[-1][:-8]}: x=[{xmin:.1f}:{xmax:.1f}], y=[{ymin:.1f}:{ymax:.1f}], z=[{zmin:.1f}:{zmax:.1f}]')
        
    if 1:
        swcfile = 'one_point_soma/797978191_transformed.CNG.swc'
        df_tree = pd.read_csv(swcfile, comment='#', sep=' ', index_col=0 ,
                     names=('id', 'type', 'x', 'y', 'z', 'r', 'pid'))

        tree_out = crop_spheric_from_soma(df_tree, CROP_RADIUS)
        tree_out.to_csv('temp.swc', sep=' ', index=True, header=False)

        
