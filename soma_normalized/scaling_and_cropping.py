##########################################################
#Author:          Yufeng Liu
#Create time:     2025-12-03
#Description:               
##########################################################

import os
import glob
import numpy as np
import pandas as pd

from swc_handler import scale_swc, crop_spheric_from_soma, write_swc

def scale_crop_neurons(
    neuron_meta_file, swc_dir, out_dir,
    size_df = pd.DataFrame(
            [[6.859441023231381, 6.013118062406376], [6.711429953871266, 6.277403685704873]], 
            index=['pyramidal', 'nonpyramidal'], columns=['normal', 'infiltration'])
):
    # loading the neurons and their meta-informations
    meta = pd.read_csv(neuron_meta_file, index_col=0)

    # processing the data
    for idx, row in meta.iterrows():
        # get the swc 
        swc_file = glob.glob(os.path.join(swc_dir, f'{idx:05d}_*swc'))
        assert (len(swc_file) == 1)
        swc_file = swc_file[0]

        size_cur = size_df.loc[row.cell_type, row.tissue_type]
        size_ref = size_df.loc['pyramidal', 'normal']
        scale = size_cur / size_ref
        
        if scale != 1:
            # No need to do scaling if scale == 1
            scaled_tree = scale_swc(swc_file, 1/scale)
            # convert tree to dataframe
            df_tree = pd.DataFrame(
                    scaled_tree, columns=('#idx', 'type', 'x', 'y', 'z', 'r', 'pid')
            ).set_index('#idx')

            # the radius should also be scaled accordingly
            df_tree.loc[:, 'r'] = df_tree['r'] / scale
        else:
            df_tree = pd.read_csv(
                    swc_file, sep=' ', names=('type', 'x', 'y', 'z', 'r', 'pid'), 
                    comment='#', index_col=0
            )
            df_tree.index.name = '#idx'

        # crop
        cropped_tree = crop_spheric_from_soma(df_tree, radius=100.0)
        
        # write to file
        filename = os.path.split(swc_file)[-1]
        out_file = os.path.join(out_dir, filename)
        cropped_tree.to_csv(out_file, index=True, sep=' ')
    
        if idx % 10 == 0:
            print(f'<-- Processed {idx+1} neurons!')

def calc_lmfeatures(file_dir, outfile, nprocessors=8):
    from global_features import calc_global_features_from_folder

    calc_global_features_from_folder(file_dir, outfile, nprocessors=nprocessors, timeout=360)
    #reindexing
    df = pd.read_csv(outfile, index_col=0)
    indices = [int(idx.split('_')[0]) for idx in df.index]
    df.index = indices
    df = df.sort_index()
    df.to_csv(outfile, index=True)

    # renaming
    dfr = df.copy()
    df.index.name = 'ID'
    dfr.columns = [
         'N_node',
         'Soma_surface',
         'N_stem',
         'Number of Bifurcations',
         'Number of Branches',
         'Number of Tips',
         'Overall Width',
         'Overall Height',
         'Overall Depth',
         'Average Diameter',
         'Total Length',
         'Total Surface',
         'Total Volume',
         'Max Euclidean Distance',
         'Max Path Distance',
         'Max Branch Order',
         'Average Contraction',
         'Average Fragmentation',
         'Average Parent-daughter Ratio',
         'Average Bifurcation Angle Local',
         'Average Bifurcation Angle Remote',
         'Hausdorff Dimension'
    ]
    dfr.to_csv(f'{outfile[:-4]}_renamed.csv', index=True)

if __name__ == '__main__':
    neuron_meta_file = '../src/tissue_cell_meta_jsp.csv'
    swc_dir = '../h01-guided-reconstruction/data/auto8.4k_0510_resample1um_mergedBranches0712'
    out_dir = './data/scale_cropped'
    lm_file = './data/lmfeatures_scale_cropped.csv'
    
    #scale_crop_neurons(neuron_meta_file, swc_dir, out_dir)

    #### calculate global features
    calc_lmfeatures(out_dir, lm_file)


