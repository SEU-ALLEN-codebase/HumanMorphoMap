##########################################################
#Author:          Yufeng Liu
#Create time:     2025-02-16
#Description:               
##########################################################
import os
import glob
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from ml.feature_processing import standardize_features

def distance_vs_similarity(meta_file_neuron, gf_file, somalist_dir, ihc=0):
    meta_n = pd.read_csv(meta_file_neuron, index_col=2, low_memory=False)
    gfs = pd.read_csv(gf_file, index_col=0)
    # extract the target neurons
    meta_n1 = meta_n.loc[gfs.index]
    meta_n1['slice'] = f'{meta_n1.patient_number}-{meta_n1.tissue_block_number}-{meta_n1.small_number}-{meta_n1.slice_number}'
    # convert the data type of immunohistochemistry
    meta_n1.loc[:,'immunohistochemistry'] = meta_n1.loc[:,'immunohistochemistry'].astype(int)

    uniq_slices, cnt_slices = np.unique(meta_n1['slice'], return_counts=True)
    # keep slices with multiple neurons
    keep_slices = uniq_slices[cnt_slices > 1]
    # estimate the distance and similarity
    meta_n2 = meta_n1[meta_n1['slice'].isin(keep_slices)]
    # standardize the features
    gfs_n2 = standardize_features(gfs.loc[meta_n2.index], gfs.columns, inplace=False)

    # iterative over slices
    for slice_i in keep_slices:
        # check if there is a image view
        somalist_files = list(glob.glob(os.path.join(f'{slice_i}*marker')))
        if len(somalist_files) == 0:
            continue

        indices_i = meta_n2.index[meta_n2['slice'] == slice_i]
        gfs_i = gfs_n2.loc[indices_i]
        ihcs = meta_n2.loc[indices_i].immunohistochemistry
        gfs_ihc0 = gfs_i[ihcs == 0]
        gfs_ihc1 = gfs_i[ihcs == 1]
        
        if ihc == 0:
            gfs_ihcs = [gfs_ihc0]
        elif ihc == 1:
            gfs_ihcs = [gfs_ihc1]
        else:
            gfs_ihcs = [gfs_ihc0, gfs_ihc1]

        for gfs_ihc in gfs_ihcs:
            if len(gfs_ihc) <= 1:
                continue
            # extract neurons in a image views
            
            

            import ipdb; ipdb.set_trace()
            print()


if __name__ == '__main__':
    meta_file_neuron = '../meta/1-50114.xlsx.csv'
    gf_file = '/data/kfchen/trace_ws/cropped_swc/proposed_1um_l_measure_total.csv'
    ihc = 2 # 0,1,2(all type)

    distance_vs_similarity(meta_file_neuron, gf_file, ihc=ihc)
    
    


