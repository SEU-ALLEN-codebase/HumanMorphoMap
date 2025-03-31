##########################################################
#Author:          Yufeng Liu
#Create time:     2025-03-18
#Description:               
##########################################################
import os
import glob
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
import umap

from morph_topo.morphology import Morphology, Topology
from ml.feature_processing import standardize_features
from plotters.customized_plotters import sns_jointplot

from config import REG2LOBE


def morphological_clustering(gf_file):
    # load all neurons
    gfs = pd.read_csv(gf_file, index_col=0, low_memory=False)
    # standardize
    gfs_s = standardize_features(gfs, gfs.columns, inplace=False)
    # do clustering
    import ipdb; ipdb.set_trace()
    print()

   

if __name__ == '__main__':
    #cell_type_file = '../meta/cell_type_annot_rating_cls2_yufeng_unique.csv'
    cell_type_file = '../meta/cell_type_annotation_8.4K_all_CLS2_unique.csv'
    meta_file_neuron = '/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta_filter/meta.csv'
    gf_file = '/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta_filter/swc_1um_cropped_150um_l_measure.csv'
    ihc = 0

    if 1:
        morphological_clustering(gf_file)


