#!/usr/bin/env python

#================================================================
#   Copyright (C) 2023 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#   
#   Filename     : feat_sel.py
#   Author       : Yufeng Liu
#   Date         : 2023-02-11
#   Description  : pymrmr may could not compatitable with other packages, may require a new env
#
#================================================================
import os
import numpy as np
import pandas as pd
import pymrmr

from config import REG2LOBE, standardize_features

def load_features(ffile, meta_file):
    df = pd.read_csv(ffile, index_col=0, low_memory=False)
    meta = pd.read_csv(meta_file, index_col=2, low_memory=False, encoding='gbk')

    # standardize
    standardize_features(df, df.columns, inplace=True)
    # get the lobe
    col_reg = 'brain_region'
    regions = [REG2LOBE[r] for r in meta[col_reg]]
    df.insert(0, 'region', regions)
    # convert to classes
    uregions = np.unique(regions)
    print(f'#classes: {len(uregions)}')
    
    rdict = dict(zip(uregions, range(len(uregions))))
    rindices = [rdict[rname] for rname in regions]

    df.loc[:, 'region'] = rindices
    return df
    

def exec_mrmr(ffile, meta_file):
    df = load_features(ffile, meta_file)
    method='MIQ'
    topk=10
    feats = pymrmr.mRMR(df, method, topk)
    print(feats)
    return feats

if __name__ == '__main__':
    #ffile = '/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta_filter/l_measure_result.csv'
    ffile = '/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta_filter/swc_1um_cropped_150um_l_measure.csv'
    meta_file = '/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta_filter/meta.csv'
    exec_mrmr(ffile, meta_file)

