##########################################################
#Author:          Yufeng Liu
#Create time:     2025-03-06
#Description:               
##########################################################
import numpy as np
import pandas as pd

gf_file = '/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta/l_measure_result.csv'
meta_file = '/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta/meta.csv'
update_file = '../meta/update_cellID_00001-50212.csv'

display_feats = [
    'N_stem',
    'Number of Branches',
    'Average Diameter',
    'Average Contraction',
    'Average Fragmentation',
    'Average Parent-daughter Ratio',
    'Average Bifurcation Angle Local',
    'Average Bifurcation Angle Remote',
]

df = pd.read_csv(gf_file, index_col=0, low_memory=False)
meta = pd.read_csv(meta_file, index_col=2, low_memory=False, encoding='gbk')
dfu = pd.read_csv(update_file, index_col='cell_id_backUp', low_memory=False)

dfs = df[display_feats]

# Extract the top N% and lowest N% neurons acoording
pn = 0.98
fn = 'Average Bifurcation Angle Remote'
thresh1 = df[fn].quantile(pn)
thresh2 = df[fn].quantile(1-pn)
print(f"Threshold for top and low {pn*100}%: {thresh1} and {thresh2}")

# check the files
top5_percent = df[df[fn] > thresh1]
low5_percent = df[df[fn] < thresh2]

#top5_names = top5_percent.index.values
top5_values = top5_percent[fn].values
top5_meta = meta.loc[top5_percent.index]
top5_meta_new_ids = dfu.loc[top5_meta.index]

#low5_names = low5_percent.index.values
low5_values = low5_percent[fn].values
low5_meta = meta.loc[low5_percent.index]
low5_meta_new_ids = dfu.loc[low5_meta.index]

import ipdb; ipdb.set_trace()
print()

