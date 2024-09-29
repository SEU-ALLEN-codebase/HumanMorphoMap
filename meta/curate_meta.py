##########################################################
#Author:          Yufeng Liu
#Create time:     2024-09-29
#Description:               
##########################################################
import numpy as np
import pandas as pd


orig_csv = 'neuron_info_9060_utf8.csv'
new_meta = 'mouse_human_tableS1_reformatted.csv'
curated_csv = 'neuron_info_9060_utf8_curated0929.csv'

df = pd.read_csv(orig_csv, index_col=0)
nm = pd.read_csv(new_meta, index_col=0)
# inplace modification iteratively
for pid in nm.index:
    pindices = np.nonzero(df['病人编号'] == pid)[0]
    iregion = df.columns.values.tolist().index('脑区')
    df.iloc[pindices,iregion] = nm.loc[pid, 'Neuron Specimen Location']
df.to_csv(curated_csv)


