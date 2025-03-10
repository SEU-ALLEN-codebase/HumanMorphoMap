##########################################################
#Author:          Yufeng Liu
#Create time:     2025-03-06
#Description:               
##########################################################
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

import matplotlib.pyplot as plt
import seaborn as sns

from config import REG2LOBE

def check_extreme_neurons(gf_file, meta_file, update_file, fn, display=False):
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

    # lobe-level distribution
    top5_meta['Brain lobe'] = [REG2LOBE[r] for r in top5_meta['brain_region']]

    dfs = df[display_feats]

    # Extract the top N% and lowest N% neurons acoording
    pn = 0.95
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

    if display:
        sns.set_theme(style='ticks', font_scale=1.8)
        # plot the distribution of top5 neurons with respect to different conditions
        # 1. age
        plt.figure(figsize=(6,6))
        sns.histplot(data=top5_meta, x='age', bins=20); plt.savefig('age.png', dpi=300); plt.close()
        # gender
        plt.figure(figsize=(6,6))
        sns.histplot(data=top5_meta, x='gender'); plt.savefig('gender.png', dpi=300); plt.close()
        # across lobes
        plt.figure(figsize=(6,6))
        sns.histplot(data=top5_meta, x='Brain lobe'); plt.savefig('lobe.png', dpi=300); plt.close()
        # across cell types
        
    

    #import ipdb; ipdb.set_trace()
    print()

def plot_stem_distributions(datasets):
    nstems = {}
    for dk, dv in datasets.items():
        print(f'--> {dk}')
        dfn = pd.read_csv(dv, index_col=0, low_memory=False)
        try:
            cur_nstems = dfn['N_stem']
        except KeyError:
            cur_nstems = dfn['Stems']
        nstems[dk] = pd.Series(cur_nstems.values)

    # plot stripplots
    df_stems = pd.DataFrame(nstems)
    
    sns.set_theme(style='ticks', font_scale=1.8)
    plt.figure(figsize=(6,6))
    sns.violinplot(df_stems, fill=False, cut=0)

    # perform statistical test
    for i in range(1, df_stems.shape[1]):
        group_a = df_stems.iloc[:,0]
        group_b = df_stems.iloc[:,i]
        u_stat, p_value = mannwhitneyu(group_a[~group_b.isna()], group_b[~group_b.isna()])
        print(u_stat, p_value)
    
    # draw the label y = 12
    ax = plt.gca()
    plt.axhline(y=12, color='red', linestyle='--', linewidth=2)
    ax.set_yticks(np.arange(2, 18, 2))
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.xaxis.set_tick_params(width=2)
    ax.yaxis.set_tick_params(width=2)

    #plt.xlabel('Dataset')
    #plt.ylabel('Number of subtrees')
    plt.savefig(f'stem_distributions_comp.png', dpi=300); plt.close()
    
    #import ipdb;ipdb.set_trace()
    print()



if __name__ == '__main__':
    gf_file = '/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta/l_measure_result.csv'
    meta_file = '/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta/meta.csv'
    update_file = '../meta/update_cellID_00001-50212.csv'

    if 1:
        fn = 'N_stem'
        check_extreme_neurons(gf_file, meta_file, update_file, fn, display=True)

    if 0:
        datasets = {
            'SEU-H8K': '/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta_filter/l_measure_result.csv',
            'DeKock': '../resources/public_data/DeKock/intermediates/gf_one_point_soma_150um.csv',
            'Allen': '../resources/public_data/allen_human_neuromorpho/intermediates/gf_one_point_soma_150um.csv',
            'H01-Skel': './data/lmeausre_pyramidal_dendrites_annotation.csv',
        }
        plot_stem_distributions(datasets)

