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

import sys
sys.path.append('../src')
from config import REG2LOBE

def check_extreme_neurons(gf_file, meta_file, update_file, fn, pn=0.95, display=False):

    ############# helper functions #############
    def histplot(top5_meta, meta, ftype='age'):
        plt.figure(figsize=(6,6))
        if ftype == 'age':
            gname = f'{ftype}_group'
            top5_meta[gname] = pd.cut(top5_meta[ftype], bins=range(18,85,10), right=False, 
                                     labels=[f'{i}-{i+9}' for i in range(18, 75, 10)])
            meta[gname] = pd.cut(meta[ftype], bins=range(18,85,10), right=False,
                                     labels=[f'{i}-{i+9}' for i in range(18, 75, 10)])
        else:
            gname = ftype
            
        ftype_dist = top5_meta.groupby(gname, observed=True).size() / meta.groupby(gname, observed=True).size() * 100
        ftype_dist.plot(kind='bar', color='lightsalmon')

        plt.ylabel('Percentage of large-trees')
        plt.xticks(rotation=0)
        plt.xlabel(gname.capitalize())

        ax = plt.gca()
        ax.spines['left'].set_linewidth(2)
        ax.spines['right'].set_linewidth(2)
        ax.spines['top'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        ax.xaxis.set_tick_params(width=2)
        ax.yaxis.set_tick_params(width=2)

        plt.subplots_adjust(left=0.15, bottom=0.15)
        plt.savefig(f'{ftype.split(" ")[-1]}.png', dpi=300); plt.close()

    ######### End of helper functions ##########

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
    meta['Brain lobe'] = [REG2LOBE[r] for r in meta['brain_region']]

    dfs = df[display_feats]

    # Extract the top N% and lowest N% neurons acoording
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

    import ipdb; ipdb.set_trace()

    #low5_names = low5_percent.index.values
    low5_values = low5_percent[fn].values
    low5_meta = meta.loc[low5_percent.index]
    low5_meta_new_ids = dfu.loc[low5_meta.index]

    if display:
        sns.set_theme(style='ticks', font_scale=1.7)
        # plot the distribution of top5 neurons with respect to different conditions
        histplot(top5_meta, meta, ftype='age')
        histplot(top5_meta, meta, ftype='gender')
        histplot(top5_meta, meta, ftype='Brain lobe')

        # TODO: across cell types
        
    #import ipdb; ipdb.set_trace()
    print()


def plot_stem_distributions(datasets, processed=False, use_h01_level1=True, meta_file=None):
    nstems = {}
    mtg_data = None
    for dk, dv in datasets.items():
        if (dk == 'ACT-H8K-O') and (not processed):
            continue
        
        print(f'--> {dk}')
        dfn = pd.read_csv(dv, index_col=0, low_memory=False)
        # whether use only level1 data of H01
        if dk == 'H01-Skel' and use_h01_level1:
            names_l1 = [name for name in dfn.index if name[-5:] != 'level']
            dfn = dfn.loc[names_l1]
            print(f'Number of neurons in H01: {dfn.shape[0]}')

        if (dk == 'ACT-H8K-O') and (meta_file is not None):
            meta = pd.read_csv(meta_file, index_col=2, low_memory=False, encoding='gbk')
            meta_n = meta.loc[dfn.index]
            mtg_mask = meta_n.brain_region.isin(['MTG', 'MTG.L', 'MTG.R'])
            dfn_mtg = dfn[mtg_mask]
            print(f'Number of MTG neurons: {len(dfn)}')

            try:
                mtg_stems = dfn_mtg['N_stem']
            except KeyError:
                mtg_stems = dfn_mtg['Stems']
            
            mtg_data = pd.Series(mtg_stems.values, name=f'{dk}-MTG')

        try:
            cur_nstems = dfn['N_stem']
        except KeyError:
            cur_nstems = dfn['Stems']

        
        nstems[dk] = pd.Series(cur_nstems.values)

    if mtg_data is not None:
        nstems['ACT-H8K-O-MTG'] = mtg_data

    # plot stripplots
    df_stems = pd.DataFrame(nstems)
    
    sns.set_theme(style='ticks', font_scale=1.5)
    if processed:
        figsize = (8,6)
    else:
        figsize = (6,6)
    plt.figure(figsize=figsize)

    sns.violinplot(df_stems, fill=False, cut=0)


    # perform statistical test
    for i in range(1, df_stems.shape[1]):
        group_a = df_stems.iloc[:,0]
        group_b = df_stems.iloc[:,i]
        group_bnz = group_b[~group_b.isna()]
        u_stat, p_value = mannwhitneyu(group_a[~group_b.isna()], group_bnz)
        print(df_stems.columns[i], len(group_bnz), u_stat, p_value)

        print(group_bnz.mean(), group_bnz.std(), group_bnz.max(), group_bnz.min(), '\n')
    
    # draw the label y = 12
    ax = plt.gca()
    plt.axhline(y=12, color='red', linestyle='--', linewidth=2)
    ax.set_yticks(np.arange(2, 20, 2))
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.xaxis.set_tick_params(width=2)
    ax.yaxis.set_tick_params(width=2)
    ax.set_ylim(0,21)

    #plt.xlabel('Dataset')
    #plt.ylabel('Number of subtrees')
    if processed:
        figname = f'stem_distributions_comp_processed.png'
    else:
        figname = f'stem_distributions_comp.png'
    plt.savefig(figname, dpi=300); plt.close()
    
    #import ipdb;ipdb.set_trace()
    print()



if __name__ == '__main__':
    gf_file = '/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta/l_measure_result.csv'
    meta_file = '/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta/meta.csv'
    update_file = '../meta/update_cellID_00001-50212.csv'

    if 0:
        fn = 'N_stem'
        check_extreme_neurons(gf_file, meta_file, update_file, fn, display=True)

    if 1:
        processed = True
        datasets = {
            'ACT-H8K': 'auto8.4k_0510_resample1um.csv',
            'DeKock': '../resources/public_data/DeKock/intermediates/gf_one_point_soma_150um.csv',
            'Allen': '../resources/public_data/allen_human_neuromorpho/intermediates/gf_one_point_soma_150um.csv',
            #'H01-Skel': '../data/lmeausre_pyramidal_dendrites_annotation.csv',
            'H01-Skel': 'gf_H01_resample1um_prune25um.csv',
            #'Mouse-9K': '../../../parcellation/BrainParcellation/microenviron/plotters/me_proj/data/lm_features_dendrites100.csv'
            'ACT-H8K-O': 'auto8.4k_0510_resample1um_mergedBranches0712.csv',
        }
        plot_stem_distributions(datasets, processed=processed, use_h01_level1=False, meta_file=meta_file)

