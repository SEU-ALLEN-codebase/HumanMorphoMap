##########################################################
#Author:          Yufeng Liu
#Create time:     2024-11-01
#Description:               
##########################################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from config import LOCAL_FEATS
from analyze_morpho_features import load_features

def regions_vs_layers(gf_file, meta_file, layer_file):
    df = load_features(gf_file, meta_file, min_neurons=0, standardize=False)
    regions = []
    for region in df.region:
        r = region.replace('.L', '').replace('.R', '')
        if '-near-' in r:
            r = r.split('-')[0]
        regions.append(r)

    df['region_nolr'] = regions
    layers = pd.read_csv(layer_file, index_col=0)

    # merge the data
    df['layer'] = layers.loc[df.index]
    # number distribution
    col_name = 'count(log10)'
    counts = df.groupby(['layer', 'region_nolr']).size().to_frame().rename(columns={0:col_name})
    log_counts = counts.apply(np.log10)
    g = sns.relplot(data=log_counts, x='layer', y='region_nolr', hue=col_name, size=col_name)
    plt.savefig('log_counts.png', dpi=300); plt.close()

    # feature visualization
    import ipdb; ipdb.set_trace()
    fgroup = df.groupby(['layer', 'region_nolr']).mean('N_stem')
    for col in fgroup.columns:
        df_i = fgroup[col].to_frame()
        g_i = sns.relplot(data=df_i, x='layer', y='region_nolr', hue=col, size=col)
        plt.savefig(f'feature_{col.replace(" ", "")}_region_layers.png', dpi=300); plt.close()

    print()



if __name__ == '__main__':
    gf_file = '/data/kfchen/trace_ws/paper_auto_human_neuron_recon/unified_recon_1um/ptls10.csv'
    meta_file = '../meta/neuron_info_9060_utf8_curated0929.csv'
    layer_file = '../resources/public_data/DeKock/predicted_layers_thresholding_outliners.csv'
    regions_vs_layers(gf_file, meta_file, layer_file)

