##########################################################
#Author:          Yufeng Liu
#Create time:     2025-03-01
#Description:               
##########################################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.stats import spearmanr, pearsonr, linregress


def cross_modality(merfile, morfile, cell_type='exc', smoothing=False):
    dfmer = pd.read_csv(merfile, index_col=0)
    dfmor = pd.read_csv(morfile, index_col=0)
    # merge the data
    df = dfmer.merge(dfmor, how='inner', on='A_bin_start')
    df = df[~(df['feature_distance_x'].isna() | df['feature_distance_y'].isna())]
    # filter by counts
    df = df[(df['count_x'] > 50) & (df['count_y'] > 50)]

    # smoothing
    df['smoothed_y'] = ndimage.convolve1d(df['feature_distance_y'], np.ones(3)/3., mode='reflect')
    
    # plotting
    sns.set_theme(style='ticks', font_scale=2.2)
    fig = plt.figure(figsize=(8,8))
    #g1 = sns.lineplot(x='feature_distance_x', y='smoothed_y', data=df, color='r', linewidth=3, alpha=0.5)
    #g2 = sns.scatterplot(x='feature_distance_x', y='smoothed_y', data=df, marker='o', color='black', s=100, alpha=0.75)

    if smoothing:
        ycol = 'smoothed_y'
    else:
        ycol = 'feature_distance_y'
    g = sns.regplot(x='feature_distance_x', y=ycol, data=df, 
                    scatter_kws={'s':100, 'alpha':0.75, 'color':'black'},
                    line_kws={'color':'red', 'alpha':0.5, 'linewidth':5})

    slope, intercept, r_value, p_value, std_err = linregress(df['feature_distance_x'], df[ycol])
    print(f'{slope:.3f}, {intercept:.3f}, {r_value:.3f}, {p_value:.3g}, {std_err:.3f}')
    #p_pearson = pearsonr(df['feature_distance_x'], df['smoothed_y'])
    #print(f'Pearson CC: {p_pearson.statistic:.3f}')

    if cell_type == 'exc':
        # estimate the linear region
        df_part = df[df['feature_distance_x'] < 8]
        slope2, intercept2, r_value2, p_value2, std_err2 = linregress(df_part['feature_distance_x'], df_part[ycol])
        print(f'{slope2:.3f}, {intercept2:.3f}, {r_value2:.3f}, {p_value2:.3g}, {std_err2:.3f}')
    
    plt.xlabel('Transcriptomic distance')
    plt.ylabel('Morphological distance')
    ax = plt.gca()
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.tick_params(width=2)
    # set the x&y range
    delta = 2.5
    xm = (df['feature_distance_x'].min() + df['feature_distance_x'].max()) / 2.
    plt.xlim(xm-delta/2, xm+delta/2)
    ym = (df['feature_distance_y'].min() + df['feature_distance_y'].max()) / 2.
    plt.ylim(ym-delta/2, ym+delta/2)
    
    plt.savefig(f'morphology_vs_merfish_{cell_type}.png', dpi=300)
    plt.close()
    
    #import ipdb; ipdb.set_trace()
    print()

if __name__ == '__main__':
    for cell_type in ['exc', 'inh']:
        if cell_type == 'exc':
            merfile = 'EXC_merfish_MTG_mean.csv'
            morfile = 'euc_feat_distances_pyramidal_nannot2_ihc0_mean.csv'
        elif cell_type == 'inh':
            merfile = 'INC_merfish_MTG_mean.csv'
            morfile = 'euc_feat_distances_nonpyramidal_nannot2_ihc0_mean.csv'
        cross_modality(merfile, morfile, cell_type=cell_type)


