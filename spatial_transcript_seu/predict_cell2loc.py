##########################################################
#Author:          Yufeng Liu
#Create time:     2025-06-09
#Description:               
##########################################################
import os
import glob
import sys
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import gc
from re import sub
import cell2location
from scipy.sparse import csr_matrix

import matplotlib as mpl
from matplotlib import rcParams
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

# this line forces theano to use the GPU and should go before importing cell2location
os.environ["THEANO_FLAGS"] = 'device=cuda0,floatX=float32,force_device=True'
# if using the CPU uncomment this:
#os.environ["THEANO_FLAGS"] = 'device=cpu,floatX=float32,openmp=True,force_device=True


def read_and_qc(sample_name, path=''):
    r""" This function reads the data for one 10X spatial experiment into the anndata object.
    It also calculates QC metrics. Modify this function if required by your workflow.

    :param sample_name: Name of the sample
    :param path: path to data
    """
    adata = sc.read_visium(path,
                           count_file='filtered_feature_bc_matrix.h5', load_images=True)

    # get the sample id
    sample_id = [k for k in adata.uns['spatial'].keys()][0]
    adata.obs['sample_id'] = sample_id
    # replace the `sample_id` by the `P-coded id` in uns-spatial
    adata.uns['spatial'][sample_name] = adata.uns['spatial'].pop(sample_id)

    adata.obs['sample'] = sample_name
    adata.var['SYMBOL'] = adata.var_names
    adata.var.rename(columns={'gene_ids': 'ENSEMBL'}, inplace=True)
    adata.var_names = adata.var['ENSEMBL']
    adata.var.drop(columns='ENSEMBL', inplace=True)

    # Remove possible duplicate genes. To avoid possible errors, I would like to just keep one copy of them.
    #dup_genes = adata.var.SYMBOL[adata.var.SYMBOL.duplicated(keep='first')].unique()
    keep_genes =  ~adata.var.SYMBOL.duplicated(keep='first')
    adata = adata[:, keep_genes]

    # Calculate QC metrics
    adata.X = adata.X.toarray()
    sc.pp.calculate_qc_metrics(adata, inplace=True)
    adata.X = csr_matrix(adata.X)

    # add sample name to obs names
    adata.obs["sample"] = [str(i) for i in adata.obs['sample']]
    adata.obs_names = adata.obs["sample"] \
                          + '_' + adata.obs_names
    adata.obs.index.name = 'spot_id'

    # Match both 'mt-' and 'MT-'
    adata.var['mt'] = [(gene.startswith('mt-') | (gene.startswith('MT-'))) for gene in adata.var['SYMBOL']]
    adata.obs['mt_frac'] = adata[:, adata.var['mt'].tolist()].X.sum(1).A.squeeze()/adata.obs['total_counts']
    
    return adata

def select_slide(adata, ss, s_col='sample'):
    r""" This function selects the data for one slide from the spatial anndata object.

    :param adata: Anndata object with multiple spatial experiments
    :param s: name of selected experiment
    :param s_col: column in adata.obs listing experiment name for each location
    """

    slide = adata[adata.obs[s_col].isin([ss]), :]
    s_keys = list(slide.uns['spatial'].keys())
    s_spatial = np.array(s_keys)[[ss in k for k in s_keys]][0]

    slide.uns['spatial'] = {s_spatial: slide.uns['spatial'][s_spatial]}

    return slide


def plot_qc(adata, slides):
    # PLOT QC FOR EACH SAMPLE
    fig, axs = plt.subplots(len(slides), 4, figsize=(15, 4*len(slides)-4))
    for i, ss in enumerate(adata.obs['sample'].unique()):
        #fig.suptitle('Covariates for filtering')

        slide = select_slide(adata, ss)
        sns.distplot(slide.obs['total_counts'],
                     kde=False, ax = axs[i, 0])
        axs[i, 0].set_xlim(0, adata.obs['total_counts'].max())
        axs[i, 0].set_xlabel(f'total_counts | {ss}')

        sns.distplot(slide.obs['total_counts']\
                     [slide.obs['total_counts']<20000],
                     kde=False, bins=40, ax = axs[i, 1])
        axs[i, 1].set_xlim(0, 20000)
        axs[i, 1].set_xlabel(f'total_counts | {ss}')

        sns.distplot(slide.obs['n_genes_by_counts'],
                     kde=False, bins=60, ax = axs[i, 2])
        axs[i, 2].set_xlim(0, adata.obs['n_genes_by_counts'].max())
        axs[i, 2].set_xlabel(f'n_genes_by_counts | {ss}')

        sns.distplot(slide.obs['n_genes_by_counts']\
                     [slide.obs['n_genes_by_counts']<6000],
                     kde=False, bins=60, ax = axs[i, 3])
        axs[i, 3].set_xlim(0, 6000)
        axs[i, 3].set_xlabel(f'n_genes_by_counts | {ss}')

    plt.savefig('qc_visium.png', dpi=300)
    plt.close()


    if 1:
        # show inter-sample variability, using two samples
        adata_vis = adata.copy()
        adata_vis.raw = adata_vis
        vis_samples = adata.obs['sample'].unique()[:2]

        adata_vis_plt = adata_vis[adata_vis.obs['sample'].isin(vis_samples), :]

        # Log-transform (log(data + 1))
        sc.pp.log1p(adata_vis_plt)

        # Find highly variable genes within each sample
        adata_vis_plt.var['highly_variable'] = False
        for ss in adata_vis_plt.obs['sample'].unique():

            adata_vis_plt_1 = adata_vis_plt[adata_vis_plt.obs['sample'].isin([ss]), :]
            sc.pp.highly_variable_genes(
                    adata_vis_plt_1, min_mean=0.0125, max_mean=5, min_disp=0.5, n_top_genes=1000
            )

            hvg_list = list(adata_vis_plt_1.var_names[adata_vis_plt_1.var['highly_variable']])
            adata_vis_plt.var.loc[hvg_list, 'highly_variable'] = True

        # Scale the data ( (data - mean) / sd )
        sc.pp.scale(adata_vis_plt, max_value=10)
        # PCA, KNN construction, UMAP
        sc.tl.pca(adata_vis_plt, svd_solver='arpack', n_comps=40, use_highly_variable=True)
        sc.pp.neighbors(adata_vis_plt, n_neighbors = 20, n_pcs = 40, metric='cosine')
        sc.tl.umap(adata_vis_plt, min_dist = 0.3, spread = 1)

        with mpl.rc_context({'figure.figsize': [8, 8],
                             'axes.facecolor': 'white'}):
            # version inconsistency between scanpy and matplotlib
            sc.pl.umap(adata_vis_plt, color=['sample'], size=30,
                       color_map = 'RdPu', ncols = 1, #legend_loc='on data',
                       legend_fontsize=10)

            plt.savefig('inter-sample_variability.png', dpi=300)
            plt.close()


def load_scRNA_data(sc_data_file):
    adata_scrna_raw = sc.read(sc_data_file, backed='r')
    import ipdb; ipdb.set_trace()
    # Column name containing cell type annotations
    covariate_col_names = 'annotation_1'

    # Extract a pd.DataFrame with signatures from anndata object
    inf_aver = adata_snrna_raw.raw.var.copy()
    inf_aver = inf_aver.loc[:, [f'means_cov_effect_{covariate_col_names}_{i}' 
                            for i in adata_snrna_raw.obs[covariate_col_names].unique()]]
    inf_aver.columns = [sub(f'means_cov_effect_{covariate_col_names}_{i}', '', i) 
                        for i in adata_snrna_raw.obs[covariate_col_names].unique()]
    inf_aver = inf_aver.iloc[:, inf_aver.columns.argsort()]

    # normalise by average experiment scaling factor (corrects for sequencing depth)
    inf_aver = inf_aver * adata_snrna_raw.uns['regression_mod']['post_sample_means']['sample_scaling'].mean()

    with mpl.rc_context({'figure.figsize': [10, 10],
                         'axes.facecolor': 'white'}):
        sc.pl.umap(adata_snrna_raw, color=['annotation_1'], size=15,
                   color_map = 'RdPu', ncols = 1, legend_loc='on data',
                   legend_fontsize=10)
        plt.savefig('scrna.png', dpi=300)
        plt.close()

    del adata_scrna_raw
    gc.collect()


if __name__ == '__main__':
    sp_data_folder = '/PBshare/SEU-ALLEN/Users/WenYe/Human-Brain-ST-data/'
    results_folder = '/data2/lyf/data/transcriptomics/human_scRNA_2023_Science/cell2loc/'
    sc_data_file = '/data2/lyf/data/transcriptomics/human_scRNA_2023_Science/data/cortical_cells.h5ad'

    regression_model_output = 'RegressionGeneBackgroundCoverageTorch'
    reg_path = f'{results_folder}regression_model/{regression_model_output}/'

    DEBUG = True
    
    if 0:   
        # preprocessing the data
        sample_data = pd.read_csv('Visium_seu.csv')

        # Read the data into anndata objects
        slides = []
        n_slides = 0
        for sample_name,sample_path in zip(sample_data['sample_name'], sample_data['sample_path']):
            slides.append(read_and_qc(sample_name, path=sample_path))
            
            n_slides += 1
            if DEBUG and n_slides >= 2:
                break


        # Combine anndata objects together
        adata = slides[0].concatenate(
            slides[1:2] if DEBUG else slides[1:],
            batch_key="sample",
            uns_merge="unique",
            batch_categories=sample_data['sample_name'],
            index_unique=None
        )

        
        plot_qc(adata, slides)
        print()

    if 1:
        # load the scRNA data
        sndata = load_scRNA_data(sc_data_file)



