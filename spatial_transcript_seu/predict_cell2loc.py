##########################################################
#Author:          Yufeng Liu
#Create time:     2025-06-09
#Description:               
##########################################################
import os
import glob
import sys
import random
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import gc
from re import sub
import cell2location
from cell2location.utils.filtering import filter_genes
from cell2location.models import Cell2location

from scipy.sparse import csr_matrix

import matplotlib as mpl
from matplotlib import rcParams
import matplotlib.pyplot as plt
import seaborn as sns

from config import PYRAMIDAL_SUPERCLUSTERS

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
    #keep_genes =  ~adata.var.SYMBOL.duplicated(keep='first')
    #adata = adata[:, keep_genes]

    if 0:
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
    #adata.var['mt'] = [(gene.startswith('mt-') | (gene.startswith('MT-'))) for gene in adata.var['SYMBOL']]
    #adata.obs['mt_frac'] = adata[:, adata.var['mt'].tolist()].X.sum(1).A.squeeze()/adata.obs['total_counts']
    # find mitochondria-encoded (MT) genes
    adata.var['MT_gene'] = [gene.startswith('MT-') | gene.startswith('mt-') for gene in adata.var['SYMBOL']]

    # remove MT genes for spatial mapping (keeping their counts in the object)
    adata.obsm['MT'] = adata[:, adata.var['MT_gene'].values].X.toarray()
    adata = adata[:, ~adata.var['MT_gene'].values]
    
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


def plot_vis_qc(adata, slides):
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


def load_scRNA_data(sc_data_file, visualize=False):
    adata = sc.read(sc_data_file)
    # Column name containing cell type annotations
    col_name = 'supercluster_term'
    
    if visualize:
        # Visualize the cell type distributions in the UMAP space
        with mpl.rc_context({'figure.figsize': [10, 10],
                             'axes.facecolor': 'white'}):
            # The sc.pl.umap recognizes only `X_umap` instead of `X_UMAP`
            adata.obsm['X_umap'] = adata.obsm['X_UMAP']
            sc.pl.umap(adata, color=col_name, size=15,
                       color_map = 'RdPu', ncols = 1, legend_loc='on data',
                       legend_fontsize=10)
            plt.savefig('scrna.png', dpi=300)
            plt.close()
 
    adata.var.index.name = 'ENSEMBL'

    if 0:
        # Randomly select a mini-set for debugging
        n_mini = 10000
        random.seed(1024)
        mini_indices = random.sample(range(adata.shape[0]), n_mini)
        adata_mini = adata[mini_indices].copy()
        adata_mini.write(f'{sc_data_file[:-5]}_mini{n_mini}.h5ad', compression=True)

    return adata


def run_cell2loc(sp_collection_file, sc_data_file, results_folder, debug=False):
    # preprocessing the spatial data
    sample_data = pd.read_csv(sp_collection_file)

    print(f'Read the data into anndata objects')
    slides = []
    n_slides = 0
    for sample_name,sample_path in zip(sample_data['sample_name'], sample_data['sample_path']):
        print(sample_name)
        slides.append(read_and_qc(sample_name, path=sample_path))
        
        n_slides += 1
        if debug and n_slides >= 2:
            break

    print('Combine anndata objects together')
    adata_sp = slides[0].concatenate(
        slides[1:2] if debug else slides[1:],
        batch_key="sample",
        uns_merge="unique",
        batch_categories=sample_data['sample_name'],
        index_unique=None
    )

    print('load the single-cell data')
    adata_sc = load_scRNA_data(sc_data_file)
    
    # training
    print('Train the model')
    import ipdb; ipdb.set_trace()
    #model = Cell2location(
    rmodel = cell2location.run_cell2location(

        # Single cell reference signatures as pd.DataFrame
        # (could also be data as anndata object for estimating signatures
        #  as cluster average expression - `sc_data=adata_snrna_raw`)
        sc_data=adata_sc,
        # Spatial data as anndata object
        sp_data=adata_sp,

        # the column in sc_data.obs that gives cluster idenitity of each cell
        summ_sc_data_args={'cluster_col': "supercluster_term",
                           'min_cells_per_cluster': 20,  # 确保稀有神经元亚型不被过滤
                          },

        train_args={'use_raw': True, # By default uses raw slots in both of the input datasets.
                    'n_iter': 40000, # Increase the number of iterations if needed (see QC below)

                    # Whe analysing the data that contains multiple experiments,
                    # cell2location automatically enters the mode which pools information across experiments
                    'sample_name_col': 'sample'}, # Column in sp_data.obs with experiment ID (see above)


        export_args={'path': results_folder, # path where to save results
                     'run_name_suffix': '', # optinal suffix to modify the name the run
                     'save_model': True,
                    },

        model_kwargs={ # Prior on the number of cells, cell types and co-located groups

                      'cell_number_prior': {
                          # - N - the expected number of cells per location:
                          'cells_per_spot': 8, # < - change this
                          # - A - the expected number of cell types per location (use default):
                          'factors_per_spot': 10,
                          # - Y - the expected number of co-located cell type groups per location (use default):
                          'combs_per_spot': 5
                      },

                       # Prior beliefs on the sensitivity of spatial technology:
                      'gene_level_prior':{
                          # Prior on the mean
                          'mean': 1/2,
                          # Prior on standard deviation,
                          # a good choice of this value should be at least 2 times lower that the mean
                          'sd': 1/6
                      }
        }
    )

    print(f'Model are saved to: {results_folder + runner["run_name"]}')
    import ipdb; ipdb.set_trace()
    print()


if __name__ == '__main__':
    sp_data_folder = '/PBshare/SEU-ALLEN/Users/WenYe/Human-Brain-ST-data/'
    sp_collection_file = 'Visium_seu.csv'
    results_folder = '/data2/lyf/data/transcriptomics/human_scRNA_2023_Science/cell2loc/'
    sc_data_file = '/data2/lyf/data/transcriptomics/human_scRNA_2023_Science/data/cortical_cells_rand30w_count5_perc0.15_nonzMean2.0.h5ad'

    regression_model_output = 'RegressionGeneBackgroundCoverageTorch'
    reg_path = f'{results_folder}regression_model/{regression_model_output}/'

    DEBUG = True
    
    if DEBUG:
        n_mini = 10000
        sc_data_file = f'{sc_data_file[:-5]}_mini{n_mini}.h5ad'
    
    run_cell2loc(sp_collection_file, sc_data_file, results_folder, debug=DEBUG)



