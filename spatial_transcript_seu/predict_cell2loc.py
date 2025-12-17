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
import torch
import cell2location
from cell2location.utils.filtering import filter_genes
from cell2location.models import RegressionModel
from cell2location.plt import plot_spatial

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

torch.set_float32_matmul_precision('medium')


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
    :param ss: name of selected experiment
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


def run_cell2loc(st_collection_file, sc_data_file, keep_all_genes, sample_name, results_folder, debug=False):
    # Initialize the path
    regression_model_output = 'model_SC'
    st_model_output = 'SpatialModel'
    reg_path = f'{results_folder}{sample_name}/{regression_model_output}'
    st_path = f'{results_folder}{sample_name}/{st_model_output}'

    os.makedirs(reg_path, exist_ok=True)
    os.makedirs(st_path, exist_ok=True)

    adata_file_sc = f"{reg_path}/sc.h5ad"

    adata_file_st = f"{st_path}/st.h5ad"
    if keep_all_genes:
        adata_file_st = f"{st_path}/st_allgenes.h5ad"

    if os.path.exists(adata_file_st):
        adata_sc = sc.read_h5ad(adata_file_sc)
        adata_st = sc.read_h5ad(adata_file_st)
        mod = cell2location.models.Cell2location.load(st_path, adata_st)
    else:
        if not os.path.exists(adata_file_sc):
            ################# Single-cell data and model training #################
            print('load the single-cell data')
            adata_sc = load_scRNA_data(sc_data_file)
            
            # training
            print('Train the model')
            # prepare anndate for the regression model
            RegressionModel.setup_anndata(adata=adata_sc, labels_key='supercluster_term')
            # create the regression model
            mod = RegressionModel(adata_sc)
            # view anndata_setup as a sanity check
            # mod.view_anndata_setup()

            # train the model
            #train_epochs_reg = max(5, int(-0.0012 * adata_sc.shape[0] + 412))    # fitted from ([1w,30w], [400,50])
            #train_epochs_reg = max(5, int(-0.0013 * adata_sc.shape[0] + 413))    # fitted from ([1w,30w], [400,20])
            train_epochs_reg = 200  # 200 should be enough for region-wise estimation
            print(f'SC training epochs: {train_epochs_reg}')
            mod.train(max_epochs=train_epochs_reg)
            # plot the loss_evaluation
            sns.lineplot(mod.history['elbo_train'])
            plt.savefig('train_loss_sc.png', dpi=300); plt.close()

            # export the estimated cell abundance (summary of the posterior distribution)
            mod.export_posterior(adata_sc, sample_kwargs={'num_samples': 1000, 'batch_size':2500})
            # save the model
            mod.save(reg_path, overwrite=True)
            # save anndate object with results
            adata_sc.write(adata_file_sc)

        else:
            adata_sc = sc.read_h5ad(adata_file_sc)
            mod = cell2location.models.RegressionModel.load(reg_path, adata_sc)

        
        # export estimated expression in each cluster
        if 'means_per_cluster_mu_fg' in adata_sc.varm.keys():
            inf_aver = adata_sc.varm['means_per_cluster_mu_fg'][[f'means_per_cluster_mu_fg_{i}'
                                            for i in adata_sc.uns['mod']['factor_names']]].copy()
        else:
            inf_aver = adata_sc.var[[f'means_per_cluster_mu_fg_{i}'
                                            for i in adata_sc.uns['mod']['factor_names']]].copy()
        inf_aver.columns = adata_sc.uns['mod']['factor_names']
        inf_aver.iloc[0:5, 0:5]


        ############## preprocessing the spatial data #################
        sample_data = pd.read_csv(st_collection_file)
        adata_st = read_and_qc(sample_name, path=sample_data['sample_path'][sample_data['sample_name'] == sample_name].iloc[0])

       
        ############## Prediction #################
        # find shared genes and subset both anndata and reference signatures
        intersect = np.intersect1d(adata_st.var_names, inf_aver.index)
        adata_st_i = adata_st[:, intersect].copy()
        inf_aver = inf_aver.loc[intersect, :].copy()

        # prepare anndata for cell2location model
        cell2location.models.Cell2location.setup_anndata(adata=adata_st_i)#, batch_key="sample")

        ## create and train the model
        mod = cell2location.models.Cell2location(
            adata_st_i, cell_state_df=inf_aver,
            # the expected average cell abundance: tissue-dependent
            # hyper-prior which can be estimated from paired histology:
            N_cells_per_location=8,
            # hyperparameter controlling normalisation of
            # within-experiment variation in RNA detection:
            detection_alpha=20
        )
        #mod.view_anndata_setup() 
        
        mod.train(max_epochs=10000,  # 30000
              # train using full data (batch_size=None)
              batch_size=None,
              # use all data points in training because
              # we need to estimate cell abundance at all locations
              train_size=1,
        )

        # plot the ST data training
        sns.lineplot(mod.history['elbo_train'])
        plt.savefig('train_loss_st.png', dpi=300); plt.close()

        # In this section, we export the estimated cell abundance (summary of the posterior distribution).
        if keep_all_genes:
            # Use the original section
            adata_st = mod.export_posterior(
                adata_st, sample_kwargs={'num_samples': 1000, 'batch_size': mod.adata.n_obs}
            )
            # Save anndata object with results
            adata_st.write(adata_file_st)
        else:
            # Use the original section
            adata_st_i = mod.export_posterior(
                adata_st_i, sample_kwargs={'num_samples': 1000, 'batch_size': mod.adata.n_obs}
            )
            # Save anndata object with results
            adata_st_i.write(adata_file_st)

        print('Save SP model...')
        mod.save(st_path, overwrite=True)

        #fig = mod.plot_spatial_QC_across_batches()
        #fig.savefig('spatial_qc.png', dpi=300)
    
    # Now we use cell2location plotter that allows showing multiple cell types in one panel
    adata_st.obs[adata_st.uns['mod']['factor_names']] = adata_st.obsm['q05_cell_abundance_w_sf']
    # select up to 5 clusters
    topk = 5
    all_labels, label_cnt = np.unique(adata_sc.obs['supercluster_term'], return_counts=True)
    cnt_thr = np.sort(label_cnt)[::-1][topk]
    clust_labels = all_labels[label_cnt > cnt_thr]
    clust_col = ['' + str(i) for i in clust_labels] # in case column names differ from labels

    print(f'Plotting for sample: {sample_name}')
    # plot in spatial coordinate
    with mpl.rc_context({'figure.figsize': (15, 15)}):
        fig = plot_spatial(
            adata=adata_st,
            # labels to show on a plot
            color=clust_col, labels=clust_labels,
            show_img=True,
            # 'fast' (white background) or 'dark_background'
            style='fast',
            # limit color scale at 99.2% quantile of cell abundance
            max_color_quantile=0.992,
            # size of locations (adjust depending on figure size)
            circle_diameter=6,
            colorbar_position='right'
        )

        fig.savefig(f'{sample_name}_top{topk}labels.png', dpi=300)

 
def get_pyramidal_cells(predicted_st_file, exist_thr=0.5):   
    # load the cell predictions
    adata_st = sc.read_h5ad(predicted_st_file)
    # import and rename
    cell_names = adata_st.uns['mod']['factor_names']
    adata_st.obs[cell_names] = adata_st.obsm['q05_cell_abundance_w_sf']
    # using 
    mcells05 = adata_st.obs[cell_names].copy()
    # thresholding to exclude low-probability cells
    mcells05[mcells05 < exist_thr] = 0

    # map to pyramidal and nonpyramidal
    # 1. 筛选出存在于DataFrame中的金字塔神经元列（避免KeyError）
    existing_pyramidal_cols = [col for col in PYRAMIDAL_SUPERCLUSTERS if col in mcells05.columns]

    # 2. 创建新的pyramidal列（对存在的列求和）
    mcells05['pyramidal'] = mcells05[existing_pyramidal_cols].sum(axis=1)

    # 3. 创建nonpyramidal列（所有非金字塔神经元列的和）
    non_pyramidal_cols = [col for col in mcells05.columns if col not in PYRAMIDAL_SUPERCLUSTERS and col != 'pyramidal']
    mcells05['nonpyramidal'] = mcells05[non_pyramidal_cols].sum(axis=1)

    # 筛选符合条件的spots
    filtered_rows_p = mcells05[
        (mcells05['pyramidal'] > exist_thr) & 
        (mcells05['pyramidal'] > 2 * mcells05['nonpyramidal'])
    ]
    filtered_rows_np = mcells05[
        (mcells05['nonpyramidal'] > exist_thr) & 
        (mcells05['nonpyramidal'] > 2 * mcells05['pyramidal'])
    ]
    mcells05['cell_code'] = -1
    mcells05.loc[filtered_rows_p.index, 'cell_code'] = 1
    mcells05.loc[filtered_rows_np.index, 'cell_code'] = 0

    # write to file
    out_path = os.path.split(os.path.split(predicted_st_file)[0])[0]
    out_file = os.path.join(out_path, 'predicted_cell_types.csv')
    mcells05.to_csv(out_file)
    


if __name__ == '__main__':
    st_collection_file = 'Visium_seu.csv'
    results_folder = './cell2loc/'
    keep_all_genes = True

    sample_dict = {
        'P00083': './data/scdata/sc_A44-A45_count5_perc0.15_nonzMean2.0.h5ad',
        'P00066': './data/scdata/sc_A5-A7+A19_count5_perc0.15_nonzMean2.0.h5ad',
        'P00065_0': './data/scdata/sc_A5-A7+A19_count5_perc0.15_nonzMean2.0.h5ad',
        'P00065_500': './data/scdata/sc_A5-A7+A19_count5_perc0.15_nonzMean2.0.h5ad'
    }
    sample_name = 'P00065_500'
    sc_data_file = sample_dict[sample_name]

    DEBUG = False
    
    if DEBUG:   # debug only
        n_mini = 10000
        #sc_data_file = f'{sc_data_file[:-5]}_mini{n_mini}.h5ad'
        sc_data_file = 'data/scdata/cortical_cells_rand30w_count5_perc0.15_nonzMean2.0_mini10000.h5ad'
    
    if 1:
        # Train the model and predict the posterior cell types
        run_cell2loc(st_collection_file, sc_data_file, keep_all_genes, sample_name, results_folder, debug=DEBUG)

    if 0:
        # extract spots mostly with pyramdial cells
        get_pyramidal_cells(predicted_st_file)


