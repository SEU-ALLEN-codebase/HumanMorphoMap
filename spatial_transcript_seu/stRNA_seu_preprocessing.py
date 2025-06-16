##########################################################
#Author:          Yufeng Liu
#Create time:     2025-06-03
#Description:               
##########################################################
import os
import h5py
import scipy.sparse as sp
import pandas as pd
import numpy as np
import json
from scipy.sparse import csc_matrix
import scanpy as sc
#from squidpy.im import smooth

def compile_10x_spatial_data(matrix_h5, positions_csv, scalefactors_json, out_adata):
    # 1. 加载表达矩阵 (CSC格式)
    with h5py.File(matrix_h5, 'r') as f:
        data = f['matrix/data'][:]
        indices = f['matrix/indices'][:]
        indptr = f['matrix/indptr'][:]
        shape = f['matrix/shape'][:]
        features = f['matrix/features/name'][:].astype(str)  # 基因名
        barcodes = f['matrix/barcodes'][:].astype(str)       # 细胞barcode
    
    # 构建稀疏矩阵（注意10x存储的是 genes × cells，需要转置为 cells × genes）
    matrix = csc_matrix((data, indices, indptr), shape=shape).T  # 关键转置！

    # 2. 加载空间坐标
    positions = pd.read_csv(positions_csv, index_col=0)
    if 'pxl_row_in_fullres' not in positions.columns:
        # in case no headers in the file
        positions = pd.read_csv(positions_csv, header=None)
        positions.columns = [
            'barcode', 'in_tissue', 'array_row', 'array_col',
            'pxl_row_in_fullres', 'pxl_col_in_fullres'
        ]
        positions = positions.set_index('barcode')
    
    # 3. 加载分辨率信息
    with open(scalefactors_json) as f:
        scalefactors = json.load(f)
    
    # 4. 合并数据构建AnnData
    # 确保barcodes顺序一致
    common_barcodes = np.intersect1d(barcodes, positions.index)
    matrix = matrix[np.isin(barcodes, common_barcodes), :]
    positions = positions.loc[common_barcodes]
    
    # 创建AnnData对象
    adata = sc.AnnData(
        X=matrix,
        obs=positions,
        var=pd.DataFrame(index=features)
    )


    # Remove possible duplicate genes. To avoid possible errors, I would like to just keep one copy of them.
    dup_genes = adata.var_names[adata.var_names.duplicated(keep='first')].unique()
    keep_genes = ~adata.var_names.duplicated(keep='first')
    adata = adata[:, keep_genes]
    
    # 5. 添加空间坐标到obsm (使用高分辨率像素坐标)
    adata.obsm['spatial'] = positions[  # physical space, in um
        ['pxl_row_in_fullres', 'pxl_col_in_fullres']
    ].values * scalefactors['tissue_hires_scalef']  # 缩放至高分辨率

    adata.obsm['spatial_pxl'] = positions[  # pixel space
        ['pxl_row_in_fullres', 'pxl_col_in_fullres']
    ].values
    
    # 6. 添加spot直径信息 (用于绘图)
    adata.uns['spot_diameter'] = scalefactors['spot_diameter_fullres'] * scalefactors['tissue_hires_scalef']
    
    # 8. save the data
    adata.write(out_adata)
    
    return adata


def preprocessing(adfile, min_genes=200, max_genes=10000, max_mt=20, min_cells=10):
    # 添加基本QC指标
    sc.pp.calculate_qc_metrics(adata, inplace=True)
    
    adata = sc.read(adfile)
    
    # 计算每个spot的基因数和总UMI
    adata.var['mt'] = adata.var_names.str.startswith('MT-')  # 线粒体基因标记
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

    # Filtering
    adata = adata[adata.obs['n_genes_by_counts'] > min_genes, :]
    adata = adata[adata.obs['n_genes_by_counts'] < max_genes, :]
    adata = adata[adata.obs['pct_counts_mt'] < max_mt, :]

    # filter genes expressed in less than `min_cells`
    sc.pp.filter_genes(adata, min_cells=min_cells)

    # 文库大小标准化(CPM/TPM)
    sc.pp.normalize_total(adata, target_sum=1e4)

    # 对数转换(注意保持稀疏性)
    sc.pp.log1p(adata)  # 自动处理稀疏矩阵

    # Extract the high-variable genes
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata = adata[:, adata.var.highly_variable]

    # 获取空间坐标(假设存储在adata.obsm['spatial']) LYF: no need to do so at this moment
    #coords = adata.obsm['spatial']

    # 标准化坐标到相同尺度
    #coords = (coords - coords.mean(0)) / coords.std(0)
    #adata.obsm['spatial'] = coords

    # 使用高斯滤波进行空间平滑 (Optional)
    #smooth(adata, sigma=1.5, mode='gauss', key_added='smooth_exp')

    # 缩放数据
    sc.pp.scale(adata, max_value=10)

    # PCA降维
    sc.tl.pca(adata, svd_solver='arpack')

    adata.write(f'{out_adata[:-5]}_processed.h5ad')


    

if __name__ == '__main__':

    if 1:
        # preprocess the data for subsequent analyses
        data_path = '/PBshare/SEU-ALLEN/Users/WenYe/Human-Brain-ST-data'
        #for sample_id in ['P065_0', 'P065_500']:
        for sample_id in ['P00117', 'P00083', 'P00089']: #['P00066', 'P00079', 'P00083', 'P00089', 'P00090', 'P00115', 'P00117']:
            print(sample_id)
            matrix_h5 = f'{data_path}/{sample_id}/filtered_feature_bc_matrix.h5'
            position_csv = f'{data_path}/{sample_id}/spatial/tissue_positions.csv'
            scalefactors_json = f'{data_path}/{sample_id}/spatial/scalefactors_json.json'

            out_adata = f'./data/spatial_adata_{sample_id}.h5ad'
            #if os.path.exists(out_adata):
            #    continue
        
            compile_10x_spatial_data(matrix_h5, position_csv, scalefactors_json, out_adata)
            #preprocessing(out_adata)

        



