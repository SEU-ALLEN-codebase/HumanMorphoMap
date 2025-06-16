##########################################################
#Author:          Yufeng Liu
#Create time:     2025-06-08
#Description:               
##########################################################
import os
import glob
import scanpy as sc
import numpy as np
import scipy.sparse as sp
import h5py
from tqdm import tqdm
import random

from cell2location.utils.filtering import filter_genes


def extract_rows_csr_memory_efficient(adata, indices, batch_size=1000):
    """
    从CSR稀疏矩阵中提取指定行，保持CSR格式，内存优化
    
    参数:
        adata: AnnData对象 (adata.X需为CSR或可转CSR)
        indices: 需要提取的行索引数组（已排序的numpy数组效率更高）
        batch_size: 分批处理的行数（控制内存峰值）
    
    返回:
        sp.csr_matrix: 提取后的CSR稀疏矩阵
    """
    # 确保输入是CSR格式（若已是CSR则无额外开销）
    X_csr = adata.X
    
    # 预处理indices（排序并去重）
    indices = np.unique(np.sort(indices))
    n_rows = len(indices)
    n_cols = X_csr.shape[1]
    
    # 预分配结果矩阵的构建组件
    all_data = []
    all_indices = []
    all_indptr = [0]  # CSR格式的indptr从0开始
    
    # 分批提取目标行
    for i in range(0, n_rows, batch_size):
        print(i)
        batch_indices = indices[i:i + batch_size]
        
        # 直接切片获取目标行（CSR格式行切片高效）
        batch_csr = X_csr[batch_indices, :]
        
        # 收集当前批次的稀疏矩阵数据
        all_data.append(batch_csr.data)
        all_indices.append(batch_csr.indices)
        
        # 更新indptr（需偏移量调整）
        batch_indptr = batch_csr.indptr[1:]  # 去掉开头的0
        adjusted_indptr = batch_indptr + all_indptr[-1]
        all_indptr.extend(adjusted_indptr)
    
    # 合并所有批次数据
    print(f'Merging the non-zero data')
    final_data = np.concatenate(all_data)
    final_indices = np.concatenate(all_indices)
    final_indptr = np.array(all_indptr)
    
    # 构建最终CSR矩阵
    # 原始创建方式（indptr会被强制转换为int32,导致溢出）
    csr_mat = sp.csr_matrix(
        (final_data, final_indices, final_indptr),
        shape=(n_rows, n_cols)
    )

    return csr_mat


if __name__ == '__main__':
    data_path = '/data2/lyf/data/transcriptomics/human_scRNA_2023_Science/data'

    if 0:
        # Extract cells
        input_file = os.path.join(data_path, 'f9ecb4ba-b033-4a93-b794-05e262dc1f59.h5ad')
        output_file = os.path.join(data_path, 'cortical_cells_rand30w.h5ad')
        sel_cells = 300000
        seed = 1024

        # 第一步：确定总细胞数和皮层细胞索引
        adata_lazy = sc.read_h5ad(input_file, backed='r')
        is_cortex = adata_lazy.obs.ROIGroupFine == 'Cerebral cortex'
        indices = np.where(is_cortex)[0]
        print(indices.shape)
        
        print(f'Random sampling {sel_cells} cells out of {indices.shape[0]}')
        random.seed(seed)
        sel_indices = random.sample(indices.tolist(), sel_cells)


        # 创建最终的 AnnData 对象
        data = extract_rows_csr_memory_efficient(adata_lazy, sel_indices, batch_size=40000)

        print('Save as AnnData...')
        adata_new = sc.AnnData(X=data)  # 自动识别CSR格式

        adata_final = adata_lazy[sel_indices, :]
        # We could not write adata_final directly, instead we construct a new AnnData object

        adata_new.obs = adata_final.obs.copy()
        adata_new.var = adata_final.var.copy()
        adata_new.uns = adata_final.uns.copy()
        adata_new.obsm = adata_final.obsm.copy()

        adata_new.write(output_file, compression=True)   

        adata_lazy.file.close()

        print(adata_new.X.indptr.shape, adata_new.X.dtype)


    if 1:
        # filter genes
        cortical_h5ad = os.path.join(data_path, 'cortical_cells_rand30w.h5ad')
        cell_count_cutoff = 5
        cell_percentage_cutoff = 0.15
        nonz_mean_cutoff = 2.
        filtered_h5ad = os.path.join(data_path, 
        f'cortical_cells_rand30w_count{cell_count_cutoff}_perc{cell_percentage_cutoff:.2f}_nonzMean{nonz_mean_cutoff:.1f}.h5ad'
        )
        batch_size = 10000  # genes

        print(f'Loading the data...')
        adata_in = sc.read_h5ad(cortical_h5ad)

        print(f'Filtering...')
        filtered_mask = filter_genes(
            adata_in, 
            cell_count_cutoff=cell_count_cutoff,
            cell_percentage_cutoff2=cell_percentage_cutoff,
            nonz_mean_cutoff=nonz_mean_cutoff,
        )

        # filter the object
        adata_in_sel = adata_in[:, filtered_mask].copy() # #genes = 11,897
        # save file temporarily
        adata_in_sel.write(filtered_h5ad, compression=True)


