##########################################################
#Author:          Yufeng Liu
#Create time:     2025-06-08
#Description:               
##########################################################

import scanpy as sc
import numpy as np
import scipy.sparse as sp
import h5py
from tqdm import tqdm
import random


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



def filter_genes_memory_efficient(adata, 
                                cell_percentage_cutoff=0.03, 
                                nonz_mean_cutoff=1.12,
                                batch_size=10000):
    """
    内存高效的基因筛选（适合超大规模稀疏矩阵）
    
    参数:
        adata: AnnData对象（adata.X需为CSR格式）
        cell_percentage_cutoff: 非零细胞占比阈值
        nonz_mean_cutoff: 非零平均表达阈值
        batch_size: 每批处理的基因数
        
    返回:
        gene_mask: 布尔数组，True表示保留的基因
    """
    # 初始化统计量存储
    n_cells = adata.shape[0]
    n_genes = adata.shape[1]
    gene_nonzero_perc = np.zeros(n_genes)
    gene_nonzero_mean = np.zeros(n_genes)
    
    # 转置矩阵为CSC格式（列切片高效）
    # 假设adata.X为CSR格式，分块处理
    chunk_size = 10000  # 根据内存调整
    csc_chunks = []
    for i in range(0, adata.shape[0], chunk_size):
        print(f'chunk index: {i}')
        i_end = min(i+chunk_size, adata.shape[0])
        chunk = adata.X[i:i_end].tocsc()  # 转换为CSC
        csc_chunks.append(chunk)
        del chunk  # 及时释放内存

    adata.X = sp.vstack(csc_chunks)  # 合并为完整CSC矩阵
    import ipdb; ipdb.set_trace()
    X_csc = adata.X.tocsc()
    
    # --- 第一步：计算非零细胞占比 ---
    print("Calculating nonzero percentages...")
    for j in tqdm(range(0, n_genes, batch_size)):
        j_end = min(j + batch_size, n_genes)
        # 获取当前批次基因的列（内存友好）
        batch = X_csc[:, j:j_end]
        # 计算非零细胞数
        nonzero_counts = np.diff(batch.indptr)
        gene_nonzero_perc[j:j_end] = nonzero_counts / n_cells
    
    # 初步筛选（减少后续计算量）
    prelim_mask = gene_nonzero_perc > cell_percentage_cutoff
    print(f"Pre-filter: {prelim_mask.sum()} genes pass percentage cutoff")
    
    # --- 第二步：计算非零平均表达 ---
    print("Calculating nonzero means...")
    for j in tqdm(np.where(prelim_mask)[0]):
        # 单独处理每个符合条件的基因列
        col = X_csc[:, j]
        nonzero_data = col.data
        gene_nonzero_mean[j] = np.mean(nonzero_data) if len(nonzero_data) > 0 else 0
    
    # 最终筛选
    final_mask = (prelim_mask) & (gene_nonzero_mean > nonz_mean_cutoff)
    print(f"Final: {final_mask.sum()} genes pass all filters")
    
    return final_mask

def filter_genes_disk_backed(adata, 
                           cell_percentage_cutoff=0.03,
                           nonz_mean_cutoff=1.12,
                           batch_size=1000):
    """
    适用于磁盘映射模式(_CSRDataset)的基因筛选
    """
    n_cells, n_genes = adata.shape
    gene_mask = np.zeros(n_genes, dtype=bool)
    
    # --- 第一步：计算非零细胞占比 ---
    print("Calculating nonzero percentages...")
    for j in tqdm(range(0, n_genes, batch_size)):
        print(j)
        # 分批读取基因列（自动转换为CSC切片）
        batch = adata.X[:, j:j+batch_size]
        nonzero_counts = np.array((batch > 0).sum(axis=0)).flatten()
        perc = nonzero_counts / n_cells
        gene_mask[j:j+batch_size] = perc > cell_percentage_cutoff
    
    # --- 第二步：计算通过初筛基因的非零均值 ---
    print("Calculating nonzero means...")
    selected_genes = np.where(gene_mask)[0]
    nonzero_means = np.zeros(len(selected_genes))
    
    for i, j in enumerate(tqdm(selected_genes)):
        print(f'[i/j]: {i}/{j}')
        col = adata.X[:, j].toarray().flatten()  # 单列转为稠密
        nonzero_data = col[col > 0]
        nonzero_means[i] = np.mean(nonzero_data) if len(nonzero_data) > 0 else 0
    
    # 更新最终掩码
    gene_mask[selected_genes] = nonzero_means > nonz_mean_cutoff
    print(f"Final: {gene_mask.sum()} genes pass filters")
    
    return gene_mask



if __name__ == '__main__':
    if 0:
        # 参数设置
        data_path = '/data2/lyf/data/transcriptomics/human_scRNA_2023_Science/data'
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
        cortical_h5ad = 'cortical_cells_rand30w.h5ad'
        cell_count_cutoff = 5
        cell_percentage_cutoff = 0.03
        nonz_mean_cutoff = 1.12
        batch_size = 10000  # genes

        adata_in = sc.read_h5ad(cortical_h5ad, backed='r')
        


        import ipdb; ipdb.set_trace()
        gene_mask = filter_genes_memory_efficient(
        #gene_mask = filter_genes_disk_backed(
            adata_in,
            cell_percentage_cutoff=cell_percentage_cutoff,
            nonz_mean_cutoff=nonz_mean_cutoff,
            batch_size=batch_size
        )

