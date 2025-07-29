##########################################################
#Author:          Yufeng Liu
#Create time:     2025-07-28
#Description:               
##########################################################
import numpy as np
import pandas as pd
from umap import UMAP
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans  # 用于辅助验证聚类结构

h01_f_file = 'data/h01_stem_features.csv'


if 0:
    ############### Visualize the joint-ADR distribution ################
    # load stem angles
    feats = pd.read_csv(h01_f_file)
    angles = feats[['num_ang_10', 'num_ang_20', 'num_ang_30', 'num_ang_40', 'num_ang_50']].copy()


    # 1. 数据标准化（UMAP对尺度敏感，建议先标准化）
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(angles)
    print(f'--> Finished data standardization')

    # 2. 初始化并拟合UMAP
    # 参数说明：
    # - n_components: 降维后的维度（2或3，方便可视化）
    # - n_neighbors: 控制局部与全局结构的平衡（默认15，样本少可调小）
    # - min_dist: 点之间的最小距离（默认0.1，值越小聚类越紧凑）
    # - metric: 距离度量（默认'euclidean'，高维数据可尝试'cosine'）
    umap = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=1024)
    embedding = umap.fit_transform(data_scaled)
    print(f'--> [UMAP projection]')

    # 3. 可视化UMAP降维结果
    sns.set_theme(style='ticks', font_scale=2.)
    plt.figure(figsize=(8, 8))
    sns.scatterplot(
        x=embedding[:, 0], y=embedding[:, 1], alpha=0.8, s=25,
        color="lightcoral", edgecolor='none', linewidth=1
    )
    # 4. 强化坐标轴和标签
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(2)      # 确认坐标轴线宽
    ax.tick_params(axis='both', which='major', width=2, color='0.2')

    plt.title("ADR distribution in H01")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.grid(False)
    sns.despine()
    plt.savefig('umap_of_ADRs_h01.png', dpi=300)
    plt.close()
    print(f'<-- UMAP visualization done')


if 1:
    ################## Find out proper examples ####################
    import os
    import glob
    
    import cv2
    import filecmp
    from tqdm import tqdm
    from scipy.spatial.distance import cdist

    from file_io import load_image
    from image_utils import get_mip_image
    from plotters.neurite_arbors import NeuriteArbors


    init_dir = './data/auto8.4k_0510_resample1um'
    final_dir = './data/auto8.4k_0510_resample1um_mergedBranches0712'
    dist_csv = 'distances_deleted_terminii.csv'

    # distance distribution from terminal points to soma
    if os.path.exists(dist_csv):
        # load existing distance file
        df_dists = pd.read_csv(dist_csv, index_col=0)
    else:
        df_dists = {}
        for step_id, init_file in enumerate(tqdm(glob.glob(os.path.join(init_dir, '*.swc')))):
            swc_name = os.path.split(init_file)[-1]
            final_file = os.path.join(final_dir, swc_name)

            # check if the two files are the same
            if not filecmp.cmp(init_file, final_file, shallow=False):
                df_init = pd.read_csv(
                        init_file, sep=' ', names=('#id', 'type', 'x', 'y', 'z', 'r', 'p'), 
                        comment='#', index_col=0
                )

                df_final = pd.read_csv(
                        final_file, sep=' ', names=('#id', 'type', 'x', 'y', 'z', 'r', 'p'), 
                        comment='#', index_col=0
                )

                # find out the deleted nodes
                deleted = df_init[~df_init.index.isin(df_final.index)]
                deleted_terminii = deleted[~deleted.index.isin(deleted.p)]
                soma = df_init[df_init.p == -1]

                # estimate the Euclidean distance
                spos = soma[['x', 'y', 'z']]
                dtpos = deleted_terminii[['x', 'y', 'z']]
                dists = cdist(spos, dtpos).ravel()

                df_dists[swc_name] = dists
                print(step_id, swc_name, dists)
                
            if step_id % 500 == 0:
                print(f'--> [{step_id}], {swc_name}')
                #break

        df_dists = pd.DataFrame({"swc_name": df_dists.keys(), "dists": df_dists.values()})
        df_dists = df_dists.explode("dists").set_index('swc_name')

        df_dists.to_csv(dist_csv)


    df_dists_neuron = df_dists.groupby(df_dists.index).mean()

    if 0:
        # visualize the distance distributions
        averaging_by_neuron = True
        sns.set_theme(style='ticks', font_scale=1.8)
        plt.figure(figsize=(8, 6))
        if averaging_by_neuron:
            df_show = df_dists_neuron
            figname = 'distance_deleted_terminii_neuron.png'
        else:
            df_show = df_dists
            figname = 'distance_deleted_terminii.png'

        sns.histplot(data=df_show, x='dists', kde=True, bins=50, stat="density")
        plt.title("Distribution of distances")
        plt.savefig(figname, dpi=300)
        plt.close()

    # find out the neurons with large distance and save the reconstruction-overlaid images
    percentile = 95
    image_dir = '/data2/kfchen/tracing_ws/14k_raw_img_data/tif'
    meta_file = '/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta_filter/meta.csv'
    display_range = 20  # radius in um


    def blend_image(img2d, swc_file, xxyy=None):
        # neurite image
        na = NeuriteArbors(swc_file)
        morph2d = na.get_morph_mip(
                type_id=None, img_shape=img2d.shape, xxyy=xxyy, bkg_transparent=True,
                color='blue'
        )

        # blending the images
        img2d_c3 = cv2.cvtColor(img2d, cv2.COLOR_GRAY2BGR)
        morph_rgb = morph2d[:, :, :3]
        alpha = morph2d[:, :, 3].astype(np.float32) / 255.0
        alpha = np.expand_dims(alpha, axis=2)

        blended = (morph_rgb * alpha + img2d_c3 * (1 - alpha)).astype(np.uint8)
        return blended

    def image_enhancing(img2d):
        mean_i = img2d.mean()
        std_i = img2d.std()
        std_i3 = std_i * 3

        new_img = img2d.astype(float)
        np.clip(new_img, mean_i-std_i3, mean_i+std_i3, new_img)
        
        # normalization
        new_img = ((new_img - new_img.min()) / (new_img.max() - new_img.min() + 1e-6) * 255.).astype(np.uint8)
        return new_img
    

    dist_percentile = np.percentile(df_dists_neuron.dists, percentile)
    neurons_large_dists = df_dists_neuron[df_dists_neuron.dists > dist_percentile]
    # we should get the resolution
    meta = pd.read_csv(meta_file, index_col='cell_id', low_memory=False, encoding='gbk')
    # plot the images, with swc skeleton overlaid
    for swc_name in neurons_large_dists.index:
        swc_id = int(swc_name.split('_')[0])
        init_swc = os.path.join(init_dir, swc_name)
        final_swc = os.path.join(final_dir, swc_name)
        image_file = os.path.join(image_dir, f'{swc_name[:-4]}.tif')    # check

        # get mips
        image = load_image(image_file)
        img2d = get_mip_image(image)
        # flip top-down for tif
        img2d = cv2.flip(img2d, flipCode=0)
        # hue normalization of image
        img2d = image_enhancing(img2d)
        rez_xy = meta.loc[swc_id, 'xy_resolution'] / 1000.
        ymax, xmax = np.array(img2d.shape) * rez_xy
        xxyy = (0, xmax, 0, ymax)

        blended_init = blend_image(img2d, init_swc, xxyy=xxyy)
        blended_final = blend_image(img2d, final_swc, xxyy=xxyy)
        # merge
        stacked_image = np.hstack((blended_init, blended_final))
        cv2.imwrite('stacked_image.png', stacked_image)
        break
        

        
    
    


