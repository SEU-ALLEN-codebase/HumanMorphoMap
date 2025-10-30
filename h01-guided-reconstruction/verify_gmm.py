##########################################################
#Author:          Yufeng Liu
#Create time:     2025-07-28
#Description:               
##########################################################
import numpy as np
import pandas as pd
from umap import UMAP
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans  # 用于辅助验证聚类结构


def features_on_umap(feat_file, feat_names, figname, scale=25):
    feats = pd.read_csv(feat_file)
    angles = feats[feat_names].copy()


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
        x=embedding[:, 0], y=embedding[:, 1], alpha=0.8, s=scale,
        color="lightcoral", edgecolor='none', linewidth=1
    )
    # 4. 强化坐标轴和标签
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(2)      # 确认坐标轴线宽
    ax.tick_params(axis='both', which='major', width=2, color='0.2')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    #plt.title(figname)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.grid(False)
    sns.despine()
    plt.savefig(figname, dpi=300)
    plt.close()
    print(f'<-- UMAP visualization done')

if __name__ == '__main__':

    if 0:
        ############### Visualize the joint-ADR distribution ################
        # load stem angles
        h01_f_file = 'data/h01_stem_features.csv'
        feat_names = ['num_ang_10', 'num_ang_20', 'num_ang_30', 'num_ang_40', 'num_ang_50']
        features_on_umap(h01_f_file, feat_names, figname='umap_of_ADRs_h01')


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
        from swc_handler import get_soma_from_swc


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
        percentile = 98
        image_dir = '/data2/kfchen/tracing_ws/14k_raw_img_data/tif'
        meta_file = '/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta_filter/meta.csv'
        display_len = 40  # radius in um


        def blend_image(img2d_c3, swc_file, xxyy=None, soma_params=None, scalef=1.0):
            # neurite image
            if soma_params is not None:
                sparams = {k:v for k,v in soma_params.items()}
                if 'size' not in sparams:
                    # soma radius
                    soma = get_soma_from_swc(swc_file)
                    srad = float(soma[5])
                    sparams['size'] = srad * scalef
            else:
                sparams = None

            na = NeuriteArbors(swc_file, soma_params=sparams, scalef=scalef)
            morph2d = na.get_morph_mip(
                    type_id=None, img_shape=img2d_c3.shape[:2], xxyy=xxyy, bkg_transparent=True,
                    color='red'
            )

            # blending the images
            morph_rgb = morph2d[:, :, :3]
            alpha = morph2d[:, :, 3].astype(np.float32) / 255.0
            alpha = np.expand_dims(alpha, axis=2)

            blended = (morph_rgb * alpha + img2d_c3 * (1 - alpha)).astype(np.uint8)

            return blended, na.soma_xyz

        def image_enhancing(img2d, sigma=4):
            mean_i = img2d.mean()
            std_i = img2d.std()
            std_i3 = std_i * sigma

            new_img = img2d.astype(float)
            np.clip(new_img, mean_i-std_i3, mean_i+std_i3, new_img)
            
            # normalization
            new_img = ((new_img - new_img.min()) / (new_img.max() - new_img.min() + 1e-6) * 255.).astype(np.uint8)
            return new_img

        def save_comp_image(
                    swc_name, init_dir, final_dir, image_dir, meta, 
                    show_raw_image=False, soma_params=None, scalef=1.0
        ):
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
            img2d_c3 = cv2.cvtColor(img2d, cv2.COLOR_GRAY2BGR)
            if scalef != 1.0:
                img2d_c3 = cv2.resize(img2d_c3, (0,0), fx=scalef, fy=scalef)
            
            rez_xy = meta.loc[swc_id, 'xy_resolution'] / 1000.
            ymax, xmax = np.array(img2d.shape) * rez_xy * scalef
            xxyy = (0, xmax, 0, ymax)

            blended_init, sxyz = blend_image(img2d_c3, init_swc, xxyy=xxyy, soma_params=soma_params, scalef=scalef)
            blended_final, _ = blend_image(img2d_c3, final_swc, xxyy=xxyy, soma_params=soma_params, scalef=scalef)

            # crop a soma-centered subregion for zoom-in view
            display_len_pixel = int(np.round(display_len / rez_xy * scalef))
            xy_pixel = np.round(np.array(sxyz[:2]) / rez_xy).astype(int)
            
            xmin, ymin = np.maximum(xy_pixel - display_len_pixel, 0)
            xmax, ymax = np.minimum(xy_pixel + display_len_pixel, img2d_c3.shape[:2][::-1])

            blended_init_sub = blended_init[ymin:ymax, xmin:xmax]
            blended_final_sub = blended_final[ymin:ymax, xmin:xmax]

            # merge
            if show_raw_image:
                stacked_image = np.hstack((img2d_c3[ymin:ymax, xmin:xmax], blended_init_sub, blended_final_sub))
            else:
                stacked_image = np.hstack((blended_init_sub, blended_final_sub))
            print(stacked_image.shape)
            cv2.imwrite(f'{swc_name[:-4]}.png', stacked_image)


        dist_percentile = np.percentile(df_dists_neuron.dists, percentile)
        neurons_large_dists = df_dists_neuron[df_dists_neuron.dists > dist_percentile]
        # we should get the resolution
        meta = pd.read_csv(meta_file, index_col='cell_id', low_memory=False, encoding='gbk')

        
        # For all neurons: plot the images, with swc skeleton overlaid
        #for iswc, swc_name in enumerate(tqdm(neurons_large_dists.index)):
        #    print(iswc, swc_name)
        #    save_comp_image(swc_name, init_dir, final_dir, image_dir, meta)
            
            
        # plot only required neurons
        swc_names = [
            '00717_P005_T01-S011_MFG_R0368_LJ-20220525_LJ.swc',
            '00985_P010_T01-S017_TP_R0490_LJ-20220607_YXQ.swc',
            '01032_P010_T01-S020_TP_R0613_LJ-20220607_YXQ.swc',
            '02820_P025_T01_-S034_LTL_R0613_RJ-20230201_YS.swc',
            '03358_P023_T01_-S007_LIFG_R0613_LJ-20221127_LD.swc',
            '08495_P051_T02_(2)_S013_-_FL.R_R0919_RJ_20230814_RJ.swc'
        ]
            
        soma_params = {
            'color': 'gray',
            'alpha': 0.75,
        }
        for swc_name in swc_names:
            save_comp_image(swc_name, init_dir, final_dir, image_dir, meta, show_raw_image=True, soma_params=soma_params, scalef=2.)
        


