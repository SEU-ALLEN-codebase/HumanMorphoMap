##########################################################
#Author:          Yufeng Liu
#Create time:     2025-01-22
#Description:               
##########################################################
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import SimpleITK as sitk
from scipy.ndimage import center_of_mass
from scipy.spatial.distance import cdist
from scipy.stats import linregress, mannwhitneyu


from config import to_PID5, to_TID3
from file_io import load_image

def keep_samples(sample_dir, meta_n):
    pnumbers = os.listdir(sample_dir)
    pnumbers2 = [pn[0] + '00' + pn[1:] for pn in pnumbers]
    
    return meta_n.patient_number.isin(pnumbers2) # patient number

def keep_neurons(recon_file, meta_n):
    gfs = pd.read_csv(recon_file, index_col=0)
    nb1 = meta_n.index.isin(gfs.index)
    return nb1


def get_neurons(brain_in_dir, meta_file_tissue, meta_file_neuron, recon_file, filter_by_samples=False, ihc=0):
    # load the meta file
    if meta_file_tissue.endswith('.xlsx'):
        meta_t = pd.read_excel(meta_file_tissue)
    else:
        meta_t = pd.read_csv(meta_file_tissue, index_col=0)
    
    if meta_file_neuron.endswith('.xlsx'):
        meta_n = pd.read_excel(meta_file_neuron)
    else:
        meta_n = pd.read_csv(meta_file_neuron, index_col=2)
    
    # get the target neurons according the criteria
    # IHC: meta_n.immunohistochemistry, TT: meta_t. vs. JSP
    if filter_by_samples:
        nb1 = keep_samples(brain_in_dir, meta_n)
    else:
        nb1 = keep_neurons(recon_file, meta_n)
    nb2 = meta_n.immunohistochemistry == ihc  # ihc: 0, 1
    # extracted neurons
    eneurons = meta_n[nb1 & nb2]
    # get the meta-informations
    eneurons['patient_number_03d'] = eneurons['patient_number'].apply(lambda x: f'P{int(x[1:]):03d}')
    eneurons['tissue_block_number_02d'] = eneurons['tissue_block_number'].apply(lambda x: f'T{int(x[1:]):02d}')
    # merge the table
    eneurons = pd.merge(eneurons, meta_t[['patient_number', 'tissue_id', 'sample_id']],
                        left_on=['patient_number_03d', 'tissue_block_number_02d'],
                        right_on=['patient_number', 'tissue_id'], how='left')
    #eneurons['patient_tissue_id'] = eneurons['patient_number_03d'] + '-' + eneurons['tissue_block_number_02d']

    eneurons['hospital'] = [hos.split('-')[1] if hos is not np.nan else np.nan for hos in eneurons.sample_id]
    # show the distribution of neurons across different hospitals
    display = True
    if display:
        uniq_regs, cnt_regs = np.unique(eneurons.brain_region, return_counts=True)
        cnt_thr = 50
        uniq_regs_thr = uniq_regs[cnt_regs > cnt_thr]
        cur_neurons = eneurons[eneurons.brain_region.isin(uniq_regs_thr)]
        for i_reg in uniq_regs_thr:
            if i_reg is np.nan: continue
            i_neurons = cur_neurons[cur_neurons.brain_region == i_reg]
            # get the clinic information
            pti = i_neurons['patient_number_03d'] + '-' + i_neurons['tissue_block_number_02d']
            pti_uniq = np.unique(pti)
            for pt in pti_uniq:
                pi = pt.split('-')[0]
                ti = pt.split('-')[1]
                meta_t_i = meta_t[(meta_t.patient_number == pi) & (meta_t.tissue_id == ti)]
                try:
                    tumor_loc = meta_t_i.tumor_location.to_numpy()[0]
                    sample_loc = meta_t_i.intracranial_location.to_numpy()[0]
                    print(pi, ti, meta_t_i.sample_id.to_numpy()[0], tumor_loc, 
                          sample_loc, meta_t_i.pathological_diagnosis.to_numpy()[0])
                except IndexError:
                    print(pi, ti, '--> Void location information')
            print(i_reg, np.unique(i_neurons.hospital.astype(str), return_counts=True), '\n')
        
    
        import ipdb; ipdb.set_trace()
        print()


def get_neurons_JSP(meta_file_tissue_JSP, meta_file_neuron, recon_file, ihc=1):
    # load the meta file
    if meta_file_tissue_JSP.endswith('.xlsx'):
        meta_t = pd.read_excel(meta_file_tissue_JSP)
    else:
        meta_t = pd.read_csv(meta_file_tissue_JSP, index_col=0, low_memory=False)
    
    if meta_file_neuron.endswith('.xlsx'):
        meta_n = pd.read_excel(meta_file_neuron)
    else:
        meta_n = pd.read_csv(meta_file_neuron, index_col=2, low_memory=False)
    
    # get the target neurons according the criteria
    # IHC: meta_n.immunohistochemistry, TT: meta_t. vs. JSP
    nb1 = keep_neurons(recon_file, meta_n)
    nb2 = meta_n.immunohistochemistry == str(ihc)  # ihc: 0, 1
    # extracted neurons
    eneurons = meta_n[nb1 & nb2]
    # get the meta-informations
    eneurons['patient_number_03d'] = eneurons['patient_number'].apply(lambda x: f'P{int(x[1:]):03d}')
    eneurons['tissue_block_number_02d'] = eneurons['tissue_block_number'].apply(lambda x: f'T{int(x[1:]):02d}')
    # merge the table
    eneurons = pd.merge(eneurons, 
            meta_t[['patient_number', 'tissue_id', 'sample_id', 'tissue_type', 'pathological_diagnosis']],
            left_on=['patient_number_03d', 'tissue_block_number_02d'],
            right_on=['patient_number', 'tissue_id'], how='left')
    #eneurons['patient_tissue_id'] = eneurons['patient_number_03d'] + '-' + eneurons['tissue_block_number_02d']
    eneurons['hospital'] = [hos.split('-')[1] if hos is not np.nan else np.nan for hos in eneurons.sample_id]
    eneurons = eneurons[eneurons.hospital == 'JSP']
    print(f'>> #Neurons: {len(eneurons)}')

    # show the distribution of neurons across different hospitals
    display = True
    if display:
        uniq_regs, cnt_regs = np.unique(eneurons.brain_region, return_counts=True)
        cnt_thr = 30
        uniq_regs_thr = uniq_regs[cnt_regs > cnt_thr]
        cur_neurons = eneurons[eneurons.brain_region.isin(uniq_regs_thr)]
        for i_reg in uniq_regs_thr:
            if i_reg is np.nan: continue
            i_neurons = cur_neurons[cur_neurons.brain_region == i_reg]
            # get the clinic information
            pti = i_neurons['patient_number_03d'] + '-' + i_neurons['tissue_block_number_02d']
            pti_uniq = np.unique(pti)
            for pt in pti_uniq:
                pi = pt.split('-')[0]
                ti = pt.split('-')[1]
                meta_t_i = meta_t[(meta_t.patient_number == pi) & (meta_t.tissue_id == ti)]
                try:
                    tumor_loc = meta_t_i.tumor_location.to_numpy()[0]
                    sample_loc = meta_t_i.intracranial_location.to_numpy()[0]
                    print(pi, ti, (pti == pt).sum(), meta_t_i.sample_id.to_numpy()[0], 
                          tumor_loc, sample_loc, meta_t_i.pathological_diagnosis.to_numpy()[0],
                          meta_t_i.tissue_type.to_numpy()[0])
                except IndexError:
                    print(pi, ti, '--> Void location information')
            print(i_reg, np.unique(i_neurons.hospital.astype(str), return_counts=True), '\n')
        
    
        import ipdb; ipdb.set_trace()
        print()

def compare_neurons(tissue_ids1, tissue_ids2, meta_file_neuron, gf_file, ihc=1, use_subset=True):
    """
    Compare the neurons from tissues of different diagnoses. tissue_ids in format of [(sample_id, tissue_id), ...]
    """
    
    ############## Helper functions ###################
    def extract_sub_neurons(tissue_ids, meta, gfs, ihc, use_subset):
        nsel = 0
        for pi, ti in tissue_ids:
            pti = (meta.patient_number == pi) & (meta.tissue_block_number == ti)
            nsel = pti | nsel
 
        nsel = nsel & (meta.immunohistochemistry == str(ihc))
        cur_gfs = gfs[gfs.index.isin(meta[nsel].index)]
        num_orig = len(cur_gfs)
        # select neurons with much fibers
        if use_subset:
            N = min(max(int(num_orig * 0.2), 10), num_orig)
            sel_ids = np.argpartition(cur_gfs['Total Length'], num_orig-N)[-N:]
            sel_gfs = cur_gfs.iloc[sel_ids]
        
        print(f'#Neurons before and after filtering: {num_orig}, {N}')

        return sel_gfs

    ############ End of helper functions ##############


    # load the meta information
    meta_n = pd.read_csv(meta_file_neuron, index_col=2, low_memory=False)
    gfs = pd.read_csv(gf_file, index_col=0)
    # extract the comparing neuron sets
    feats_n1 = extract_sub_neurons(tissue_ids1, meta_n, gfs, ihc=ihc, use_subset=use_subset)
    feats_n2 = extract_sub_neurons(tissue_ids2, meta_n, gfs, ihc=ihc, use_subset=use_subset)

    # visualize the differences
    sns.set_theme(style='ticks', font_scale=1.8)
    
    # pair-by-pair feature comparison
    for col in feats_n1.columns:
        xname = 'Normal'
        yname = 'Infiltrating'
        f_cur = pd.DataFrame({
            xname: feats_n1[col],
            yname: feats_n2[col],
        })
        sns.boxplot(data=f_cur, width=0.35, color='black', fill=False)
        #sns.stripplot(data=f_cur)
        cname = col.replace(" ", "")
        plt.title(cname)
        plt.savefig(f'{xname}_{yname}_{cname}.png', dpi=300); plt.close()
    

    #import ipdb; ipdb.set_trace()
    print()

def get_physical_coordinates_from_mask(mask, image):
    """
    从二值掩膜（numpy数组，Z,Y,X顺序）中提取所有点的物理坐标。
    """
    # 获取图像的元数据
    spacing = np.array(image.GetSpacing())    # (X,Y,Z) 物理分辨率
    origin = np.array(image.GetOrigin())     # 物理原点
    direction = np.array(image.GetDirection()).reshape(3, 3)  # 方向矩阵 (3x3)

    # 获取掩膜中所有点的体素坐标 (Z,Y,X 顺序)
    voxel_coords_zyx = np.argwhere(mask)  # shape: (N, 3)

    if len(voxel_coords_zyx) == 0:
        raise ValueError("掩膜为空，请检查输入！")

    # 将体素坐标从 (Z,Y,X) 转为 (X,Y,Z)（SimpleITK 默认顺序）
    voxel_coords_xyz = voxel_coords_zyx[:, [2, 1, 0]]  # (N, 3) 的 (X,Y,Z)

    # 计算物理坐标: physical_xyz = origin + direction @ (spacing * voxel_xyz)
    physical_coords = origin + (direction @ ((spacing * voxel_coords_xyz).T)).T
    return physical_coords

def calculate_min_distance_to_tumor(tumor_mask, sample_mask, image):
    """
    计算 sample_mask 的质心到 tumor_mask 的最小物理距离。
    """
    # 1. 获取 sample_mask 的质心（物理坐标）
    sample_physical_coords = get_physical_coordinates_from_mask(sample_mask, image)
    centroid = np.mean(sample_physical_coords, axis=0)  # (3,) 物理坐标

    # 2. 获取 tumor_mask 的所有物理坐标
    tumor_physical_coords = get_physical_coordinates_from_mask(tumor_mask, image)

    # 3. 计算质心到 tumor 点的最小欧氏距离
    distances = cdist(centroid.reshape(1, 3), tumor_physical_coords)
    min_distance = np.min(distances)
    return min_distance


def calculate_distance2tumor(seg_dir, seg_ann_file, meta_file_sample, dist2tumor_file):
    # parse seg_ann_file
    seg_ann = pd.read_csv(seg_ann_file, index_col='subject', usecols=(1,2,3,4,5))
    # get all the segmentations and estimate the distances
    meta_s = pd.read_csv(meta_file_sample, index_col=0, low_memory=False)

    JSP_problematics = ['NanJ-JSP-ZDL-01', 'NanJ-JSP-ZXB-01']

    # reformat the sample-informations from JSP
    seg2PT = []
    subject_names = []
    for irow, row in seg_ann.iterrows():
        sub_id, sub_name = irow.split('_')
        if ' ' in sub_name:
            city = sub_name.split(' ')[0]
            hospital, operator = sub_name.split(' ')[1].split('-')[:2]
        else:
            city, hospital, operator = sub_name.split('-')[:3]
        
        # we use JSP only for this analyses
        if hospital != 'JSP':
            continue
        # get the path
        seg_path = os.path.join(seg_dir, irow, 'Segmentation.seg.nrrd')
        if not os.path.exists(seg_path):
            seg_path = os.path.join(seg_dir, irow, 'Segmentation_1.seg.nrrd')

        seg2PT.append([sub_id, irow, city, hospital, operator, row.label, row['name'], row['sample_id_seu'], row['channel'], seg_path])
        subject_names.append(sub_name)

    seg2PT = pd.DataFrame(seg2PT, index=subject_names,
                columns=('subject_id', 'subject', 'city', 'hospital', 'operator', 'label', 'seg_name', 'sample_id_seu', 'channel', 'seg_file'))
    
    # Find out the correspondence of seg to PTRSB
    matched = []
    unmatched = []
    matched_seg2PT = []
    seg_ptrsb = []
    for irow, row in meta_s.iterrows():
        if row.patient_number is np.NaN or row.patient_number in(['-', '--']):
            continue
        # discard the problematic samples
        if row.sample_id in JSP_problematics:
            continue

        pid, tid = row.patient_number, row.tissue_id
        # matching
        sample_id = row.sample_id
        k1 = sample_id
        k2 = '-'.join(sample_id.split('-')[:-1])
        k3 = sample_id.replace('-', ' ', 1)
        k4 = '-'.join(sample_id.replace('-', ' ', 1).split('-')[:-1])

        found = False
        for ki in [k1, k2, k3, k4]:
            if ki in seg2PT.index:
                cur_segs = seg2PT.loc[ki]
                matched.append(cur_segs.index[0])
                matched_seg2PT.append(sample_id)
                
                found = True
                break

        if (not found) and (sample_id in ['NanJ-JSP-XFY-01', 'NanJ-JSP-XFY-02']):
            cur_segs = seg2PT.loc['NanJ-JSP-XFY-01-02']
            matched.append(cur_segs.index[0])
            matched_seg2PT.append('NanJ-JSP-XFY-01-02')

            found = True

        if not found:
            unmatched.append(sample_id)
            
        # tissue matching
        if found:
            if cur_segs.ndim == 1:
                print(f'!!![Error] Only one mask is found in {cur_segs.subject}')
                continue

            if (cur_segs.seg_name == 'tumor').sum() != 1:
                print(f'!!![Error] The current sample {row.sample_id} has incorrect segmentation mask: {cur_segs.seg_name}')
                continue
            
            tumor_label = cur_segs[cur_segs.seg_name == 'tumor'].label.iloc[0]
            samples = cur_segs[cur_segs.seg_name == 'sample']
            if samples.shape[0] == 1:
                # The information are matched
                sample_label = samples.iloc[0].label
            elif samples.shape[0] == 0:
                print(f'!!![Error] No sample mask is found in {cur_segs.iloc[0].subject}')
                continue
            elif samples.shape[0] > 1:
                # match according to T-coding
                tid3 = to_TID3(tid)
                if (samples.sample_id_seu == tid3).sum() == 1: 
                    # found the correspondence
                    sample_label = samples[samples.sample_id_seu == tid3]['label'].values[0]
                else:
                    print(f'--> {cur_segs.iloc[0].subject}')
                    continue

            # get the information
            seg_ptrsb.append([row.sample_id, cur_segs.iloc[0]['subject'], to_PID5(pid), to_TID3(tid),
                              tumor_label, sample_label, cur_segs.iloc[0]['seg_file']])


    seg_ptrsb = pd.DataFrame(seg_ptrsb, columns=('sample_id', 'seg_id', 'pid5', 'tid3', 
                                'tumor_label', 'sample_label', 'seg_file'))
    seg_ptrsb = seg_ptrsb.set_index('sample_id')
    
    # estimate the distance from sample to tumor
    dists = []
    for irow, row in seg_ptrsb.iterrows():
        seg_id, pid5, tid5, tumor_label, sample_label, seg_file = row
        print(irow)
        # load the segmentation
        img_seg = sitk.ReadImage(seg_file)
        img_arr = sitk.GetArrayFromImage(img_seg)

        if seg_id in ['281_NanJ-JSP-JLX-01', '274_NanJ-JSP-WZB-01']:
            tumor_mask = img_arr[:,:,:,0] == tumor_label
            sample_mask = img_arr[:,:,:,1] == sample_label
        else:
            tumor_mask = img_arr == tumor_label
            sample_mask = img_arr == sample_label

        # check if there are different
        if np.array_equal(tumor_mask, sample_mask):
            raise ValueError('The tumor mask and sample mask are the same!')
        
        # calculate the distance
        dist = calculate_min_distance_to_tumor(tumor_mask, sample_mask, img_seg)
        dists.append(dist)

    seg_ptrsb['dist2tumor'] = dists

    # save to file
    seg_ptrsb.to_csv(dist2tumor_file, index=True)
    

def analyze_morph_by_distance(meta_file_neuron, gf_file, dist2tumor_file, ctype_file, ihc=0):

    ############### Helper functions ################
    # 标注显著性
    def get_stars(p):
        if p < 0.001:
            return '***'
        elif p < 0.01:
            return '**'
        elif p < 0.05:
            return '*'
        else:
            return 'ns'


    def _plot(gfs_cur, figname, min_counts=10, bins=[0,5,100]):
        sns.set_theme(style='ticks', font_scale=1.6)
        display_features = {
            'Soma_surface': 'Soma surface',
            'N_stem': 'Number of Stems',
            'Number of Branches': 'Number of Branches',
            #'Number of Tips': 'Number of Tips',
            'Average Diameter': 'Avg. Diameter',
            'Total Length': 'Total Length',
            'Max Branch Order': 'Max Branch Order',
            'Average Contraction': 'Avg. Straightness',
            'Average Fragmentation': 'Avg. Branch Length',
            'Average Parent-daughter Ratio': 'Avg. Parent-daughter Ratio',
            'Average Bifurcation Angle Local': 'Avg. Bif. Angle Local',
            'Average Bifurcation Angle Remote': 'Avg. Bif. Angle Remote', 
            'Hausdorff Dimension': 'Hausdorff Dimension',
        }

        # 数据准备
        # rename the features
        gfs_cur.rename(columns=display_features, inplace=True)
        features = display_features.values()   #gfs_cur.columns[:-1]  # 所有特征列（排除 distance）
        gfs_cur['distance_bin'] = pd.cut(
            gfs_cur['distance'],
            #bins=np.arange(gfs_cur['distance'].min(), gfs_cur['distance'].max() + bin_width, bin_width),
            bins = bins,
            right=False
        )
        intervals, interval_counts = np.unique(gfs_cur.distance_bin, return_counts=True)
        gfs_cur = gfs_cur[gfs_cur.distance_bin.isin(intervals[interval_counts >= min_counts])]
        print(f"Number of samples per distance_bin: {gfs_cur.groupby('distance_bin', observed=False)['pt_code'].nunique()}")
         
        
        # 计算全局 y 轴范围（排除异常值）
        y_limits = {}
        for feature in features:
            q1 = gfs_cur[feature].quantile(0.25)
            q3 = gfs_cur[feature].quantile(0.75)
            iqr = q3 - q1
            y_min = q1 - 2.0 * iqr
            y_max = q3 + 2.0 * iqr
            y_limits[feature] = (y_min, y_max)

        # 设置图形（4 列子图）
        n_features = len(features)
        n_cols = 4
        n_rows = int(np.ceil(n_features / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows), sharex=True)
        axes = axes.flatten()

        # 为每个特征绘制箱线图和回归线
        for i, feature in enumerate(features):
            ax = axes[i]
            
            # 箱线图（调整宽度为 0.5）
            sns.boxplot(
                data=gfs_cur,
                x='distance_bin',
                y=feature,
                ax=ax,
                width=0.4,  # 更窄的箱体
                color='skyblue',
                showmeans=False,
                linewidth=3,
                meanprops={'marker': 'o', 'markerfacecolor': 'red', 'linewidth': 3}
            )
            
            
            # 计算每个分箱的均值并绘制回归线
            bin_means = gfs_cur.groupby('distance_bin', observed=False)[feature].median().reset_index()
            bin_means['bin_mid'] = bin_means['distance_bin'].apply(lambda x: x.mid)  # 取区间中点
            bin_means = bin_means[~bin_means[feature].isna()]

            # 在每组箱体的中位数位置添加红色圆点
            x_positions = np.arange(len(bin_means))  # 箱线图的x轴位置（0, 1, 2,...）
            #x_positions = [ax.get_xticks()[i] for i in range(len(bin_means))]
            #print(x_positions, bin_means[feature])
            ax.scatter(
                x_positions, 
                bin_means[feature], 
                color="red", 
                marker="o", 
                zorder=50,  # 确保圆点显示在最上层
                s=50,
            )

            # 绘制基线：第一个串口均值为准
            baseline = bin_means[feature].iloc[0]
            ax.axhline(y=baseline, color='orange', linestyle='--', linewidth=2)
            
            '''
            # 线性回归
            slope, intercept, r_value, p_value, std_err = linregress(
                bin_means['bin_mid'],
                bin_means[feature]
            )
            reg_line = intercept + slope * bin_means['bin_mid'].astype(float)
            
            # 绘制回归线
            ax.plot(
                bin_means.index,  # x 轴为分箱序号
                reg_line,
                color='green',
                linestyle='--',
                #label=f'Slope: {slope:.2f}\nR²: {r_value**2:.2f}'
            )
            '''

            # do statistical test
            group1 = gfs_cur[gfs_cur['distance_bin'] == bin_means['distance_bin'][0]][feature]
            group2 = gfs_cur[gfs_cur['distance_bin'] == bin_means['distance_bin'][1]][feature]
            u_stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
            # 绘制横线和星号
            x1, x2 = 0.15, 0.85
            y_min, y_max = y_limits[feature]  # 标注的y轴位置
            y_delta = (y_max - y_min)
            
            y1, y2 = y_max-0.15*y_delta, y_max-0.11*y_delta
            ax.plot([x1, x1, x2, x2], [y1, y2, y2, y1], lw=2, color='red')
            stars = get_stars(p_value)
            y_text = y2 if stars == 'ns' else y1
            ax.text((x1+x2)*0.5, y_text, stars, 
                   ha='center', va='bottom', color='red')
            

            # 设置 y 轴范围（排除异常值）
            ax.set_ylim(y_limits[feature])
            #ax.set_xlim(0.5, 5.5)
            
            # 标签和标题
            ax.set_title(feature)
            ax.set_ylabel('')
            ax.set_xlabel('Distance to tumor (mm)')
            ax.tick_params(axis='x', rotation=0)
            ax.tick_params(axis='y', direction='in')
            ax.set_xticks([0, 1])
            ax.set_xticklabels(('<= 5', '> 5'), ha="center")
            
            # bold
            ax.spines['left'].set_linewidth(2)
            ax.spines['right'].set_linewidth(2)
            ax.spines['top'].set_linewidth(2)
            ax.spines['bottom'].set_linewidth(2)
            #ax.legend()

        # 隐藏多余子图
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.savefig(figname, dpi=300)
        plt.close()

    ############# End of Helper functions ###########


    meta_n = pd.read_csv(meta_file_neuron, index_col=0, low_memory=False, encoding='gbk')
    gfs = pd.read_csv(gf_file, index_col=0, low_memory=False)
    dists = pd.read_csv(dist2tumor_file, index_col=0)
    ctypes = pd.read_csv(ctype_file, index_col=0)

    # extract neurons
    # 1. ihc extraction
    ihc_mask = meta_n.immunohistochemistry == ihc
    # 2. tissue extraction
    dists['pt_code'] = dists['pid5'] + '-' + dists['tid3']
    meta_n['pt_code'] = meta_n['patient_number'] + '-' + meta_n['tissue_block_number']
    tissue_mask = (meta_n['pt_code']).isin(dists['pt_code'])
    # 3. cell type extraction: to be added
    ctypes_idxs = [int(name.split('_')[0]) for name in ctypes.index]
    ctypes = ctypes.reset_index()
    ctypes.index = ctypes_idxs
    # get the cell types
    ctypes_ = ctypes.loc[gfs.index]
    py_mask = (ctypes_.num_annotator >= 2) & (ctypes_.CLS2 == '0')
    nonpy_mask = (ctypes_.num_annotator >= 2) & (ctypes_.CLS2 == '1')

    ctype_dict = {
        'pyramidal': py_mask,
        'nonpyramidal': nonpy_mask,
    }
    
    # morphological analysis
    for ctype, ctype_mask in ctype_dict.items():
        c_mask = (ihc_mask & tissue_mask).values & ctype_mask.values
        gfs_cur = gfs[c_mask]
        meta_n_cur = meta_n[c_mask]
        print(f'Number of {ctype} cells: {gfs_cur.shape[0]}')
    
        # calculate the feature versus the distance
        dists_re = dists.set_index('pt_code')
        dists_to_tumor = dists_re.loc[meta_n_cur['pt_code'], 'dist2tumor']
        gfs_cur['distance'] = dists_to_tumor.values
        gfs_cur['pt_code'] = dists_re.loc[meta_n_cur['pt_code']].index.values
    
        # visualization
        _plot(gfs_cur, f'morph_vs_dist2tumor_{ctype}.png')


if __name__ == '__main__':
    indir = '/PBshare/SEU-ALLEN/Users/ZhixiYun/data/human_regi'
    meta_file_tissue = '../meta/样本信息表10302024.xlsx.csv'
    meta_file_tissue_JSP = '../meta/江苏省人民医院样本登记表-修改版-20250211.xlsx.csv'
    meta_file_neuron = '../meta/1-50114.xlsx.csv'
    gf_file = '/data/kfchen/trace_ws/cropped_swc/proposed_1um_l_measure_total.csv'
    ihc = 1
    
    if 0:
        #get_neurons(indir, meta_file_tissue, meta_file_neuron, recon_file=gf_file, ihc=ihc)
        get_neurons_JSP(meta_file_tissue_JSP, meta_file_neuron, recon_file=gf_file, ihc=0)

    if 0:
        tissue_ids1, tissue_ids2 = [('P00065', 'T001')], [('P00066', 'T001')]
        #tissue_ids1, tissue_ids2 = [('P00057', 'T001'), ('P00070', 'T001')], [('P00062', 'T001'), ('P00064', 'T001')]
        compare_neurons(tissue_ids1, tissue_ids2, meta_file_neuron, gf_file, ihc=ihc)

    if 1:
        meta_file_neuron = '/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta_filter/meta.csv'
        gf_file = '/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta_filter/l_measure_result.csv'
        seg_dir = '/PBshare/SEU-ALLEN/Users/ZhixiYun/data/HumanNeurons/sample_annotation'
        seg_ann_file = '../meta/seg_info_250317_fromDB_yufeng0324.csv'
        dist2tumor_file = './caches/dist2tumor_0325.csv'
        ctype_file = '../meta/cell_type_annotation_8.4K_all_CLS2_unique.csv'
        #calculate_distance2tumor(seg_dir, seg_ann_file, meta_file_tissue_JSP, dist2tumor_file)
        analyze_morph_by_distance(meta_file_neuron, gf_file, dist2tumor_file, ctype_file, ihc=1)


