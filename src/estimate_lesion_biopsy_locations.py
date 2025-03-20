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

#from config import IMMUNO_ID
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
    
def morphology_vs_distance(seg_dir, seg_ann_file, meta_file_tissue):
    # parse seg_ann_file
    seg_ann = pd.read_csv(seg_ann_file, index_col='subject', usecols=(1,2,3))
    # get all the segmentations and estimate the distances
    meta_t = pd.read_csv(meta_file_tissue, index_col=0)
    import ipdb; ipdb.set_trace()
    print()
    


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
        meta_file_neuron = '/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta/meta.csv'
        seg_dir = '/data/PBS/SEU-ALLEN/Users/ZhixiYun/data/HumanNeurons/sample_annotation'
        seg_ann_file = '../meta/seg_info_250317_fromDB.csv'
        morphology_vs_distance(seg_dir, seg_ann_file, meta_file_tissue)


