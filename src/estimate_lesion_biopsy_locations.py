##########################################################
#Author:          Yufeng Liu
#Create time:     2025-01-22
#Description:               
##########################################################
import os
import glob
import numpy as np
import pandas as pd

from config import IMMUNO_ID
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
                    print(pi, ti, meta_t_i.sample_id.to_numpy()[0], tumor_loc, sample_loc, meta_t_i.pathological_diagnosis.to_numpy()[0])
                except IndexError:
                    print(pi, ti, '--> Void location information')
            print(i_reg, np.unique(i_neurons.hospital.astype(str), return_counts=True), '\n')
        
    
        import ipdb; ipdb.set_trace()
        print()

if __name__ == '__main__':
    indir = '/PBshare/SEU-ALLEN/Users/ZhixiYun/data/human_regi'
    meta_file_tissue = '../meta/样本信息表10302024.xlsx.csv'
    meta_file_neuron = '../meta/1-50114.xlsx.csv'
    gf_file = '/data/kfchen/trace_ws/cropped_swc/proposed_1um_l_measure_total.csv'
    ihc = 1
    
    get_neurons(indir, meta_file_tissue, meta_file_neuron, recon_file=gf_file, ihc=ihc)

