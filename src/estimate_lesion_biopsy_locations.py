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

def get_neurons(brain_in_dir, meta_file_tissue, meta_file_neuron):
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
    pnumbers = os.listdir(brain_in_dir)
    pnumbers2 = [pn[0] + '00' + pn[1:] for pn in pnumbers]
    # extract the neurons
    nb1 = meta_n.patient_number.isin(pnumbers2) # patient number
    nb2 = meta_n.immunohistochemistry == 1  # IHC positive
    # extracted neurons
    eneurons = meta_n[nb1 & nb2]
    # get the meta-informations
    eneurons['patient_number_03d'] = eneurons['patient_number'].apply(lambda x: f'P{int(x[1:]):03d}')
    eneurons['tissue_block_number_03d'] = eneurons['tissue_block_number'].apply(lambda x: f'T{int(x[1:]):02d}')
    # merge the table
    eneurons = pd.merge(eneurons, meta_t[['patient_number', 'tissue_id', 'sample_id']],
                        left_on=['patient_number_03d', 'tissue_block_number_03d'],
                        right_on=['patient_number', 'tissue_id'], how='left')
    eneurons['hospital'] = [hos.split('-')[1] for hos in eneurons.sample_id]
    # show the distribution of neurons across different hospitals
    display = True
    if display:
        uniq_regs, cnt_regs = np.unique(eneurons.brain_region, return_counts=True)
        cnt_thr = 50
        uniq_regs_thr = uniq_regs[cnt_regs > cnt_thr]
        cur_neurons = eneurons[eneurons.brain_region.isin(uniq_regs_thr)]
        
    
        import ipdb; ipdb.set_trace()
        print()

if __name__ == '__main__':
    indir = '/PBshare/SEU-ALLEN/Users/ZhixiYun/data/human_regi'
    meta_file_tissue = '../meta/样本信息表10302024.xlsx.csv'
    meta_file_neuron = '../meta/1-50114.xlsx.csv'

    get_neurons(indir, meta_file_tissue, meta_file_neuron)

