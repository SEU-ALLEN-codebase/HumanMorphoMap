##########################################################
#Author:          Yufeng Liu
#Create time:     2024-05-16
#Description:               
##########################################################

import os
import glob
import sys
import subprocess
import time
import pandas as pd


def resample_swc(swc_in, swc_out, step=2, vaa3d='/opt/Vaa3D_x.1.1.4_ubuntu/Vaa3D-x', correction=True):
    cmd_str = f'xvfb-run -a -s "-screen 0 640x480x16" {vaa3d} -x resample_swc -f resample_swc -i "{swc_in}" -o "{swc_out}" -p {step}'
    print(swc_in, swc_out)
    p = subprocess.check_output(cmd_str, shell=True)
    if correction:
        # The built-in resample_swc has a bug: the first node is commented out, and there are two additional columns
        subprocess.run(f"sed -i 's/pid1/pid\\n1/g' '{swc_out}'; sed -i 's/ -1 -1//g' '{swc_out}'", shell=True)
    return True

def sort_swc(swc_in, swc_out=None, vaa3d='/opt/Vaa3D_x.1.1.4_ubuntu/Vaa3D-x'):
    cmd_str = f'xvfb-run -a -s "-screen 0 640x480x16" {vaa3d} -x sort_neuron_swc -f sort_swc -i "{swc_in}" -o "{swc_out}"'
    p = subprocess.check_output(cmd_str, shell=True)

    # retype
    df = pd.read_csv(swc_out, sep=' ', names=('#id', 'type', 'x', 'y', 'z', 'r', 'p'), comment='#', index_col=False)
    df['type'] = 3
    df.loc[0, 'type'] = 1
    df.to_csv(swc_out, sep=' ', index=False)

    return True

def resample_sort_swc(swc_in, swc_out):
    resample_swc(swc_in, swc_out)
    sort_swc(swc_out, swc_out)


if __name__ == '__main__':
    if 1: 
        # resample manual morphologies
        #indir = '/PBshare/SEU-ALLEN/Users/XiaoqinGu/dataset/H01_pyramidal_w_dendriteano/cut_new/level3'    # 25um
        #outdir = './data/H01_resample1um_prune25um'
        
        #indir = '/data/kfchen/trace_ws/paper_trace_result/final_data_and_meta_filter/swc_1um'
        #indir = './data/auto8.4k_0510_pruned'
        #outdir = './data/auto8.4k_0510_pruned_resample1um'
        
        # manual data
        indir = '/data/kfchen/trace_ws/paper_auto_human_neuron_recon/swc_label/1um_swc_lab'
        outdir = './data/manual_resampled1um'

        step = 1    # 1um
        args_list = []
        for swcfile in glob.glob(os.path.join(indir, '*.swc')):
            fn = os.path.split(swcfile)[-1]
            oswcfile = os.path.join(outdir, fn)
            if not os.path.exists(oswcfile):
                args_list.append((swcfile, oswcfile, step))


    # multiprocessing
    from multiprocessing import Pool
    pool = Pool(processes=16)
    #pool.starmap(resample_sort_swc, args_list)
    pool.starmap(resample_swc, args_list)
    pool.close()
    pool.join()
