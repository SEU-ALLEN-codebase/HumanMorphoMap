##########################################################
#Author:          Yufeng Liu
#Create time:     2024-05-16
#Description:               
##########################################################

import os
import glob
import sys
import numpy as np
import pandas as pd
from swc_handler import resample_swc


if __name__ == '__main__':

    if 0:
        # resample manual morphologies
        indir = 'merged'
        step = 1
        outdir = f'merged_{step}um'
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

    if 1:
        # Estimate the layer information based on statistics
        coarse_boarders = np.array([252, 1201, 1582, 2772])
        coarse_layers = ['L1', 'L2/3', 'L4', 'L5/6']
        # layer thickness ratio in DOI: 10.1023/a:1024130211265
        fine_boarders = np.array([252, 514, 1201, 1582, 2219, 2772])
        fine_layers = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6']
        
        # screen over all neurons
        swc_dir = 'merged_1um'
        meta_file = 'meta_DeKock.csv'
        
        meta = []
        for swcfile in glob.glob(os.path.join(swc_dir, '*swc')):
            swc_name = os.path.split(swcfile)[-1]
            if swc_name == 'H42-06.swc':    # unknown yet
                meta.append([swc_name[:-4], '', ''])
                continue
            pia_soma_dist = int(swc_name.split('_')[0])
            # assign cortical layers
            # coarse
            cidx = min((coarse_boarders < pia_soma_dist).sum(), coarse_boarders.shape[0] - 1)
            clayer = coarse_layers[cidx]
            # fine
            fidx = min((fine_boarders < pia_soma_dist).sum(), fine_boarders.shape[0] - 1)
            flayer = fine_layers[fidx]

            meta.append([swc_name[:-4], clayer, flayer])

        # save to file
        meta = pd.DataFrame(meta, columns=('name', 'coarse_layer', 'fine_layer'))
        meta.to_csv(meta_file, index=False)

