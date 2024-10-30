##########################################################
#Author:          Yufeng Liu
#Create time:     2024-10-08
#Description:               
##########################################################
import os
import glob
import numpy as np
import pandas as pd

def soma_converter(swcfile, out_dir):
    outfile = os.path.join(out_dir, os.path.split(swcfile)[-1])
    if os.path.exists(outfile):
        return

    try:
        df = pd.read_csv(swcfile, comment='#', sep=' ', usecols=range(1,8), 
                     names=('id', 'type', 'x', 'y', 'z', 'r', 'pid'), index_col=0)
    except pd.errors.ParserError:
        df = pd.read_csv(swcfile, comment='#', sep=' ', usecols=range(0,7), 
                     names=('id', 'type', 'x', 'y', 'z', 'r', 'pid'), index_col=0)

    # find out the non-center soma point
    ncs = (df.type == 1) & (df.pid != -1)
    if ncs.sum() == 1:
        raise ValueError
        
        # the SWC is fragmentated in this case!
        idx = df.index[np.nonzero(ncs)[0]][0]
        df = df.iloc[np.nonzero(~ncs)[0]]
        
        assert (df.pid == idx).sum() == 0
        indices = df.index.values.copy()
        indices[indices > idx] = indices[indices > idx] - 1
        df.index = indices

        pindices = df.pid.values.copy()
        pindices[pindices > idx] = pindices[pindices > idx] - 1
        df.pid = pindices
    elif ncs.sum() == 2:
        # re-ordering all nodes
        id1, id2 = df.index[np.nonzero(ncs)[0]]
        # make sure no nodes are connected to these two points
        assert (df.pid == id1).sum() == 0
        assert (df.pid == id2).sum() == 0
        if id1 > id2:
            id1, id2 = id2, id1

        df = df.iloc[np.nonzero(~ncs)[0]]
        indices = df.index.values.copy()
        indices[indices > id2] = indices[indices > id2] - 1
        indices[indices > id1] = indices[indices > id1] - 1
        df.index = indices

        # also for the parent indices
        pindices = df.pid.values.copy()
        pindices[pindices > id2] = pindices[pindices > id2] - 1
        pindices[pindices > id1] = pindices[pindices > id1] - 1
        df.pid = pindices
    else:
        raise ValueError

    # save out
    df.to_csv(outfile, sep=' ', index=True, header=False)


in_dir = './CNG_version'
out_dir = './one_point_soma'
error_files = set(
    ['H16-03-007-01-01-08-01_564395300_m.swc', # broken file
     'cell1_sorted_Scaled.swc', # those cells are problematics: 1) incorrect soma format; 2) wrong radius of soma
     'cell3_sorted_Scaled.swc', 
     'cell5_sorted_Scaled.swc',
     'cell2_sorted_Scaled.swc', 
     'cell4_PV_sorted_Scaled.swc']
)

for infile in glob.glob(os.path.join(in_dir, '*.swc')):
    if os.path.split(infile)[-1] in error_files:
        continue
    
    print(infile)
    soma_converter(infile, out_dir)
    
    
