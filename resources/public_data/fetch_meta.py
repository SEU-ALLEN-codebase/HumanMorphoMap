import os
import glob
import subprocess
import json
import numpy as np
import pandas as pd

dataset = 'allman'
swc_dir = f'{dataset}/CNG_version'
meta_dir = f'{dataset}/meta'

if not os.path.exists(meta_dir):
    os.mkdir(meta_dir)

if 1:
    neurons = {}
    for swcfile in glob.glob(os.path.join(swc_dir, '*swc')):
        neuron_name = os.path.split(swcfile)[-1][:-4]
        print(neuron_name)
        # check if it has been dumped
        cur_meta_file = os.path.join(meta_dir, f'{neuron_name}.json')
        if os.path.exists(cur_meta_file):
            with open(cur_meta_file, 'r') as f0:
                neuron = json.load(f0)
        else:
            info_str = subprocess.check_output(f'curl -X GET "http://cng.gmu.edu:8080/api/neuron/name/{neuron_name}"', shell=True)
            # decode as json
            neuron = json.loads(info_str.decode('utf-8'))
            
            if 'neuron_name' not in neuron:
                raise ValueError("CURL error: info_str")
            # dump to file
            with open(cur_meta_file, 'w') as f1:
                json.dump(neuron, f1, indent=4)
        
        neurons[neuron['neuron_name']] = neuron

    # save to file
    with open(f'{dataset}/meta_info.json', 'w') as fp:
        json.dump(neurons, fp, indent=4)

if 0:
    # extract the meta-informations, including layers, brain regions, and cell types.
    meta_file = 'meta_info.json'
    with open(meta_file, 'r') as fp:
        meta = json.load(fp)

    meta_extracted = []
    for fn, meta_i in meta.items():
        print(fn)
        region = meta_i['brain_region']
        cell_type = meta_i['cell_type']

        lr, layer, reg = '', '', ''
        for region_i in region:
            if (region_i == 'left') or (region_i == 'right'):
                lr = region_i
            elif region_i.startswith('layer'):
                layer = region_i
            elif (len(region_i.split(' ')) == 3) or (region_i in ['primary visual', 'superior gyrus']):
                reg = region_i

        # cell type information
        pyr_c, exc_c = '', ''
        for ct in cell_type:
            if (pyr_c == '') and (ct in ['pyramidal', 'Spiny', 'interneuron', 'Aspiny']):
                pyr_c = ct
                continue

            if ct in ['Excitatory', 'Inhibitory']:
                exc_c = ct
                
        meta_extracted.append([fn, lr, layer, reg, pyr_c, exc_c])
    
    df_meta = pd.DataFrame(meta_extracted, 
        columns=('name', 'lr', 'layer', 'region', 'cell_type', 'neuron_type')
    )
    # assume all interneuron are inhibitory
    #df_meta.loc[np.nonzero(df_meta.cell_type == 'interneuron')[0], 'neuron_type'] = 'Inhibitory'
    df_meta.loc[np.nonzero(df_meta.cell_type == 'pyramidal')[0], 'neuron_type'] = 'Excitatory'
    df_meta.reset_index('name', inplace=True)
    df_meta.to_csv('meta_regions_types.csv')

