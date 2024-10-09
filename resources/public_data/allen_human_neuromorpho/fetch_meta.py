import os
import glob
import subprocess
import json

swc_dir = 'CNG_version'
meta_dir = 'meta'

neurons = {}
for swcfile in glob.glob(os.path.join(swc_dir, '*swc')):
    neuron_name = os.path.split(swcfile)[-1][:-8]
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
        # dump to file
        with open(cur_meta_file, 'w') as f1:
            json.dump(neuron, f1, indent=4)
    
    neurons[neuron['neuron_name']] = neuron

if 1:
    # save to file
    with open('meta_info.json', 'w') as fp:
        json.dump(neurons, fp, indent=4)

