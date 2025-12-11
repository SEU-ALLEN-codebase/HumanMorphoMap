##########################################################
#Author:          Yufeng Liu
#Create time:     2025-12-10
#Description:               
##########################################################
import os
import glob
import warnings
import struct
import pickle
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from scipy.spatial.distance import pdist

os.environ["TIFFWARNINGS"] = "0"
import SimpleITK as sitk

warnings.filterwarnings("ignore")
sitk.ProcessObject.GlobalWarningDisplayOff()  # 关闭全局警告显示

def get_v3draw_size(rawfile):
    # get the image size
    with open(rawfile, 'rb') as f:
        formatkey = "raw_image_stack_by_hpeng"
        f.seek(len(formatkey))
        endianCodeData = f.read(1).decode('utf-8')
        if endianCodeData == 'B':
            endian = '>'
        elif endianCodeData == 'L':
            endian = '<'
        else:
            print('A')

        print(endian)
        f.seek(len(formatkey) + 1 + 2)
        width, height, depth, channel = struct.unpack(endian + 'iiii', f.read(4*4))

    return width, height, depth, channel

def get_distances(meta_n_file, image_dir, out_file):
    hospitals = ['BJ-TT', 'NanJ-JSP', 'NanJ-JZ', 'NanJ-NK', 'SH-HS']
    meta_n = pd.read_csv(meta_n_file, index_col=2, low_memory=False)

    dists_dict = {}
    for hospital in hospitals:
        hos_dir = os.path.join(image_dir, hospital)
        for oper_dir in glob.glob(os.path.join(hos_dir, '*')):
            # each operator
            if not os.path.isdir(oper_dir):
                continue
            oper = os.path.split(oper_dir)[-1]
            print(hospital, oper)

            for sdir in glob.glob(os.path.join(oper_dir, '*')):
                if not os.path.isdir(sdir):
                    continue
                sname = os.path.split(sdir)[-1]

                #if len(dists_dict) > 2:   # debug
                #    break

                # check if there are images
                zseries = list(glob.glob(os.path.join(sdir, 'ZSeries*')))
                atlvol= list(glob.glob(os.path.join(sdir, 'AtlasVolume*')))
                images = zseries + atlvol
                if len(images) < 2: # discard if < 2 images, but it may contain multiple cells.
                    continue

                # match the the meta table
                matched_images = {}
                for image in images:
                    docname = image.split('-')[-1]
                    uploaded = meta_n.document_name == docname
                    if uploaded.sum() == 1:
                        # matched, get the exact coordinates
                        matched_images[image] = meta_n.index[uploaded][0]
                    elif uploaded.sum() > 1:
                        # There are duplicate document name, be aware of that!
                        cur_matches = meta_n[uploaded].shooting_staff == oper
                        if cur_matches.sum() != 1:
                            print(f'   --- Duplicate document name {docname} for image: {image}')
                            continue
                        else:
                            cur_idx = np.nonzero(cur_matches)[0][0]
                            matched_images[image] = meta_n.index[uploaded][cur_idx]
                    else:
                        continue

                # skip samples with matching cells < 2
                if len(matched_images) < 2:
                    continue
                print(f'   {sname}: {len(matched_images)}')

                # double check into subsets by shooting date
                cur_images = meta_n.loc[matched_images.values()]
                matched_images_names = np.array([*matched_images.keys()])
                shooting_dates = cur_images.shooting_date
                dates, ndates = np.unique(cur_images.shooting_date.values, return_counts=True)

                i_metas_l = []
                i_images_l = []
                if len(dates) > 1:
                    print(f'   >>> Multiple dates: {dates} for {sname}')
                    # splitting
                    for date, ndate in zip(dates, ndates):
                        if ndate < 2:
                            continue
                        i_mask = cur_images.shooting_date == date
                        i_metas_l.append(cur_images[i_mask])
                        i_images_l.append(matched_images_names[i_mask])
                else:
                    i_images_l.append(matched_images_names)
                    i_metas_l.append(cur_images)

                # iterative over all neurons
                for i_images, i_metas in zip(i_images_l, i_metas_l):
                    # This is a separate imaging set within a slice
                    if isinstance(i_images, np.ndarray):
                        tmp_name = os.path.split(i_images[0])[-1]
                    elif isinstance(i_images, str):
                        tmp_name = os.path.split(i_images)[-1]
                    else:
                        raise ValueError("Error HHH")

                    img_key = f'{hospital}_{oper}_{sname}_{tmp_name}'
                    coords = []
                    indices = []
                    for ii in range(len(i_images)):
                        # get the coordinate
                        i_image = i_images[ii]
                        slice_name, cell_id = os.path.split(i_image)
                        i_meta = i_metas.iloc[ii]
                        # get the coordinate from xml file
                        xmlfile = os.path.join(i_image, f'{os.path.split(i_image)[-1]}.xml')
                        if not os.path.exists(xmlfile):
                            continue

                        image_xyz_um = []
                        for event, elem in ET.iterparse(xmlfile, events=('end',)):
                            if elem.tag == 'PVStateValue' and elem.attrib['key'] == 'positionCurrent':
                                for sub_elem in elem.findall(".//SubindexedValue"):
                                    image_xyz_um.append(float(sub_elem.attrib['value']))
                                elem.clear()
                                break

                        # get the image size
                        image_t = cell_id.split('-')[0]
                        if image_t == 'ZSeries':
                            # infer the image size from the tif files
                            tifs = list(glob.glob(os.path.join(i_image, '*.tif')))
                            depth = len(tifs)
                            if depth < 50:
                                continue
                            # get the x, y
                            tif = sitk.ReadImage(tifs[0])
                            width, height = tif.GetWidth(), tif.GetHeight()
                        elif image_t == 'AtlasVolume':
                            # The image size should be inferred from the image stack
                            # first, find the image stack
                            raw_file1 = os.path.join(slice_name, 'v3draw_16bit', cell_id.split('-')[0],
                                                    f'{cell_id}.v3draw')
                            raw_file2 = f'{i_image}.v3draw'
                            raw_file3 = f'{i_image}/{cell_id}.v3draw'
                            raw_file = None
                            for rfile in [raw_file1, raw_file2, raw_file3]:
                                if os.path.exists(rfile):
                                    raw_file = rfile
                                    break
                            if raw_file is None:
                                print(f'   --> Not found: {i_image}')
                                continue

                            width, height, depth, channel = get_v3draw_size(raw_file)

                        else:
                            raise TypeError()

                        # calculate the coordinates
                        try:
                            sx, sy, sz = i_meta[['soma_x', 'soma_y', 'soma_z']].astype(float)
                        except ValueError:
                            continue

                        dx, dy, dz = width/2 - sx, height/2 - sy, depth/2 - sz
                        xi = image_xyz_um[0] - dx*float(i_meta.xy_resolution)/1000.
                        yi = image_xyz_um[1] - dy*float(i_meta.xy_resolution)/1000.
                        zi = image_xyz_um[2] - dz*float(i_meta.z_resolution)/1000.

                        coords.append((xi,yi,zi))
                        indices.append(i_meta.name)

                    # estimate the similarity between them
                    if len(coords) < 2:
                        continue

                    # pairwise distance
                    dists = pdist(coords)
                    if dists.max() > 5000: # mostly error
                        continue

                    dists_dict[img_key] = (indices, dists)

    # save the data to file
    with open(out_file, 'wb') as fp:
        pickle.dump(dists_dict, fp)

if __name__ == '__main__':
    meta_file_neuron = '../meta/1-50114.xlsx.csv'
    image_dir = '/human402/Human_Single_Imaging_Raw_Data/Manual_cell_data/After_20220715'
    out_file = '../data/pairwise_distance_in_slice.pkl'
    
    get_distances(meta_file_neuron, image_dir, out_file)
    
    
