#!/usr/bin/env python
# Copyright 2018 Division of Medical Image Computing, German Cancer Research Center (DKFZ).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os, time
import numpy as np
import pandas as pd
import pickle
import argparse
from multiprocessing import Pool
from skimage import io
from scipy.ndimage.measurements import label
from scipy.ndimage.interpolation import zoom

DO_MP = True


def process_image(out_dir, pid, image_in, label_in, is_cell, min_volume):
    img = np.swapaxes(image_in, 0, 2)
    labels = np.swapaxes(label_in, 0, 2)
    seg = np.zeros_like(labels)
    label_ids = range(1, labels.max() + 1)
    print(f'Processing chunk with {labels.max()} labels')
    # Allow diagonal connections between components
    structure = np.ones((3, 3, 3))
    n_segs = 0
    for l in label_ids:
        mask = labels == l
        ccs, n_ccs = label(mask, structure)
        for c in range(1, n_ccs + 1):
            cc_mask = ccs == c
            cc_vol = cc_mask.sum()
            if cc_vol >= min_volume:
                n_segs += 1
                seg[cc_mask] = n_segs
    print(f'Found {n_segs} labeled components')

    mi = np.percentile(img, 0.5)
    ma = np.percentile(img, 99.5)
    img = np.float32(np.clip((img - mi) / (ma - mi), 0, 1))
    # if is_cell:
    #     class_id = 0
    # else:
    #     class_id = 1
    class_id = 0
    out = np.concatenate((img[None], seg[None]))
    out_path = os.path.join(out_dir, '{}.npy'.format(pid))
    np.save(out_path, out)

    with open(os.path.join(out_dir, 'meta_info_{}.pickle'.format(pid)), 'wb') as handle:
        pickle.dump([out_path, class_id, str(pid)], handle)

    pass


def generate_dataset(cf):
    train_image = io.imread(os.path.join(cf.root_dir, 'images', 'train', '0000.tif'))
    eval_image = io.imread(os.path.join(cf.root_dir, 'images', 'eval', '0000.tif'))
    images = {'train': train_image, 'eval': eval_image}

    out_shape_z = 32
    overlap = 8
    dz = out_shape_z - overlap

    info = []
    pid = 0
    for label_type in ['cell', 'organelle']:
        pid = 0
        for subset in ['train', 'eval']:
            ss = 'train' if subset == 'train' else 'test'
            labels = io.imread(os.path.join(cf.root_dir, 'annotations', subset, label_type, '0000.tif')).astype(np.uint16)
            labels = zoom(labels, (4, 1, 1), order=0)
            save_dir = os.path.join(cf.root_dir, 'mdt', label_type, ss)
            if os.path.isdir(save_dir):
                raise Exception("A dataset directory already exists at {}. ".format(cf.root_dir) +
                                "Please make sure to generate data in an empty or new directory.")
            os.makedirs(save_dir, exist_ok=False)

            image = images[subset]
            dim_z = image.shape[0]
            zs = list(range(0, dim_z - out_shape_z, dz))
            if dim_z - out_shape_z not in zs:
                zs.append(dim_z-out_shape_z)
            for z in zs:
                z_range = slice(z, z + out_shape_z)
                info += [[save_dir, pid, image[z_range], labels[z_range],
                          label_type == 'cell', cf.min_volume]]
                pid += 1

    print('starting creation of {} images'.format(len(info)))
    if DO_MP:
        pool = Pool(processes=os.cpu_count()-1)
        pool.starmap(process_image, info, chunksize=1)
        pool.close()
        pool.join()
    else:
        for inputs in info:
            process_image(*inputs)

    for label_type in ['cell', 'organelle']:
        for subset in ['train', 'test']:
            save_dir = os.path.join(cf.root_dir, 'mdt', label_type, subset)
            aggregate_meta_info(save_dir)
    pass


def aggregate_meta_info(exp_dir):

    files = [os.path.join(exp_dir, f) for f in os.listdir(exp_dir) if 'meta_info' in f]
    df = pd.DataFrame(columns=['path', 'class_id', 'pid'])
    for f in files:
        with open(f, 'rb') as handle:
            df.loc[len(df)] = pickle.load(handle)

    df.to_pickle(os.path.join(exp_dir, 'info_df.pickle'))
    print("aggregated meta info to df with length", len(df))


if __name__ == '__main__':
    stime = time.time()
    import sys
    sys.path.append("../..")
    import utils.exp_utils as utils

    cf_file = utils.import_module("cf", "configs.py")
    cf = cf_file.configs()

    generate_dataset(cf)

    mins, secs = divmod((time.time() - stime), 60)
    h, mins = divmod(mins, 60)
    t = "{:d}h:{:02d}m:{:02d}s".format(int(h), int(mins), int(secs))
    print("{} total runtime: {}".format(os.path.split(__file__)[1], t))


