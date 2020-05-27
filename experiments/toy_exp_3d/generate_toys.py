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
from scipy.ndimage.measurements import label

DO_MP = True


def sphere_mask(im_shape, center, diameter):
    ndims = len(im_shape)
    coords = [list(range(s)) for s in im_shape]
    if ndims == 2:
        raise NotImplementedError('Not implemented')
    if ndims == 3:
        mx, my, mz = np.meshgrid(*coords)
        mask = (mx - center[0]) ** 2 + (my - center[1]) ** 2 + \
               (mz - center[2]) ** 2 < diameter ** 2
        return mask
    else:
        raise ValueError('ndims must be 2 or 3')


def create_image(out_dir, six, foreground_margin, class_diameters, mode, noisy_bg, max_n_objs):

    print('\rprocessing {} {}'.format(out_dir, six), end="", flush=True)
    im_shape = (156, 156, 96)
    img = noisy_bg * np.random.rand(*im_shape) if noisy_bg > 0 else np.zeros(im_shape)
    seg = np.zeros(im_shape).astype('uint8')

    n_objs = np.random.randint(1, max_n_objs)

    class_id = np.random.randint(0, 2)
    # class_ids = []
    for n in range(n_objs):
        # class_id = np.random.randint(0, 2)
        center_x = np.random.randint(foreground_margin, img.shape[0] - foreground_margin)
        center_y = np.random.randint(foreground_margin, img.shape[1] - foreground_margin)
        center_z = np.random.randint(foreground_margin // 2, img.shape[2] - foreground_margin // 2)

        mask = sphere_mask(im_shape, (center_x, center_y, center_z), class_diameters[class_id])
        img[mask] += 0.2
        seg[mask] = n + 1

        if 'donuts' in mode:
            hole_diameter = 6
            if class_id == 1:
                hole_mask = sphere_mask(im_shape, (center_x, center_y, center_z), hole_diameter)
                img[hole_mask] -= 0.2
                seg[hole_mask] = 0

    # cluster, nclusters = label(seg)
    out = np.concatenate((img[None], seg[None]))
    out_path = os.path.join(out_dir, '{}.npy'.format(six))
    np.save(out_path, out)

    with open(os.path.join(out_dir, 'meta_info_{}.pickle'.format(six)), 'wb') as handle:
        pickle.dump([out_path, class_id, str(six)], handle)


def generate_dataset(cf, exp_name, n_train_images, n_test_images, mode, class_diameters=(20, 20), noisy_bg=0, max_n_objs=10):

    train_dir = os.path.join(cf.root_dir, exp_name, 'train')
    test_dir = os.path.join(cf.root_dir, exp_name, 'test')
    if os.path.isdir(train_dir) or os.path.isdir(test_dir):
        raise Exception("A dataset directory already exists at {}. ".format(cf.root_dir)+
                        "Please make sure to generate data in an empty or new directory.")
    os.makedirs(train_dir, exist_ok=False)
    os.makedirs(test_dir, exist_ok=False)

    # enforced distance between object center and image edge.
    foreground_margin = int(np.ceil(np.max(class_diameters) / 1.25))

    info = []
    info += [[train_dir, six, foreground_margin, class_diameters, mode, noisy_bg, max_n_objs] for six in range(n_train_images)]
    info += [[test_dir, six, foreground_margin, class_diameters, mode, noisy_bg, max_n_objs] for six in range(n_test_images)]

    print('starting creation of {} images'.format(len(info)))
    if DO_MP:
        pool = Pool(processes=os.cpu_count()-1)
        pool.starmap(create_image, info, chunksize=1)
        pool.close()
        pool.join()
    else:
        for inputs in info:
            create_image(*inputs)
    print()
    aggregate_meta_info(train_dir)
    aggregate_meta_info(test_dir)


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

    # parser = argparse.ArgumentParser()
    # mode_choices = ['donuts_shape_3d', 'donuts_pattern_3d', 'circles_scale_3d']
    # parser.add_argument('-m', '--modes', nargs='+', type=str, default=mode_choices, choices=mode_choices)
    # parser.add_argument('--noise', action='store_true', help="if given, add noise to the sample bg.")
    # parser.add_argument('--n_train', type=int, default=400, help="Nr. of train images to generate.")
    # parser.add_argument('--n_test', type=int, default=400, help="Nr. of test images to generate.")
    # args = parser.parse_args()

    args_noise = 0.5
    args_n_train = 400
    args_n_test = 100
    args_mode = 'donuts_shape_3d'
    args_n_objs = 11


    cf_file = utils.import_module("cf", "configs.py")
    cf = cf_file.configs()

    class_diameters = {
        'donuts_shape_3d': (12, 20),
        'donuts_pattern_3d': (12, 20),
        'circles_scale_3d': (19, 20)
    }

    # for mode in args.modes:
    #     generate_dataset(cf, mode + ("_noise" if args.noise else ""), n_train_images=args.n_train, n_test_images=args.n_test, mode=mode,
    #                         class_diameters=class_diameters[mode], noisy_bg=args.noise)
    generate_dataset(cf, args_mode + ("_noise" if args_noise else ""), n_train_images=args_n_train, n_test_images=args_n_test, mode=args_mode,
                        class_diameters=class_diameters[args_mode], noisy_bg=args_noise,
                        max_n_objs=args_n_objs)


    mins, secs = divmod((time.time() - stime), 60)
    h, mins = divmod(mins, 60)
    t = "{:d}h:{:02d}m:{:02d}s".format(int(h), int(mins), int(secs))
    print("{} total runtime: {}".format(os.path.split(__file__)[1], t))


