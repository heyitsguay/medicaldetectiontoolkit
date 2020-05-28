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

import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import numpy as np
from default_configs import DefaultConfigs

CLASS_SPECIFIC_SEG = False


class configs(DefaultConfigs):

    def __init__(self, server_env=False):

        #########################
        #    Preprocessing      #
        #########################

        self.root_dir = '/home/matthew/datasets/platelet-instance-3d'
        self.min_volume = 20

        #########################
        #         I/O           #
        #########################

        # Either 'cell' or 'organelle'
        self.label_type = 'organelle'

        # one out of [2, 3]. dimension the model operates in.
        self.dim = 3

        # one out of ['mrcnn', 'retina_net', 'retina_unet', 'detection_unet', 'ufrcnn'].
        self.model = 'mrcnn'

        DefaultConfigs.__init__(self, self.model, server_env, self.dim)

        # int [0 < dataset_size]. select n patients from dataset for prototyping.
        self.select_prototype_subset = None
        self.hold_out_test_set = True
        # including val set. will be 3/4 train, 1/4 val.
        self.n_train_val_data = 8

        # path to preprocessed data.
        self.input_df_name = 'info_df.pickle'
        self.pp_name = os.path.join('mdt', self.label_type, 'train')
        self.pp_data_path = os.path.join(self.root_dir, self.pp_name)
        self.pp_test_name = os.path.join('mdt', self.label_type, 'test')
        self.pp_test_data_path = os.path.join(self.root_dir, self.pp_test_name)

        # settings for deployment in cloud.
        if server_env:
            # path to preprocessed data.
            pp_root_dir = '/datasets/datasets_ramien/toy_exp/data'
            self.pp_name = os.path.join('mdt', self.label_type, 'train')
            self.pp_data_path = os.path.join(pp_root_dir, self.pp_name)
            self.pp_test_name = os.path.join('mdt', self.label_type, 'test')
            self.pp_test_data_path = os.path.join(pp_root_dir, self.pp_test_name)
            self.select_prototype_subset = None

        #########################
        #      Data Loader      #
        #########################

        # select modalities from preprocessed data
        self.channels = [0]
        self.n_channels = len(self.channels)

        # patch_size to be used for training. pre_crop_size is the patch_size before data augmentation.
        self.pre_crop_size_3D = [256, 256, 32]
        self.patch_size_3D = [256, 256, 32]

        self.patch_size = self.patch_size_3D
        self.pre_crop_size = self.pre_crop_size_3D

        # ratio of free sampled batch elements before class balancing is triggered
        # (>0 to include "empty"/background patches.)
        self.batch_sample_slack = 0.2

        # set 2D network to operate in 3D images.
        self.merge_2D_to_3D_preds = False

        # feed +/- n neighbouring slices into channel dimension. set to None for no context.
        self.n_3D_context = None
        if self.n_3D_context is not None and self.dim == 2:
            self.n_channels *= (self.n_3D_context * 2 + 1)


        #########################
        #      Architecture      #
        #########################

        self.start_filts = 48 if self.dim == 2 else 24
        self.end_filts = self.start_filts * 4 if self.dim == 2 else self.start_filts * 2
        self.res_architecture = 'resnet50' # 'resnet101', 'resnet50'
        self.norm = None # one of None, 'instance_norm', 'batch_norm'
        # 0 for no weight decay
        self.weight_decay = 3e-8
        # which weights to exclude from weight decay, options: ["norm", "bias"].
        self.exclude_from_wd = ("norm",)

        # one of 'xavier_uniform', 'xavier_normal', or 'kaiming_normal', None (= default = 'kaiming_uniform')
        self.weight_init = None

        #########################
        #  Schedule / Selection #
        #########################

        self.num_epochs = 25
        self.num_train_batches = 50
        self.batch_size = 4

        self.do_validation = True
        # decide whether to validate on entire patient volumes (like testing) or sampled patches (like training)
        # the former is more accurate, while the latter is faster (depending on volume size)
        self.val_mode = 'val_sampling' # one of 'val_sampling' , 'val_patient'
        if self.val_mode == 'val_patient':
            self.max_val_patients = None  # if 'None' iterates over entire val_set once.
        if self.val_mode == 'val_sampling':
            self.num_val_batches = 8

        # set dynamic_lr_scheduling to True to apply LR scheduling with below settings.
        self.dynamic_lr_scheduling = True
        self.lr_decay_factor = 0.7
        self.scheduling_patience = np.ceil(7200 / (self.num_train_batches * self.batch_size))
        self.scheduling_criterion = 'all_ap'
        self.scheduling_mode = 'min' if "loss" in self.scheduling_criterion else 'max'

        #########################
        #   Testing / Plotting  #
        #########################

        # set the top-n epochs to be saved for temporal averaging in testing.
        self.save_n_models = 5
        self.test_n_epochs = 5

        # set a minimum epoch number for saving in case of instabilities in the first phase of training.
        self.min_save_thresh = 0 if self.dim == 2 else 0

        self.report_score_level = ['patient', 'rois']  # choose list from 'patient', 'rois'
        self.class_dict = {1: 'all'}  # 0 is background.
        self.patient_class_of_interest = 1  # patient metrics are only plotted for one class.
        self.ap_match_ious = [0.1, 0.5, 0.7, 0.8, 0.9]  # list of ious to be evaluated for ap-scoring.

        self.model_selection_criteria = ['all_ap']  # criteria to average over for saving epochs.
        self.min_det_thresh = 0.1  # minimum confidence value to select predictions for evaluation.

        # threshold for clustering predictions together (wcs = weighted cluster scoring).
        # needs to be >= the expected overlap of predictions coming from one model (typically NMS threshold).
        # if too high, preds of the same object are separate clusters.
        self.wcs_iou = 1e-2

        self.plot_prediction_histograms = True
        self.plot_stat_curves = True

        #########################
        #   Data Augmentation   #
        #########################

        self.da_kwargs={
            'do_elastic_deform': True,
            'alpha':(0., 150.),
            'sigma':(10., 30.),
            'do_rotation':False,
            'angle_x': (0., 2 * np.pi),
            'angle_y': (0., 0),
            'angle_z': (0., 0),
            'do_scale': False,
            'scale':(0.8, 1.1),
            'random_crop':False,
            'rand_crop_dist':  (self.patch_size[0] / 2. - 3, self.patch_size[1] / 2. - 3),
            'border_mode_data': 'constant',
            'border_cval_data': 0,
            'order_data': 1
        }

        if self.dim == 3:
            self.da_kwargs['do_elastic_deform'] = False
            self.da_kwargs['angle_x'] = (0, 0.0)
            self.da_kwargs['angle_y'] = (0, 0.0) #must be 0!!
            self.da_kwargs['angle_z'] = (0., 2 * np.pi)


        #########################
        #   Add model specifics #
        #########################

        {'detection_unet': self.add_det_unet_configs,
         'mrcnn': self.add_mrcnn_configs,
         'ufrcnn': self.add_mrcnn_configs,
         'retina_net': self.add_mrcnn_configs,
         'retina_unet': self.add_mrcnn_configs,
        }[self.model]()


    def add_det_unet_configs(self):

        self.learning_rate = [1e-4] * self.num_epochs

        # aggregation from pixel prediction to object scores (connected component). One of ['max', 'median']
        self.aggregation_operation = 'max'

        # max number of roi candidates to identify per image (slice in 2D, volume in 3D)
        self.n_roi_candidates = 100

        # loss mode: either weighted cross entropy ('wce'), batch-wise dice loss ('dice), or the sum of both ('dice_wce')
        self.seg_loss_mode = 'dice_wce'

        # if <1, false positive predictions in foreground are penalized less.
        self.fp_dice_weight = 1 if self.dim == 2 else 1

        self.wce_weights = [0.3, 1, 1]
        self.detection_min_confidence = self.min_det_thresh

        # if 'True', loss distinguishes all classes, else only foreground vs. background (class agnostic).
        self.class_specific_seg_flag = CLASS_SPECIFIC_SEG
        self.num_seg_classes = 2
        self.head_classes = self.num_seg_classes

    def add_mrcnn_configs(self):

        # learning rate is a list with one entry per epoch.
        self.learning_rate = [3e-4] * self.num_epochs

        # disable mask head loss. (e.g. if no pixelwise annotations available)
        self.frcnn_mode = False

        # disable the re-sampling of mask proposals to original size for speed-up.
        # since evaluation is detection-driven (box-matching) and not instance segmentation-driven (iou-matching),
        # mask-outputs are optional.
        self.return_masks_in_val = True
        self.return_masks_in_test = True

        # set number of proposal boxes to plot after each epoch.
        self.n_plot_rpn_props = 0 if self.dim == 2 else 2

        # number of classes for head networks: n_foreground_classes + 1 (background)
        self.head_classes = 2

        # seg_classes hier refers to the first stage classifier (RPN)
        self.num_seg_classes = 2  # foreground vs. background

        # feature map strides per pyramid level are inferred from architecture.
        self.backbone_strides = {'xy': [4, 8, 16, 32], 'z': [1, 2, 4, 8]}

        # anchor scales are chosen according to expected object sizes in data set. Default uses only one anchor scale
        # per pyramid level. (outer list are pyramid levels (corresponding to BACKBONE_STRIDES), inner list are scales per level.)
        self.rpn_anchor_scales = {'xy': [[8], [16], [32], [64]], 'z': [[2], [4], [8], [16]]}

        # choose which pyramid levels to extract features from: P2: 0, P3: 1, P4: 2, P5: 3.
        self.pyramid_levels = [0, 1, 2, 3]

        # number of feature maps in rpn. typically lowered in 3D to save gpu-memory.
        self.n_rpn_features = 512 if self.dim == 2 else 256

        # anchor ratios and strides per position in feature maps.
        self.rpn_anchor_ratios = [0.5, 1., 2.]
        self.rpn_anchor_stride = 1

        # Threshold for first stage (RPN) non-maximum suppression (NMS):  LOWER == HARDER SELECTION
        self.rpn_nms_threshold = 0.7 if self.dim == 2 else 0.7

        # loss sampling settings.
        self.rpn_train_anchors_per_image = 100 #per batch element
        self.train_rois_per_image = 100 #per batch element
        self.roi_positive_ratio = 0.5
        self.anchor_matching_iou = 0.7

        # factor of top-k candidates to draw from  per negative sample (stochastic-hard-example-mining).
        # poolsize to draw top-k candidates from will be shem_poolsize * n_negative_samples.
        self.shem_poolsize = 10

        self.pool_size = (7, 7) if self.dim == 2 else (7, 7, 3)
        self.mask_pool_size = (14, 14) if self.dim == 2 else (14, 14, 5)
        self.mask_shape = (28, 28) if self.dim == 2 else (28, 28, 10)

        self.rpn_bbox_std_dev = np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2])
        self.bbox_std_dev = np.array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2])
        self.window = np.array([0, 0, self.patch_size[0], self.patch_size[1], 0, self.patch_size[2]])
        self.scale = np.array([self.patch_size[0], self.patch_size[1], self.patch_size[0], self.patch_size[1], self.patch_size[2], self.patch_size[2]])

        if self.dim == 2:
            self.rpn_bbox_std_dev = self.rpn_bbox_std_dev[:4]
            self.bbox_std_dev = self.bbox_std_dev[:4]
            self.window = self.window[:4]
            self.scale = self.scale[:4]

        # pre-selection in proposal-layer (stage 1) for NMS-speedup. applied per batch element.
        self.pre_nms_limit = 3000 if self.dim == 2 else 6000

        # n_proposals to be selected after NMS per batch element. too high numbers blow up memory if "detect_while_training" is True,
        # since proposals of the entire batch are forwarded through second stage in as one "batch".
        self.roi_chunk_size = 2500 if self.dim == 2 else 600
        self.post_nms_rois_training = 500 if self.dim == 2 else 150
        self.post_nms_rois_inference = 500

        # Final selection of detections (refine_detections)
        self.model_max_instances_per_batch_element = 80  # per batch element and class.
        self.detection_nms_threshold = 1e-2  # needs to be > 0, otherwise all predictions are one cluster.
        self.model_min_confidence = 0.1

        if self.dim == 2:
            self.backbone_shapes = np.array(
                [[int(np.ceil(self.patch_size[0] / stride)),
                  int(np.ceil(self.patch_size[1] / stride))]
                 for stride in self.backbone_strides['xy']])
        else:
            self.backbone_shapes = np.array(
                [[int(np.ceil(self.patch_size[0] / stride)),
                  int(np.ceil(self.patch_size[1] / stride)),
                  int(np.ceil(self.patch_size[2] / stride_z))]
                 for stride, stride_z in zip(self.backbone_strides['xy'], self.backbone_strides['z']
                                             )])
        if self.model == 'ufrcnn':
            self.operate_stride1 = True
            self.class_specific_seg_flag = CLASS_SPECIFIC_SEG
            self.num_seg_classes = 3 if self.class_specific_seg_flag else 2
            self.frcnn_mode = True

        if self.model == 'retina_net' or self.model == 'retina_unet':
            # implement extra anchor-scales according to retina-net publication.
            self.rpn_anchor_scales['xy'] = [[ii[0], ii[0] * (2 ** (1 / 3)), ii[0] * (2 ** (2 / 3))] for ii in
                                            self.rpn_anchor_scales['xy']]
            self.rpn_anchor_scales['z'] = [[ii[0], ii[0] * (2 ** (1 / 3)), ii[0] * (2 ** (2 / 3))] for ii in
                                           self.rpn_anchor_scales['z']]
            self.n_anchors_per_pos = len(self.rpn_anchor_ratios) * 3

            self.n_rpn_features = 256 if self.dim == 2 else 64

            # pre-selection of detections for NMS-speedup. per entire batch.
            self.pre_nms_limit = 10000 if self.dim == 2 else 50000

            # anchor matching iou is lower than in Mask R-CNN according to https://arxiv.org/abs/1708.02002
            self.anchor_matching_iou = 0.5

            # if 'True', seg loss distinguishes all classes, else only foreground vs. background (class agnostic).
            self.num_seg_classes = 3 if self.class_specific_seg_flag else 2

            if self.model == 'retina_unet':
                self.operate_stride1 = True
