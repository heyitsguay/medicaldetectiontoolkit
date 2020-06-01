"""Run inference on a piece of data

"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

import utils.exp_utils as utils
import utils.model_utils as mutils


def lcimb_predict(save_file, exp_dir, weight_path, x_path, cmap='prism'):
    """

    :param save_file:
    :param exp_dir:
    :param weight_path:
    :param x_path:
    :param cmap:
    :return:
    """
    cf = utils.prep_exp(exp_dir, exp_dir, server_env=False, is_training=False,
                        use_stored_settings=True)
    logger = utils.get_logger(cf.exp_dir, False)
    model = utils.import_module('module', cf.model_path)
    net = model.net(cf, logger).cuda()
    net.load_state_dict(torch.load(weight_path))

    x = np.load(x_path)
    img_rgb = np.stack([x.astype(np.float32)]*3, axis=0)
    x = torch.from_numpy(x)[0]
    x = x[None, None].cuda()
    y = net.forward(x)
    detections = y[3].cpu().data.numpy()
    masks = y[4].permute(0, 2, 3, 4, 1).cpu().data.numpy()

    batch_ixs = detections[:, 2 * cf.dim]
    detections = [detections[batch_ixs == ix] for ix in range(y.shape[0])]
    masks = [masks[batch_ixs == ix] for ix in range(y.shape[0])]

    for ix in range(y.shape[0]):
        boxes = detections[ix][:, :2 * cf.dim].astype(np.int32)
        class_ids = detections[ix][:, 2 * cf.dim + 1].astype(np.int32)
        scores = detections[ix][:, 2 * cf.dim + 2]
        masks = masks[ix][np.arange(boxes.shape[0]), ..., class_ids]
        exclude_ix = np.where(
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 5] - boxes[:, 4]) <= 0)[0]
        if exclude_ix.shape[0] > 0:
            boxes = np.delete(boxes, exclude_ix, axis=0)
            class_ids = np.delete(class_ids, exclude_ix, axis=0)
            scores = np.delete(scores, exclude_ix, axis=0)
            masks = np.delete(masks, exclude_ix, axis=0)

        full_masks = []
        permuted_image_shape = list(y.shape[2:]) + [y.shape[1]]
        for i in range(masks.shape[0]):
            full_masks.append(mutils.unmold_mask_3D(masks[i], boxes[i], permuted_image_shape))
        n_masks = len(full_masks)
        cmap = plt.get_cmap(cmap)
        for n in range(n_masks):
            mask = full_masks[n]
            reg = mask > 0.5
            mask_rgb = cmap(reg * (n / n_masks))[..., :3]
            img_rgb[reg, :] = 0.5 * mask_rgb[reg, :] + 0.5 * img_rgb[reg, :]
    img_out = np.clip(255 * img_rgb, 0, 255).astype(np.uint8)
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    plt.imsave(save_file, img_out)
    return img_out


if __name__ == '__main__':
    args = sys.argv
    save_file = args[0]
    exp_dir = args[1]
    weight_path = args[2]
    x_path = args[3]
    if len(args) > 4:
        cmap = args[4]
    else:
        cmap = 'prism'
    lcimb_predict(save_file, exp_dir, weight_path, x_path, cmap)
