# Copyright (c) 2024 Valeo
# See LICENSE.md for details.

# pylint: skip-file
# flake8 : noqa

import argparse
import json
import logging
import tarfile

import cv2
import numpy as np
from sklearn import metrics
from tqdm import tqdm


DEBUG = False
CS_STUFF_CLASSES = [11, 12, 13, 14, 15, 16, 17, 18, 19]

logger = logging.getLogger('bravo_toolkit')


def tqqdm(iterable, *args, **kwargs):
    if logger.getEffectiveLevel() <= logging.INFO:
        return tqdm(iterable, *args, **kwargs)
    return iterable


def load_tar(tar_path):
    return tarfile.open(tar_path)


def read_image_from_tar(tar, member, flag=cv2.IMREAD_GRAYSCALE):
    try:
        f = tar.extractfile(member)
        content = f.read()
        f.close()
        file_bytes = np.asarray(bytearray(content), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, flag)
        if img is None:
            raise ValueError('error decoding image')
        return img
    except:
        logger.error('Unable to load file {}'.format(member.name))
        raise IOError('Unable to load file {}'.format(member.name))


def read_image_from_tar_conf(tar, member, flag=cv2.IMREAD_UNCHANGED):
    try:
        f = tar.extractfile(member)
        content = f.read()
        f.close()
        file_bytes = np.asarray(bytearray(content), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, flag)
        if img is None:
            raise ValueError('error decoding image')
        return img
    except:
        logger.error('Unable to load file {}'.format(member.name))
        raise IOError('Unable to load file {}'.format(member.name))


def fast_hist(a, b, n):
    # Fast calculation of the confusion matrix per frame
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def check_conds(split_name, gt_suffix, additional_conds, name):
    if split_name is not None and split_name not in name:
        return False
    if gt_suffix is not None and gt_suffix not in name:
        return False
    if len(additional_conds) == 0:
        return True
    else:
        for cond in additional_conds:
            if cond not in name:
                return False
        return True


def semantic_filtering(seg_pred, ori_ood_conf):
    conf = np.zeros_like(ori_ood_conf)
    for c in CS_STUFF_CLASSES:
        mask = np.where(seg_pred == c, 1, 0)
        conf += mask * ori_ood_conf
    conf += 0.01 * ori_ood_conf
    return np.clip(conf, 0, 0.99)


# This is a slow alternative implementation of compute_ood_scores2, kept as reference for testing
# It is not used in the final evaluation
def compute_ood_scores(flat_labels, flat_pred, num_points=50):
    # From fishycapes code
    pos = flat_labels == 1
    valid = flat_labels <= 1  # filter out void
    gt = pos[valid]
    del pos
    uncertainty = flat_pred[valid].reshape(-1).astype(np.float32, copy=False)
    del valid

    # Sort the classifier scores (uncertainties)
    sorted_indices = np.argsort(uncertainty, kind='mergesort')[::-1]
    uncertainty, gt = uncertainty[sorted_indices], gt[sorted_indices]
    del sorted_indices

    # Remove duplicates along the curve
    distinct_value_indices = np.where(np.diff(uncertainty))[0]
    threshold_idxs = np.r_[distinct_value_indices, gt.size - 1]
    del distinct_value_indices, uncertainty

    # Accumulate TPs and FPs
    tps = np.cumsum(gt, dtype=np.uint64)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    del threshold_idxs

    # Compute Precision and Recall
    precision = tps / (tps + fps)
    precision[np.isnan(precision)] = 0
    recall = tps / tps[-1]
    # stop when full recall attained and reverse the outputs so recall is decreasing
    sl = slice(tps.searchsorted(tps[-1]), None, -1)
    precision = np.r_[precision[sl], 1]
    recall = np.r_[recall[sl], 0]
    average_precision = -np.sum(np.diff(recall) * precision[:-1])

    # select num_points values for a plotted curve
    interval = 1.0 / num_points
    curve_precision = [precision[-1]]
    curve_recall = [recall[-1]]
    idx = recall.size - 1
    for p in range(1, num_points):
        while recall[idx] < p * interval:
            idx -= 1
        curve_precision.append(precision[idx])
        curve_recall.append(recall[idx])
    curve_precision.append(precision[0])
    curve_recall.append(recall[0])
    del precision, recall

    if tps.size == 0 or fps[0] != 0 or tps[0] != 0:
        # Add an extra threshold position if necessary
        # to make sure that the curve starts at (0, 0)
        tps = np.r_[0., tps]
        fps = np.r_[0., fps]

    # Compute TPR and FPR
    tpr = tps / tps[-1]
    del tps
    fpr = fps / fps[-1]
    del fps

    # Compute AUROC
    auroc = np.trapz(tpr, fpr)

    # Compute FPR@95%TPR
    fpr_tpr95 = fpr[np.searchsorted(tpr, 0.95)]
    results = {
        'auroc': auroc,
        'AP': average_precision,
        'FPR@95%TPR': fpr_tpr95,
        # 'recall': np.array(curve_recall),
        # 'precision': np.array(curve_precision),
        # 'fpr': fpr,
        # 'tpr': tpr
        }

    return results


def compute_scores(conf, pred, label, ECE_NUM_BINS=15, CONF_NUM_BINS=65535, tpr_th=0.95):
    conf = conf.astype(np.float32)
    # normalize conf to [0, 1]
    conf = conf / CONF_NUM_BINS
    tau_tab = np.linspace(0, 1, ECE_NUM_BINS + 1)  # Confidence bins
    nb_items_bin = np.zeros(ECE_NUM_BINS)
    acc_tab = np.zeros(ECE_NUM_BINS)  # Empirical (true) confidence
    mean_conf = np.zeros(ECE_NUM_BINS)  # Predicted confidence
    for i in np.arange(ECE_NUM_BINS):  # Iterates over the bins
        # Selects the items where the predicted max probability falls in the bin
        # [tau_tab[i], tau_tab[i + 1)]
        sec = (tau_tab[i + 1] > conf) & (conf >= tau_tab[i])
        nb_items_bin[i] = np.sum(sec)  # Number of items in the bin
        # Selects the predicted classes, and the true classes
        class_pred_sec, y_sec = pred[sec], label[sec]
        # Averages of the predicted max probabilities
        mean_conf[i] = np.mean(conf[sec]) if nb_items_bin[i] > 0 else np.nan
        # Computes the empirical confidence
        acc_tab[i] = np.mean(class_pred_sec == y_sec) if nb_items_bin[i] > 0 else np.nan
    mean_conf = mean_conf[nb_items_bin > 0]
    acc_tab = acc_tab[nb_items_bin > 0]
    nb_items_bin = nb_items_bin[nb_items_bin > 0]
    if sum(nb_items_bin) != 0:
        ece = np.average(np.absolute(mean_conf - acc_tab), weights=nb_items_bin.astype(np.float32) / np.sum(nb_items_bin))
    else:
        ece = 0.0
    # TODO: wouldn't it be better to call compute_ood_scores2 here?
    # auroc, aupr_success, aupr_error, fpr95 = compute_ood_scores2(-conf, pred == label)
    # In this postprocessing we assume ID samples have larger "conf" values than OOD samples => we negate "conf" values
    # such that higher (signed) values correspond to detecting OOD samples
    fpr_list, tpr_list, thresholds = metrics.roc_curve(pred == label, conf)
    fpr = fpr_list[np.argmax(tpr_list >= tpr_th)]
    precision_in, recall_in, thresholds_in = metrics.precision_recall_curve(pred == label, conf)
    precision_out, recall_out, thresholds_out = metrics.precision_recall_curve(pred != label, -conf)  # TODO: this formula differs from the one in compute_ood_scores2? Is this intended? => possibly, this came as this, to double-check
    auroc = metrics.auc(fpr_list, tpr_list)
    aupr_success = metrics.auc(recall_in, precision_in)
    aupr_error = metrics.auc(recall_out, precision_out)
    return ece, auroc, aupr_success, aupr_error, fpr


def compute_ood_scores2(conf, ood_label, tpr_th=0.95):
    valid = ood_label <= 1  # filter out void
    ood_label = ood_label[valid] == 1
    conf = conf[valid]
    # TODO: double-check the logic of negating the confs, especially because the confs are already being negated by the caller
    # In this postprocessing we assume ID samples have larger "conf" values than OOD samples => we negate "conf" values
    # such that higher (signed) values correspond to detecting OOD samples
    fpr_list, tpr_list, thresholds = metrics.roc_curve(ood_label, conf, drop_intermediate=False)
    fpr = fpr_list[np.argmax(tpr_list >= tpr_th)]
    precision_in, recall_in, thresholds_in = metrics.precision_recall_curve(ood_label, conf)
    precision_out, recall_out, thresholds_out = metrics.precision_recall_curve(ood_label, -conf)  # TODO: see above
    auroc = metrics.auc(fpr_list, tpr_list)
    aupr_success = metrics.auc(recall_in, precision_in)
    aupr_error = metrics.auc(recall_out, precision_out)
    return auroc, aupr_success, aupr_error, fpr


def evaluate_bravo(gt_data,
                   submission_data,
                   split_name,                     # ACDC, SMIYC, outofcontext, synflare, synobjs, synrain
                   additional_conds,               # list of naming conditions for loading ground-truths
                   gt_suffix,                      # suffix of ground-truths
                   pred_suffix,                    # suffix for loading predictions
                   conf_suffix,                    # suffix for loading confidences
                   invalid_mask_suffix=None,       # suffix for loading valid masks
                   compute_semantic_metrics=True,  # compute semantic metrics
                   compute_semantic_metrics_invalid_area=False,
                   compute_OOD=False,              # compute OOD scores
                   NUM_CLASSES=19,
                   ECE_NUM_BINS=15,
                   CONF_NUM_BINS=65535,
                   SAMPLES_PER_IMG=20000,
                   strict=False
                   ):
    gts = [mem for mem in gt_data.getmembers() if check_conds(split_name, gt_suffix, additional_conds, mem.name)]
    n_images = len(gts)
    str_conds = ' '.join(additional_conds)
    logger.info(f'{split_name}-{str_conds}: evaluation on {n_images} images')
    hist = np.zeros((NUM_CLASSES, NUM_CLASSES))
    hist_valid = np.zeros((NUM_CLASSES, NUM_CLASSES))
    hist_invalid = np.zeros((NUM_CLASSES, NUM_CLASSES))
    all_gt_labels = []
    all_preds = []
    all_confs = []
    all_valid_indices = []
    all_invalid_indices = []

    all_gt_labels_valid = []
    all_preds_valid = []
    all_confs_valid = []
    all_gt_labels_invalid = []
    all_preds_invalid = []
    all_confs_invalid = []

    for idx, gt_mem in enumerate(tqqdm(gts)):

        np.random.seed(42)  # CHANGED: Takes the same samples from all images

        if DEBUG and idx >= 2:
            break

        gt_name = gt_mem.name
        pred_name = gt_name.replace(gt_suffix, pred_suffix)
        conf_name = gt_name.replace(gt_suffix, conf_suffix)

        # Read current ground truth and prediction files.
        try:
            img_gt_label_np = read_image_from_tar(gt_data, gt_mem)
        except:
            logger.error(f'Unable to load ground truth file {gt_name}')
            raise IOError(f'Unable to load ground truth file {gt_name}')

        try:
            pred = submission_data.getmember(pred_name)
            img_pred_label_np = read_image_from_tar(submission_data, pred)
        except:
            logger.error(f'Unable to load prediction file {pred_name}')
            raise IOError(f'Unable to load prediction file {pred_name}')

        try:
            conf = submission_data.getmember(conf_name)
            img_conf_np = read_image_from_tar_conf(submission_data, conf)
        except:
            logger.error(f'Unable to load prediction file {conf_name}')
            raise IOError(f'Unable to load prediction file {conf_name}')

        # TODO: clarify the resizing rules in the submission instructions
        # Ensures that dimensions of the two images match exactly.
        img_gt_label_shape = img_gt_label_np.shape
        img_pred_label_shape = img_pred_label_np.shape
        if len(img_pred_label_shape) != 2:
            logger.error(f'Prediction is not a proper 2D matrix for file {gt_name}')
            raise ValueError(f'Prediction is not a proper 2D matrix for file {gt_name}')
        if img_pred_label_shape != img_gt_label_shape:
            # resize img_gt_label_np to img_pred_label_shape
            img_pred_label_np = cv2.resize(img_pred_label_np, img_gt_label_shape[::-1], interpolation=cv2.INTER_LINEAR)
            img_conf_np = cv2.resize(img_conf_np, img_gt_label_shape[::-1], interpolation=cv2.INTER_LINEAR)
            img_pred_label_shape = img_pred_label_np.shape
            logger.warning(f'Resized prediction to match ground truth dimensions for file {pred_name}')
            # raise ValueError(f'Image dimensions mismatch for file {pred_name}')

        if strict:
            # STRICTLY SIMILAR TO THE OLD SCRIPT
            if invalid_mask_suffix is not None:
                mask_name = gt_name.replace(gt_suffix, invalid_mask_suffix)
                try:
                    img_invalid_mask_np = read_image_from_tar(gt_data, mask_name)
                except:
                    logger.error(f'Unable to load invalid mask file {mask_name}')
                    raise IOError(f'Unable to load invalid mask file {mask_name}')

                # TODO: can we do this once outside the server? => probably, let's check
                # resize img_invalid_mask_np to img_gt_label_shape
                img_invalid_mask_np = cv2.resize(img_invalid_mask_np, img_gt_label_shape[::-1], interpolation=cv2.INTER_NEAREST)

                # convert the invalid binary mask to indices of invalid and valid pixels
                img_invalid_indices = np.where(img_invalid_mask_np == 1)
                img_valid_indices = np.where(img_invalid_mask_np == 0)
                # get 1D indices of invalid and valid pixels
                img_invalid_indices_1D = np.ravel_multi_index(img_invalid_indices, img_invalid_mask_np.shape)
                img_valid_indices_1D = np.ravel_multi_index(img_valid_indices, img_invalid_mask_np.shape)
            else:
                mask_name = img_invalid_mask_np = img_valid_indices = img_invalid_indices = img_valid_indices_1D = img_invalid_indices_1D = None

            # Calculate the intersection and union counts per class for the two images.
            img_gt_label_np = img_gt_label_np.flatten()
            img_pred_label_np = img_pred_label_np.flatten()
            img_conf_np = img_conf_np.flatten()
            hist += fast_hist(img_gt_label_np, img_pred_label_np, NUM_CLASSES)

            # Sample pixels for ECE and AUROC
            nsamples = min(SAMPLES_PER_IMG, img_gt_label_np.shape[0])
            samples_indices = np.random.choice(range(img_gt_label_np.shape[0]), nsamples, replace=False)

            valid_indices = np.zeros_like(img_gt_label_np)
            valid_indices[img_valid_indices_1D] = 1  # all indices are valid if img_valid_indices_1D is None

            invalid_indices = np.zeros_like(img_gt_label_np)
            invalid_indices[img_invalid_indices_1D] = 1

            all_gt_labels.append(img_gt_label_np[samples_indices])
            all_preds.append(img_pred_label_np[samples_indices])
            all_confs.append(img_conf_np[samples_indices])
            all_valid_indices.append(valid_indices[samples_indices])
            all_invalid_indices.append(invalid_indices[samples_indices])

            # Compute scores for valid/invalid pixels
            if invalid_mask_suffix is not None:
                hist_valid += fast_hist(img_gt_label_np[img_valid_indices_1D], img_pred_label_np[img_valid_indices_1D], NUM_CLASSES)
                hist_invalid += fast_hist(img_gt_label_np[img_invalid_indices_1D], img_pred_label_np[img_invalid_indices_1D], NUM_CLASSES)
                nsamples_valid = min(SAMPLES_PER_IMG, len(img_valid_indices_1D))
                samples_indices_valid = np.random.choice(range(len(img_valid_indices_1D)), nsamples_valid, replace=False)
                samples_indices_valid = img_valid_indices_1D[samples_indices_valid]
                all_gt_labels_valid.append(img_gt_label_np[samples_indices_valid])
                all_preds_valid.append(img_pred_label_np[samples_indices_valid])
                all_confs_valid.append(img_conf_np[samples_indices_valid])

                if compute_semantic_metrics_invalid_area:
                    nsamples_invalid = min(SAMPLES_PER_IMG, len(img_invalid_indices_1D))
                    samples_indices_invalid = np.random.choice(range(len(img_invalid_indices_1D)), nsamples_invalid, replace=False)
                    samples_indices_invalid = img_invalid_indices_1D[samples_indices_invalid]
                    all_gt_labels_invalid.append(img_gt_label_np[samples_indices_invalid])
                    all_preds_invalid.append(img_pred_label_np[samples_indices_invalid])
                    all_confs_invalid.append(img_conf_np[samples_indices_invalid])
        else:
            # COMPATIBILITY MODE WITH NEW SCRIPT
            # Removes all pixels of class 255
            # Because of that, the treatment of indices is modified, with all arrays being flattened from start

            # Calculate the intersection and union counts per class for the two images.
            img_gt_label_np = img_gt_label_np.flatten()
            img_pred_label_np = img_pred_label_np.flatten()
            img_conf_np = img_conf_np.flatten()

            non_void = img_gt_label_np != 255
            img_gt_label_np = img_gt_label_np[non_void]
            img_pred_label_np = img_pred_label_np[non_void]
            img_conf_np = img_conf_np[non_void]

            if invalid_mask_suffix is not None:
                mask_name = gt_name.replace(gt_suffix, invalid_mask_suffix)
                try:
                    img_invalid_mask_np = read_image_from_tar(gt_data, mask_name)
                except:
                    logger.error(f'Unable to load invalid mask file {mask_name}')
                    raise IOError(f'Unable to load invalid mask file {mask_name}')

                # TODO: can we do this once outside the server? => probably, let's check
                # resize img_invalid_mask_np to img_gt_label_shape
                img_invalid_mask_np = cv2.resize(img_invalid_mask_np, img_gt_label_shape[::-1], interpolation=cv2.INTER_NEAREST)

                img_invalid_mask_np = img_invalid_mask_np.flatten()
                img_invalid_mask_np = img_invalid_mask_np[non_void]

                # convert the invalid binary mask to indices of invalid and valid pixels
                img_invalid_indices_1D = np.nonzero(img_invalid_mask_np == 1)[0]
                img_valid_indices_1D = np.nonzero(img_invalid_mask_np == 0)[0]
            else:
                mask_name = img_invalid_mask_np = img_valid_indices_1D = img_invalid_indices_1D = None

            hist += fast_hist(img_gt_label_np, img_pred_label_np, NUM_CLASSES)

            # Sample pixels for ECE and AUROC
            nsamples = min(SAMPLES_PER_IMG, img_gt_label_np.shape[0])
            samples_indices = np.random.choice(range(img_gt_label_np.shape[0]), nsamples, replace=False)

            valid_indices = np.zeros_like(img_gt_label_np)
            valid_indices[img_valid_indices_1D] = 1  # all indices are valid if img_valid_indices_1D is None

            invalid_indices = np.zeros_like(img_gt_label_np)
            invalid_indices[img_invalid_indices_1D] = 1

            all_gt_labels.append(img_gt_label_np[samples_indices])
            all_preds.append(img_pred_label_np[samples_indices])
            all_confs.append(img_conf_np[samples_indices])
            all_valid_indices.append(valid_indices[samples_indices])
            all_invalid_indices.append(invalid_indices[samples_indices])

            # Compute scores for valid/invalid pixels
            if invalid_mask_suffix is not None:
                hist_valid += fast_hist(img_gt_label_np[img_valid_indices_1D], img_pred_label_np[img_valid_indices_1D], NUM_CLASSES)
                hist_invalid += fast_hist(img_gt_label_np[img_invalid_indices_1D], img_pred_label_np[img_invalid_indices_1D], NUM_CLASSES)
                nsamples_valid = min(SAMPLES_PER_IMG, len(img_valid_indices_1D))
                samples_indices_valid = np.random.choice(range(len(img_valid_indices_1D)), nsamples_valid, replace=False)
                samples_indices_valid = img_valid_indices_1D[samples_indices_valid]
                all_gt_labels_valid.append(img_gt_label_np[samples_indices_valid])
                all_preds_valid.append(img_pred_label_np[samples_indices_valid])
                all_confs_valid.append(img_conf_np[samples_indices_valid])

                if compute_semantic_metrics_invalid_area:
                    nsamples_invalid = min(SAMPLES_PER_IMG, len(img_invalid_indices_1D))
                    samples_indices_invalid = np.random.choice(range(len(img_invalid_indices_1D)), nsamples_invalid, replace=False)
                    samples_indices_invalid = img_invalid_indices_1D[samples_indices_invalid]
                    all_gt_labels_invalid.append(img_gt_label_np[samples_indices_invalid])
                    all_preds_invalid.append(img_pred_label_np[samples_indices_invalid])
                    all_confs_invalid.append(img_conf_np[samples_indices_invalid])

    # calculate mIoU
    if compute_semantic_metrics:
        perclass_iou = per_class_iu(hist).tolist()
        miou = np.nanmean(perclass_iou)
        logger.info(f'{split_name}-{str_conds}: mIoU = {miou}')
    else:
        miou = None
        perclass_iou = None
    # calculate mIoU for valid/invalid pixels
    if compute_semantic_metrics and invalid_mask_suffix is not None:
        perclass_iou_valid = per_class_iu(hist_valid).tolist()
        miou_valid = np.nanmean(perclass_iou_valid)
        perclass_iou_invalid = per_class_iu(hist_invalid).tolist()
        miou_invalid = np.nanmean(perclass_iou_invalid)
        logger.info(f'{split_name}-{str_conds}: mIoU-valid = {miou_valid}')
        logger.info(f'{split_name}-{str_conds}: mIoU-invalid = {miou_invalid}')
        logger.info('====================================================')
    else:
        miou_valid = miou_invalid = perclass_iou_valid = perclass_iou_invalid = None

    # calculate ECE, AUROC, AUPR_success, AUPR_error, FPR_at_95TPR

    all_gt_labels = np.concatenate(all_gt_labels)
    all_preds = np.concatenate(all_preds)
    all_confs = np.concatenate(all_confs).astype(np.float32)
    all_valid_indices = np.concatenate(all_valid_indices)
    all_invalid_indices = np.concatenate(all_invalid_indices)

    if compute_semantic_metrics:
        ece, auroc, aupr_success, aupr_error, fpr95 = compute_scores(all_confs, all_preds, all_gt_labels, ECE_NUM_BINS=ECE_NUM_BINS, tpr_th=0.95)
        logger.info(f'{split_name}-{str_conds}: ECE = {ece}')
        logger.info(f'{split_name}-{str_conds}: AUROC = {round(auroc * 100,2)}')
        logger.info(f'{split_name}-{str_conds}: AUPR_success = {round(aupr_success * 100,2)}')
        logger.info(f'{split_name}-{str_conds}: AUPR_error = {round(aupr_error * 100,2)}')
        logger.info(f'{split_name}-{str_conds}: FPR_at_95TPR = {round(fpr95 * 100,2)}')
        logger.info('====================================================')
    else:
        ece = auroc = aupr_success = aupr_error = fpr95 = None

    if compute_semantic_metrics and invalid_mask_suffix is not None:
        all_gt_labels_valid = np.concatenate(all_gt_labels_valid)
        all_preds_valid = np.concatenate(all_preds_valid)
        all_confs_valid = np.concatenate(all_confs_valid).astype(np.float32)
        ece_valid, auroc_valid, aupr_success_valid, aupr_error_valid, fpr95_valid = compute_scores(all_confs_valid, all_preds_valid, all_gt_labels_valid, ECE_NUM_BINS=ECE_NUM_BINS, tpr_th=0.95)
        logger.info(f'{split_name}-{str_conds}: ECE_valid = {ece_valid}')
        logger.info(f'{split_name}-{str_conds}: AUROC_valid = {round(auroc_valid * 100,2)}')
        logger.info(f'{split_name}-{str_conds}: AUPR_success_valid = {round(aupr_success_valid * 100,2)}')
        logger.info(f'{split_name}-{str_conds}: AUPR_error_valid = {round(aupr_error_valid * 100,2)}')
        logger.info(f'{split_name}-{str_conds}: FPR_at_95TPR_valid = {round(fpr95_valid * 100,2)}')
        logger.info('====================================================')

        if compute_semantic_metrics_invalid_area:
            all_gt_labels_invalid = np.concatenate(all_gt_labels_invalid)
            all_preds_invalid = np.concatenate(all_preds_invalid)
            all_confs_invalid = np.concatenate(all_confs_invalid).astype(np.float32)
            ece_invalid, auroc_invalid, aupr_success_invalid, aupr_error_invalid, fpr95_invalid = compute_scores(all_confs_invalid, all_preds_invalid, all_gt_labels_invalid, ECE_NUM_BINS=ECE_NUM_BINS, tpr_th=0.95)
            logger.info(f'{split_name}-{str_conds}: ECE_invalid = {ece_invalid}')
            logger.info(f'{split_name}-{str_conds}: AUROC_invalid = {round(auroc_invalid * 100,2)}')
            logger.info(f'{split_name}-{str_conds}: AUPR_success_invalid = {round(aupr_success_invalid * 100,2)}')
            logger.info(f'{split_name}-{str_conds}: AUPR_error_invalid = {round(aupr_error_invalid * 100,2)}')
            logger.info(f'{split_name}-{str_conds}: FPR_at_95TPR_invalid = {round(fpr95_invalid * 100,2)}')
            logger.info('====================================================')
        else:
            ece_invalid = auroc_invalid = aupr_success_invalid = aupr_error_invalid = fpr95_invalid = None

    else:
        ece_valid = auroc_valid = aupr_success_valid = aupr_error_valid = fpr95_valid = None
        ece_invalid = auroc_invalid = aupr_success_invalid = aupr_error_invalid = fpr95_invalid = None

    # calculate OOD detection scores
    if compute_OOD and invalid_mask_suffix is not None:
        ood_labels = np.zeros_like(all_confs)
        ood_labels[all_invalid_indices == 1] = 1
        ood_labels[all_gt_labels == 255] = 255

        auroc_ood, aupr_ood, aupr_ood_error, fpr95_ood = compute_ood_scores2(-all_confs, ood_labels)
        logger.info(f'{split_name}-{str_conds}: AUROC_ood = {round(auroc_ood * 100,2)}')
        logger.info(f'{split_name}-{str_conds}: AUPR_ood = {round(aupr_ood * 100,2)}')
        logger.info(f'{split_name}-{str_conds}: FPR_at_95TPR_ood = {round(fpr95_ood * 100,2)}')

        # apply semantic filtering
        all_confs_semfilt = semantic_filtering(all_preds, 1-all_confs/CONF_NUM_BINS)
        auroc_ood_semfilt, aupr_ood_semfilt, aupr_ood_error_semfilt, fpr95_ood_semfilt = compute_ood_scores2(all_confs_semfilt, ood_labels)
        logger.info(f'{split_name}-{str_conds}: AUROC_ood_semfilt = {round(auroc_ood_semfilt * 100,2)}')
        logger.info(f'{split_name}-{str_conds}: AUPR_ood_semfilt = {round(aupr_ood_semfilt * 100,2)}')
        logger.info(f'{split_name}-{str_conds}: FPR_at_95TPR_ood_semfilt = {round(fpr95_ood_semfilt * 100,2)}')
    else:
        auroc_ood = aupr_ood = fpr95_ood = auroc_ood_semfilt = aupr_ood_semfilt = fpr95_ood_semfilt = None

    metrics = {
        'miou': miou,
        'iou': perclass_iou,
        'ece': ece,
        'auroc': auroc,
        'auprc_success': aupr_success,
        'auprc_error': aupr_error,
        'fpr95': fpr95,
        'miou_valid': miou_valid,
        'iou_valid': perclass_iou_valid,
        'ece_valid': ece_valid,
        'auroc_valid': auroc_valid,
        'auprc_success_valid': aupr_success_valid,
        'auprc_error_valid': aupr_error_valid,
        'fpr95_valid': fpr95_valid,
        'miou_invalid': miou_invalid,
        'iou_invalid': perclass_iou_invalid,
        'ece_invalid': ece_invalid,
        'auroc_invalid': auroc_invalid,
        'auprc_success_invalid': aupr_success_invalid,
        'auprc_error_invalid': aupr_error_invalid,
        'fpr95_invalid': fpr95_invalid,
        'auroc_ood': auroc_ood,
        'auprc_ood': aupr_ood,
        'fpr95_ood': fpr95_ood,
        }
    if strict:
        metrics.update({
            'auroc_ood_semfilt': auroc_ood_semfilt,
            'auprc_ood_semfilt': aupr_ood_semfilt,
            'fpr95_ood_semfilt': fpr95_ood_semfilt
        })
    return metrics


BRAVO_SUBSETS = ['ACDC_fog', 'ACDC_rain', 'ACDC_night', 'ACDC_snow', 'synrain', 'SMIYC', 'synobjs', 'synflare',
                 'outofcontext']


def evaluate_method(gtFilePath, submFilePath, evaluationParams):
    gt_data = load_tar(gtFilePath)
    submission_data = load_tar(submFilePath)

    NUM_CLASSES = 19
    ECE_NUM_BINS = 15
    CONF_NUM_BINS = 65535
    SAMPLES_PER_IMG = 100000
    # SAMPLES_PER_IMG = np.inf

    subsets = evaluationParams.get('subsets', BRAVO_SUBSETS)
    strict = evaluationParams.get('strict', False)
    results_dict = {}

    if 'ACDC_fog' in subsets:
        acdcFogMetrics = evaluate_bravo(gt_data=gt_data,
                                        submission_data=submission_data,
                                        split_name='ACDC',
                                        additional_conds=['/fog/'],
                                        gt_suffix='_gt_labelTrainIds.png',
                                        invalid_mask_suffix='_gt_invIds.png',
                                        pred_suffix='_rgb_anon_pred.png',
                                        conf_suffix='_rgb_anon_conf.png',
                                        NUM_CLASSES=NUM_CLASSES,
                                        ECE_NUM_BINS=ECE_NUM_BINS,
                                        CONF_NUM_BINS=CONF_NUM_BINS,
                                        SAMPLES_PER_IMG=SAMPLES_PER_IMG,
                                        strict=strict)
        results_dict['ACDC_fog'] = acdcFogMetrics

    if 'ACDC_rain' in subsets:
        acdcRainMetrics = evaluate_bravo(gt_data=gt_data,
                                         submission_data=submission_data,
                                         split_name='ACDC',
                                         additional_conds=['/rain/'],
                                         gt_suffix='_gt_labelTrainIds.png',
                                         invalid_mask_suffix='_gt_invIds.png',
                                         pred_suffix='_rgb_anon_pred.png',
                                         conf_suffix='_rgb_anon_conf.png',
                                         NUM_CLASSES=NUM_CLASSES,
                                         ECE_NUM_BINS=ECE_NUM_BINS,
                                         CONF_NUM_BINS=CONF_NUM_BINS,
                                         SAMPLES_PER_IMG=SAMPLES_PER_IMG,
                                         strict=strict)
        results_dict['ACDC_rain'] = acdcRainMetrics

    if 'ACDC_night' in subsets:
        acdcNightMetrics = evaluate_bravo(gt_data=gt_data,
                                          submission_data=submission_data,
                                          split_name='ACDC',
                                          additional_conds=['/night/'],
                                          gt_suffix='_gt_labelTrainIds.png',
                                          invalid_mask_suffix='_gt_invIds.png',
                                          pred_suffix='_rgb_anon_pred.png',
                                          conf_suffix='_rgb_anon_conf.png',
                                          NUM_CLASSES=NUM_CLASSES,
                                          ECE_NUM_BINS=ECE_NUM_BINS,
                                          CONF_NUM_BINS=CONF_NUM_BINS,
                                          SAMPLES_PER_IMG=SAMPLES_PER_IMG,
                                          strict=strict)
        results_dict['ACDC_night'] = acdcNightMetrics

    if 'ACDC_snow' in subsets:
        acdcSnowMetrics = evaluate_bravo(gt_data=gt_data,
                                         submission_data=submission_data,
                                         split_name='ACDC',
                                         additional_conds=['/snow/'],
                                         gt_suffix='_gt_labelTrainIds.png',
                                         invalid_mask_suffix='_gt_invIds.png',
                                         pred_suffix='_rgb_anon_pred.png',
                                         conf_suffix='_rgb_anon_conf.png',
                                         NUM_CLASSES=NUM_CLASSES,
                                         ECE_NUM_BINS=ECE_NUM_BINS,
                                         CONF_NUM_BINS=CONF_NUM_BINS,
                                         SAMPLES_PER_IMG=SAMPLES_PER_IMG,
                                         strict=strict)
        results_dict['ACDC_snow'] = acdcSnowMetrics

    if 'synrain' in subsets:
        synrainMetrics = evaluate_bravo(gt_data=gt_data,
                                        submission_data=submission_data,
                                        split_name='synrain',
                                        additional_conds=[],
                                        gt_suffix='_gt_labelTrainIds.png',
                                        pred_suffix='_leftImg8bit_pred.png',
                                        conf_suffix='_leftImg8bit_conf.png',
                                        invalid_mask_suffix='_gt_invIds.png',
                                        compute_semantic_metrics=True,
                                        compute_semantic_metrics_invalid_area=True,
                                        compute_OOD=True,
                                        NUM_CLASSES=NUM_CLASSES,
                                        ECE_NUM_BINS=ECE_NUM_BINS,
                                        CONF_NUM_BINS=CONF_NUM_BINS,
                                        SAMPLES_PER_IMG=SAMPLES_PER_IMG,
                                        strict=strict)
        results_dict['synrain'] = synrainMetrics

    if 'SMIYC' in subsets:
        smiycMetrics = evaluate_bravo(gt_data=gt_data,
                                      submission_data=submission_data,
                                      split_name='SMIYC',
                                      additional_conds=[],
                                      gt_suffix='_labels_semantic_fake.png',
                                      pred_suffix='_pred.png',
                                      conf_suffix='_conf.png',
                                      invalid_mask_suffix='_labels_semantic.png',
                                      compute_semantic_metrics=False,
                                      compute_OOD=True,
                                      NUM_CLASSES=NUM_CLASSES,
                                      ECE_NUM_BINS=ECE_NUM_BINS,
                                      CONF_NUM_BINS=CONF_NUM_BINS,
                                      SAMPLES_PER_IMG=np.inf,
                                      strict=strict)
        results_dict['SMIYC'] = smiycMetrics

    if 'synobjs' in subsets:
        synobjsMetrics = evaluate_bravo(gt_data=gt_data,
                                        submission_data=submission_data,
                                        split_name='synobjs',
                                        additional_conds=[],
                                        gt_suffix='_gt.png',
                                        pred_suffix='_pred.png',
                                        conf_suffix='_conf.png',
                                        invalid_mask_suffix='_mask.png',
                                        compute_semantic_metrics=True,
                                        compute_OOD=True,
                                        NUM_CLASSES=NUM_CLASSES,
                                        ECE_NUM_BINS=ECE_NUM_BINS,
                                        CONF_NUM_BINS=CONF_NUM_BINS,
                                        SAMPLES_PER_IMG=SAMPLES_PER_IMG,
                                        strict=strict)
        results_dict['synobjs'] = synobjsMetrics

    if 'synflare' in subsets:
        synflareMetrics = evaluate_bravo(gt_data=gt_data,
                                         submission_data=submission_data,
                                         split_name='synflare',
                                         additional_conds=[],
                                         gt_suffix='_gt_labelTrainIds.png',
                                         pred_suffix='_leftImg8bit_pred.png',
                                         conf_suffix='_leftImg8bit_conf.png',
                                         invalid_mask_suffix='_gt_invIds.png',
                                         compute_semantic_metrics=True,
                                         compute_OOD=True,
                                         NUM_CLASSES=NUM_CLASSES,
                                         ECE_NUM_BINS=ECE_NUM_BINS,
                                         CONF_NUM_BINS=CONF_NUM_BINS,
                                         SAMPLES_PER_IMG=SAMPLES_PER_IMG,
                                         strict=strict)
        results_dict['synflare'] = synflareMetrics

    if 'outofcontext' in subsets:
        outofcontextMetrics = evaluate_bravo(gt_data=gt_data,
                                             submission_data=submission_data,
                                             split_name='outofcontext',
                                             additional_conds=[],
                                             gt_suffix='_gt_labelTrainIds.png',
                                             pred_suffix='_leftImg8bit_pred.png',
                                             conf_suffix='_leftImg8bit_conf.png',
                                             invalid_mask_suffix='_gt_invIds.png',
                                             compute_semantic_metrics=True,
                                             compute_OOD=False,
                                             NUM_CLASSES=NUM_CLASSES,
                                             ECE_NUM_BINS=ECE_NUM_BINS,
                                             CONF_NUM_BINS=CONF_NUM_BINS,
                                             SAMPLES_PER_IMG=SAMPLES_PER_IMG,
                                             strict=strict)
        results_dict['outofcontext'] = outofcontextMetrics

    return results_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
         description='Evaluates submissions for the ELSA BRAVO Challenge.',
         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('submission', help='path to submission tar file')
    parser.add_argument('--gt', required=True, help='path to ground-truth tar file')
    parser.add_argument('--results', default='results_old.json', help='JSON file to store the computed metrics')
    parser.add_argument('--subsets', default=['ALL'], nargs='+', choices=BRAVO_SUBSETS+['ALL'], metavar='SUBSETS',
                        help='tasks to evaluate: ALL for all tasks, or one or more items from this list, separated '
                             'by spaces: ' + ' '.join(BRAVO_SUBSETS))
    parser.add_argument('--strict', action='store_true', help='disables comparison mode, where the script will '
                        'process the void class (label 255) like the new script.')
    parser.add_argument('--debug', action='store_true', help='enables extra verbose debug output')
    parser.add_argument('--quiet', action='store_true', help='prints only errors and warnings')
    args = parser.parse_args()

    level = logging.WARNING if args.quiet else (logging.DEBUG if args.debug else logging.INFO)
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')

    if args.strict:
        logger.warning('comparison mode is disabled, results will not be comparable to the new script.')

    logger.info('evaluating...')
    if args.subsets == ['ALL']:
        args.subsets = BRAVO_SUBSETS
    results = evaluate_method(args.gt, args.submission,
                              evaluationParams={'subsets': args.subsets, 'strict': args.strict})

    if not args.quiet:
        logger.info('results:')
        print(json.dumps(results, indent=4))

    logger.info('saving results...')
    with open(args.results, 'wt', encoding='utf-8') as json_file:
        json.dump(results, json_file, indent=4)

    logger.info('done.')
