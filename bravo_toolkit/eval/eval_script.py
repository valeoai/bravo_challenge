# Copyright (c) 2024 Valeo
# See LICENSE.md for details.

import argparse
from contextlib import closing
from functools import cache
import json
import logging
import os
import sys
import tarfile

import numpy as np
from tqdm import tqdm
import scipy.stats as stats

from bravo_toolkit.codec.bravo_codec import bravo_decode
from bravo_toolkit.codec.bravo_tarfile import (SAMPLES_SUFFIX, SPLIT_PREFIX, SPLIT_TO_CONF_SUFFIX, SPLIT_TO_GT_SUFFIX,
                                               SPLIT_TO_MASK_SUFFIX, SPLIT_TO_PRED_SUFFIX, SUBMISSION_SUFFIX,
                                               tar_extract_file, tar_extract_grayscale, tar_extract_image)
from bravo_toolkit.eval.metrics import fast_cm, get_auprc, get_auroc, get_ece, get_tp_fp_counts, per_class_iou
from bravo_toolkit.util.sample_gt_pixels import SAMPLES_PER_IMG, decode_indices


DEBUG = False

NUM_CLASSES = 19
CONF_N_LEVELS = 65536
eps = 1. / CONF_N_LEVELS
bias = eps / 2.
CONF_VALUES = np.linspace(0., 1.-eps, CONF_N_LEVELS) + bias

logger = logging.getLogger('bravo_toolkit')


def tqqdm(iterable, *args, **kwargs):
    if logger.getEffectiveLevel() <= logging.INFO:
        return tqdm(iterable, *args, **kwargs)
    return iterable


@cache
def get_ground_truth_info(gt_path):
    '''Loads information from the ground-truth tar file.'''
    gt_truths = []
    gt_invalids = []
    with closing(tarfile.open(gt_path, 'r')) as gt_data:
        split = prefix = '//////////'
        for member in gt_data.getmembers():
            if not member.isfile():
                continue
            name = member.name

            # Find the split name
            if not name.startswith(prefix):  # Prefixes tend to occur in runs, so makes a quick check
                for split in SPLIT_TO_GT_SUFFIX:
                    prefix = SPLIT_PREFIX.format(split=split)
                    if name.startswith(prefix):
                        break
                else:
                    split = prefix = name
                    logger.warning('Unexpected file prefix in ground-truth tarfile: %s', name)
                    continue

            # Find the file type
            if name.endswith(SPLIT_TO_GT_SUFFIX[split]):
                gt_truths.append(member)
            elif name.endswith(SPLIT_TO_MASK_SUFFIX[split]):
                gt_invalids.append(member)
            else:
                logger.warning('Unexpected file in ground-truth tarfile: %s', name)

    if len(gt_truths) != len(gt_invalids):
        raise RuntimeError(f'Invalid ground-truth tarfile at `{gt_path}`: # of labelTrainIds files ({len(gt_truths)}) '
                           f'!= invIds files {len(gt_invalids)}.')
    return gt_truths, gt_invalids


def validate_data(gt_path, submission_path, _extra_params):
    '''
    Validates API call for the ELSA Challenges server.

    Because the submission files are so large, pre-validating the data would be more expensive than just doing it in
    the evaluation, so this function is a no-op.

    Args:
        gt_path (str): path to the ground-truth tar file
        submission_path (str): path to the submission tar file
        _extra_params (dict): additional parameters, not used

    Raises:
        ValueError: if the submission is invalid
    '''
    logger.info('validate_data started')

    gt_truths, _ = get_ground_truth_info(gt_path)
    submissions = { member.name: False for member in gt_truths }

    logger.info('validate_data ground truth read')

    with closing(tarfile.open(submission_path, 'r')) as submission_data:
        split = prefix = '//////////'
        for member in submission_data.getmembers():
            if not member.isfile():
                continue
            name = member.name

            # Find the split name
            if not name.startswith(prefix):  # Prefixes tend to occur in runs, so makes a quick check
                for split in SPLIT_TO_GT_SUFFIX:
                    prefix = SPLIT_PREFIX.format(split=split)
                    if name.startswith(prefix):
                        break
                else:
                    split = prefix = name
                    logger.warning('Unexpected file prefix in submissions tarfile: %s', name)
                    continue

            # Maps submission file to a ground-truth file
            if name.endswith(SUBMISSION_SUFFIX):
                gt_name = name[:-len(SUBMISSION_SUFFIX)] + SPLIT_TO_GT_SUFFIX[split]
            else:
                logger.warning('Unexpected file suffix in submissions tarfile: %s', name)
                continue

            # Checks if ground-truth file exists and marks it as found
            if gt_name not in submissions:
                logger.warning('Submission file name has no matches in ground-truth tarfile: %s -> %s', name, gt_name)
            else:
                submissions[gt_name] = True

    # Checks if all ground-truth files have been found
    missing = [name for name, found in submissions.items() if not found]
    if missing:
        first_missing = ', '.join(f'`{mf}`' for mf in missing[:3])
        error_msg = f'Missing {len(missing)} files in submission. First missing files: {first_missing}...'
        logger.error('validate_data error: %s', error_msg)
        raise ValueError(error_msg)

    logger.info('validate_data passed')


def get_curve_metrics(tp_counts, fp_counts, tpr_th=0.95):
    '''
    Computes the curve-based scores for the given ground-truth and confidence values.

    Args:
        tp_counts (np.ndarray): raw counts of true positives for each confidence level
        fp_counts (np.ndarray): raw counts of false positives for each confidence level
        tpr_th (float): threshold for TPR, default is 0.95

    Returns:
        auroc (float): Area Under the Receiver Operating Characteristic curve for the positive class
        fpr_at_tpr_th (float): False Positive Rate at the given True Positive Rate threshold in the ROC curve
        auprc_pos (float): Area Under the Precision-Recall curve for the positive class
        auprc_neg (float): Area Under the Precision-Recall curve for the negative class
        ece (float): Expected Calibration Error for positive class
    '''
    auroc, tprs, fprs = get_auroc(tp_counts, fp_counts)
    tpr_th_i = np.searchsorted(tprs, tpr_th, 'left')  # index of the first element >= tpr_th, assumes tprs is sorted
    fpr_at_tpr_th = fprs[tpr_th_i]
    auprc_pos, _, _ = get_auprc(tp_counts, fp_counts)
    ece = get_ece(tp_counts, tp_counts+fp_counts, confidence_values=CONF_VALUES, bins=15)
    # To obtain the negative counts, we have to integrate the curve, invert it, and differentiate it from the other end
    tp_cumm = np.concatenate(([0], np.cumsum(tp_counts)))
    fp_cumm = np.concatenate(([0], np.cumsum(fp_counts)))
    tn_cumm = fp_cumm[-1] - fp_cumm  # Total of negatives - false positives
    fn_cumm = tp_cumm[-1] - tp_cumm  # Total of positives - true positives
    tn_counts = np.diff(tn_cumm[::-1])  # We would reverse the arrays again to get the negative counts parallel to the
    fn_counts = np.diff(fn_cumm[::-1])  # positive counts, but in this case we actually need the reversed values...
    auprc_neg, _, _ = get_auprc(tn_counts, fn_counts)  # ...for the reversed arrays act as if negating confidence values
    return auroc, fpr_at_tpr_th, auprc_pos, auprc_neg, ece


def evaluate_bravo(*,
                   gt_data,
                   samples_data,
                   submission_data,
                   split_name,
                   other_names=tuple(),
                   gt_suffix=None,
                   mask_suffix=None,
                   submission_suffix=SUBMISSION_SUFFIX,
                   samples_suffix=SAMPLES_SUFFIX,
                   semantic_metrics=True,
                   ood_metrics=False,
                   compare_old=False,
                   compare_old_seed=1,
                   show_counters=False,
                   ):
    '''
    Evaluate submission_data against gt_data for the given split_name and additional conditions.

    Args:
        gt_data (tarfile): ground truth data
        samples_data (tarfile): sample data (ignored if compare_old is True)
        submission_data (tarfile): submission data
        split_name (str): name of the split (ACDC, SMIYC, outofcontext, synflare, synobjs, synrain)
        gt_suffix (str): suffix for loading ground truths, default is derived from split_name
        mask_suffix (str): suffix for loading invalid masks, default is derived from split_name
        submission_suffix (str): suffix for loading submission files, default is `SUBMISSION_SUFFIX`
        samples_suffix (str): suffix for loading sample files, default is `SAMPLES_SUFFIX`
        other_names (iterable of str): iterable of additional name substrings that must be present in ground truths
        semantic_metrics (bool): compute semantic metrics, default is True
        ood_metrics (bool): compute OOD scores, default is False
        compare_old (bool): comparison mode (old submission format and 100k pixels for curve-based metrics, used for
                test purposes only, do not use in production), default is False
        compare_old_seed (int): seed for the comparison mode, default is 1
        show_counters (bool): show debug counters, default is False
    Returns:
        dict: evaluation results
    '''
    # Get suffixes from the split name if not provided
    gt_suffix = gt_suffix or SPLIT_TO_GT_SUFFIX[split_name]
    mask_suffix = mask_suffix or SPLIT_TO_MASK_SUFFIX[split_name]

    def substring_in_name(name, *conditions):
        for c in conditions:
            if c is not None and c not in name:
                return False
        return True

    gts = [mem for mem in gt_data.getmembers() if substring_in_name(mem.name, split_name, gt_suffix, *other_names)]
    n_images = len(gts)

    other_names_str = ' '.join(other_names)
    logger.info('%s-%s: evaluation on %d images', split_name, other_names_str, n_images)

    # Acquire data from the tar files...
    # ...ground truth files
    logger.info('evaluate_bravo - reading ground truth files...')
    gt_files = {}
    mask_files = {}
    samples_files = {}
    sub_files = {}
    evaluation_tuples = []
    for idx, gt_mem in enumerate(tqqdm(gts)):
        if DEBUG and idx >= 2:
            break
        # The ground truth files are in the "right" order and may be read directly
        gt_name = gt_mem.name
        gt_files[gt_name] = tar_extract_grayscale(gt_data, gt_mem, 'ground-truth')
        # The submission files are not necessarily in the order: store them to read sequentially in the next loop
        base_name = gt_name[:-len(gt_suffix)]
        sub_name = base_name + submission_suffix
        mask_name = base_name + mask_suffix
        samples_name = base_name + samples_suffix
        sub_files[sub_name] = None
        mask_files[mask_name] = None
        samples_files[samples_name] = None
        evaluation_tuples.append((gt_name, sub_name, mask_name, samples_name))

    # ...mask ground truth files
    logger.info('evaluate_bravo - reading ground-truth mask files...')
    for mask_mem in tqqdm(gt_data.getmembers()):
        sub_name = mask_mem.name
        if sub_name in mask_files:
            mask_files[sub_name] = tar_extract_grayscale(gt_data, mask_mem, 'mask')
    missing_mask_files = [n for n, f in mask_files.items() if f is None]
    if missing_mask_files:
        error_msg = f'{len(missing_mask_files)} mask files not found in ground-truth tar: ' + \
                    ', '.join(missing_mask_files[:5]) + '...'
        logger.error('evaluate_bravo - %s', error_msg)
        raise ValueError(error_msg)

    # ...samples files
    if not compare_old:
        logger.info('evaluate_bravo - reading samples files...')
        for samp_mem in tqqdm(samples_data.getmembers()):
            samp_name = samp_mem.name
            if samp_name in samples_files:
                samples_files[samp_name] = decode_indices(tar_extract_file(samples_data, samp_mem))

    # ...submission files
    pred_files = {}
    conf_files = {}
    if not compare_old:
        logger.info('evaluate_bravo - reading submission files...')
        for sub_mem in tqqdm(submission_data.getmembers()):
            sub_name = sub_mem.name
            if sub_name in sub_files:
                f = submission_data.extractfile(sub_mem)
                submission_raw = f.read()
                f.close()
                pred_files[sub_name], conf_files[sub_name], header = bravo_decode(submission_raw, dequantize=False)
                if header['quantize_levels'] != CONF_N_LEVELS:
                    logger.error('evaluate_bravo - invalid header in submission file `%s`: %s', sub_name, header)
                    raise ValueError(f'Invalid header in submission file `{sub_name}`: {header}')

    # Performs evaluation

    # ...accumulated confusion matrices for class labels, valid pixels, and invalid pixels (in pixel counts)
    cm_valid = np.zeros((NUM_CLASSES, NUM_CLASSES))

    # ...accumulated true positive/false positive pixel counts for semantic curve-based metrics
    # ......class labels, valid pixels
    gt_tp_valid = np.zeros(CONF_N_LEVELS)
    gt_fp_valid = np.zeros(CONF_N_LEVELS)
    # ......ood detection
    ood_tp = np.zeros(CONF_N_LEVELS)  # Prediction == OOD (invalid zone)
    ood_fp = np.zeros(CONF_N_LEVELS)

    log_p = log_mv = log_nv = log_mvnv = log_i = 0  # DEBUG counters, do not affect the results
    all_invalid = 0

    logger.info('evaluate_bravo - accumulating statistics...')
    for evaluation_tuple in tqqdm(evaluation_tuples):
        # Gets the data
        gt_name, sub_name, mask_name, samples_name = evaluation_tuple
        logger.debug('evaluate_bravo - processing files of `%s`', gt_name)
        gt_file = gt_files[gt_name]
        mask_file = mask_files[mask_name]
        if compare_old:
            pred_name = gt_name[:-len(gt_suffix)] + SPLIT_TO_PRED_SUFFIX[split_name]
            conf_name = gt_name[:-len(gt_suffix)] + SPLIT_TO_CONF_SUFFIX[split_name]
            pred_member = submission_data.getmember(pred_name)
            conf_member = submission_data.getmember(conf_name)
            pred_file = tar_extract_grayscale(submission_data, pred_member)
            conf_file = tar_extract_image(submission_data, conf_member)
            conf_indices = None
        else:
            pred_file = pred_files[sub_name]
            conf_file = conf_files[sub_name]
            conf_indices = samples_files[samples_name]

        # Check the dimensions of the images
        if gt_file.shape != mask_file.shape:
            logger.error('evaluate_bravo - ground-truth and mask dimensions mismatch for file `%s`: %s vs %s',
                         gt_name, gt_file.shape, mask_file.shape)
            raise ValueError(f'Ground-truth and mask dimensions mismatch for file `{gt_name}`: '
                             f'{gt_file.shape} vs {mask_file.shape}')
        if gt_file.shape != pred_file.shape:
            logger.error('evaluate_bravo - ground-truth and prediction dimensions mismatch for file `%s`: %s vs %s',
                         gt_name, gt_file.shape, pred_file.shape)
            raise ValueError(f'Ground-truth and prediction dimensions mismatch for file `{gt_name}`: '
                             f'{gt_file.shape} vs {pred_file.shape}')

        # Converts everything to 1D arrays
        gt_file = gt_file.ravel()
        mask_valid = mask_file.ravel() == 0
        pred_file = pred_file.ravel()
        conf_file = conf_file.ravel() if compare_old else conf_file  # already 1D in new format

        # Computes and accumulates the per-pixel predicted class-labels confusion matrices
        # ...filters-out void class
        non_void = gt_file != 255
        gt_file_nv = gt_file[non_void]
        pred_file_nv = pred_file[non_void]
        mask_valid_nv = mask_valid[non_void]
        mask_invalid_nv = ~mask_valid_nv
        # ...computes matrices
        cm_valid += fast_cm(gt_file_nv[mask_valid_nv], pred_file_nv[mask_valid_nv], NUM_CLASSES)

        # Debug counters, no effect on metrics
        if show_counters:
            log_p += gt_file.size
            log_mv += np.sum(mask_valid)
            log_nv += np.sum(non_void)
            log_mvnv += np.sum(mask_valid_nv)
            log_i += 1

        # Computes and accumulates the counts for curve-based metrics...
        # ...subsample arrays
        if compare_old:
            # ...nothing is subsampled: samples everything
            all_indices = np.arange(gt_file_nv.size)
            if gt_file_nv.size > SAMPLES_PER_IMG:
                # For the comparison with the old script, we need to subsample the data in this order
                if log_i == 1:
                    logger.info('evaluate_bravo - compare_old with %s samples per image and seed %d',
                                f'{SAMPLES_PER_IMG:_}', compare_old_seed)
                np.random.seed(compare_old_seed)
                all_indices = np.random.choice(all_indices, SAMPLES_PER_IMG, replace=False)
            valid_indices = np.nonzero(mask_valid_nv)[0]
            if valid_indices.size > SAMPLES_PER_IMG:
                valid_indices = np.random.choice(valid_indices, SAMPLES_PER_IMG, replace=False)

            # ...gets derived subsampled arrays (all, valid, and invalid independent)
            conf_file_nv = conf_file[non_void]
            class_right_valid = gt_file_nv[valid_indices] == pred_file_nv[valid_indices]
            conf_all = conf_file_nv[all_indices]
            conf_valid = conf_file_nv[valid_indices]
            gt_ood = mask_invalid_nv[all_indices]

        else:
            # ...confidences are already subsampled: subsamples other data on the same indices
            assert conf_indices is not None
            gt_file = gt_file[conf_indices]
            pred_file = pred_file[conf_indices]
            mask_valid = mask_valid[conf_indices]
            mask_invalid = ~mask_valid

            assert gt_file.shape == conf_file.shape, f'{gt_file.shape} != {conf_file.shape} ' \
                f'({conf_indices.size if conf_indices is not None else None})'

            # ...gets derived subsampled arrays (all aligned to the confidences)
            class_right_valid = (gt_file == pred_file)[mask_valid]
            conf_all = conf_file
            conf_valid = conf_file[mask_valid]
            gt_ood = mask_invalid

        all_invalid += np.sum(gt_ood)

        # ...gets cummulative true positives and false positives pixel counts for each confidence level
        # ......class labels, valid pixels
        get_tp_fp_counts(class_right_valid, conf_valid, gt_tp_valid, gt_fp_valid, score_levels=CONF_N_LEVELS)
        # ......ood detection
        doubt = CONF_N_LEVELS - 1 - conf_all
        get_tp_fp_counts(gt_ood, doubt, ood_tp, ood_fp, score_levels=CONF_N_LEVELS)

    logger.log(logging.INFO if show_counters else logging.DEBUG,
                'log_p: %d, log_mv: %d, log_nv: %d, log_mvnv: %d, log_i: %d', log_p, log_mv, log_nv, log_mvnv, log_i)

    logger.info('evaluate_bravo - computing metrics...')

    miou_valid = iou_valid = auroc_valid = fpr95_valid = auprc_success_valid = auprc_error_valid = ece_valid = None
    auroc_ood = fpr95_ood = auprc_ood = None

    if semantic_metrics:
        # Metrics based on the ground-truth class labels <= cm, cm_valid, cm_invalid
        # ...mean intersection over union (mIoU) for all pixels, valid pixels, and invalid pixels
        iou_valid = per_class_iou(cm_valid).tolist()
        miou_valid = np.nanmean(iou_valid)
        # ...curve-based metrics <= gt_tp, gt_fp, gt_tp_valid, gt_fp_valid, gt_tp_invalid, gt_fp_invalid
        logger.debug('valid -  tp: %s, fp: %s', np.sum(gt_tp_valid), np.sum(gt_fp_valid))
        auroc_valid, fpr95_valid, auprc_success_valid, auprc_error_valid, ece_valid = \
                get_curve_metrics(gt_tp_valid, gt_fp_valid)

    if ood_metrics:
        if all_invalid == 0:
            logger.error('evaluate_bravo - no invalid pixels found for OOD detection')
            raise ValueError('No invalid pixels found for OOD detection')

        # Curve-metrics based on ood detection <= ood_tp, ood_fp
        logger.debug('ood -  tp: %s, fp: %s', np.sum(ood_tp), np.sum(ood_fp))
        auroc_ood, fpr95_ood, auprc_ood, _, _ = get_curve_metrics(ood_tp, ood_fp)

    computed_metrics = {
        'miou_valid': miou_valid,
        'iou_valid': iou_valid,
        'ece_valid': ece_valid,
        'auroc_valid': auroc_valid,
        'auprc_success_valid': auprc_success_valid,
        'auprc_error_valid': auprc_error_valid,
        'fpr95_valid': fpr95_valid,
        'auroc_ood': auroc_ood,
        'auprc_ood': auprc_ood,
        'fpr95_ood': fpr95_ood,
        }
    return computed_metrics


BRAVO_SUBSETS = ['ACDCfog', 'ACDCrain', 'ACDCnight', 'ACDCsnow', 'synrain', 'SMIYC', 'synobjs', 'synflare',
                 'outofcontext']


def update_results(all_results, new_results, new_key):
    all_results.update((f'{new_key}_{k}', v) for k, v in new_results.items() if v is not None)


def summarize_results(all_results, subsets):
    scalars = {k: v if '_ece_' not in k and '_fpr95_' not in k else 1.-v  # ECE and FPR95 are lower-is-better
               for k, v in all_results.items()
               if np.isscalar(v)}
    for s in subsets:
        subset_scalars = np.array([v for k, v in scalars.items() if k.startswith(f'{s}_')])
        all_results[f'{s}_hmean'] = stats.hmean(subset_scalars)
    all_semantic = [v for k, v in scalars.items() if k.endswith('_valid')]
    all_ood = [v for k, v in scalars.items() if k.endswith('_ood')]
    semantic_hmean = stats.hmean(all_semantic)
    ood_hmean = stats.hmean(all_ood)
    all_results['semantic_hmean'] = semantic_hmean
    all_results['ood_hmean'] = ood_hmean
    all_results['bravo_index'] = stats.hmean([semantic_hmean, ood_hmean])


def default_evaluation_params():
    return {
        'subsets': BRAVO_SUBSETS,
        'compare_old': False,
        'compare_old_seed': 1,
        'samples_path': None,
        'show_counters': False,
    }


def evaluate_method(gt_path, submission_path, extra_params=None):
    logger.info('evaluate_method - computing metrics...')
    extra_params = extra_params or {}
    subsets = extra_params.pop('subsets', BRAVO_SUBSETS)
    compare_old = extra_params.get('compare_old', False)
    gt_data = tarfile.open(gt_path, 'r')
    submission_data = tarfile.open(submission_path, 'r')
    samples_path = extra_params.pop('samples_path')
    if samples_path == '<PRODUCTION>':
        samples_path = os.path.join(os.path.dirname(gt_path), 'bravo_SAMPLING.tar')
    if compare_old:
        samples_data = None
    else:
        samples_data = tarfile.open(samples_path, 'r')
    extra_params['samples_data'] = samples_data
    all_results = {}

    if compare_old:
        logger.warning('COMPARISON MODE: using old submission format and %s pixels for curve-based metrics.',
                       f'{SAMPLES_PER_IMG:_}')

    logger.info('...1 of 9 - ACDCfog')
    if 'ACDCfog' in subsets:
        results = evaluate_bravo(gt_data=gt_data,
                                 submission_data=submission_data,
                                 split_name='ACDC',
                                 other_names=['/fog/'],
                                 **extra_params)
        update_results(all_results, results, 'ACDCfog')

    logger.info('...2 of 9 - ACDCrain')
    if 'ACDCrain' in subsets:
        results = evaluate_bravo(gt_data=gt_data,
                                 submission_data=submission_data,
                                 split_name='ACDC',
                                 other_names=['/rain/'],
                                 **extra_params)
        update_results(all_results, results, 'ACDCrain')

    logger.info('...3 of 9 - ACDCnight')
    if 'ACDCnight' in subsets:
        results = evaluate_bravo(gt_data=gt_data,
                                 submission_data=submission_data,
                                 split_name='ACDC',
                                 other_names=['/night/'],
                                 **extra_params)
        update_results(all_results, results, 'ACDCnight')

    logger.info('...4 of 9 - ACDCsnow')
    if 'ACDCsnow' in subsets:
        results = evaluate_bravo(gt_data=gt_data,
                                 submission_data=submission_data,
                                 split_name='ACDC',
                                 other_names=['/snow/'],
                                 **extra_params)
        update_results(all_results, results, 'ACDCsnow')

    logger.info('...5 of 9 - synrain')
    if 'synrain' in subsets:
        results = evaluate_bravo(gt_data=gt_data,
                                 submission_data=submission_data,
                                 split_name='synrain',
                                 **extra_params)
        update_results(all_results, results, 'synrain')

    logger.info('...6 of 9 - SMIYC')
    if 'SMIYC' in subsets:
        results = evaluate_bravo(gt_data=gt_data,
                                 submission_data=submission_data,
                                 split_name='SMIYC',
                                 semantic_metrics=False,
                                 ood_metrics=True,
                                 **extra_params)
        update_results(all_results, results, 'SMIYC')

    logger.info('...7 of 9 - synobjs')
    if 'synobjs' in subsets:
        results = evaluate_bravo(gt_data=gt_data,
                                 submission_data=submission_data,
                                 split_name='synobjs',
                                 ood_metrics=True,
                                 **extra_params)
        update_results(all_results, results, 'synobjs')

    logger.info('...8 of 9 - synflare')
    if 'synflare' in subsets:
        results = evaluate_bravo(gt_data=gt_data,
                                 submission_data=submission_data,
                                 split_name='synflare',
                                 **extra_params)
        update_results(all_results, results, 'synflare')

    logger.info('...9 of 9 - outofcontext')
    if 'outofcontext' in subsets:
        results = evaluate_bravo(gt_data=gt_data,
                                 submission_data=submission_data,
                                 split_name='outofcontext',
                                 **extra_params)
        update_results(all_results, results, 'outofcontext')

    summarize_results(all_results, subsets)

    # This format is expected by the ELSA BRAVO Challenge server
    return {
        'method': all_results,
        'result': True,
    }


def main():
    parser = argparse.ArgumentParser(
         description='Evaluates submissions for the ELSA BRAVO Challenge.',
         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('submission', help='path to submission tar file')
    parser.add_argument('--gt', required=True, help='path to ground-truth tar file')
    parser.add_argument('--samples', help='path to pixel samples tar file, required unles --compare_old is set')
    parser.add_argument('--results', default='results.json', help='JSON file to store the computed metrics')
    parser.add_argument('--skip_val', action='store_true', help='skips validation of the submission')
    parser.add_argument('--skip_eval', action='store_true', help='skips computing the metrics')
    parser.add_argument('--debug', action='store_true', help='enables extra verbose debug output')
    parser.add_argument('--quiet', action='store_true', help='prints only errors and warnings')
    parser.add_argument('--show_counters', action='store_true', help='shows debug counters out of debug mode')
    parser.add_argument('--compare_old', nargs='?', default=False, const=True, help='enables comparison mode, where '
                        'the script will expect submissions on the old format, and subsample 100 000 pixels per image '
                        'for the curve-based metrics. It accepts an optional argument as the seed for the random '
                        'sampling.')
    parser.add_argument('--subsets', default=['ALL'], nargs='+', choices=BRAVO_SUBSETS+['ALL'], metavar='SUBSETS',
                        help='tasks to evaluate: ALL for all tasks, or one or more items from this list, separated '
                             'by spaces: ' + ' '.join(BRAVO_SUBSETS))
    args = parser.parse_args()

    level = logging.WARNING if args.quiet else (logging.DEBUG if args.debug else logging.INFO)
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')

    if args.skip_val and args.skip_eval:
        logger.error('both --skip_val and --skip_eval are set, nothing to do.')
        sys.exit(1)

    seed = 1
    if args.compare_old:
        logger.warning('comparison mode is enabled, this is for testing purposes only, do not use in production')
        if args.compare_old is not True:
            seed = int(args.compare_old)
            args.compare_old = True
    else:
        args.compare_old = False
        if args.samples is None:
            logger.error('--samples is required unless --compare_old is set')
            exit(1)

    if not (args.skip_val or args.compare_old):
        logger.info('validating...')
        validate_data(args.gt, args.submission, None)

    if not args.skip_eval:
        logger.info('evaluating...')
        if args.subsets == ['ALL']:
            args.subsets = BRAVO_SUBSETS
        res = evaluate_method(args.gt, args.submission,
                              extra_params={'subsets': args.subsets,
                                            'compare_old': args.compare_old,
                                            'compare_old_seed': seed,
                                            'samples_path': args.samples,
                                            'show_counters': args.show_counters})
        res = res['method']

        if not args.quiet:
            logger.info('results:')
            print(json.dumps(res, indent=4))

        logger.info('saving results...')
        with open(args.results, 'wt', encoding='utf-8') as json_file:
            json.dump(res, json_file, indent=4)

    logger.info('done.')


if __name__ == '__main__':
    main()
