# Copyright (c) 2024 Valeo
# See LICENSE.md for details.

import json
import os

import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_equal
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from bravo_toolkit.codec.bravo_tarfile import extract_grayscale, extract_image
from bravo_toolkit.eval.metrics import (_get_ece_original, _get_ece_reference, get_auprc, get_auroc, get_ece,
                                        get_tp_fp_counts)


# --------- Utilities ---------

def get_random_data(seed, allow_duplicates=False):
    '''Simulates array of ground-truth labels and scores.'''
    # Samples simulated data parameters
    np.random.seed(seed)
    while True:
        sample_size = np.random.randint(2, 100)
        alpha = np.random.randint(1, 10)
        beta = np.random.randint(1, 10)
        # Samples ground truth labels, and ensures that at least one positive and one negative example are present
        y_true = np.random.randint(0, 2, size=sample_size)
        y_true[np.random.randint(sample_size//2)] = 0
        y_true[np.random.randint(sample_size//2) + sample_size//2] = 1
        # Samples scores
        y_alphas = np.where(y_true == 1, alpha, beta)
        y_betas = np.where(y_true == 1, beta, alpha)
        y_score = np.random.beta(y_alphas, y_betas)
        if not allow_duplicates:
            # Eliminates entries with duplicate scores
            y_score, indices = np.unique(y_score, return_index=True)
            y_true = y_true[indices]
        if y_score.size >= 2:
            break
    return y_true, y_score


def get_raw_counts(y_true, y_score, deduplicate=False):
    '''Gets the raw true and false positive counts from ground-truth labels and scores.'''
    # Sorts data by score
    sorted_indices = np.argsort(y_score)
    y_true = y_true[sorted_indices]
    y_score = y_score[sorted_indices]
    # Computes true and false positive counts
    tp_counts = y_true
    fp_counts = 1 - y_true
    if deduplicate:
        # Merge counts for equal scores
        _y_unique_scores, unique_indices = np.unique(y_score, return_index=True)
        tp_counts = np.add.reduceat(tp_counts, unique_indices)
        fp_counts = np.add.reduceat(fp_counts, unique_indices)
    return tp_counts, fp_counts


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(SCRIPT_DIR, "bravo_test_images")
IMAGES_METADATA_FILE = os.path.join(SCRIPT_DIR, 'bravo_test_images.json')
with open(IMAGES_METADATA_FILE, 'rt', encoding='utf-8') as imff:
    IMAGES_METADATA = json.load(imff)
    IMAGES_LIST = list(IMAGES_METADATA.keys())


def get_real_data_raw(pred_file):
    '''Gets the ground-truth labels, class predictions, and confidence scores from a real data case.'''
    conf_file = pred_file.replace('_pred.png', '_conf.png')
    gt_file = pred_file.replace('_pred.png', '_gt.png')

    # Loads ground-truth and confidence images
    with open(os.path.join(IMAGES_DIR, pred_file), 'rb') as f:
        pred = extract_grayscale(f, 'class prediction')
    with open(os.path.join(IMAGES_DIR, conf_file), 'rb') as f:
        conf = extract_image(f, 'confidence')
    with open(os.path.join(IMAGES_DIR, gt_file), 'rb') as f:
        gt = extract_grayscale(f, 'gt')

    return gt, pred, conf


def get_real_data(pred_file):
    '''Gets the binarized labels (hit/miss) and confidence scores from a real data case.'''
    gt, pred, conf = get_real_data_raw(pred_file)
    return gt.ravel() == pred.ravel(), conf.ravel()


BIG = 1000000000  # Coefficient for big numbers

# --------- AUROC ---------

auroc_test_cases = [
        # tp_counts, fp_counts, expected_auc
        # tp_counts => exact (not cummulative) counts of true positives for each increasing confidence level
        # fp_counts => exact (not cummulative) counts of false positives for each increasing confidence level
        # Perfectly wrong predictions
        (1, np.array([100, 0, 0, 0, 0]), np.array([0, 0, 0, 0, 100]), 0.0),
        (2, np.array([13, 1, 45, 57, 0]), np.array([0, 0, 0, 0, 100]), 0.0),
        (3, np.array([13, 1, 45, 1, 57, 0, 0, 0, 0]), np.array([0, 0, 0, 0, 0, 30, 2, 4, 16]), 0.0),
        # Mostly wrong predictions
        (4, np.array([20, 20, 20, 20, 20]), np.array([0, 0, 0, 0, 100]), 0.1),  # 20/100/2 (/2 due to trapezoidal rule)
        (5, np.array([13, 1, 45, 1, 57]), np.array([0, 0, 0, 0, 100]), 0.24358974358974358),  # 57/sum(tp)/2
        (6, np.array([100, 0, 0, 0, 0]), np.array([20, 20, 20, 20, 20]), 0.1),  # 1/5/2
        # Perfectly correct predictions
        (7, np.array([0, 0, 0, 0, 100]), np.array([100, 0, 0, 0, 0]), 1.0),
        (8, np.array([0, 0, 0, 0, 0, 30, 2, 4, 16]), np.array([13, 1, 45, 1, 57, 0, 0, 0, 0]), 1.0),
        (9, np.array([0, 0, 0, 0, 100]), np.array([20, 20, 20, 20, 0]), 1.0),
        (10, np.array([0, 0, 0, 0, 100]), np.array([13, 1, 45, 57, 0]), 1.0),
        # Mostly correct predictions
        (11, np.array([0, 0, 0, 0, 100]), np.array([20, 20, 20, 20, 20]), 0.9),  # 1 - 1/5/2
        (12, np.array([0, 0, 0, 0, 100]), np.array([13, 1, 45, 1, 57]), 0.7564102564102564),  # 1 - 57/sum(tp)/2
        # Corner cases
        (13, np.array([100, 0, 0, 0, 0]), np.array([100, 0, 0, 0, 0]), 0.5),  # Everything happens at the last step
        (14, np.array([0, 0, 0, 0, 100]), np.array([0, 0, 0, 0, 100]), 0.5),  # Everything happens at the first step
        # Perfect ignorance
        (15, np.array([517, 517, 517, 517, 517]), np.array([517, 517, 517, 517, 517]), 0.5),
        # Basic test case
        (16, np.array([40, 10, 30, 20]), np.array([15, 25, 5, 35]), 0.3875),
        # Basic test case - different shapes
        (17, np.array([[40, 10, 30, 20]]), np.array([[15, 25, 5, 35]]), 0.3875),
        (18, np.array([[40, 10], [30, 20]]), np.array([[15, 25], [5, 35]]), 0.3875),
        # Basic test case - heterogeneos shapes
        (19, np.array([[40, 10], [30, 20]]), np.array([15, 25, 5, 35]), 0.3875),
        # Basic test case - large numbers
        (20, np.array([40*BIG, 10*BIG, 30*BIG, 20*BIG]), np.array([15*BIG, 25*BIG, 5*BIG, 35*BIG]), 0.3875),
    ]


@pytest.mark.parametrize('_n, tp, fp, expected_auc', auroc_test_cases)
def test_metrics_get_auroc_basic(_n, tp, fp, expected_auc):
    auc, tpr, fpr = get_auroc(tp, fp)
    expected_tpr = np.cumsum(tp.ravel()[::-1]) / np.sum(tp)
    expected_fpr = np.cumsum(fp.ravel()[::-1]) / np.sum(fp)
    expected_tpr = np.concatenate(([0.], expected_tpr))
    expected_fpr = np.concatenate(([0.], expected_fpr))
    assert_almost_equal(auc, expected_auc, decimal=5)
    assert_array_almost_equal(tpr, expected_tpr)
    assert_array_almost_equal(fpr, expected_fpr)


@pytest.mark.parametrize('seed', range(100))
def test_metrics_get_auroc_reference(seed):
    y_true, y_score = get_random_data(seed)
    # Computes reference AUC
    expected_auc = roc_auc_score(y_true, y_score)
    # Computes toolkit AUC
    tp_counts, fp_counts = get_raw_counts(y_true, y_score)
    auc, tpr, fpr = get_auroc(tp_counts, fp_counts)
    assert_almost_equal(auc, expected_auc, decimal=5)
    assert_equal(tpr[0], 0.)
    assert_equal(fpr[0], 0.)
    assert_equal(tpr[-1], 1.)
    assert_equal(fpr[-1], 1.)
    assert_equal(tpr.size, fpr.size)
    assert_equal(tpr, np.sort(tpr))
    assert_equal(fpr, np.sort(fpr))


def test_metrics_get_auroc_empty_arrays():
    tp = np.array([])
    fp = np.array([])
    with pytest.raises(ValueError):
        get_auroc(tp, fp)


def test_metrics_get_auroc_single_element():
    tp = np.array([1])
    fp = np.array([1])
    with pytest.raises(ValueError):
        get_auroc(tp, fp)


def test_metrics_get_auroc_mismatched_lengths():
    np.random.seed(0)
    tp = np.random.randint(1, 1000, size=100)
    fp = np.random.randint(1, 1000, size=99)
    with pytest.raises(ValueError):
        get_auroc(tp, fp)


def test_metrics_get_auroc_zero_arrays():
    tp = np.zeros(17)
    fp = np.zeros(17)
    with pytest.raises(ValueError):
        get_auroc(tp, fp)


@pytest.mark.parametrize('pred_file', IMAGES_LIST)
def test_metrics_get_auroc_real_data(pred_file):
    y_true, y_score = get_real_data(pred_file)
    # Computes reference AUC
    auc_ref = roc_auc_score(y_true, y_score)
    # Gets toolkit AUC with reference counts
    tp_counts, fp_counts = get_raw_counts(y_true, y_score, deduplicate=True)
    auc, _, _ = get_auroc(tp_counts, fp_counts)
    # Gets toolkit AUC with toolkit counts
    tp_counts2, fp_counts2 = get_tp_fp_counts(y_true, y_score, score_levels=65536)
    auc2, _, _ = get_auroc(tp_counts2, fp_counts2)
    # Compares values
    assert_almost_equal(auc, auc_ref, decimal=5)
    assert_almost_equal(auc, auc2, decimal=6)


def test_metrics_get_auroc_aggregated():
    '''Check that computing the metric pixel-wise over many images and aggregating the counts gives the same result.'''
    score_levels = 65536
    tp_counts = np.zeros(score_levels)
    fp_counts = np.zeros(score_levels)
    y_trues = []
    y_scores = []
    for pred_file in IMAGES_LIST:
        y_true, y_score = get_real_data(pred_file)
        get_tp_fp_counts(y_true, y_score, tp_counts, fp_counts, score_levels=score_levels)
        y_trues.append(y_true)
        y_scores.append(y_score)
    # Computes reference AUC using all data
    y_trues = np.concatenate(y_trues)
    y_scores = np.concatenate(y_scores)
    auc_ref = roc_auc_score(y_trues, y_scores)
    # Gets toolkit AUC with aggregated reference counts
    auc, _, _ = get_auroc(tp_counts, fp_counts)
    # Compares values
    assert_almost_equal(auc, auc_ref, decimal=5)


@pytest.mark.skip(reason="this test currently fails, because aggressively subsampling individual images is not stable")
@pytest.mark.parametrize('pred_file', IMAGES_LIST)
def test_metrics_get_auroc_subsamples(pred_file):
    sample_conf = 8  # This picks only 1/(sample_conf*sample_conf) of the data
    sample_offsets = (0, sample_conf//2)

    y_true, y_score = get_real_data(pred_file)
    aucs = []
    for sample_offset in sample_offsets:
        # Subsamples data
        y_true = y_true[sample_offset::sample_conf]
        y_score = y_score[sample_offset::sample_conf]
        # Gets toolkit AUC with toolkit counts
        tp_counts, fp_counts = get_tp_fp_counts(y_true, y_score, score_levels=65536)
        auc, _, _ = get_auroc(tp_counts, fp_counts)
        aucs.append(auc)
        # Compares values
    assert_almost_equal(aucs[0], aucs[1], decimal=3)


def test_metrics_get_auroc_subsamples_aggregated():
    sample_conf = 8
    sample_offsets = (0, sample_conf//2)
    score_levels = 65536

    tp_counts_aggregated = [np.zeros(score_levels) for _ in sample_offsets]
    fp_counts_aggregated = [np.zeros(score_levels) for _ in sample_offsets]
    aucs = []
    for pred_file in IMAGES_LIST:
        y_true, y_score = get_real_data(pred_file)
        # Aggregation for AUROC
        for s, sample_offset in enumerate(sample_offsets):
            # Subsamples for AUROC
            y_true_subsample = y_true[sample_offset::sample_conf]
            y_score_subsample = y_score[sample_offset::sample_conf]
            tp_counts_subsample, fp_counts_subsample = get_tp_fp_counts(y_true_subsample, y_score_subsample,
                                                                        score_levels=score_levels)
            tp_counts_aggregated[s] += tp_counts_subsample
            fp_counts_aggregated[s] += fp_counts_subsample

    # Aggregated AUROC Assertions
    aucs = []
    for s, _ in enumerate(sample_offsets):
        auc, _, _ = get_auroc(tp_counts_aggregated[s], fp_counts_aggregated[s])
        aucs.append(auc)

    # Subsampled AUROC Assertions
    assert_almost_equal(aucs[0], aucs[1], decimal=2)


# --------- FPR@95 ---------

@pytest.mark.parametrize('pred_file', IMAGES_LIST)
def test_metrics_get_auroc_fpr_at_tpr_th(pred_file, tpr_th=0.95):
    y_true, y_score = get_real_data(pred_file)
    # Computes reference AUC
    fpr_ref, tpr_ref, _ = roc_curve(y_true, y_score)
    fpr_at_th_ref = fpr_ref[np.argmax(tpr_ref >= tpr_th)]
    # Gets toolkit AUC with reference counts
    tp_counts, fp_counts = get_raw_counts(y_true, y_score, deduplicate=True)
    _auc, tpr, fpr = get_auroc(tp_counts, fp_counts)
    tpr_th_i = np.searchsorted(tpr, tpr_th, 'left')
    fpr_at_th = fpr[tpr_th_i]
    # Compares values
    assert_almost_equal(fpr_at_th, fpr_at_th_ref, decimal=5)


# --------- AUPRC ---------

AUPRC = 0.5096161616161616


auprc_test_cases = [
        # tp_counts, fp_counts, expected_auc
        # tp_counts => exact (not cummulative) counts of true positives for each increasing confidence level
        # fp_counts => exact (not cummulative) counts of false positives for each increasing confidence level
        # Mostly wrong predictions
        (1, np.array([10, 0, 0, 0, 0]), np.array([0, 0, 0, 0, 10]), 0.5),  # AUPRC uses step-function integral
        (2, np.array([40, 0, 0, 0, 0]), np.array([0, 10, 10, 10, 10]), 0.5),
        # Perfectly correct predictions
        (3, np.array([0, 0, 0, 0, 100]), np.array([100, 0, 0, 0, 0]), 1.0),
        (4, np.array([0, 0, 0, 0, 0, 30, 2, 4, 16]), np.array([13, 1, 45, 1, 57, 0, 0, 0, 0]), 1.0),
        (5, np.array([0, 0, 0, 0, 100]), np.array([20, 20, 20, 20, 0]), 1.0),
        (6, np.array([0, 0, 0, 0, 100]), np.array([13, 1, 45, 57, 0]), 1.0),
        # Mostly correct predictions
        (7, np.array([0, 0, 0, 40, 0]), np.array([30, 0, 0, 0, 10]), 0.8),
        # Corner cases
        (8, np.array([100, 0, 0, 0, 0]), np.array([100, 0, 0, 0, 0]), 0.5),  # Everything happens at the last step
        (9, np.array([0, 0, 0, 0, 100]), np.array([0, 0, 0, 0, 100]), 0.5),  # Everything happens at the first step
        (10, np.array([0, 0, 0, 0, 0]), np.array([0, 0, 0, 0, 100]), 0.),  # No true positives
        (11, np.array([0, 0, 0, 0, 100]), np.array([0, 0, 0, 0, 0]), 1.),  # No false positives
        (12, np.array([0, 0, 0, 0, 0]), np.array([0, 0, 0, 0, 0]), 0.),  # No samples
        # Perfect ignorance
        (13, np.array([517, 517, 517, 517, 517]), np.array([517, 517, 517, 517, 517]), 0.5),
        # Basic test case
        (14, np.array([40, 10, 30, 20]), np.array([15, 25, 5, 35]), AUPRC),
        # Basic test case - different shapes
        (15, np.array([[40, 10, 30, 20]]), np.array([[15, 25, 5, 35]]), AUPRC),
        (16, np.array([[40, 10], [30, 20]]), np.array([[15, 25], [5, 35]]), AUPRC),
        # Basic test case - heterogeneous shapes
        (17, np.array([[40, 10], [30, 20]]), np.array([15, 25, 5, 35]), AUPRC),
        # Basic test case - large numbers
        (18, np.array([40*BIG, 10*BIG, 30*BIG, 20*BIG]), np.array([15*BIG, 25*BIG, 5*BIG, 35*BIG]), AUPRC),
    ]


@pytest.mark.filterwarnings("ignore:.*divide:RuntimeWarning")
@pytest.mark.parametrize('_n, tp, fp, expected_auprc', auprc_test_cases)
def test_metrics_get_auprc_basic(_n, tp, fp, expected_auprc):
    auprc, precision, recall = get_auprc(tp, fp)
    tp_counts = np.cumsum(tp.ravel()[::-1])
    fp_counts = np.cumsum(fp.ravel()[::-1])
    pp_counts = tp_counts + fp_counts
    expected_precision = np.where(pp_counts == 0, 1., tp_counts / pp_counts)
    expected_recall = (tp_counts / tp_counts[-1]) if tp_counts[-1] > 0 else tp_counts
    expected_precision = np.concatenate(([1.], expected_precision))
    expected_recall = np.concatenate(([0.], expected_recall))
    assert_almost_equal(auprc, expected_auprc, decimal=5)
    assert_array_almost_equal(precision, expected_precision)
    assert_array_almost_equal(recall, expected_recall)


@pytest.mark.parametrize('seed', range(100))
def test_metrics_get_auprc_reference(seed):
    y_true, y_score = get_random_data(seed)
    # Computes reference auprc
    expected_auprc = average_precision_score(y_true, y_score)
    # Computes toolkit auprc
    tp_counts, fp_counts = get_raw_counts(y_true, y_score)
    auprc, precision, recall = get_auprc(tp_counts, fp_counts)
    assert_almost_equal(auprc, expected_auprc, decimal=5)
    assert_equal(precision[0], 1.)
    assert_equal(recall[0], 0.)
    assert_equal(recall[-1], 1.)
    assert_equal(precision.size, recall.size)
    assert_equal(recall, np.sort(recall))


def test_metrics_get_auprc_empty_arrays():
    tp = np.array([])
    fp = np.array([])
    with pytest.raises(ValueError):
        get_auprc(tp, fp)


def test_metrics_get_auprc_single_element():
    tp = np.array([1])
    fp = np.array([1])
    with pytest.raises(ValueError):
        get_auprc(tp, fp)


def test_metrics_get_auprc_mismatched_lengths():
    np.random.seed(0)
    tp = np.random.randint(1, 1000, size=100)
    fp = np.random.randint(1, 1000, size=99)
    with pytest.raises(ValueError):
        get_auprc(tp, fp)


@pytest.mark.parametrize('pred_file', IMAGES_LIST)
def test_metrics_get_auprc_real_data(pred_file):
    y_true, y_score = get_real_data(pred_file)
    # Computes reference auprc
    auprc_ref = average_precision_score(y_true, y_score)
    # Gets toolkit auprc
    tp_counts, fp_counts = get_raw_counts(y_true, y_score, deduplicate=True)
    auprc, _, _ = get_auprc(tp_counts, fp_counts)
    # Compares values
    assert_almost_equal(auprc, auprc_ref, decimal=5)


def test_metrics_get_auprc_aggregated():
    '''Check that computing the metric pixel-wise over many images and aggregating the counts gives the same result.'''
    score_levels = 65536
    tp_counts = np.zeros(score_levels)
    fp_counts = np.zeros(score_levels)
    y_trues = []
    y_scores = []
    for pred_file in IMAGES_LIST:
        y_true, y_score = get_real_data(pred_file)
        get_tp_fp_counts(y_true, y_score, tp_counts, fp_counts, score_levels=score_levels)
        y_trues.append(y_true)
        y_scores.append(y_score)
    # Computes reference auprc using all data
    y_trues = np.concatenate(y_trues)
    y_scores = np.concatenate(y_scores)
    auprc_ref = average_precision_score(y_trues, y_scores)
    # Gets toolkit auprc with aggregated reference counts
    auprc, _, _ = get_auprc(tp_counts, fp_counts)
    # Compares values
    assert_almost_equal(auprc, auprc_ref, decimal=5)


@pytest.mark.skip(reason="this test currently fails, because aggressively subsampling individual images is not stable")
@pytest.mark.parametrize('pred_file', IMAGES_LIST)
def test_metrics_get_auprc_subsamples(pred_file):
    sample_conf = 8  # This picks only 1/(sample_conf*sample_conf) of the data
    sample_offsets = (0, sample_conf//2)

    y_true, y_score = get_real_data(pred_file)
    auprcs = []
    for sample_offset in sample_offsets:
        # Subsamples data
        y_true = y_true[sample_offset::sample_conf]
        y_score = y_score[sample_offset::sample_conf]
        # Gets toolkit PRC with toolkit counts
        tp_counts, fp_counts = get_tp_fp_counts(y_true, y_score, score_levels=65536)
        auprc, _, _ = get_auprc(tp_counts, fp_counts)
        auprcs.append(auprc)
        # Compares values
    assert_almost_equal(auprcs[0], auprcs[1], decimal=4)


def test_metrics_get_auprc_subsamples_aggregated():
    sample_conf = 8
    sample_offsets = (0, sample_conf//2)
    score_levels = 65536

    tp_counts_aggregated = [np.zeros(score_levels) for _ in sample_offsets]
    fp_counts_aggregated = [np.zeros(score_levels) for _ in sample_offsets]
    auprcs = []

    for pred_file in IMAGES_LIST:
        y_true, y_score = get_real_data(pred_file)
        # Aggregation for AUPRC
        for s, sample_offset in enumerate(sample_offsets):
            # Subsamples for AUPRC
            y_true_subsample = y_true[sample_offset::sample_conf]
            y_score_subsample = y_score[sample_offset::sample_conf]
            tp_counts_subsample, fp_counts_subsample = get_tp_fp_counts(y_true_subsample, y_score_subsample,
                                                                        score_levels=score_levels)
            tp_counts_aggregated[s] += tp_counts_subsample
            fp_counts_aggregated[s] += fp_counts_subsample

    # Aggregated AUPRC Assertions
    for s, _ in enumerate(sample_offsets):
        auprc, _, _ = get_auprc(tp_counts_aggregated[s], fp_counts_aggregated[s])
        auprcs.append(auprc)

    # Subsampled AUPRC Assertions
    assert_almost_equal(auprcs[0], auprcs[1], decimal=2)


# --------- ECE ---------

@pytest.mark.parametrize('pred_file', IMAGES_LIST)
def test_metrics_get_ece_real_data(pred_file):
    score_levels = 65536
    y_true, y_pred, y_score = get_real_data_raw(pred_file)

    # Computes original ece
    ece_original = _get_ece_original(label=y_true, pred=y_pred, conf=y_score, ECE_NUM_BINS=15,
                                     CONF_NUM_BINS=score_levels, DEBIAS=True)
    # Computes reference ece
    y_score_continuous = (y_score.astype(np.float32) + 0.5) / score_levels
    ece_ref15 = _get_ece_reference(y_true, y_pred, y_score_continuous, ece_bins=15)
    ece_ref32 = _get_ece_reference(y_true, y_pred, y_score_continuous, ece_bins=32)

    # Convert to 1D
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    y_score = y_score.ravel()

    # Gets toolkit ece
    y_score = y_score.ravel()
    d_counts = np.zeros(score_levels, dtype=np.int64)
    t_counts = np.zeros(score_levels, dtype=np.int64)
    np.add.at(d_counts, y_score, y_true == y_pred)
    np.add.at(t_counts, y_score, 1)
    confidence_values = (np.linspace(0, score_levels-1, score_levels) + 0.5) / score_levels
    ece15 = get_ece(d_counts, t_counts, confidence_values, bins=15)
    ece32 = get_ece(d_counts, t_counts, confidence_values, bins=32)

    # Compares values
    assert_almost_equal(ece_ref15, ece_original, decimal=6)
    assert_almost_equal(ece15, ece_ref15, decimal=6)
    assert_almost_equal(ece32, ece_ref32, decimal=6)


def test_metrics_get_ece_aggregated():
    '''Check that computing the metric pixel-wise over many images and aggregating the counts gives the same result.'''
    score_levels = 65536

    y_trues = []
    y_preds = []
    y_scores = []
    for pred_file in IMAGES_LIST:
        y_t, y_p, y_s = get_real_data_raw(pred_file)
        y_t = y_t.ravel()
        y_p = y_p.ravel()
        y_s = y_s.ravel()
        y_trues.append(y_t)
        y_preds.append(y_p)
        y_scores.append(y_s)
    y_trues = np.concatenate(y_trues)
    y_preds = np.concatenate(y_preds)
    y_scores = np.concatenate(y_scores)

    # Computes original ece using all data
    ece_original = _get_ece_original(label=y_trues, pred=y_preds, conf=y_scores, ECE_NUM_BINS=15,
                                     CONF_NUM_BINS=score_levels, DEBIAS=True)

    # Computes reference ece using all data
    y_scores_continuous = (y_scores.astype(np.float32) + 0.5) / score_levels
    ece_ref15 = _get_ece_reference(y_trues, y_preds, y_scores_continuous, ece_bins=15)
    ece_ref32 = _get_ece_reference(y_trues, y_preds, y_scores_continuous, ece_bins=32)

    # Gets toolkit ece with aggregated reference counts
    y_scores = y_scores.ravel()
    d_counts = np.zeros(score_levels, dtype=np.int64)
    t_counts = np.zeros(score_levels, dtype=np.int64)
    np.add.at(d_counts, y_scores, y_trues == y_preds)
    np.add.at(t_counts, y_scores, 1)
    confidence_values = (np.linspace(0, score_levels-1, score_levels) + 0.5) / score_levels
    ece15 = get_ece(d_counts, t_counts, confidence_values, bins=15)
    ece32 = get_ece(d_counts, t_counts, confidence_values, bins=32)

    # Compares values
    assert_almost_equal(ece_ref15, ece_original, decimal=6)
    assert_almost_equal(ece15, ece_ref15, decimal=6)
    assert_almost_equal(ece32, ece_ref32, decimal=6)


@pytest.mark.skip(reason="this test currently fails, because aggressively subsampling individual images is not stable")
@pytest.mark.parametrize('pred_file', IMAGES_LIST)
def test_metrics_get_ece_subsamples(pred_file):
    sample_conf = 8  # This picks only 1/(sample_conf*sample_conf) of the data
    sample_offsets = (0, sample_conf//2)
    score_levels = 65536

    y_true, y_pred, y_score = get_real_data_raw(pred_file)
    eces = []
    for sample_offset in sample_offsets:
        # Subsamples data
        y_true_sample = y_true[sample_offset::sample_conf].ravel()
        y_pred_sample = y_pred[sample_offset::sample_conf].ravel()
        y_score_sample = y_score[sample_offset::sample_conf].ravel()
        # Computes toolkit ECE with subsampled data
        y_score_sample = y_score_sample.ravel()
        d_counts = np.zeros(score_levels, dtype=np.int64)
        t_counts = np.zeros(score_levels, dtype=np.int64)
        np.add.at(d_counts, y_score_sample, y_true_sample == y_pred_sample)
        np.add.at(t_counts, y_score_sample, 1)
        confidence_values = (np.linspace(0., score_levels-1, score_levels) + 0.5) / score_levels
        ece = get_ece(d_counts, t_counts, confidence_values, bins=15)
        eces.append(ece)

    # Compares values
    assert_almost_equal(eces[0], eces[1], decimal=4)


def test_metrics_get_ece_subsamples_aggregated():
    sample_conf = 8
    sample_offsets = (0, sample_conf//2)
    score_levels = 65536

    d_counts_aggregated = [np.zeros(score_levels, dtype=np.int64) for _ in sample_offsets]
    t_counts_aggregated = [np.zeros(score_levels, dtype=np.int64) for _ in sample_offsets]
    eces = []
    for pred_file in IMAGES_LIST:
        y_true, y_pred, y_score = get_real_data_raw(pred_file)
        # Aggregation for ECE
        for s, sample_offset in enumerate(sample_offsets):
            # Subsamples for ECE
            y_true_subsample = y_true[sample_offset::sample_conf].ravel()
            y_pred_subsample = y_pred[sample_offset::sample_conf].ravel()
            y_score_subsample = y_score[sample_offset::sample_conf].ravel()
            np.add.at(d_counts_aggregated[s], y_score_subsample, y_true_subsample == y_pred_subsample)
            np.add.at(t_counts_aggregated[s], y_score_subsample, 1)

    # Aggregated ECE Assertions
    confidence_values = (np.linspace(0., score_levels-1, score_levels) + 0.5) / score_levels
    for s, _ in enumerate(sample_offsets):
        ece = get_ece(d_counts_aggregated[s], t_counts_aggregated[s], confidence_values)
        eces.append(ece)

    # Subsampled ECE Assertions
    assert_almost_equal(eces[0], eces[1], decimal=2)
