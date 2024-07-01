# Copyright (c) 2024 Valeo
# See LICENSE.md for details.
import numpy as np
from numpy import ndarray


def _get_ece_original(*, conf, pred, label, ECE_NUM_BINS=15, CONF_NUM_BINS=65535, DEBIAS=False):
    '''
    Original implementation of the Expected Calibration Error (ECE) metric, before refactoring.

    Used for test purposes only. Use _get_ece_reference() instead for an equivalent reference implementation.
    Use get_ece() on production code.

    The default parameters are set to match the original implementation. Set CONF_NUM_BINS=65536, DEBIAS=True for a
    closer match to the new implementations
    '''
    conf = conf.astype(np.float32)
    # normalize conf to [0, 1]
    if DEBIAS:
        conf = (conf + 0.5) / CONF_NUM_BINS
    else:
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
        ece = np.average(np.absolute(mean_conf-acc_tab), weights=nb_items_bin.astype(np.float32)/np.sum(nb_items_bin))
    else:
        raise ValueError('No samples found for ECE calculation.')
    return ece


def _get_ece_reference(y_true, y_pred, y_conf, bin_min=0., bin_max=1., ece_bins=15):
    '''
    Evaluates the Expected Calibration Error (ECE) - Reference implementation for testing and debugging.

    The data is split into `ece_bins` bins based on their confidence values.

    For each bin, the average confidence and accuracy are computed. The ECE is then computed by:
        ECE = sum_i (|avg_acc_i - avg_conf_i| * bin_count_i/n_samples)

    In short, the ECE checks if the confidence values are well-calibrated in comparison to the accuracies, in an
    average weighted by the observations. In this implementation, the bins are equally spaced in the confidence range
    [1./num_classes, 1.]

    The input data is not verified, and is assumed to be valid.

    Args:
        y_true (np.ndarray): ground-truth class labels
        y_pred (np.ndarray): predicted class labels
        y_conf (np.ndarray): confidence for predicted class labels, in the range [1./num_classes, 1.]
        ece_bins (int): number of bins for ECE metric, default is 15

    Returns:
        float: ECE
    '''
    # Convert to 1D arrays
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    y_conf = y_conf.ravel()

    # Quantize confidence values into bins
    bin_limits = np.linspace(bin_min, bin_max, ece_bins + 1)
    bin_counts = np.zeros(ece_bins)
    bin_indices = np.digitize(y_conf, bin_limits) - 1  # Allocate each sample to its bin
    bin_indices = np.clip(bin_indices, 0, ece_bins - 1)  # Clip the samples to valid bins
    bin_counts = np.bincount(bin_indices, minlength=ece_bins)

    # Compute mean accuracy and confidence for each bin
    bin_accs = np.zeros(ece_bins)  # Empirical accuracy
    bin_confs = np.zeros(ece_bins)  # Predicted confidence
    for i in range(ece_bins):
        if bin_counts[i] == 0:
            continue
        bin_indices_i = bin_indices == i
        bin_accs[i] = np.mean((y_pred == y_true)[bin_indices_i])
        bin_confs[i] = np.mean(y_conf[bin_indices_i])

    # Gets average difference between confidence and accuracy, weighted by bin counts
    has_data = bin_counts > 0
    bin_confs = bin_confs[has_data]
    bin_accs = bin_accs[has_data]
    bin_counts = bin_counts[has_data]
    n_samples = np.sum(bin_counts)
    if n_samples != 0:
        ece_value = np.average(np.absolute(bin_confs - bin_accs), weights=bin_counts.astype(np.float32) / n_samples)
    else:
        raise ValueError('No samples found for ECE calculation.')

    return ece_value


def _ece_bin_subsample(all_counts, bins=15):
    '''
    Subsamples the counts of samples in each bin to a smaller number of bins.

    If the number of bins is not a divisor of the original number of bins:
    - The remaining bins are merged into the last bin, if the number of remaining bins is less than half the target.
    - Otherwise the remaining bins are given an ind

    Args:
        counts (tuple of np.ndarray): Raw counts to be subsampled (see `get_ece` for details on the counts format)
            All arrays will be raveled to 1D and must have the same size.
        bins (int): The target number of bins. Default is 15.

    Returns:
        tuple of np.ndarray: The subsampled counts.
    '''

    # Get the number of bins in the original data and ensure that all counts have the same length
    n = all_counts[0].size
    if not all(c.size == n for c in all_counts):
        raise ValueError('All arrays must have the same size.')
    if bins <= 0 or bins > n:
        raise ValueError('The number of bins must be a positive integer <= the size of the arrays.')

    # Calculate the subsampling factor
    bin_limits = np.round(np.linspace(0, n, bins + 1)).astype(np.int64)

    # Use np.add.reduceat to compute the sums for each bin
    binned = [np.add.reduceat(c.ravel(), bin_limits[:-1]) for c in all_counts]

    return binned


def get_ece(d_counts, t_counts, confidence_values, *, bins=15):
    '''
    Calculate the Expected Calibration Error (ECE) for a binary classifier given the counts of true and false positive
    samples at different levels of confidence for the classifier. The counts are assumed to be ordered by increasing
    confidence level and to be for exact values of confidence (i.e. not cumulative counts).

    Args:
        d_counts (np.ndarray): The number of correctly classified (diagonal) samples at each confidence level.
        t_counts (np.ndarray): The number of total samples at each confidence level.
        confidence_values (np.ndarray): The confidence values corresponding to each level, in the range [0, 1].
        bins (int): Desired number of bins for the ECE calculation. Default is 15.

    Returns:
        float: The ECE of the classifier.
    '''
    N = np.sum(t_counts)
    if N == 0:
        raise ValueError('No samples found for ECE calculation.')

    weighted_confidences = t_counts * confidence_values
    d_counts, t_counts, weighted_confidences = _ece_bin_subsample((d_counts, t_counts, weighted_confidences), bins=bins)

    # Computes statistics and groups them into bins
    weighted_confidences = weighted_confidences / t_counts
    weights = t_counts.astype(np.float32) / N
    accuracies = d_counts / t_counts

    # Remove levels with no samples
    has_data = t_counts > 0
    weighted_confidences = weighted_confidences[has_data]
    weights = weights[has_data]
    accuracies = accuracies[has_data]

    # Compute ECE
    ece_value = np.average(np.absolute(weighted_confidences - accuracies), weights=weights)
    return ece_value


def get_auroc(tp_counts: ndarray, fp_counts: ndarray) -> tuple[float, ndarray, ndarray]:
    '''
    Calculate the ROC curve and AUC for a binary classifier given the counts of true and false positive samples at
    different levels of confidence for the classifier. The counts are assumed to be ordered by increasing confidence
    level and to be for exact values of confidence (i.e. not cumulative counts).

    The counts are assumed to be non-negative: behavior is undefined if this is not the case.

    Args:
        tp_counts (np.ndarray): The number of true positive samples at each confidence level.
        fp_counts (np.ndarray): The number of false positive samples at each confidence level.

    Raise:
        ValueError: If the counts have different sizes, if they have less than two elements, or if they are entirely
                    zero.

    Returns:
        float: The AUC of the ROC curve.
        np.ndarray: The true positive rates at each confidence level, starting from 0 and not decreasing.
        np.ndarray: The false positive rates at each confidence level, starting from 0 and not decreasing.
    '''
    if tp_counts.size != fp_counts.size:
        raise ValueError('tp_counts and fp_counts must have the same length.')
    if tp_counts.size <= 1:
        raise ValueError('tp_counts and fp_counts must have at least two elements.')

    # Reverse cumulative sums get the counts up-to-and-above each confidence level
    tp_counts = tp_counts.ravel()
    fp_counts = fp_counts.ravel()
    tp_cumsum = np.cumsum(tp_counts[::-1])
    fp_cumsum = np.cumsum(fp_counts[::-1])

    # The last element of the cumulative sums is the total
    p_total = tp_cumsum[-1]
    n_total = fp_cumsum[-1]
    if p_total == 0 or n_total == 0:
        raise ValueError('tp_counts and fp_counts must have at least one non-zero element.')

    # Calculate TPR and FPR
    tpr = tp_cumsum / p_total
    fpr = fp_cumsum / n_total

    # Starts from an implicit (0, 0) point
    tpr = np.concatenate(([0.], tpr))
    fpr = np.concatenate(([0.], fpr))

    # Calculate AUC using the trapezoidal rule
    auc = np.trapz(y=tpr, x=fpr)

    return auc, tpr, fpr


def get_auprc(tp_counts, fp_counts):
    '''
    Calculate the PR curve and AUC for a binary classifier given the counts of true and false positive samples at
    different levels of confidence for the classifier. The counts are assumed to be ordered by increasing confidence
    level and to be for exact values of confidence (i.e. not cumulative counts).

    Args:
        tp_counts (np.ndarray): The number of true positive samples at each confidence level.
        fp_counts (np.ndarray): The number of false positive samples at each confidence level.

    Returns:
        float: The AUC of the PR curve.
        np.ndarray: The precision values at each confidence level, starting from 1. The values are not monotonic.
        np.ndarray: The recall values at each confidence level, starting from 0 and not decreasing.
    '''
    if tp_counts.size != fp_counts.size:
        raise ValueError('tp_counts and fp_counts must have the same length.')
    if tp_counts.size <= 1:
        raise ValueError('tp_counts and fp_counts must have at least two elements.')
    tp_counts = tp_counts.ravel()
    fp_counts = fp_counts.ravel()

    # Reverse cumulative sums get the counts up-to-and-above each confidence level
    tp_counts = tp_counts.ravel()
    fp_counts = fp_counts.ravel()
    tp_cumsum = np.cumsum(tp_counts[::-1])
    fp_cumsum = np.cumsum(fp_counts[::-1])

    # The last element of the cumulative sums is the total
    p_total = tp_cumsum[-1]

    # Calculate precision and recall
    pp_cumsum = tp_cumsum + fp_cumsum  # Cumulative sum of positive predictions
    with np.errstate(divide='ignore', invalid='ignore'):
        precision = np.where(pp_cumsum == 0, 1., tp_cumsum / pp_cumsum)
    recall = (tp_cumsum / p_total) if p_total > 0 else tp_cumsum

    # Starts from an implicit (0, 1) point
    precision = np.concatenate(([1.], precision))
    recall = np.concatenate(([0.], recall))

    # Calculate AUC using step-function integral (recommended instead of trapezoidal rule for PR curves in Scikit-learn.
    # See https://github.com/scikit-learn/scikit-learn/blob/8721245511de2f225ff5f9aa5f5fadce663cd4a3/sklearn/metrics/_ranking.py#L236C9-L236C67
    auc = np.sum(np.diff(recall) * precision[1:])  # precision[0] is always 1, making the integration formula correct

    return auc, precision, recall


def get_tp_fp_counts(y_true, y_score, tp_counts=None, fp_counts=None, *, score_levels=128):
    '''
    Gets the true and false positive counts from ground-truth labels and scores.

    Args:
        y_true (np.ndarray): Ground-truth labels, False for negatives and True for positives.
        y_score (np.ndarray): Predicted scores, each in the interval [0, score_levels-1]
        tp_counts (np.ndarray): Initial True positive counts at each score level. Default is None, which creates new
            array initialized to zeros.
        fp_counts (np.ndarray): Same, for False positive counts.
        score_levels (int): Number of quantized score levels to use. Default is 128.

    Returns:
        np.ndarray: True positive counts at each score level. Same as input tp_counts if not None.
        np.ndarray: False positive counts at each score level. Same as input fp_counts if not None.
    '''
    if y_true.shape != y_score.shape:
        raise ValueError('y_true and y_score must have the same shape.')
    y_true = y_true.ravel()
    y_score = y_score.ravel()
    # Sorts data by score
    sorted_indices = np.argsort(y_score)
    y_true = y_true[sorted_indices]
    y_score = y_score[sorted_indices]
    # Computes true and false positive counts
    tp_counts = np.zeros(score_levels) if tp_counts is None else tp_counts
    fp_counts = np.zeros(score_levels) if fp_counts is None else fp_counts
    np.add.at(tp_counts, y_score, y_true)
    np.add.at(fp_counts, y_score, ~y_true)
    return tp_counts, fp_counts


def fast_cm(y_true, y_pred, n):
    '''
    Fast computation of a confusion matrix from two arrays of labels.

    Args:
        y_true  (np.ndarray): array of true labels
        y_pred (np.ndarray): array of predicted labels
        n (int): number of classes

    Returns:
        np.ndarray: confusion matrix, where rows are true labels and columns are predicted labels
    '''
    y_true = y_true.ravel().astype(int)
    y_pred = y_pred.ravel().astype(int)
    k = (y_true < 0) | (y_true > n) | (y_pred < 0) | (y_pred > n)
    if np.any(k):
        raise ValueError('Invalid class values in ground-truth or prediction: '
                         f'{np.unique(np.concatenate((y_true[k], y_pred[k])))}')
    # Convert class numbers into indices of a simulated 2D array of shape (n, n) flattened into 1D, row-major
    effective_indices = n * y_true + y_pred
    # Count the occurrences of each index, reshaping the 1D array into a 2D array
    return np.bincount(effective_indices, minlength=n ** 2).reshape(n, n)


def per_class_iou(cm):
    ''''
    Compute the Intersection over Union (IoU) for each class from a confusion matrix.

    Args:
        cm (np.ndarray): n x n 2D confusion matrix (the orientation is not important, as the formula is symmetric)

    Returns:
        np.ndarray: 1D array of IoU values for each of the n classes
    '''
    # The diagonal contains the intersection of predicted and true labels
    # The sum of rows (columns) is the union of predicted (true) labels (or vice-versa, depending on the orientation)
    return np.diag(cm) / (cm.sum(1) + cm.sum(0) - np.diag(cm))
