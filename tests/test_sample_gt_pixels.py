# Copyright (c) 2024 Valeo
# See LICENSE.md for details.

from itertools import chain, product

import numpy as np
from numpy.testing import assert_equal
import pytest

from bravo_toolkit.util.sample_gt_pixels import decode_indices, encode_indices, sample_gt_pixels


@pytest.mark.parametrize('samples_per_image, seed1, seed2',
                         chain(product([50_000, 100_000, 500_000], [1, 2, 3], [1, 2, 3])))
def test_sample_gt_pixels_cases(samples_per_image, seed1, seed2):
    np.random.seed(seed1)
    gt_file = np.random.randint(0, 254, 2000_000).astype(np.uint8)
    gt_file[np.random.choice(2000_000, np.random.randint(50_000, 200_000), replace=False)] = 255
    sampled_indices = sample_gt_pixels(gt_file, samples_per_image, seed2)
    assert sampled_indices.size == samples_per_image
    encoded = encode_indices(sampled_indices)
    decoded = decode_indices(encoded)
    assert_equal(sampled_indices, decoded)


@pytest.mark.skip(reason="with the encoding with 3 bytes instead of 2 the encoding does not overflow anymore")
def test_sample_gt_pixels_overflow():
    np.random.seed(1)
    gt_file = np.random.randint(0, 256, 10_000_000).astype(np.uint8)
    samples_per_image = 100
    sampled_indices = sample_gt_pixels(gt_file, samples_per_image, seed=1)
    with pytest.raises( OverflowError):
        _ = encode_indices(sampled_indices)


@pytest.mark.parametrize('fraction, seed1, seed2',
                         chain(product([0.55, 0.75], [1, 2, 3], [1, 2, 3])))
def test_sample_gt_pixels_small(fraction, seed1, seed2):
    np.random.seed(seed1)
    gt_file = np.random.randint(0, 2, 100*100).astype(np.uint8) * 255
    samples_per_image = int(100*100*fraction)
    sampled_indices = sample_gt_pixels(gt_file, samples_per_image, seed2)
    encoded = encode_indices(sampled_indices)
    decoded = decode_indices(encoded)
    assert_equal(sampled_indices, decoded)
