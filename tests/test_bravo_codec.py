import json
import os

import numpy as np
import pytest

from bravo_toolkit.codec.bravo_codec import bravo_decode, bravo_encode
from bravo_toolkit.codec.bravo_tarfile import extract_grayscale, extract_image
from bravo_toolkit.util.sample_gt_pixels import SAMPLES_PER_IMG, sample_gt_pixels


# --------- Utilities ---------

def bravo_simulation_test(*, seed=42, array_shape=(1000, 2000), n_classes=19, n_regions=50, n_indices=SAMPLES_PER_IMG,
                          void_chance=0.2, void_class=255, input_images=None):
    np.random.seed(seed)

    if input_images is None:
        # Creates a random but "realistic" class array with a Voronoi tessellation
        n_rows, n_cols = array_shape
        seeds = np.column_stack([
            np.random.randint(0, n_cols, n_regions),
            np.random.randint(0, n_rows, n_regions)
        ])
        classes = np.random.randint(0, n_classes, n_regions)
        classes = np.where(np.random.rand(n_regions) < void_chance, void_class, classes)
        rows = np.arange(n_rows)
        cols = np.arange(n_cols)
        # ...computes the distances of coordinates to each seed and finds the closest one
        row_distances = (rows[:, None] - seeds[:, 0])**2
        col_distances = (cols[:, None] - seeds[:, 1])**2
        distances = row_distances[:, None, :] + col_distances[None, :, :]  # Squared Euclidean distance
        voronoi = np.argmin(distances, axis=2)
        # ...assigns the class to each region
        class_array = classes[voronoi].astype(np.uint8)

        # Generate a somewhat "realistic" confidence array: random but smooth
        confidences = np.random.rand(n_regions) * (1. - 1./n_classes) + 1./n_classes
        confidence_array = confidences[voronoi]
        confidence_array += np.random.normal(0, 0.02, size=confidence_array.shape)
        confidence_array = np.clip(confidence_array, 1./n_classes, 1.)
        confidence_array = confidence_array.astype(np.float32)
        confidence_indices = sample_gt_pixels(confidence_array, n_indices, seed=seed)
    else:
        class_array, confidence_array, confidence_indices = input_images
        confidence_array = confidence_array.astype(np.float32)
        confidence_array = np.clip(confidence_array, 1./n_classes, 1.)

    confidence_array = np.floor(confidence_array * 65536).astype(np.uint16)

    # Encode the arrays
    encoded_bytes = bravo_encode(class_array, confidence_array, confidence_indices=confidence_indices)
    confidence_slice = slice(None) if confidence_indices is None else confidence_indices
    confidence_sample = confidence_array.ravel()[confidence_slice]

    # The computations below have to be reverified if the data types change
    assert confidence_array.dtype == np.uint16 and confidence_sample.dtype == np.uint16
    original_size = class_array.nbytes + confidence_array.nbytes
    raw_size = class_array.nbytes + confidence_array.nbytes
    sampled_size = class_array.nbytes + confidence_sample.nbytes
    encoded_size = len(encoded_bytes)

    results = {
        "original size": original_size,
        "raw size": raw_size,
        "sampled size": sampled_size,
        "encoded size": encoded_size,
        "original/encoded ratio": original_size / encoded_size,
        "raw/encoded ratio": raw_size / encoded_size,
        "sampled/encoded ratio": sampled_size / encoded_size,
    }

    # Decode the arrays
    decoded_class_array, decoded_confidence_array, _ = bravo_decode(encoded_bytes)

    # Verify that the decoded class array matches the original
    assert np.all(decoded_class_array == class_array), "Class arrays do not match"

    confidence_sample = confidence_sample / 65536.0
    decoded_confidence_array = decoded_confidence_array / 65536.0

    # Verify that the decoded confidence array is close to the original within the quantization tolerance
    tolerance = 1 / 65536.0
    results["tolerance"] = tolerance
    results["max_abs_diff"] = np.max(np.abs(decoded_confidence_array - confidence_sample))
    results["match_within_tolerance"] = np.allclose(decoded_confidence_array, confidence_sample, atol=tolerance)

    assert results["match_within_tolerance"], f"codec failed within tolerance of {results['tolerance']}: " \
                                              f"max. diff: {results['max_abs_diff']}"
    return results

# --------- Utilities ---------


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(SCRIPT_DIR, "bravo_test_images")
IMAGES_METADATA_FILE = os.path.join(SCRIPT_DIR, 'bravo_test_images.json')
with open(IMAGES_METADATA_FILE, 'rt', encoding='utf-8') as imff:
    IMAGES_METADATA = json.load(imff)
    IMAGES_LIST = list(IMAGES_METADATA.keys())


def get_real_data(pred_file):
    '''Gets the ground-truth labels and scores from a real data case.'''
    conf_file = pred_file.replace('_pred.png', '_conf.png')

    # Loads ground-truth and confidence images
    with open(os.path.join(IMAGES_DIR, pred_file), 'rb') as f:
        pred = extract_grayscale(f, 'class prediction')
    with open(os.path.join(IMAGES_DIR, conf_file), 'rb') as f:
        conf = extract_image(f, 'confidence')

    # Compare toolkit auc with reference auc
    return pred, conf


# --------- BRAVO_CODEC ---------

def test_bravo_codec_default_test():
    results = bravo_simulation_test()
    assert results["original/encoded ratio"] > 10
    assert results["sampled/encoded ratio"] > 5


bravo_codec_test_cases = [
    # seed, array_shape, n_classes, n_regions, sample_size,
    # Default test case
    (42, (1000, 2000), 19, 50, 100_000),
    # Different seeds and sizes
    (43, (256, 128),   19, 50, 100_000),
    (44, (1001, 64),   19, 50, 100_000),
    (45, (317, 2030),  19, 50, 100_000),
    (46, (31, 510),    19, 50, 100_000),
    # Different number of classes, regions, and sample sizes
    (47, (1000, 2000),  13, 50, 100_000),
    (48, (1000, 2000), 167, 50, 100_000),
    (49, (1000, 2000),  19, 17, 100_000),
    (50, (1000, 2000),  19, 99, 100_000),
    (51, (1000, 2000),  19, 99, 10_000),
    (52, (1000, 2000),  19, 99, 1000_000),
    # Extreme cases
    (53, (1, 1),       19,  50, 100_000),
    (54, (1, 1024),    19,  50, 100_000),
    (55, (1024, 1),    19,  50, 100_000),
    (56, (2, 2),       19,  50, 100_000),
    (57, (1024, 1024),  2,  50, 100_000),
    (58, (1024, 1024), 19,   1, 100_000),
    (59, (1024, 1024), 19, 500, 100_000),
]


@pytest.mark.parametrize('seed, array_shape, n_classes, n_regions, sample_size', bravo_codec_test_cases)
def test_bravo_codec_test_cases(seed, array_shape, n_classes, n_regions, sample_size):
    results = bravo_simulation_test(seed=seed, array_shape=array_shape, n_classes=n_classes, n_regions=n_regions,
                                    n_indices=sample_size)
    if array_shape[0] * array_shape[1] > 100:
        assert results["original/encoded ratio"] > array_shape[0] * array_shape[1] / sample_size / 2
        assert results["sampled/encoded ratio"] > 1


def test_bravo_codec_large_array():
    results = bravo_simulation_test(array_shape=(4096, 4097), n_classes=19, n_regions=50)
    assert results["original/encoded ratio"] > 10
    assert results["sampled/encoded ratio"] > 5


@pytest.mark.parametrize('pred_file', IMAGES_LIST)
def test_bravo_codec_true_images(pred_file):
    pred_image, conf_image = get_real_data(pred_file)
    confidence_indices = sample_gt_pixels(conf_image, SAMPLES_PER_IMG, seed=1)
    results = bravo_simulation_test(input_images=(pred_image, conf_image, confidence_indices))
    assert results["original/encoded ratio"] > 10
    assert results["sampled/encoded ratio"] > 5
