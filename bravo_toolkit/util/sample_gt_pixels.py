# Copyright (c) 2024 Valeo
# See LICENSE.md for details.

import argparse
from contextlib import closing
import io
import logging
import os
import sys
import tarfile
import time

import numpy as np
from tqdm import tqdm

from bravo_toolkit.codec.bravo_codec import _compress, _decompress
from bravo_toolkit.codec.bravo_tarfile import (SAMPLES_SUFFIX, SPLIT_PREFIX, SPLIT_TO_GT_SUFFIX, SPLIT_TO_MASK_SUFFIX,
                                               tar_extract_file, tar_extract_grayscale)


SAMPLES_PER_IMG = 100_000


logger = logging.getLogger('bravo_toolkit')


def tqqdm(iterable, *args, **kwargs):
    if logger.getEffectiveLevel() <= logging.INFO:
        return tqdm(iterable, *args, **kwargs)
    return iterable


def sample_gt_pixels(gt_file, samples_per_image, seed):
    '''
    Sample the ground-truth pixels from a single image. The samples are taken after filtering out void pixels.

    Args:
        gt_file: 2D numpy array of ground-truth pixel values
        samples_per_image: number of samples to take
        seed: random seed for sampling

    Returns:
        sampled_indices: unsorted 1D numpy array of sampled indices
    '''
    gt_file = gt_file.ravel()
    non_void = np.nonzero(gt_file != 255)[0]
    if non_void.size > samples_per_image:
        np.random.seed(seed)
        non_void = np.random.choice(non_void, samples_per_image, replace=False)
        non_void = np.sort(non_void)
    return non_void


def encode_indices(indices):
    '''
    Encode the sampled 1D indices into a byte array.
    - The indices are assumed to be sorted
    - The first index will be stored as as a 16-bit unsigned integer (little-endian)
    - The differences between the indices will be stored as 8-bit unsigned integers
    - If a difference is greater than 255, it will be stored as a zero followed by a 16-bit unsigned integer
    '''
    # Encode the differences
    differences = np.diff(indices)
    # logger.debug('encode_indices - differences: first=%d, min=%d, max=%d',
    #              indices[0], differences.min(), differences.max())
    encoded = bytearray()
    encoded.extend(int(indices.size).to_bytes(3, byteorder='little'))
    encoded.extend(int(indices[0]).to_bytes(3, byteorder='little'))
    for diff in differences:
        if diff == 0:
            raise ValueError('repeated values in input array `indices` are not allowed')
        elif diff <= 255:
            encoded.append(diff)
        else:
            # 3-byte encoding for differences greater than 255 (0, low byte, high byte)
            encoded.append(0)
            encoded.extend(int(diff).to_bytes(3, byteorder='little'))
    encoded = _compress(encoded)
    return encoded


def decode_indices(encoded):
    '''
    Decode a byte array into the sampled 1D indices, using the encoding scheme in `_encode_indices`.
    The length of the indices is assumed to be known a priori.
    '''
    encoded = _decompress(encoded)
    length = int.from_bytes(encoded[:3], byteorder='little')
    indices = np.empty(length, dtype=np.int32)
    indices[0] = index = int.from_bytes(encoded[3:6], byteorder='little')
    encoded_len = len(encoded)
    i = 1
    e = 6
    while e < encoded_len:
        diff = encoded[e]
        if diff == 0:
            diff = int.from_bytes(encoded[e+1:e+4], byteorder='little')
            e += 4
        else:
            e += 1
        index += diff
        indices[i] = index
        i += 1
    if i < length:
        raise ValueError('decode_indices - unexpected end of encoded data '
                            f'(encoded_len={encoded_len}, i={i}, length={length})')
    return indices


def sample_all_gt_pixels(gt_tar_path, sample_tar_path, samples_per_image, seed, check=False):
    # Acquire data from the tar files...
    with closing(tarfile.open(gt_tar_path, "r")) as gt_tar:
        logger.info("Listing input files...")
        gt_tar_members = gt_tar.getmembers()

        logger.info('sample_gt_pixels - reading ground truth files...')
        gt_files = []
        for member in tqqdm(gt_tar_members):
            # Skip non-files (directories, etc.)
            if not member.isfile():
                continue

            # Determine the split and base path for prediction files
            base_split = ""
            base_path = ""
            for split, gt_suffix in SPLIT_TO_GT_SUFFIX.items():
                if member.name.startswith(SPLIT_PREFIX.format(split=split)):
                    base_split = split
                    if member.name.endswith(gt_suffix):
                        base_path = member.name[:-len(gt_suffix)]
                        break
            if not base_path:
                if not base_split or not member.name.endswith(SPLIT_TO_MASK_SUFFIX[base_split]):
                    logger.error("Unexpected file: `%s`", member.name)
                    sys.exit(1)
                continue

            # Determine the file base and corresponding ground-truth file
            gt_name = base_path + SPLIT_TO_GT_SUFFIX[base_split]
            gt_member = gt_tar.getmember(gt_name)
            gt_file = tar_extract_grayscale(gt_tar, gt_member, 'ground-truth')
            gt_files.append((gt_name, gt_file, base_path))

    if check:
        logger.info('sample_gt_pixels - checking %s pixels with seed %d...', f'{samples_per_image:_}', seed)
    else:
        logger.info('sample_gt_pixels - sampling %s pixels with seed %d...', f'{samples_per_image:_}', seed)
    all_nan = all_filled = 0
    with tarfile.open(sample_tar_path, "r" if check else "w") as sample_tar:
        for gt_name, gt_file, base_path in tqqdm(gt_files):
            assert base_path.startswith('bravo_')
            # Sample and compress the indexes
            gt_samples = sample_gt_pixels(gt_file, samples_per_image, seed)
            # gt_samples = gt_samples.astype(np.int32).tobytes()
            gt_encoded = encode_indices(gt_samples)

            encoded_filename = base_path + SAMPLES_SUFFIX
            if check:
                # Read and decode the samples from the tar file
                sample_member = sample_tar.getmember(encoded_filename)
                sample_data = tar_extract_file(sample_tar, sample_member)
                if gt_file.size > samples_per_image:
                    assert gt_encoded == sample_data
                    sample_decoded = decode_indices(sample_data)
                    assert np.array_equal(gt_samples, sample_decoded)
                    all_filled += 1
                else:
                    assert np.isnan(gt_encoded)
                    assert np.isnan(sample_data)
                    all_nan += 1
            else:
                # Write the samples to the tar file
                tarinfo = tarfile.TarInfo(name=encoded_filename)
                tarinfo.size = len(gt_encoded)
                # tarinfo.uid = current_uid
                # tarinfo.gid = current_gid
                tarinfo.mtime = time.time()
                sample_tar.addfile(tarinfo, io.BytesIO(gt_encoded))
    if check:
        logger.info('sample_gt_pixels - %d small images, %d large images, %d total images - all matched',
                    all_nan, all_filled, all_nan + all_filled)


def main():
    parser = argparse.ArgumentParser(
         description='Evaluates submissions for the ELSA BRAVO Challenge.',
         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('submission', default=default_submission, help='path to submission tar file')
    # parser.add_argument('--gt', default=default_gt, help='path to ground-truth tar file')
    parser.add_argument('--gt', help='path to ground-truth tar file')
    parser.add_argument('--results',  help='tar file to store the samples')
    parser.add_argument('--check',  help='checks the samples in the tar file')
    parser.add_argument('--samples_per_image', type=int, default=SAMPLES_PER_IMG, help='number of samples per image')
    parser.add_argument('--seed', type=int, default=1, help='seed for the random sampling')
    parser.add_argument('--debug', action='store_true', help='enables extra verbose debug output')
    parser.add_argument('--quiet', action='store_true', help='prints only errors and warnings')
    args = parser.parse_args()

    level = logging.WARNING if args.quiet else (logging.DEBUG if args.debug else logging.INFO)
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')

    if args.gt is None or (args.results is None and args.check is None):
        logger.error('--gt and --results are required unless --test is specified')
        sys.exit(1)
    if args.results is not None and args.check is not None:
        logger.error('--results and --check are mutually exclusive')
        sys.exit(1)
    if args.results is not None:
        try:
            sample_all_gt_pixels(args.gt, args.results, args.samples_per_image, args.seed)
        except Exception as e:
            if os.path.exists(args.results):
                os.remove(args.results)
            raise e
    elif args.check is not None:
        sample_all_gt_pixels(args.gt, args.check, args.samples_per_image, args.seed, check=True)
    else:
        assert False, 'unreachable code'

    logger.info('done.')


if __name__ == '__main__':
    main()
