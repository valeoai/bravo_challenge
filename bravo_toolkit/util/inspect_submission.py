# Copyright (c) 2024 Valeo
# See LICENSE.md for details.

import argparse
import logging
import sys
import tarfile

import numpy as np
from tqdm import tqdm

from bravo_toolkit.codec.bravo_codec import bravo_encode
from bravo_toolkit.codec.bravo_tarfile import (SPLIT_PREFIX, SPLIT_TO_CONF_SUFFIX, SPLIT_TO_PRED_SUFFIX,
                                               tar_extract_grayscale, tar_extract_image)


def process_tar_files(input_tar_path, rounded_quantization=False):
    '''
    Converts all pairs of prediction and confidence images in input tar file (old submission format) into a single
    encoded binary file in output tar file (new submission format) with `bravo_codec.bravo_encode`.
    '''
    # current_uid = os.getuid()
    # current_gid = os.getgid()

    print("Opening tar files...")
    all_shapes = []
    with tarfile.open(input_tar_path, 'r') as input_tar:

        print("Listing input files...")
        input_tar_members = input_tar.getmembers()

        print("Inspecting submission...")
        for member in tqdm(input_tar_members):
            # Skip non-files (directories, etc.)
            if not member.isfile():
                continue

            # Determine the split and base path for prediction files
            base_split = ''
            base_path = ''
            for split, pred_suffix in SPLIT_TO_PRED_SUFFIX.items():
                if member.name.startswith(SPLIT_PREFIX.format(split=split)):
                    base_split = split
                    if member.name.endswith(pred_suffix):
                        base_path = member.name[:-len(pred_suffix)]
                        break
            if not base_path:
                if not base_split or not member.name.endswith(SPLIT_TO_CONF_SUFFIX[base_split]):
                    logging.warning('Unexpected file: `%s`', member.name)
                continue

            # Determine the file base and corresponding CONF file
            conf_filename = base_path + SPLIT_TO_CONF_SUFFIX[base_split]
            conf_member = input_tar.getmember(conf_filename)

            # Extract PRED and CONF images
            pred_image = tar_extract_grayscale(input_tar, member, 'prediction')
            conf_image = tar_extract_image(input_tar, conf_member, 'confidence')
            if conf_image.dtype != np.uint16:
                logging.error('Confidence image is not uint16: `%s`', conf_filename)
                sys.exit(1)
            conf_image = conf_image.astype(np.float32) / 65536  # This preserves the mantissa and shifts the exponent

            if pred_image.shape != conf_image.shape:
                logging.error('Prediction and confidence images have different shapes: `%s` and `%s`',
                              member.name, conf_filename)
                sys.exit(1)

            all_shapes.append(pred_image.shape)

        np.set_printoptions(suppress=True, precision=1)
        all_shapes = np.array(all_shapes)
        print('Image shape statistics:')
        print('Avg: ', np.mean(all_shapes, axis=0))
        print('Max: ', np.max(all_shapes, axis=0))
        print('75p: ', np.percentile(all_shapes, 75, axis=0))
        print('Med: ', np.median(all_shapes, axis=0))
        print('25p: ', np.percentile(all_shapes, 25, axis=0))
        print('Min: ', np.min(all_shapes, axis=0))


    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Process TAR files for encoding.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input_tar_path", help="Path to the input TAR file.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    process_tar_files(args.input_tar_path)


if __name__ == "__main__":
    main()
