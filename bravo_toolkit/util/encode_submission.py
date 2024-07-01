# Copyright (c) 2024 Valeo
# See LICENSE.md for details.

import argparse
from contextlib import closing
import glob
import io
import logging
import os
import sys
import tarfile
import time

import numpy as np
from tqdm import tqdm

from bravo_toolkit.codec.bravo_codec import bravo_encode
from bravo_toolkit.codec.bravo_tarfile import (SAMPLES_SUFFIX, SPLIT_PREFIX, SPLIT_TO_CONF_SUFFIX, SPLIT_TO_PRED_SUFFIX,
                                               SUBMISSION_SUFFIX, tar_extract_file, tar_extract_grayscale,
                                               tar_extract_image)
from bravo_toolkit.util.sample_gt_pixels import decode_indices


logger = logging.getLogger('bravo_toolkit')


def tqqdm(iterable, *args, **kwargs):
    if logger.getEffectiveLevel() <= logging.INFO:
        return tqdm(iterable, *args, **kwargs)
    return iterable


class DirectoryAsTarMember:
    def __init__(self, name, parent):
        self.name = name
        self.parent = parent

    def isfile(self):
        return os.path.isfile(os.path.join(self.parent.root, self.name))


class DirectoryAsTar:
    def __init__(self, root):
        self.root = os.path.abspath(root)

    def close(self):
        pass

    def getmembers(self):
        members = []
        for file_path in glob.iglob(self.root + '/**/*', recursive=True):
            inside_path = os.path.relpath(file_path, self.root)
            members.append(DirectoryAsTarMember(inside_path, self))
        return members

    def getmember(self, name):
        if os.path.exists(os.path.join(self.root, name)):
            return DirectoryAsTarMember(name, self)

    def extractfile(self, member):
        if member.parent is not self:
            raise ValueError("The member does not belong to this instance.")
        file_path = os.path.join(self.root, member.name)
        if not os.path.isfile(file_path):
            raise ValueError(f"The member {member.name} is not a file.")
        return open(file_path, 'rb')


def process_tar_files(input_path, output_tar_path, *, samples_tar_path):
    """
    Converts all pairs of prediction and confidence images in input tar file (old submission format) into a single
    encoded binary file in output tar file (new submission format) with `bravo_codec.bravo_encode`.
    """
    # current_uid = os.getuid()
    # current_gid = os.getgid()

    def open_input_path():
        if os.path.isdir(input_path):
            return DirectoryAsTar(input_path)
        return tarfile.open(input_path, "r")

    logger.info("Opening tar files...")
    with closing(open_input_path()) as input_tar, \
         closing(tarfile.open(output_tar_path, "w")) as output_tar, \
         closing(tarfile.open(samples_tar_path, "r")) as samples_tar:

        logger.info("Listing input files...")
        input_tar_members = input_tar.getmembers()

        logger.info("Reencoding submission...")
        for member in tqqdm(input_tar_members):
            # Skip non-files (directories, etc.)
            if not member.isfile():
                continue

            # Determine the split and base path for prediction files
            base_split = ""
            base_path = ""
            for split, pred_suffix in SPLIT_TO_PRED_SUFFIX.items():
                if member.name.startswith(SPLIT_PREFIX.format(split=split)):
                    base_split = split
                    if member.name.endswith(pred_suffix):
                        base_path = member.name[:-len(pred_suffix)]
                        break
            if not base_path:
                if not base_split or not member.name.endswith(SPLIT_TO_CONF_SUFFIX[base_split]):
                    logger.warning("Unexpected file: `%s`", member.name)
                continue

            # Determine the file base and corresponding CONF file
            conf_filename = base_path + SPLIT_TO_CONF_SUFFIX[base_split]
            conf_member = input_tar.getmember(conf_filename)

            # Extract PRED and CONF images
            pred_image = tar_extract_grayscale(input_tar, member, "prediction")
            conf_image = tar_extract_image(input_tar, conf_member, "confidence")
            if conf_image.dtype != np.uint16:
                logger.error("Confidence image is not uint16: `%s`", conf_filename)
                sys.exit(1)
            if pred_image.shape != conf_image.shape:
                logger.error("Prediction and confidence images have different shapes: `%s`, `%s`",
                              member.name, conf_filename)
                sys.exit(1)

            # Extract samples index
            confidence_indices_bytes = tar_extract_file(samples_tar, base_path + SAMPLES_SUFFIX)
            confidence_indices = decode_indices(confidence_indices_bytes)

            # Encode the images
            encoded_data = bravo_encode(pred_image, conf_image, confidence_indices=confidence_indices)

            # Add to output tarfile
            encoded_filename = base_path + SUBMISSION_SUFFIX
            tarinfo = tarfile.TarInfo(name=encoded_filename)
            tarinfo.size = len(encoded_data)
            # tarinfo.uid = current_uid
            # tarinfo.gid = current_gid
            tarinfo.mtime = time.time()
            output_tar.addfile(tarinfo, io.BytesIO(encoded_data))

    logger.info("Done!")


def main():
    parser = argparse.ArgumentParser(description="Process TAR files for encoding.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input_path", help="Path to the input directory or TAR file.")
    parser.add_argument("output_tar_path", help="Path to the output TAR file.")
    parser.add_argument("--samples", help="Path to the pixel samples TAR file.")
    parser.add_argument('--debug', action='store_true', help='enables extra verbose debug output')
    parser.add_argument('--quiet', action='store_true', help='prints only errors and warnings')
    args = parser.parse_args()

    level = logging.WARNING if args.quiet else (logging.DEBUG if args.debug else logging.INFO)
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')

    try:
        process_tar_files(args.input_path, args.output_tar_path, samples_tar_path=args.samples)
    except Exception as e:
        if os.path.exists(args.output_tar_path):
            os.remove(args.output_tar_path)
        raise e


if __name__ == "__main__":
    main()
