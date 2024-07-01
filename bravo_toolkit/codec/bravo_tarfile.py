# Copyright (c) 2024 Valeo
# See LICENSE.md for details.

import logging
from contextlib import closing

import cv2
import numpy as np


logger = logging.getLogger('bravo_toolkit')


# Suffixes for the ground-truth tar files

SPLIT_TO_GT_SUFFIX = {
    'ACDC':         '_gt_labelTrainIds.png',
    'SMIYC':        '_labels_semantic_fake.png',
    'outofcontext': '_gt_labelTrainIds.png',
    'synobjs':      '_gt.png',
    'synflare':     '_gt_labelTrainIds.png',
    'synrain':      '_gt_labelTrainIds.png',
}
SPLIT_TO_MASK_SUFFIX = {
    'ACDC':         '_gt_invIds.png',
    'SMIYC':        '_labels_semantic.png',
    'outofcontext': '_gt_invIds.png',
    'synobjs':      '_mask.png',
    'synflare':     '_gt_invIds.png',
    'synrain':      '_gt_invIds.png',
}

# Suffixes for the new-style submission tar files

SUBMISSION_SUFFIX = '_encoded.bin'
SAMPLES_SUFFIX = '_samples.bin'

# Suffixes for the old-style submission tar files

SPLIT_TO_PRED_SUFFIX = {
    'ACDC':         '_rgb_anon_pred.png',
    'SMIYC':        '_pred.png',
    'outofcontext': '_leftImg8bit_pred.png',
    'synobjs':      '_pred.png',
    'synflare':     '_leftImg8bit_pred.png',
    'synrain':      '_leftImg8bit_pred.png',
}

SPLIT_TO_CONF_SUFFIX = {
    'ACDC':         '_rgb_anon_conf.png',
    'SMIYC':        '_conf.png',
    'outofcontext': '_leftImg8bit_conf.png',
    'synobjs':      '_conf.png',
    'synflare':     '_leftImg8bit_conf.png',
    'synrain':      '_leftImg8bit_conf.png',
}

# Directories inside the tar files

SPLIT_PREFIX = 'bravo_{split}/'


# Helper functions for extracting images from tar files

def tar_extract_grayscale(tar, member, image_type='image', flag=cv2.IMREAD_GRAYSCALE):
    '''Helper function for `tar_extract_image` with default flag=cv2.IMREAD_GRAYSCALE.'''
    return tar_extract_image(tar, member, image_type, flag)


def tar_extract_image(tar, member, image_type='image', flag=cv2.IMREAD_UNCHANGED):
    '''
    Extracts an image from a tar file member.

    Args:
        tar (tarfile): tar file object
        member (tarfile.TarInfo): tar file member
        image_type (str): type of image to extract, for logging purposes, does not affect the extraction
        flag (int): flag for cv2.imdecode, default is cv2.IMREAD_UNCHANGED

    Returns:
        np.ndarray: image data
    '''
    with closing(tar.extractfile(member)) as f:
        img = extract_image(f, image_type=image_type, flag=flag)
    return img


def extract_grayscale(reader, image_type='image', flag=cv2.IMREAD_GRAYSCALE):
    '''Helper function for `extract_image` with default flag=cv2.IMREAD_GRAYSCALE.'''
    return extract_image(reader, image_type, flag)


def extract_image(reader, image_type='image', flag=cv2.IMREAD_UNCHANGED):
    '''
    Extracts an image from a reader object.

    Args:
        reader (io.BufferedReader): reader object
        image_type (str): type of image to extract, for logging purposes, does not affect the extraction
        flag (int): flag for cv2.imdecode, default is cv2.IMREAD_UNCHANGED

    Returns:
        np.ndarray: image data
    '''
    content = reader.read()
    file_bytes = np.asarray(bytearray(content), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, flag)
    if img is None:
        raise ValueError(f'Failed to decode {image_type} image')
    return img


def tar_extract_file(tar, member):
    '''
    Extracts a file from a tar file member.

    Args:
        tar (tarfile): tar file object
        member (tarfile.TarInfo): tar file member

    Returns:
        bytes: file data
    '''
    with closing(tar.extractfile(member)) as f:
        file_data = f.read()
    return file_data
