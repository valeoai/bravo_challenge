# Copyright (c) 2024 Valeo
# See LICENSE.md for details.

import os
import json
import tarfile


# Configuration variables
script_dir = os.path.dirname(__file__)
submission_dir = os.path.join(script_dir, 'bravo_test_images')
gt_dir = os.path.join(script_dir, 'bravo_test_images')
metadata_file = os.path.join(script_dir, 'bravo_test_images.json')

submission_tarfile = os.path.expanduser('~/shared/thvu/BRAVO/challenge/toolkit/submissions_dgssclip/bravo_submission.tar')
gt_tarfile = os.path.expanduser('~/shared/thvu/BRAVO/challenge/toolkit/bravo_GT.tar')

# Suffix and prefix mappings
SPLIT_TO_GT_SUFFIX = {
    'ACDC': '_gt_labelTrainIds.png',
    'SMIYC': '_labels_semantic_fake.png',
    'outofcontext': '_gt_labelTrainIds.png',
    'synobjs': '_gt.png',
    'synflare': '_gt_labelTrainIds.png',
    'synrain': '_gt_labelTrainIds.png',
}
SPLIT_TO_MASK_SUFFIX = {
    'ACDC': '_gt_invIds.png',
    'SMIYC': '_labels_semantic.png',
    'outofcontext': '_gt_invIds.png',
    'synobjs': '_mask.png',
    'synflare': '_gt_invIds.png',
    'synrain': '_gt_invIds.png',
}
SPLIT_TO_PRED_SUFFIX = {
    'ACDC': '_rgb_anon_pred.png',
    'SMIYC': '_pred.png',
    'outofcontext': '_leftImg8bit_pred.png',
    'synobjs': '_pred.png',
    'synflare': '_leftImg8bit_pred.png',
    'synrain': '_leftImg8bit_pred.png',
}
SPLIT_PREFIX = 'bravo_{split}/'


BRAVO_CODEC_TEST_IMAGES = [
    'bravo_ACDC/snow/test/GP010122/GP010122_frame_000085_rgb_anon_pred.png',
    'bravo_ACDC/fog/test/GOPR0478/GOPR0478_frame_000451_rgb_anon_pred.png',
    'bravo_ACDC/night/test/GOPR0594/GOPR0594_frame_000715_rgb_anon_pred.png',
    'bravo_ACDC/rain/test/GOPR0572/GOPR0572_frame_000692_rgb_anon_pred.png',
    'bravo_SMIYC/RoadAnomaly21/images/airplane0001_pred.png',
    'bravo_SMIYC/RoadAnomaly21/images/boat_trailer0004_pred.png',
    'bravo_SMIYC/RoadAnomaly21/images/carriage0001_pred.png',
    'bravo_SMIYC/RoadAnomaly21/images/cow0013_pred.png',
    'bravo_SMIYC/RoadAnomaly21/images/tent0000_pred.png',
    'bravo_SMIYC/RoadAnomaly21/images/validation0007_pred.png',
    'bravo_SMIYC/RoadAnomaly21/images/zebra0000_pred.png',
    'bravo_outofcontext/munster/munster_000026_000019_leftImg8bit_pred.png',
    'bravo_outofcontext/frankfurt/frankfurt_000001_048654_leftImg8bit_pred.png',
    'bravo_outofcontext/lindau/lindau_000024_000019_leftImg8bit_pred.png',
    'bravo_synflare/munster/munster_000111_000019_leftImg8bit_pred.png',
    'bravo_synflare/frankfurt/frankfurt_000001_014565_leftImg8bit_pred.png',
    'bravo_synflare/lindau/lindau_000015_000019_leftImg8bit_pred.png',
    'bravo_synobjs/elephant/127_pred.png',
    'bravo_synobjs/toilet/487_pred.png',
    'bravo_synobjs/tiger/437_pred.png',
    'bravo_synobjs/flamingo/9_pred.png',
    'bravo_synobjs/sofa/299_pred.png',
    'bravo_synobjs/billboard/212_pred.png',
    'bravo_synrain/munster/munster_000018_000019_leftImg8bit_pred.png',
    'bravo_synrain/frankfurt/frankfurt_000001_046272_leftImg8bit_pred.png',
    'bravo_synrain/lindau/lindau_000023_000019_leftImg8bit_pred.png',
]


def extract_file(tar, path_in_tar, target_path):
    """ Helper function to extract a file from a tar file """
    member = tar.getmember(path_in_tar)
    f = tar.extractfile(member)
    contents = f.read()
    f.close()
    with open(target_path, 'wb') as f:
        f.write(contents)
    print(f"    {member.name} -> {target_path}")


print("Extracting files...")
metadata = {}
with tarfile.open(submission_tarfile, 'r') as sub_tar, tarfile.open(gt_tarfile, 'r') as gt_tar:
    for pred_path in BRAVO_CODEC_TEST_IMAGES:
        print(f"Processing {pred_path}...")
        for s in SPLIT_TO_GT_SUFFIX:
            if pred_path.startswith(SPLIT_PREFIX.format(split=s)):
                split = s
                break
        else:
            raise ValueError(f"Could not determine split for file {pred_path}")

        conf_path = pred_path.replace('_pred.png', '_conf.png')

        # Extract prediction and confidence files
        pred_target_file = (pred_path[6:-len(SPLIT_TO_PRED_SUFFIX[split])] + '_pred.png').replace('/', '_')
        conf_target_file = pred_target_file.replace('_pred.png', '_conf.png')
        extract_file(sub_tar, pred_path, os.path.join(submission_dir, pred_target_file))
        extract_file(sub_tar, conf_path, os.path.join(submission_dir, conf_target_file))

        # Extracts ground-truth files
        gt_file = pred_path[:-len(SPLIT_TO_PRED_SUFFIX[split])] + SPLIT_TO_GT_SUFFIX[split]
        mask_file = pred_path[:-len(SPLIT_TO_PRED_SUFFIX[split])] + SPLIT_TO_MASK_SUFFIX[split]
        gt_target_file = pred_target_file.replace('_pred.png', '_gt.png')
        mask_target_file = pred_target_file.replace('_pred.png', '_mask.png')
        extract_file(gt_tar, gt_file, os.path.join(gt_dir, gt_target_file))
        extract_file(gt_tar, mask_file, os.path.join(gt_dir, mask_target_file))

        metadata[pred_target_file] = split

print("Writing metadata file...")
with open(metadata_file, 'wt', encoding='utf-8') as meta:
    json.dump(metadata, meta, indent=4)

print("Done.")
