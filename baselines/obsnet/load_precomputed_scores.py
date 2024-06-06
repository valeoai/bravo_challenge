import os
import h5py
import numpy as np
import cv2
from tqdm import tqdm
# path to precomputed scores
precomputed_score_path = '/home/thvu/shared/thvu/cache/obsnet/AnomalyTrack-all'
 
# go through each hdf5 file in the path
files = []
for (dirpath, dirnames, filenames) in os.walk(precomputed_score_path):
    for filename in filenames:
        if filename.endswith('.hdf5'):
            files.append(dirpath + '/' + filename)

# read each hdf5 file and store the scores in a dictionary
scores = {}
destfile_conf_path = '/home/thvu/shared/thvu/BRAVO/challenge/toolkit/submissions_obsnet_512x1024_precomputed_scaled/bravo_SMIYC/RoadAnomaly21/images'
# make dir destfile_conf_path
if not os.path.exists(destfile_conf_path):
    os.makedirs(destfile_conf_path)
oriimg_path = '/home/thvu/shared/thvu/BRAVO/challenge/final_release_08_2023/bravo_SMIYC/RoadAnomaly21/images'
for file in tqdm(files):
    with h5py.File(file, 'r') as f:
        conf_float = 1-np.array(f['value'])
        conf = (conf_float*65535).astype(np.uint16)
        oriimg_shape = cv2.imread(os.path.join(oriimg_path, os.path.basename(file).replace('.hdf5', '.jpg'))).shape
        # reshape conf to oriimg_shape
        conf = cv2.resize(conf, (oriimg_shape[1], oriimg_shape[0]), interpolation=cv2.INTER_LINEAR)
        destfile_pred = os.path.join(destfile_conf_path, os.path.basename(file).replace('.hdf5', '_pred.png'))
        destfile_conf = os.path.join(destfile_conf_path, os.path.basename(file).replace('.hdf5', '_conf.png'))
        # pred = np.random.randint(0, 19, (oriimg_shape[0], oriimg_shape[1])).astype(np.uint8)
        pred = np.random.randint(0, 19, conf.shape).astype(np.uint8)
        cv2.imwrite(destfile_pred, pred)
        cv2.imwrite(destfile_conf, conf)
        np.save(destfile_conf.replace('.png', '.npy'), conf_float)