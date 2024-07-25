import os
import cv2
import glob
import time
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torchvision import transforms
import torchvision.transforms.functional as tf

from Models.segnet import SegNet
from Models.obsnet import Obsnet_Seg as ObsNet
from Models.deeplab_v3plus import deeplab_v3plus
from Models.road_anomaly_networks.deepv3 import DeepWV3Plus, DeepWV3Plus_Obsnet



def img2tensor(file, args):
    try:
        img = Image.open(file)
        imsize = img.size
        img = img.resize((args.w, args.h), Image.BILINEAR)
        img = tf.to_tensor(img)
        if img.shape[0] == 4:
            img = img[:3]
        img = tf.normalize(img, args.mean, args.std, False)
        img = img.unsqueeze(0).to(args.device)
    except:
        breakpoint()
    return img, imsize


def test(args):

    if args.model == "segnet":
        segnet = SegNet(3, args.nclass, init_vgg=False).to(args.device)
        obsnet = ObsNet(input_channels=3, output_channels=1).to(args.device)

    elif args.model == "deeplabv3plus":
        segnet = deeplab_v3plus('resnet101', num_classes=args.nclass, output_stride=16,
                                pretrained_backbone=True).to(args.device)
        obsnet = deeplab_v3plus('resnet101', num_classes=args.nclass, output_stride=16,
                                pretrained_backbone=True, obsnet=True).to(args.device)

    elif args.model == "road_anomaly":
        segnet = DeepWV3Plus(args.nclass).to(args.device)
        obsnet = DeepWV3Plus_Obsnet(num_classes=1).to(args.device)
    else:
        raise NameError("type of model not understood")
    
    # check if args.dest_folder exists, if not make it
    if not os.path.exists(args.dest_folder):
        os.makedirs(args.dest_folder)

    segnet.load_state_dict(torch.load(args.segnet_file))
    segnet.eval()

    obsnet.load_state_dict(torch.load(args.obsnet_file))
    obsnet.eval()

    total_time = 0
    splits = ['ACDC', 'SMIYC', 'outofcontext', 'synflare', 'synobjs', 'synrain']
    modified_suffix_from = ['.png', '.jpg', '.png', '.png', '.png', '.png']
    # find the index of args.split in splits
    split_idx = splits.index(args.split)
    img_suffix = modified_suffix_from[split_idx]
    
    with torch.no_grad():
        for i, file in enumerate(tqdm(args.imgs)):
            img, im_orisize = img2tensor(file, args)
            # Inference
            start = time.time()
            _seg_pred, _obs_pred = _inference(img, segnet, obsnet)

            total_time += time.time() - start
            img = (img - img.min()) / (img.max() - img.min())
            
            _obs_pred = torch.sigmoid(_obs_pred)
            
            # Post processing (removed)! 
            # BRAVO note: remove the trick in obsnet that filters OOD pixels using semantic results
            #
            # _seg_pred_tmp = torch.argmax(_seg_pred, dim=1)
            # obs_pred = torch.zeros_like(_obs_pred)
            # for c in args.stuff_classes:
            #     mask = torch.where(_seg_pred_tmp[0] == c, args.one, args.zero)
            #     obs_pred += mask * _obs_pred
            # obs_pred += 0.01 * _obs_pred
            # obs_pred = torch.clamp(obs_pred, 0, 0.99).cpu().numpy()
            
            _obs_pred = _obs_pred.cpu().numpy()
            
            # upscale prediction to original size
            _obs_pred = cv2.resize(_obs_pred, (im_orisize[1], im_orisize[0]), interpolation=cv2.INTER_LINEAR)
            _seg_pred = torch.nn.functional.interpolate(_seg_pred, size=(im_orisize[1], im_orisize[0]), mode='bilinear', align_corners=False)
            _seg_pred = torch.argmax(_seg_pred, dim=1)
            
            filepath = file[len(args.img_folder):]
            if filepath[0] == '/':
                filepath = filepath[1:]
            destfile_pred = os.path.join(args.dest_folder, filepath).replace(img_suffix, '_pred.png')
            pred = _seg_pred[0].cpu().numpy().astype(np.uint8)
            if not os.path.exists(os.path.dirname(destfile_pred)):
                os.makedirs(os.path.dirname(destfile_pred))
            cv2.imwrite(destfile_pred, pred)
            
            destfile_conf = os.path.join(args.dest_folder, filepath).replace(img_suffix, '_conf.png')
            conf = ((1-_obs_pred) * 65535).astype(np.uint16)
            cv2.imwrite(destfile_conf, conf)
        
            print(f"Test: image: {i}, progression: {i/len(args.imgs)*100:.1f} %, img = {file.split('/')[-1]} ")
    print(f"Inference run time: {len(args.imgs)/total_time:.1f} FPS")


def _inference(img, segnet, obsnet):
    seg_feat = segnet(img, return_feat=True)
    obs_pred = torch.sigmoid(obsnet(img, seg_feat)).squeeze()
    return seg_feat[-1], obs_pred


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",        type=str, default="CityScapes", help="Type of dataset")
    parser.add_argument("--model",       type=str, default="road_anomaly", help="Segnet|deeplabv3plus|road_anomaly")
    parser.add_argument("--split",  type=str, default="SMIYC", help="bravo split")
    parser.add_argument("--segnet_file", type=str, default="./cache/obsnet/WideResnet_DeepLabv3plus_CityScapes.pth", help="path to segnet")
    parser.add_argument("--obsnet_file", type=str, default="./cache/obsnet/ObsNet_CityScapes.pth", help="path to obsnet")
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.one = torch.FloatTensor([1.]).to(args.device)
    args.zero = torch.FloatTensor([0.]).to(args.device)
    
    splits = ['ACDC', 'SMIYC', 'outofcontext', 'synflare', 'synobjs', 'synrain']
        
    for split in splits:
        args.split = split
        args.img_folder = f'./bravo_dataset/bravo_{args.split}'
        args.dest_folder = f'./output/bravo_{args.split}'

        if args.data == "CamVid":
            args.mean = [0.4108, 0.4240, 0.4316]
            args.std = [0.3066, 0.3111, 0.3068]
            args.h, args.w = [360, 480]
            args.nclass = 12
            args.colors = np.array([
                [128, 128, 128],  # sky
                [128, 0, 0],  # building
                [192, 192, 128],  # column_pole
                [128, 64, 128],  # road
                [0, 0, 192],  # sidewalk
                [128, 128, 0],  # Tree
                [192, 128, 128],  # SignSymbol
                [64, 64, 128],  # Fence
                [64, 0, 128],  # Car
                [64, 64, 0],  # Pedestrian
                [0, 128, 192],  # Bicyclist
                [0, 0, 0],  # Void
            ])

            args.stuff_classes = [8, 9, 10]

        elif args.data == "BddAnomaly":
            args.mean = [0.3698, 0.4145, 0.4247]
            args.std = [0.2525, 0.2695, 0.2870]
            args.h, args.w = [360, 640]  # Original size [720, 1280]
            args.nclass = 19
            args.colors = np.array([
                [128, 64, 128],  # road
                [244, 35, 232],  # sidewalk
                [70, 70, 70],  # building
                [102, 102, 156],  # wall
                [190, 153, 153],  # fence
                [153, 153, 153],  # pole
                [250, 170, 30],  # traffic_light
                [220, 220, 0],  # traffic_sign
                [107, 142, 35],  # vegetation
                [152, 251, 152],  # terrain
                [0, 130, 180],  # sky
                [220, 20, 60],  # person
                [255, 0, 0],  # rider
                [0, 0, 142],  # car
                [0, 0, 70],  # truck
                [0, 60, 100],  # bus
                [0, 80, 100],  # train
                [0, 0, 230],  # motorcycle
                [119, 11, 32],  # bicycle
                [0, 0, 0]])  # unlabelled
            args.stuff_classes = [11, 12, 13, 14, 15, 16, 17, 18, 19]

        elif args.data == "CityScapes":
            args.h, args.w = [512, 1024]   # original size [1024, 2048]
            args.mean = (0.485, 0.456, 0.406)
            args.std = (0.229, 0.224, 0.225)
            args.nclass = 19
            args.colors = np.array([[128, 64, 128],                     # 0: road
                                    [244, 35, 232],                     # 1: sidewalk
                                    [70, 70, 70],                       # 2: building
                                    [102, 102, 156],                    # 3: wall
                                    [190, 153, 153],                    # 4: fence
                                    [153, 153, 153],                    # 5: pole
                                    [250, 170, 30],                     # 6: traffic_light
                                    [220, 220, 0],                      # 7: traffic_sign
                                    [107, 142, 35],                     # 8: vegetation
                                    [152, 251, 152],                    # 9: terrain
                                    [0, 130, 180],                      # 10: sky
                                    [220, 20, 60],                      # 11: person
                                    [255, 0, 0],                        # 12: rider
                                    [0, 0, 142],                        # 13: car
                                    [0, 0, 70],                         # 14: truck
                                    [0, 60, 100],                       # 15: bus
                                    [0, 80, 100],                       # 16: train
                                    [0, 0, 230],                        # 17: motorcycle
                                    [119, 11, 32],                      # 18: bicycle
                                    [0, 0, 0]])                         # 19: unlabelled
            args.stuff_classes = [11, 12, 13, 14, 15, 16, 17, 18, 19]
        else:
            raise NameError("Data not known")
        
        imgs = []
        for (dirpath, dirnames, filenames) in os.walk(args.img_folder):
            for filename in filenames:
                if filename.endswith('.png') or filename.endswith('.jpg'):
                    imgs.append(dirpath + '/' + filename)

        args.imgs = imgs
        args.cmap = dict(zip(range(len(args.colors)), args.colors))
        args.yellow = torch.FloatTensor([1, 1, 0]).to(args.device).view(1, 3, 1, 1).expand(1, 3, args.h, args.w)
        args.blue = torch.FloatTensor([0, 0, .4]).to(args.device).view(1, 3, 1, 1).expand(1, 3, args.h, args.w)

        test(args)

