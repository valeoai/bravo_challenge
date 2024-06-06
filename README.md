# Working with the bravo_toolkit package

## Preparing the environment

Clone the repository to `local_repo_root` and create an enviroment with the requirements

```bash
cd bravo_toolkit
conda create -n bravo python=3.9
conda activate bravo
python -m pip install -r requirements.txt
```

Remember to always activate the environment and add `local_repo_root` to `PYTHONPATH` before running the commands in the sections below.
```bash
conda activate bravo
export PYTHONPATH=<local_repo_root>
```

## Encoding the submission files to the submission format

To encode the submission, you'll need to download the sampling file [bravo_SAMPLING.tar](https://github.com/valeoai/bravo_challenge/releases/download/v0.1.0/bravo_SAMPLING.tar).

The submission files must be in a directory tree or in a .tar file. Use one of the commands below:

```bash
python -m bravo_toolkit.util.encode_submission <submission-root-directory> <encoded-submission-output.tar> --samples bravo_SAMPLING.tar
```

or

```bash
python -m bravo_toolkit.util.encode_submission <submission-raw-files.tar> <encoded-submission-output.tar> --samples bravo_SAMPLING.tar
```

## Expected format for the raw input images

For the class prediction files (`_pred.png`): PNG format, 8-bits, grayscale, with each pixel with a value from 0 to 19 corresponding to the 19 classes of Cityscapes.

For the confidence files (`_conf.png`): PNG format, 16-bits, grayscale, with each pixel with a value from 0 to 65535 corresponding to the confidence on the prediction (for the predicted class). For confidences originally computed on a continuous [0.0, 1.0] interval, we suggest discretizing them using the formula: `min(floor(conf*65536), 65535)`

## Expected input directory tree for the submission

The submission directory, or raw input tar file expected by `encode_submission` should have the following structure:

```
submission_directory_root or submission_raw.tar
├── bravo_ACDC
│   ├── fog
│   │   └── test
│   │       ├── GOPR0475
│   │       │   ├── GOPR0475_frame_000247_rgb_anon_conf.png
│   │       │   ├── GOPR0475_frame_000247_rgb_anon_pred.png
│   │       │   ├── ...
│   │       │   ├── GOPR0475_frame_001060_rgb_anon_conf.png
│   │       │   └── GOPR0475_frame_001060_rgb_anon_pred.png
│   │       ├── GOPR0477
│   │       │   ├── GOPR0477_frame_000794_rgb_anon_conf.png
│   │       │   ├── GOPR0477_frame_000794_rgb_anon_pred.png
│   │       │   ├── ...
│   │       │   ├── GOPR0477_frame_001032_rgb_anon_conf.png
│   │       │   └── GOPR0477_frame_001032_rgb_anon_pred.png
│   │       ├── GOPR0478
│   │       │   ├── GOPR0478_frame_000259_rgb_anon_conf.png
│   │       │   ├── GOPR0478_frame_000259_rgb_anon_pred.png
│   │       │   ├── ...
│   │       │   ├── GOPR0478_frame_001023_rgb_anon_conf.png
│   │       │   └── GOPR0478_frame_001023_rgb_anon_pred.png
│   │       ├── GP010475
│   │       │   ├── GP010475_frame_000006_rgb_anon_conf.png
│   │       │   ├── GP010475_frame_000006_rgb_anon_pred.png
│   │       │   ├── ...
│   │       │   ├── GP010475_frame_000831_rgb_anon_conf.png
│   │       │   └── GP010475_frame_000831_rgb_anon_pred.png
│   │       ├── GP010477
│   │       │   ├── GP010477_frame_000001_rgb_anon_conf.png
│   │       │   ├── GP010477_frame_000001_rgb_anon_pred.png
│   │       │   ├── ...
│   │       │   ├── GP010477_frame_000224_rgb_anon_conf.png
│   │       │   └── GP010477_frame_000224_rgb_anon_pred.png
│   │       ├── GP010478
│   │       │   ├── GP010478_frame_000032_rgb_anon_conf.png
│   │       │   ├── GP010478_frame_000032_rgb_anon_pred.png
│   │       │   ├── ...
│   │       │   ├── GP010478_frame_001061_rgb_anon_conf.png
│   │       │   └── GP010478_frame_001061_rgb_anon_pred.png
│   │       └── GP020478
│   │           ├── GP020478_frame_000001_rgb_anon_conf.png
│   │           ├── GP020478_frame_000001_rgb_anon_pred.png
│   │           ├── ...
│   │           ├── GP020478_frame_000042_rgb_anon_conf.png
│   │           └── GP020478_frame_000042_rgb_anon_pred.png
│   ├── night
│   │   └── test
│   │       ├── GOPR0355
│   │       │   ├── GOPR0355_frame_000138_rgb_anon_conf.png
│   │       │   ├── GOPR0355_frame_000138_rgb_anon_pred.png
│   │       │   ├── ...
│   │       │   ├── GOPR0355_frame_000214_rgb_anon_conf.png
│   │       │   └── GOPR0355_frame_000214_rgb_anon_pred.png
│   │       ├── GOPR0356
│   │       │   ├── GOPR0356_frame_000065_rgb_anon_conf.png
│   │       │   ├── GOPR0356_frame_000065_rgb_anon_pred.png
│   │       │   ├── ...
│   │       │   ├── GOPR0356_frame_001001_rgb_anon_conf.png
│   │       │   └── GOPR0356_frame_001001_rgb_anon_pred.png
│   │       ├── GOPR0364
│   │       │   ├── GOPR0364_frame_000001_rgb_anon_conf.png
│   │       │   ├── GOPR0364_frame_000001_rgb_anon_pred.png
│   │       │   ├── ...
│   │       │   ├── GOPR0364_frame_001053_rgb_anon_conf.png
│   │       │   └── GOPR0364_frame_001053_rgb_anon_pred.png
│   │       ├── GOPR0594
│   │       │   ├── GOPR0594_frame_000114_rgb_anon_conf.png
│   │       │   ├── GOPR0594_frame_000114_rgb_anon_pred.png
│   │       │   ├── ...
│   │       │   ├── GOPR0594_frame_001060_rgb_anon_conf.png
│   │       │   └── GOPR0594_frame_001060_rgb_anon_pred.png
│   │       ├── GP010364
│   │       │   ├── GP010364_frame_000009_rgb_anon_conf.png
│   │       │   ├── GP010364_frame_000009_rgb_anon_pred.png
│   │       │   ├── ...
│   │       │   ├── GP010364_frame_000443_rgb_anon_conf.png
│   │       │   └── GP010364_frame_000443_rgb_anon_pred.png
│   │       └── GP010594
│   │           ├── GP010594_frame_000003_rgb_anon_conf.png
│   │           ├── GP010594_frame_000003_rgb_anon_pred.png
│   │           ├── ...
│   │           ├── GP010594_frame_000087_rgb_anon_conf.png
│   │           └── GP010594_frame_000087_rgb_anon_pred.png
│   ├── rain
│   │   └── test
│   │       ├── GOPR0572
│   │       │   ├── GOPR0572_frame_000145_rgb_anon_conf.png
│   │       │   ├── GOPR0572_frame_000145_rgb_anon_pred.png
│   │       │   ├── ...
│   │       │   ├── GOPR0572_frame_001035_rgb_anon_conf.png
│   │       │   └── GOPR0572_frame_001035_rgb_anon_pred.png
│   │       ├── GOPR0573
│   │       │   ├── GOPR0573_frame_000180_rgb_anon_conf.png
│   │       │   ├── GOPR0573_frame_000180_rgb_anon_pred.png
│   │       │   ├── ...
│   │       │   ├── GOPR0573_frame_001046_rgb_anon_conf.png
│   │       │   └── GOPR0573_frame_001046_rgb_anon_pred.png
│   │       ├── GP010400
│   │       │   ├── GP010400_frame_000616_rgb_anon_conf.png
│   │       │   ├── GP010400_frame_000616_rgb_anon_pred.png
│   │       │   ├── ...
│   │       │   ├── GP010400_frame_001057_rgb_anon_conf.png
│   │       │   └── GP010400_frame_001057_rgb_anon_pred.png
│   │       ├── GP010402
│   │       │   ├── GP010402_frame_000326_rgb_anon_conf.png
│   │       │   ├── GP010402_frame_000326_rgb_anon_pred.png
│   │       │   ├── ...
│   │       │   ├── GP010402_frame_001046_rgb_anon_conf.png
│   │       │   └── GP010402_frame_001046_rgb_anon_pred.png
│   │       ├── GP010571
│   │       │   ├── GP010571_frame_000077_rgb_anon_conf.png
│   │       │   ├── GP010571_frame_000077_rgb_anon_pred.png
│   │       │   ├── ...
│   │       │   ├── GP010571_frame_001050_rgb_anon_conf.png
│   │       │   └── GP010571_frame_001050_rgb_anon_pred.png
│   │       ├── GP010572
│   │       │   ├── GP010572_frame_000027_rgb_anon_conf.png
│   │       │   ├── GP010572_frame_000027_rgb_anon_pred.png
│   │       │   ├── ...
│   │       │   ├── GP010572_frame_000916_rgb_anon_conf.png
│   │       │   └── GP010572_frame_000916_rgb_anon_pred.png
│   │       ├── GP010573
│   │       │   ├── GP010573_frame_000001_rgb_anon_conf.png
│   │       │   ├── GP010573_frame_000001_rgb_anon_pred.png
│   │       │   ├── ...
│   │       │   ├── GP010573_frame_001056_rgb_anon_conf.png
│   │       │   └── GP010573_frame_001056_rgb_anon_pred.png
│   │       ├── GP020400
│   │       │   ├── GP020400_frame_000001_rgb_anon_conf.png
│   │       │   ├── GP020400_frame_000001_rgb_anon_pred.png
│   │       │   ├── ...
│   │       │   ├── GP020400_frame_000142_rgb_anon_conf.png
│   │       │   └── GP020400_frame_000142_rgb_anon_pred.png
│   │       ├── GP020571
│   │       │   ├── GP020571_frame_000001_rgb_anon_conf.png
│   │       │   ├── GP020571_frame_000001_rgb_anon_pred.png
│   │       │   ├── ...
│   │       │   ├── GP020571_frame_000248_rgb_anon_conf.png
│   │       │   └── GP020571_frame_000248_rgb_anon_pred.png
│   │       ├── GP020573
│   │       │   ├── GP020573_frame_000001_rgb_anon_conf.png
│   │       │   ├── GP020573_frame_000001_rgb_anon_pred.png
│   │       │   ├── ...
│   │       │   ├── GP020573_frame_000887_rgb_anon_conf.png
│   │       │   └── GP020573_frame_000887_rgb_anon_pred.png
│   │       └── GP030573
│   │           ├── GP030573_frame_000073_rgb_anon_conf.png
│   │           ├── GP030573_frame_000073_rgb_anon_pred.png
│   │           ├── ...
│   │           ├── GP030573_frame_000914_rgb_anon_conf.png
│   │           └── GP030573_frame_000914_rgb_anon_pred.png
│   └── snow
│       └── test
│           ├── GOPR0122
│           │   ├── GOPR0122_frame_000651_rgb_anon_conf.png
│           │   ├── GOPR0122_frame_000651_rgb_anon_pred.png
│           │   ├── ...
│           │   ├── GOPR0122_frame_001054_rgb_anon_conf.png
│           │   └── GOPR0122_frame_001054_rgb_anon_pred.png
│           ├── GOPR0176
│           │   ├── GOPR0176_frame_000394_rgb_anon_conf.png
│           │   ├── GOPR0176_frame_000394_rgb_anon_pred.png
│           │   ├── ...
│           │   ├── GOPR0176_frame_000884_rgb_anon_conf.png
│           │   └── GOPR0176_frame_000884_rgb_anon_pred.png
│           ├── GOPR0494
│           │   ├── GOPR0494_frame_000020_rgb_anon_conf.png
│           │   ├── GOPR0494_frame_000020_rgb_anon_pred.png
│           │   ├── ...
│           │   ├── GOPR0494_frame_001056_rgb_anon_conf.png
│           │   └── GOPR0494_frame_001056_rgb_anon_pred.png
│           ├── GOPR0496
│           │   ├── GOPR0496_frame_000663_rgb_anon_conf.png
│           │   ├── GOPR0496_frame_000663_rgb_anon_pred.png
│           │   ├── ...
│           │   ├── GOPR0496_frame_001033_rgb_anon_conf.png
│           │   └── GOPR0496_frame_001033_rgb_anon_pred.png
│           ├── GP010122
│           │   ├── GP010122_frame_000001_rgb_anon_conf.png
│           │   ├── GP010122_frame_000001_rgb_anon_pred.png
│           │   ├── ...
│           │   ├── GP010122_frame_000223_rgb_anon_conf.png
│           │   └── GP010122_frame_000223_rgb_anon_pred.png
│           ├── GP010176
│           │   ├── GP010176_frame_000001_rgb_anon_conf.png
│           │   ├── GP010176_frame_000001_rgb_anon_pred.png
│           │   ├── ...
│           │   ├── GP010176_frame_001057_rgb_anon_conf.png
│           │   └── GP010176_frame_001057_rgb_anon_pred.png
│           ├── GP010494
│           │   ├── GP010494_frame_000001_rgb_anon_conf.png
│           │   ├── GP010494_frame_000001_rgb_anon_pred.png
│           │   ├── ...
│           │   ├── GP010494_frame_000242_rgb_anon_conf.png
│           │   └── GP010494_frame_000242_rgb_anon_pred.png
│           ├── GP010496
│           │   ├── GP010496_frame_000001_rgb_anon_conf.png
│           │   ├── GP010496_frame_000001_rgb_anon_pred.png
│           │   ├── ...
│           │   ├── GP010496_frame_000883_rgb_anon_conf.png
│           │   └── GP010496_frame_000883_rgb_anon_pred.png
│           ├── GP010606
│           │   ├── GP010606_frame_000001_rgb_anon_conf.png
│           │   ├── GP010606_frame_000001_rgb_anon_pred.png
│           │   ├── ...
│           │   ├── GP010606_frame_001054_rgb_anon_conf.png
│           │   └── GP010606_frame_001054_rgb_anon_pred.png
│           ├── GP020176
│           │   ├── GP020176_frame_000001_rgb_anon_conf.png
│           │   ├── GP020176_frame_000001_rgb_anon_pred.png
│           │   ├── ...
│           │   ├── GP020176_frame_001060_rgb_anon_conf.png
│           │   └── GP020176_frame_001060_rgb_anon_pred.png
│           ├── GP020606
│           │   ├── GP020606_frame_000021_rgb_anon_conf.png
│           │   ├── GP020606_frame_000021_rgb_anon_pred.png
│           │   ├── ...
│           │   ├── GP020606_frame_000558_rgb_anon_conf.png
│           │   └── GP020606_frame_000558_rgb_anon_pred.png
│           └── GP030176
│               ├── GP030176_frame_000001_rgb_anon_conf.png
│               ├── GP030176_frame_000001_rgb_anon_pred.png
│               ├── ...
│               ├── GP030176_frame_000369_rgb_anon_conf.png
│               └── GP030176_frame_000369_rgb_anon_pred.png
├── bravo_SMIYC
│   └── RoadAnomaly21
│       └── images
│           ├── airplane0000_conf.png
│           ├── airplane0000_pred.png
│           ├── ...
│           ├── zebra0001_conf.png
│           └── zebra0001_pred.png
├── bravo_outofcontext
│   ├── frankfurt
│   │   ├── frankfurt_000000_000576_leftImg8bit_conf.png
│   │   ├── frankfurt_000000_000576_leftImg8bit_pred.png
│   │   ├── ...
│   │   ├── frankfurt_000001_082466_leftImg8bit_conf.png
│   │   └── frankfurt_000001_082466_leftImg8bit_pred.png
│   ├── lindau
│   │   ├── lindau_000000_000019_leftImg8bit_conf.png
│   │   ├── lindau_000000_000019_leftImg8bit_pred.png
│   │   ├── ...
│   │   ├── lindau_000058_000019_leftImg8bit_conf.png
│   │   └── lindau_000058_000019_leftImg8bit_pred.png
│   └── munster
│       ├── munster_000000_000019_leftImg8bit_conf.png
│       ├── munster_000000_000019_leftImg8bit_pred.png
│       ├── ...
│       ├── munster_000172_000019_leftImg8bit_conf.png
│       └── munster_000172_000019_leftImg8bit_pred.png
├── bravo_synflare
│   ├── frankfurt
│   │   ├── frankfurt_000000_000294_leftImg8bit_conf.png
│   │   ├── frankfurt_000000_000294_leftImg8bit_pred.png
│   │   ├── ...
│   │   ├── frankfurt_000001_082466_leftImg8bit_conf.png
│   │   └── frankfurt_000001_082466_leftImg8bit_pred.png
│   ├── lindau
│   │   ├── lindau_000000_000019_leftImg8bit_conf.png
│   │   ├── lindau_000000_000019_leftImg8bit_pred.png
│   │   ├── ...
│   │   ├── lindau_000058_000019_leftImg8bit_conf.png
│   │   └── lindau_000058_000019_leftImg8bit_pred.png
│   └── munster
│       ├── munster_000000_000019_leftImg8bit_conf.png
│       ├── munster_000000_000019_leftImg8bit_pred.png
│       ├── ...
│       ├── munster_000172_000019_leftImg8bit_conf.png
│       └── munster_000172_000019_leftImg8bit_pred.png
├── bravo_synobjs
│   ├── armchair
│   │   ├── 1_conf.png
│   │   ├── 1_pred.png
│   │   ├── ...
│   │   ├── 504_conf.png
│   │   ├── 504_pred.png
│   ├── baby
│   │   ├── 49_conf.png
│   │   ├── 49_pred.png
│   │   ├── ...
│   │   ├── 421_conf.png
│   │   ├── 421_pred.png
│   ├── bathtub
│   │   ├── 16_conf.png
│   │   ├── 16_pred.png
│   │   ├── ...
│   │   ├── 501_conf.png
│   │   ├── 501_pred.png
│   ├── bench
│   │   ├── 0_conf.png
│   │   ├── 0_pred.png
│   │   ├── ...
│   │   ├── 423_conf.png
│   │   ├── 423_pred.png
│   ├── billboard
│   │   ├── 134_conf.png
│   │   ├── 134_pred.png
│   │   ├── ...
│   │   ├── 461_conf.png
│   │   └── 461_pred.png
│   ├── box
│   │   ├── 58_conf.png
│   │   ├── 58_pred.png
│   │   ├── ...
│   │   ├── 381_conf.png
│   │   ├── 381_pred.png
│   ├── cheetah
│   │   ├── 14_conf.png
│   │   ├── 14_pred.png
│   │   ├── ...
│   │   ├── 500_conf.png
│   │   ├── 500_pred.png
│   ├── chimpanzee
│   │   ├── 0_conf.png
│   │   ├── 0_pred.png
│   │   ├── ...
│   │   ├── 468_conf.png
│   │   ├── 468_pred.png
│   ├── elephant
│   │   ├── 9_conf.png
│   │   └── 9_pred.png
│   │   ├── ...
│   │   ├── 441_conf.png
│   │   ├── 441_pred.png
│   ├── flamingo
│   │   ├── 5_conf.png
│   │   ├── 5_pred.png
│   │   ├── ...
│   │   ├── 482_conf.png
│   │   ├── 482_pred.png
│   ├── giraffe
│   │   ├── 8_conf.png
│   │   └── 8_pred.png
│   │   ├── ...
│   │   ├── 510_conf.png
│   │   ├── 510_pred.png
│   ├── gorilla
│   │   ├── 4_conf.png
│   │   ├── 4_pred.png
│   │   ├── ...
│   │   ├── 493_conf.png
│   │   ├── 493_pred.png
│   ├── hippopotamus
│   │   ├── 29_conf.png
│   │   ├── 29_pred.png
│   │   ├── ...
│   │   ├── 442_conf.png
│   │   ├── 442_pred.png
│   ├── kangaroo
│   │   ├── 6_conf.png
│   │   ├── 6_pred.png
│   │   ├── ...
│   │   ├── 495_conf.png
│   │   ├── 495_pred.png
│   ├── koala
│   │   ├── 0_conf.png
│   │   ├── 0_pred.png
│   │   ├── ...
│   │   ├── 489_conf.png
│   │   ├── 489_pred.png
│   ├── lion
│   │   ├── 7_conf.png
│   │   ├── 7_pred.png
│   │   ├── ...
│   │   ├── 503_conf.png
│   │   ├── 503_pred.png
│   ├── panda
│   │   ├── 5_conf.png
│   │   ├── 5_pred.png
│   │   ├── ...
│   │   ├── 494_conf.png
│   │   ├── 494_pred.png
│   ├── penguin
│   │   ├── 5_conf.png
│   │   ├── 5_pred.png
│   │   ├── ...
│   │   ├── 465_conf.png
│   │   ├── 465_pred.png
│   ├── plant
│   │   ├── 3_conf.png
│   │   ├── 3_pred.png
│   │   ├── ...
│   │   ├── 400_conf.png
│   │   ├── 400_pred.png
│   ├── polar bear
│   │   ├── 4_conf.png
│   │   ├── 4_pred.png
│   │   ├── ...
│   │   ├── 501_conf.png
│   │   ├── 501_pred.png
│   ├── sofa
│   │   ├── 3_conf.png
│   │   ├── 3_pred.png
│   │   ├── ...
│   │   ├── 453_conf.png
│   │   ├── 453_pred.png
│   ├── table
│   │   ├── 0_conf.png
│   │   ├── 0_pred.png
│   │   ├── ...
│   │   ├── 461_conf.png
│   │   └── 461_pred.png
│   ├── tiger
│   │   ├── 28_conf.png
│   │   ├── 28_pred.png
│   │   ├── ...
│   │   ├── 450_conf.png
│   │   ├── 450_pred.png
│   ├── toilet
│   │   ├── 15_conf.png
│   │   ├── 15_pred.png
│   │   ├── ...
│   │   ├── 504_conf.png
│   │   ├── 504_pred.png
│   ├── vase
│   │   ├── 3_conf.png
│   │   ├── 3_pred.png
│   │   ├── ...
│   │   ├── 506_conf.png
│   │   ├── 506_pred.png
│   └── zebra
│       ├── 5_conf.png
│       ├── 5_pred.png
│       ├── ...
│       ├── 499_conf.png
│       ├── 499_pred.png
└── bravo_synrain
    ├── frankfurt
    │   ├── frankfurt_000000_000294_leftImg8bit_conf.png
    │   ├── frankfurt_000000_000294_leftImg8bit_pred.png
    │   ├── ...
    │   ├── frankfurt_000001_083852_leftImg8bit_conf.png
    │   └── frankfurt_000001_083852_leftImg8bit_pred.png
    ├── lindau
    │   ├── lindau_000000_000019_leftImg8bit_conf.png
    │   ├── lindau_000000_000019_leftImg8bit_pred.png
    │   ├── ...
    │   ├── lindau_000058_000019_leftImg8bit_conf.png
    │   └── lindau_000058_000019_leftImg8bit_pred.png
    └── munster
        ├── munster_000000_000019_leftImg8bit_conf.png
        ├── munster_000000_000019_leftImg8bit_pred.png
        ├── ...
        ├── munster_000173_000019_leftImg8bit_conf.png
        └── munster_000173_000019_leftImg8bit_pred.png
88 directories, 7802 files
```

