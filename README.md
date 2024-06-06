<style>
.image-container {
    display: flex;
    justify-content: space-between;
    overflow-x: auto;
    white-space: nowrap;
}

.image-container img {
    max-width: calc(95% / 5);
    height: auto;
}
</style>

# BRAVO Challenge

The 2024 BRAVO Challenge aims to benchmark segmentation models on urban scenes undergoing diverse forms of natural degradation and realistic-looking synthetic corruption. We propose two tracks.

### Track 1 – Single-domain training

In this track, you must train your models exclusively on the [Cityscapes dataset](https://www.cityscapes-dataset.com/). This track evaluates the robustness of models trained with limited supervision and geographical diversity when facing unexpected corruptions observed in real-world scenarios.

### Track 2 – Multi-domain training

In this track, you must train your models over a mix of datasets, whose choice is strictly limited to the list provided below, comprising both natural and synthetic domains. This track assesses the impact of fewer constraints on the training data on robustness.

Allowed training datasets for Track 2:
- [Cityscapes](https://www.cityscapes-dataset.com/)
- [BDD100k](https://bdd-data.berkeley.edu/)
- [Mapillary Vistas](https://www.mapillary.com/datasets)
- [India Driving Dataset](https://idd.insaan.iiit.ac.in/)
- [WildDash 2](https://www.wilddash.cc/)
- [GTA5 Dataset](https://download.visinf.tu-darmstadt.de/data/from_games/) (synthetic)
- [SHIFT Dataset](https://www.vis.xyz/shift/) (synthetic)

## 1. General rules

1. Models in each track must be trained using only the datasets allowed for that track.
2. Employing generative models for synthetic data augmentation is strictly forbidden.
3. All results must be reproducible. Participants must submit a white paper containing comprehensive technical details alongside their results. Participants must make models and inference code accessible.
4. Evaluation will consider the 19 classes of Cityscapes (see below).


## 2. The BRAVO Benchmark Dataset

We created the benchmark dataset with real, captured images and realistic-looking synthetic augmentations, repurposing existing datasets and combining them with newly generated data. The benchmark dataset comprises images from [ACDC](https://acdc.vision.ee.ethz.ch/), [SegmentMeIfYouCan](https://segmentmeifyoucan.com/), [Out-of-context Cityscapes](https://arxiv.org/abs/2108.00968), and new synthetic data.

Get the full benchmark dataset at the following link: [full BRAVO Dataset download link](https://drive.google.com/drive/u/4/folders/11-dnlbMjm8O_ynq1REuDYKOmHLqEhGYP).

The dataset includes the following subsets (with individual download links):

**bravo-ACDC:** real scenes captured in adverse weather conditions, i.e., fog, night, rain, and snow. ([download link](https://drive.google.com/drive/u/4/folders/1IW6-Tdfk2At6CrIIrA-QJF6CEcHgqqha) or directly from [ACDC website](https://acdc.vision.ee.ethz.ch/download))
    <p class="image-container">
        <img src="images/bravobenchmark/acdc/acdc1.png" alt="Image 1">
        <img src="images/bravobenchmark/acdc/acdc2.png" alt="Image 2">
        <img src="images/bravobenchmark/acdc/acdc3.png" alt="Image 3">
        <img src="images/bravobenchmark/acdc/acdc4.png" alt="Image 4">
        <img src="images/bravobenchmark/acdc/acdc5.png" alt="Image 5">
    </p>

**bravo-SMIYC:** real scenes featuring out-of-distribution (OOD) objects rarely encountered on the road. ([download link](https://drive.google.com/drive/u/4/folders/1XnC9_7RzwZCWaDpP3iETbGt7Y
    <p class="image-container">
        <img src="images/bravobenchmark/smiyc/smiyc1.jpg" alt="Image 1">
        <img src="images/bravobenchmark/smiyc/smiyc2.jpg" alt="Image 2">
        <img src="images/bravobenchmark/smiyc/smiyc3.jpg" alt="Image 3">
        <img src="images/bravobenchmark/smiyc/smiyc4.jpg" alt="Image 4">
        <img src="images/bravobenchmark/smiyc/smiyc5.jpg" alt="Image 5">
    </p>

**bravo-synrain:** augmented scenes with synthesized raindrops on the camera lens. We augmented the validation images of Cityscapes and generated 500 images with raindrops. ([download link](https://drive.google.com/drive/u/4/folders/1onP6tUVSjV-qKWWLm6wiOZCB9U14_gQ6))
    <p class="image-container">
        <img src="images/bravobenchmark/synrain/rain1.png" alt="Image 1">
        <img src="images/bravobenchmark/synrain/rain2.png" alt="Image 2">
        <img src="images/bravobenchmark/synrain/rain3.png" alt="Image 3">
        <img src="images/bravobenchmark/synrain/rain4.png" alt="Image 4">
        <img src="images/bravobenchmark/synrain/rain5.png" alt="Image 5">
    </p>

**bravo-synobjs:** augmented scenes with inpainted synthetic OOD objects. We augmented the validation images of Cityscapes and generated 656 images with 26 OOD objects. ([download link](https://drive.google.com/drive/u/4/folders/1KKt_25S69DBf8ZTxhOhELpLgS2gyyGnf))
    <p class="image-container">
        <img src="images/bravobenchmark/synobjs/cheetah.png" alt="Image 1">
        <img src="images/bravobenchmark/synobjs/chimpanzee.png" alt="Image 2">
        <img src="images/bravobenchmark/synobjs/lion.png" alt="Image 3">
        <img src="images/bravobenchmark/synobjs/panda.png" alt="Image 4">
        <img src="images/bravobenchmark/synobjs/penguine.png" alt="Image 5">
    </p>

**bravo-synflare:** augmented scenes with synthesized light flares. We augmented the validation images of Cityscapes and generated 308 images with random light flares. ([download link](https://drive.google.com/drive/u/4/folders/13EpBXUY8BChoqfMxR5JhiyhqrzqLAO2y))
    <p class="image-container">
        <img src="images/bravobenchmark/synflare/flare1.png" alt="Image 1">
        <img src="images/bravobenchmark/synflare/flare2.png" alt="Image 2">
        <img src="images/bravobenchmark/synflare/flare3.png" alt="Image 3">
        <img src="images/bravobenchmark/synflare/flare4.png" alt="Image 4">
        <img src="images/bravobenchmark/synflare/flare5.png" alt="Image 5">
    </p>

**bravo-outofcontext:** augmented scenes with random backgrounds. We augmented the validation images of Cityscapes and generated 329 images with random random backgrounds. ([download link](https://drive.google.com/drive/u/4/folders/1NoXqTQWxrj_yKMNRKLOd1rnn2TjqIaU5))
    <p class="image-container">
        <img src="images/bravobenchmark/synooc/ooc1.png" alt="Image 1">
        <img src="images/bravobenchmark/synooc/ooc2.png" alt="Image 2">
        <img src="images/bravobenchmark/synooc/ooc3.png" alt="Image 3">
        <img src="images/bravobenchmark/synooc/ooc4.png" alt="Image 4">
        <img src="images/bravobenchmark/synooc/ooc5.png" alt="Image 5">
    </p>


## 3. Metrics

The metrics are computed separately for 9 subsets of the benchmark dataset: ACDCfog, ACDCrain, ACDCnight, ACDCsnow, SMIYC, synrain, synobjs, synflare, and outofcontext.

The metrics are computed pixel-wise and averaged over the pixels of the images belonging to a subset.

For each subset, the following metrics are computed:

### 3.1. Semantic metrics

- **mIoU**: mean Intersection Over Union, the rate of corrected labeled  pixels over all pixels. Evaluated on subsets: ACDC*, synrain, synflare, outofcontext.
- **ECE**: Expected Calibration Error, quantifying the mismatch between predicted confidence and actual accuracy. Evaluated on subsets: ACDC*, synrain, synflare, outofcontext.
- **AUROC**: Area Under the ROC Curve, over the binary criterion of a pixel being accurate, ranked by the predicted confidence level for the pixel. Evaluated on subsets: ACDC*, synrain, synflare, synobjs, SMIYC.
- **FPR@95**: False Positive Rate when True Positive Rate is 95% computed in the ROC curve above. Evaluated on subsets: ACDC*, synrain, synflare, synobjs, SMIYC.
- **AUPR-Success**: Area Under the Precision-Recall curve, over the same data as the AUROC. Evaluated on subsets: ACDC*, synrain, synflare, outofcontext.
- **AUPR-Error**: Area Under the Precision-Recall, on the reversed data (pixel being inaccurate, ranked by 1-confidence). Evaluated on subsets: ACDC*, synrain, synflare, outofcontext.

Those metrics are computed for all pixels, all "invalid pixels" (e.g., those behind a flare or an unknown object), and all valid pixels (those unaffected by the "invalidating" transformations).

### 3.2. Out-of-distribution metrics

Those are the AUROC, AUPRC-Success, and FPR@95 metrics above, but over different data: the (ground-truth) status of a pixel being invalid ranked by reversed predicted confidence (1-confidence). Those metrics quantify whether the model attributes, as expected, less confidence to the invalid pixels.

## 4. Creating a submission

For each input image "source.png", we require two submitted files "source_pred.png" for the semantic prediction and "source_conf.png" for the confidences.

### 4.1 Expected format for the raw input images

For the class prediction files (`_pred.png`): PNG format, 8-bits, grayscale, with each pixel with a value from 0 to 19 corresponding to the 19 classes of Cityscapes, which are, in order:

0. road
1. sidewalk
2. building
3. wall
4. fence
5. pole
6. traffic light
7. traffic sign
8. vegetation
9. terrain
10. sky
11. person
12. rider
13. car
14. truck
15. bus
16. train
17. motorcycle
18. bicycle.

For the confidence files (`_conf.png`): PNG format, 16-bits, grayscale, with each pixel's value from 0 to 65535 corresponding to the confidence in the prediction (for the predicted class).
For confidences  computed initially on a continuous [0.0, 1.0] interval, we suggest discretizing them using the formula: `min(floor(conf*65536), 65535)`

Each prediction and confidence image should have exactly the same dimensions as the corresponding input image. The evaluation is made pixel-wise.

### 4.2. Assembling the submission

The submission files should be assembled on a tarfile or directory tree corresponding to the original hierarchical organization of the full BRAVO Dataset Benchmark (see Appendix 1).

### 4.3. Encoding the submission

The submission has to be compressed and encoded before being uploaded to the Evaluation Server. This requires using the utilities in this repository.

#### 4.3.1 Preparing the environment

Clone this repository to `local_repo_root` and create an environment with the requirements:

```bash
cd bravo_toolkit
conda create -n bravo python=3.9
conda activate bravo
python -m pip install -r requirements.txt
```

Always activate the environment and add `local_repo_root` to `PYTHONPATH` before running the commands in the sections below.
```bash
conda activate bravo
export PYTHONPATH=<local_repo_root>
```

We use Anaconda in the examples above, but its use is optional. Any manager providing an isolated Python 3.9 environment (e.g., virtualenv) is acceptable.

#### 4.3.2. Encoding the submission files to the submission format

To encode the submission, you'll need to download the sampling file [bravo_SAMPLING.tar](https://github.com/valeoai/bravo_challenge/releases/download/v0.1.0/bravo_SAMPLING.tar).

The submission files must be in a directory tree or in a .tar file. Use one of the commands below:

```bash
python -m bravo_toolkit.util.encode_submission <submission-root-directory> <encoded-submission-output.tar> --samples bravo_SAMPLING.tar
```

or

```bash
python -m bravo_toolkit.util.encode_submission <submission-raw-files.tar> <encoded-submission-output.tar> --samples bravo_SAMPLING.tar
```

### 4.4. Uploading the submission to the evaluation server

We are excited to unveil the BRAVO Challenge as an initiative within [ELSA — European Lighthouse on Secure and Safe AI](https://www.elsa-ai.eu/), a network of excellence funded by the European Union. The BRAVO Challenge is officially featured on the [ELSA Benchmarks website](https://benchmarks.elsa-ai.eu/) as the Autonomous Driving/Robust Perception task.

Please refer to the [task website](https://benchmarks.elsa-ai.eu/?ch=1&com=introduction) for more instructions on uploading the submission.


## Acknowledgements

We extend our heartfelt gratitude to the authors of [ACDC](https://acdc.vision.ee.ethz.ch/contact/), [SegmentMeIfYouCan](https://segmentmeifyoucan.com/), and [Out-of-context Cityscapes](https://arxiv.org/abs/2108.00968) for generously permitting us to repurpose their benchmarking data. We are also thankful to the authors of [GuidedDisent](https://github.com/astra-vision/GuidedDisent) and [Flare Removal](https://github.com/google-research/google-research/tree/master/flare_removal) for providing the amazing toolboxes that helped synthesize realistic-looking raindrops and light flares. All those people collectively contributed to creating BRAVO, a unified benchmark for robustness in autonomous driving.

## A1. Expected input directory tree for the submission

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

