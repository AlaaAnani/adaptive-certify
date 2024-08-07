# Adaptive Hierarchical Certification for Segmentation using Randomized Smoothing 

[paper PDF](https://raw.githubusercontent.com/mlresearch/v235/main/assets/anani24a/anani24a.pdf) | [arXiv](https://arxiv.org/abs/2402.08400) | [poster](https://icml.cc/media/PosterPDFs/ICML%202024/33356.png?t=1720897031.2040431)

Code for the paper: 

Alaa Anani, Tobias Lorenz, Bernt Schiele and Mario Fritz. Adaptive Hierarchical Certification for Segmentation using Randomized Smoothing. In _International Conference on Machine Learning (ICML)_, 2024. 



<figure>
  <img src="images/ICML24-Anani.png" alt="ImageAltText">
</figure>

## Abstract
Certification for machine learning is proving that no adversarial sample can evade a model within a range under certain conditions, a necessity for safety-critical domains. Common certification methods for segmentation use a flat set of fine-grained classes, leading to high abstain rates due to model uncertainty across many classes. We propose a novel, more practical setting, which certifies pixels within a multi-level hierarchy, and adaptively relaxes the certification to a coarser level for unstable components classic methods would abstain from, effectively lowering the abstain rate whilst providing more certified semantically meaningful information. We mathematically formulate the problem setup, introduce an adaptive hierarchical certification algorithm and prove the correctness of its guarantees. Since certified accuracy does not take the loss of information into account for coarser classes, we introduce the Certified Information Gain ($\mathrm{CIG}$) metric, which is proportional to the class granularity level. Our extensive experiments on the datasets Cityscapes, PASCAL-Context, ACDC and COCO-Stuff demonstrate that our adaptive algorithm achieves a higher $\mathrm{CIG}$ and lower abstain rate compared to the current state-of-the-art certification method.

<figure>
  <img src="images/teaser.png" alt="ImageAltText">
  <figcaption>The certified segmentation outputs on input images (a) and (d) from SegCertify in (b) and (e), and AdaptiveCertify in (c) and (f) with their corresponding certified information gain (CIG) and abstain rate. Our method provides more meaningful certified output in pixels the state-of-the-art abstains from (white pixels), with a much lower abstain rate, and higher certified information gain.</figcaption>
</figure>

## Installation
Clone the repository:
```
git clone --recurse-submodules https://github.com/AlaaAnani/adaptive-certify.git
cd adaptive-certify/
```
Setup the environment by running the following:

```
conda create -n cert python=3.8
conda activate cert
```
Install PyTorch. We used pytorch==1.7.1:
```
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
```
Install the rest of the libraries
```
pip install -r requirements.txt
```
## Datasets Preperation
We place the datasets in the following directory format under `HRNet-Semantic-Segmentation/`:
```
HRNet-Semantic-Segmentation
├── data
│   ├── acdc
│   │   ├── gt
│   │   ├── rgb_anon
│   ├── cityscapes
│   │   ├── gtFine
│   │   ├── leftImg8bit
│   ├── cocostuff
│   │   ├── cocostuff-10k-v1.1.json
│   │   ├── train
│   │   └── val
│   └── pascal_ctx
│   |   ├── trainval
│   |   └── VOCdevkit
│   ├── list
│   │   ├── acdc
│   │   ├── ade20k
│   │   ├── cityscapes
│   │   └── cocostuff

```
### Cityscapes
Download the dataset from the [official Cityscapes dataset website](https://www.cityscapes-dataset.com/downloads/).
### ACDC
Download the dataset from the [official ACDC dataset website](https://acdc.vision.ee.ethz.ch/download)
### PASCAL-Context
Follow the [mmsegmentation instructions](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#pascal-context) to download the dataset API and do the conversions.
### COCO-Stuff-10K
Follow the [mmsegmentation instructions](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#coco-stuff-10k) to download the dataset and run the necessary label conversions.


### Hierarchies
We already upload our pre-defined hierarchies JSONs in `configs/<dataset>/<dataset>_hierarchy.json`. 

However, if you would like to create them from scratch, we also provide the script for this under `annotations/<dataset>.py`

For example, to create the hierarchy JSON for Cityscapes, run the following:
```
python annotations/cityscapes.py
```
which will create `configs/cityscapes/cityscapes_hierarchy.json`.

Feel free to experiment with different hierarchy DAG structures on top every dataset fine-grained classes in level 0. To do so, change the nodes arrangement in a dataset's script `annotations/<dataset>.py`.


# Model Weights
### Cityscapes, ACDC, PASCAL-Context:
For models trained on Cityscapes and PASCAL-Context with a noise of $\sigma=0.25$, we use the weights provided by the repository [segmentation-smoothing](https://github.com/eth-sri/segmentation-smoothing/tree/main/code)


### COCO-Stuff-10K: 
#### Training 
To train HrNetV2 on COCO-Stuff-10K with a noise of $\sigma=0.25$, run the following:
```
python -m torch.distributed.launch --nproc_per_node=1 tools/train.py --cfg experiments/cocostuff/seg_hrnet_ocr_adv025_3965_alt_w48_520x520_ohem_sgd_lr1e-3_wd1e-4_bs_16_epoch110_paddle.yaml
```
### Download all model weights

Alternatively, download all the needed weights from GDrive [here](https://drive.google.com/drive/folders/1MiciR1oJJaSYb4EDTKn207YYsww7Luo_?usp=sharing).

The pretrained_models directory under `HRNet-Semantic-Segmentation/` should look like this:
```
HRNet-Semantic-Segmentation/
├── pretrained_models
│   ├── cityscapes.pth
│   ├── cocostuff10k_025.pth
│   ├── hrnetv2_w48_imagenet_pretrained.pth
│   ├── HRNet_W48_C_ssld_pretrained.pth
│   └── pascal.pth
```
## Run Certification
### Sigma inference
To run certification on AdaptiveCertify, and also the baseline SegCertify:
```
python tools/test_adaptivecert.py --cfg configs/cityscapes/cityscapes.yaml --exp inference
```
Possible values for cfg depending on the dataset are:
- 'configs/cocostuff/cocostuff.yaml'
- 'configs/pascal_ctx/pascal_ctx.yaml'
- 'configs/acdc/acdc.yaml'
- 'configs/cityscapes/cityscapes.yaml'

### Experiments reproduction
To replicate the experiment figures we have in the paper, you need to run the tools/test_adaptivecert.py script.
1. You can assign to the experiment argument (--exp) either of these values: `['inference', 'table', 'distribution', 'images', 'fluctuations', 'find_best_threshold']`
```
python tools/test_adaptivecert.py --exp <experiment name> --cfg <dataset config file>
```
For example, to reproduce the table in our results on Cityscapes:
```
python tools/test_adaptivecert.py --exp table --cfg configs/cityscapes/cityscapes.yaml
```
This creates a .pkl file per image: `logs/<dataset_name>/<experiment name>/<image name>.pkl` containing the necessary data to create the figure.

2. To graph out the logged outputs, or produce corresponding latex code for the figures, use the IPYNB Notebooks we have in `experiments/`. They scan the generated data in the logs/ directory created by step 1, and use them to generate the needed visuals.
   
# Contributors
[Alaa Anani](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/people/alaa-anani) (aanani@mpi-inf.mpg.de)

[Tobias Lorenz](https://www.t-lorenz.com/) (tobias.lorenz@cispa.de)

[Prof. Dr. Bernt Schiele](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/people/bernt-schiele) (schiele@mpi-inf.mpg.de)

[Prof. Dr. Mario Fritz](https://cispa.saarland/group/fritz/) (fritz@cispa.de)


