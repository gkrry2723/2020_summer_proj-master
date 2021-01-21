# Hair Removal via GAN
This repository contains source codes to train hair-removal network using GAN. In this work, [Wasserstein GAN-GP(WGAN-GP)](https://arxiv.org/abs/1704.00028) is used for the generative adversarial loss and WGAN-GP parts of the codes are builtd based on [the Pytorch implementation of WGAN-GP by Erik Linder-Norén](https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan_gp/wgan_gp.py).

## Prerequisites
- Melanoma Hair Removal Dataset

## Installation
- Download the repository
```
git clone https://github.com/joyfuldahye/imagelab_cau_2020_summer_proj.git
```
- Install required libraries
```
pip install -r hair_removal/requirements.txt
```

## Training by running a shellscript file
1. Set dataset(`--dir_train_data_image`), csv(`--dir_data_csv_hair`, `--dir_data_csv_hair`) and result path(`--dir_output`) in `hair_removal/src/scripts/train_hair_removal.sh` according to your environment
2. Modify `hair_removal/src/scripts/train_hair_removal.sh` as you want to try
3. Run the following commands in your terminal
```
cd hair_removal/src/scripts
chmod +x train_hair_removal.sh
./train_hair_removal.sh
```

## Monitoring training and output images
Current training codes save learning curves and output images in `hair_removal/result`. Please note that you are free to change the path to save result outputs and methods to monitor the results.
- `hair_removal/result/output/'your-experiment-name(e.g., 20200724_l1_lamdis10_lamgp10_lrd00001_lrg00001_bs4_ndisc5_nep10000_ex1_monitor_train)'`: contains training curves and output images.
    * The training curves are simultaneously updated as the training progresses with the interval(`--sample_interval`) you've set in `hair_removal/src/scripts/train_hair_removal.sh`.
    * `hair_removal/result/output/'your-experiment-name/learn-curve-distance`: shows a training curve of distance term in the generator loss
    * `hair_removal/result/output/'your-experiment-name/learn-curve-loss-d`: shows a training curve of discriminator loss
    * `hair_removal/result/output/'your-experiment-name/learn-curve-loss-g`: shows a training curve of generator loss
- `hair_removal/result/train_info/'your-experiment-name(e.g., 20200724_l1_lamdis10_lamgp10_lrd00001_lrg00001_bs4_ndisc5_nep10000_ex1_monitor_train)'`: contains the configuration and model architecture details of the experiment 'your-experiment-name'.