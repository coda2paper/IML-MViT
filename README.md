# This is the implementation of our paper：MEViT

## Our environment
CUDA 11.3<br>
Python 3.8<br>
PyTorch 1.11.0<br>

## Usage
You can follow the steps below to test our model. The training code will be released after the paper is accepted.<br>
Step 1: install the packages in requirements.txt with ```pip install -r requirements.txt``` first.<br>
Step 2: Download the casia v1 dataset and put it under ```./dataset/casia```.<br>
Step 3: Download the pre-trained weights from [Google Drive](https://drive.google.com/drive/folders/17ylPvK9qxU4oDYIfqqQKfhItOcoGx5dR?usp=drive_link) and place them under ```./checkpoints/checkpoint.pth``` and ```./checkpoints/mae_pretrain_vit_base.pth```.<br>
Step 4: run ```python prediction.py``` and you can get the F1 value and AUC score.<br>
