# Small Object Few-shot Segmentation for Vision-based Industrial Inspection

This is an official PyTorch implementation of the paper [Small Object Few-shot Segmentation for Vision-based Industrial Inspection](https://arxiv.org/abs/2407.21351).

<p align="center">
  <img src=assets/SOFS.jpg width="100%">
</p>

We present SOFS to solve problems that various and sufficient defects are difficult to obtain and anomaly detection cannot detect specific defects in Vision-based Industrial Inspection. 
SOFS can quickly adapt to unseen classes without retraining, achieving few-shot semantic segmentation (FSS) and few-shot anomaly detection (FAD).
SOFS can segment the small defects conditioned on the support sets, e.g., it segments the defects with area proportions less than 0.03%.
Some visualizations are shown in the figure below.


### Installation
1. The default python version is python 3.8.
2. Follow the installation of [DINO v2](https://github.com/facebookresearch/dinov2), please download DINO v2 ViT-B/14 distilled (without registers) pre-trained model.
3. Please download the pretrained model weight (https://drive.google.com/drive/folders/1HlGakaTWR77MKnCwzqBNsFjgk6Qhinym?usp=sharing)
4. Use the following commands:
```
conda create -n sofs python=3.8 -y
conda activate sofs
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch -c nvidia -y
pip install -r requirements.txt
pip uninstall -y opencv-python opencv-python-headless
pip install opencv-python==4.5.5.64
```

### Train 
사전학습하기.
```
bash train_dusan_hole.sh
```


### Test 
새로운 도메인 데이터로 inference.
```
bash test_dusan_hole.sh

# 사전 학습된 체크포인트 지정해서 test
CKPT_PATH=/nas_homes/dongjin/iipl_project/last.pth bash test_dusan_hole.sh
```
