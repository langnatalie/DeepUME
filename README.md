# DeepUME: Learning the Universal Manifold Embedding for Robust Point Cloud Registration
<!--Created by Natalie Lang and Joseph M. Francos from Ben-Gurion University.-->
[[paper]](https://github.com/langnatalie/DeepUME/files/7360160/DeepUME_paper_for_Yossi__revised_.2.pdf)
[[supplementary]](https://github.com/langnatalie/DeepUME/files/7360122/DeepUME_supplementary_for_Yossi__revised_.pdf)
[[thesis]](https://github.com/langnatalie/DeepUME/files/6937954/DeepUME_thesis.pdf)
[[data]](https://drive.google.com/drive/folders/1E6muJwx3WONbMJnFunywdmtTDofBZj_L?usp=sharing)

![teaser-1](https://user-images.githubusercontent.com/55830582/128337630-a6d48728-b933-4593-a7da-819e488298ad.png)

## Introduction
Deep Universal Manifold Embedding (DeepUME) is a learning-based point cloud registration algorithm which achieves fast and accurate global regitration. This repository contains a basic PyTorch implementation of DeepUME. Please refer to our [paper](https://github.com/langnatalie/DeepUME/files/6937947/DeepUME_paper.pdf) for more details.

## Usage
This code has been tested on Python 3.6.13, PyTorch 1.4.0 and CUDA 10.1.

### Prerequisite
1. PyTorch=1.4.0: https://pytorch.org
2. h5py
3. open3d
4. TensorboardX: https://github.com/lanpa/tensorboardX
3. Download [data](https://drive.google.com/drive/folders/1E6muJwx3WONbMJnFunywdmtTDofBZj_L?usp=sharing) to data/.

### Training
```
python main.py --exp_name=deepume --noise=sampling
```

### Testing
```
python main.py --exp_name=deepume --eval 
or
python main.py --exp_name=pretrained --eval --pretrained='pretrained/deepume.t7' --noise=zero_intersec --test_dataset=FAUST
```
