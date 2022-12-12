# DeepUME: Learning the Universal Manifold Embedding for Robust Point Cloud Registration [(BMVC 2021)](https://www.bmvc2021-virtualconference.com/conference/papers/paper_0948.html)
<!--Created by Natalie Lang and Joseph M. Francos from Ben-Gurion University.-->
[[paper]](http://arxiv.org/abs/2112.09938)
[[thesis]](https://github.com/langnatalie/DeepUME/files/6937954/DeepUME_thesis.pdf)

![teaser-1](https://user-images.githubusercontent.com/55830582/128337630-a6d48728-b933-4593-a7da-819e488298ad.png)

## Introduction
Deep Universal Manifold Embedding (DeepUME) is a learning-based point cloud registration algorithm which achieves fast and accurate global regitration. This repository contains a basic PyTorch implementation of DeepUME. Please refer to our [paper](http://arxiv.org/abs/2112.09938) for more details.

## Usage
This code has been tested on Python 3.6.13, PyTorch 1.4.0 and CUDA 10.1.

### Prerequisite
1. PyTorch=1.4.0: https://pytorch.org
2. h5py
3. open3d
4. TensorboardX: https://github.com/lanpa/tensorboardX

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
