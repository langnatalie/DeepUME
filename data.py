#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import glob
import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
import open3d as o3d


# Part of the code is referred from: https://github.com/charlesq34/pointnet

def normalize(pc):
    """
    :param pc: size Nx3
    :return: pc centered and rescaled to fit in the unit sphere
    """
    mean = np.mean(pc, axis=0)
    centered_pc = pc - mean
    pc_norms = np.linalg.norm(centered_pc, axis=1)
    return centered_pc / max(pc_norms)


def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048*')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        www += ' --no-check-certificate'
        os.system('wget %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))


def load_h5(partition):
    download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []

    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition)):
        f = h5py.File(h5_name, 'r')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)

    return all_data, all_label


def load_ply(type, unit_sphare=False):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')

    if unit_sphare:
        all_data = {i: normalize(np.asarray(o3d.io.read_point_cloud(file).points)).astype('float32')
                    for i, file in enumerate(sorted(glob.glob(os.path.join(DATA_DIR, type, '*.ply'))))}
    else:
        all_data = {i: np.asarray(o3d.io.read_point_cloud(file).points).astype('float32')
                    for i, file in enumerate(glob.glob(os.path.join(DATA_DIR, type, '*.ply')))}

    return all_data


class Data(Dataset):
    def __init__(self, type='ModelNet40', partition='train', num_points=1024, noise='sampling', sigma=None):
        if type == 'ModelNet40':
            self.data, self.label = load_h5(partition)
        elif type == 'FAUST':
            self.data = load_ply(type, unit_sphare=False)
            self.partition = 'test'
        elif type == 'Stanford':
            self.data = load_ply(type, unit_sphare=True)
            self.partition = 'test'
        else:
            assert 'Data type not implemented'
        self.num_points = num_points
        self.noise = noise
        self.sigma = sigma
        self.partition = partition

    def __getitem__(self, item):
        pointcloud = np.random.permutation(self.data[item])[:2048]

        if self.partition != 'train':
            np.random.seed(item)
        pc_size = pointcloud.shape[0]

        anglex = np.random.uniform() * np.pi / 2
        angley = np.random.uniform() * np.pi / 2
        anglez = np.random.uniform() * np.pi

        cosx = np.cos(anglex)
        cosy = np.cos(angley)
        cosz = np.cos(anglez)
        sinx = np.sin(anglex)
        siny = np.sin(angley)
        sinz = np.sin(anglez)
        Rx = np.array([[1, 0, 0],
                       [0, cosx, -sinx],
                       [0, sinx, cosx]])
        Ry = np.array([[cosy, 0, siny],
                       [0, 1, 0],
                       [-siny, 0, cosy]])
        Rz = np.array([[cosz, -sinz, 0],
                       [sinz, cosz, 0],
                       [0, 0, 1]])
        R_ab = Rx.dot(Ry).dot(Rz)
        translation_ab = np.array([np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5),
                                   np.random.uniform(-0.5, 0.5)])

        pointcloud1 = pointcloud.T

        rotation_ab = Rotation.from_euler('zyx', [anglez, angley, anglex])
        pointcloud2 = rotation_ab.apply(pointcloud1.T).T + np.expand_dims(translation_ab, axis=1)

        if self.noise == 'gaussian':
            if self.sigma is not None:
                sigma = self.sigma
                pointcloud2 = pointcloud2 + np.random.normal(0, sigma, pointcloud2.shape)  # Add gaussian noise.
            else:
                sigma = 0.04 * np.random.random_sample()  # Generate random variance value b/w 0 to 0.1
                pointcloud2 = pointcloud2 + np.random.normal(0, sigma, pointcloud2.shape)  # Add gaussian noise.

            pointcloud1 = np.random.permutation(pointcloud1[:, :self.num_points].T).T
            pointcloud2 = np.random.permutation(pointcloud2[:, :self.num_points].T).T

        elif self.noise == '':
            pointcloud1 = np.random.permutation(pointcloud1[:, :self.num_points].T).T
            pointcloud2 = np.random.permutation(pointcloud2[:, :self.num_points].T).T

        elif self.noise == 'zero_intersec':
            pointcloud1 = np.random.permutation(pointcloud1[:, :self.num_points].T).T
            pointcloud2 = np.random.permutation(pointcloud2[:, self.num_points:].T).T
        else:
            pointcloud1 = pointcloud1[:, :self.num_points]
            pointcloud2 = pointcloud2[:, np.random.permutation(pc_size)[:self.num_points]]

        return pointcloud1.astype('float32'), pointcloud2.astype('float32'), R_ab.astype('float32'), \
            translation_ab.astype('float32')

    def __len__(self):
        return len(self.data)

