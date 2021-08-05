#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function
import torch
import numpy as np
import math

try:
    from .chamfer_distance import ChamferDistance
except (ModuleNotFoundError, ImportError) as err:
    from chamfer_distance import ChamferDistance

# Part of the code is referred from: https://github.com/WangYueFt/dcp, https://github.com/itailang/SampleNet,
# https://www.codegrepper.com/code-examples/python/Rotation+matrix+to+Euler+angles+Python.


def quat2mat(quat):
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat


def transform_point_cloud(point_cloud, rotation, translation):
    return torch.matmul(rotation, point_cloud) + translation.unsqueeze(2)


def reverse_transform_point_cloud(point_cloud, rotation, translation):
    return torch.matmul(rotation.permute(0, 2, 1).contiguous(), point_cloud - translation.unsqueeze(2))


def pca(pc, directions=None):
    # A is of size B (batch) X n (features) X m (samples)
    b, n, m = pc.shape
    mass = torch.mean(pc, dim=2, keepdim=True)
    pc = pc - mass
    cov_pc = torch.matmul(pc, pc.permute(0, 2, 1).contiguous()) / (m - 1)
    e, v = torch.symeig(cov_pc, eigenvectors=True)

    if directions is not None:
        signs = torch.sign(
            torch.diagonal(torch.matmul(directions.permute(0, 2, 1).contiguous(), v), dim1=1, dim2=2)).unsqueeze(1)
        v *= signs

    return pc, v, mass


def invariant_coordinates_pca_chamfer(pc1, pc2):
    pc1_centered, vecs1, mass1 = pca(pc1)
    pc2_centered, vecs2, mass2 = pca(pc2)

    batch_size = pc1.shape[0]
    device = pc1.device
    totensor = lambda vec: torch.tensor(vec, device=device)

    quadrants = totensor([[1, 1, 1], [-1, -1, -1], [-1, 1, 1], [1, -1, 1],
                          [1, 1, -1], [-1, -1, 1], [1, -1, -1], [-1, 1, -1]])

    def project(pc, axes):
        if len(pc.shape) == 3:
            return torch.matmul(axes.permute(0, 2, 1).contiguous(), pc)
        if len(pc.shape) == 2:
            return torch.matmul(axes.T, pc)
        else:
            assert 'pc shape length is either 2 or 3'

    projected1 = project(pc1_centered, vecs1)
    new_vecs2 = torch.zeros_like(vecs2)
    projected2 = torch.zeros_like(pc2_centered)
    quadrants_mats = torch.diag_embed(quadrants).type(torch.float).to(device)

    for i in range(batch_size):
        distances = []
        for idx, option in enumerate(quadrants_mats):
            distances.append(chamfer(projected1[i].unsqueeze(0),
                                     project(pc2_centered[i], torch.matmul(vecs2[i], option)).unsqueeze(0))[0])
        chosen_quadrant = torch.argmin(totensor(distances))
        new_vecs2[i] = torch.matmul(vecs2[i], quadrants_mats[chosen_quadrant])
        projected2[i] = project(pc2_centered[i], new_vecs2[i])

    return projected1, vecs1, mass1, \
           projected2, new_vecs2, mass2


def chamfer(pc1, pc2):
    # ref_pc and samp_pc are B x N x 3 matrices
    if pc1.shape[2] is not 3:
        pc1 = pc1.permute(0, 2, 1).contiguous()
        pc2 = pc2.permute(0, 2, 1).contiguous()

    cost_pc1_pc2, cost_pc2_pc1 = ChamferDistance()(pc1, pc2)
    max_cost = torch.mean(torch.max(cost_pc1_pc2, dim=1)[0] + torch.max(cost_pc2_pc1, dim=1)[0])  # furthest points
    cost_pc1_pc2 = torch.mean(cost_pc1_pc2)
    cost_pc2_pc1 = torch.mean(cost_pc2_pc1)
    loss = cost_pc1_pc2 + cost_pc2_pc1
    return loss, max_cost


def RRMSE(R_pred, R_gt):
    R_pred_euler = npmat2euler_fullrange(R_pred.detach().cpu().numpy())
    R_gt_euler = npmat2euler_fullrange(R_gt.detach().cpu().numpy())
    return np.sqrt(np.mean((np.degrees(R_pred_euler) - np.degrees(R_gt_euler)) ** 2))


def rotationMatrixToEulerAngles(R):
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    x = math.atan2(R[2, 1], R[2, 2])
    y = math.atan2(-R[2, 0], sy)
    z = math.atan2(R[1, 0], R[0, 0])
    return np.array([z, y, x])


def npmat2euler_fullrange(mats):
    eulers = []
    for i in range(mats.shape[0]):
        eulers.append(rotationMatrixToEulerAngles(mats[i]))
    return np.asarray(eulers, dtype='float32')


def timer(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    return int(hours), int(minutes), int(seconds)