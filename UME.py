#!/usr/bin/env python
# -*- coding: utf-8 -*-


from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader

from data import Data
from util import chamfer, transform_point_cloud, RRMSE


def DFMC(pc):
    """
    :param pc: Bx3xN
    :return: for each point in pc, its distance from mass-center: Bx1xN
    """
    return torch.norm(pc - torch.mean(pc, dim=2, keepdim=True),
                      dim=1, keepdim=True)


def hard_indicators(point_features, num_ws=8):
    """
    :param point_features: Bx1xN (f(pc))
    :param num_ws: number of indicator functions
    :return: indicators function activation: BxNxnum_ws (w(f(pc)))
    """
    point_features = point_features.squeeze(1)
    batch_size, num_point = point_features.shape

    # Create the result of caracteristic W functions "on" (composition) pc
    ws_composition_pc = torch.zeros([batch_size, num_point, num_ws], device=point_features.device)

    for object in range(batch_size):
        min_val = torch.min(point_features[object, :]).item()
        max_val = torch.max(point_features[object, :]).item()
        quantization_levels = torch.linspace(start=min_val, end=max_val, steps=num_ws + 1)
        for w in range(num_ws):
            lower_tensor = torch.ge(point_features[object, :], quantization_levels[w])
            upper_tensor = torch.lt(point_features[object, :], quantization_levels[w + 1])
            in_range = torch.logical_and(lower_tensor, upper_tensor)
            ws_composition_pc[object, :, w] = in_range.type(torch.float32)

    return ws_composition_pc


def ume_matrix(pc, ws_activations, num_ws=8):
    """
    :param pc: Bx3xN
    :param ws_activations: BxNxnum_ws
    :return: ume_matrix: Bxnum_wsx3
    """

    def calc_ume_col(coordinates, ws_activations, num_ws):  # do point wize multiplication
        return torch.sum(coordinates * ws_activations, dim=1).view(-1, num_ws)

    def create_coordinates_matrix(pc, axis, num_ws):
        return pc[:, axis, :].unsqueeze(2).repeat(1, 1, num_ws)

    batch_size, _, num_points = pc.shape
    ume_matrix = torch.zeros(size=[batch_size, num_ws, 3], device=pc.device)
    pc = pc - torch.mean(pc, dim=2, keepdim=True)

    for i in range(3):
        ume_matrix[:, :, i] = calc_ume_col(create_coordinates_matrix(pc, i, num_ws), ws_activations, num_ws)

    return ume_matrix


def ume_no_indicators(pc, pc_inv_func):
    pc_inv_func = pc_inv_func.permute(0, 2, 1).contiguous()
    return ume_matrix(pc, pc_inv_func, num_ws=pc_inv_func.shape[2]) / \
           pc.shape[2]


def horn_for_ume(pc1, pc1_ume, pc2, pc2_ume, pc1_mass=None, pc2_mass=None):
    pc1_ume = pc1_ume.permute(0, 2, 1).contiguous()
    pc2_ume = pc2_ume.permute(0, 2, 1).contiguous()
    batch_size, _, cols = pc1_ume.shape

    l_points = pc2_ume
    r_points = pc1_ume

    center_r = torch.mean(r_points, dim=2, keepdim=True)
    center_l = torch.mean(l_points, dim=2, keepdim=True)

    r_points = r_points - center_r
    l_points = l_points - center_l

    M = torch.matmul(l_points, r_points.permute(0, 2, 1).contiguous())

    [U, _, V] = torch.svd(M)
    c = torch.eye(3, device=pc1_ume.device).unsqueeze(0).repeat(batch_size, 1, 1)
    c[:, 2, 2] = torch.det(torch.matmul(U, V.permute(0, 2, 1).contiguous()))
    R = torch.matmul(torch.matmul(U, c), V.permute(0, 2, 1).contiguous())

    if (pc1_mass is None) and (pc2_mass is None):
        t = torch.mean(pc2, dim=2, keepdim=True) - torch.matmul(R, torch.mean(pc1, dim=2, keepdim=True))
    else:
        t = pc2_mass - torch.matmul(R, pc1_mass)

    return R, t.view(batch_size, 3)


def UME(type='ModelNet40', noise='sampling', batch_size=1):
    testset = Data(type=type, partition='test', noise=noise)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, drop_last=False)

    total_chamfer = 0
    total_maxmin = 0
    total_R_RMSE = 0
    total_t_RMSE = 0
    num_examples = 0
    for i, (src, target, rotation_ab, translation_ab) in enumerate(tqdm(test_loader)):

        if noise == 'bernoulli':
            q1, q2 = np.random.uniform(low=0.2, high=1, size=2)
            indexes_1 = [bool(np.random.binomial(n=1, p=q1)) for _ in range(1024)]
            indexes_2 = [bool(np.random.binomial(n=1, p=q2)) for _ in range(1024)]
            # stay with at least 200 points
            while sum(indexes_1) < 200:
                indexes_1 = [bool(np.random.binomial(n=1, p=q1)) for _ in range(1024)]
            while sum(indexes_2) < 200:
                indexes_2 = [bool(np.random.binomial(n=1, p=q2)) for _ in range(1024)]
            src = src[:, :, indexes_1]
            target = target[:, :, indexes_2]

        src = src.cuda()
        target = target.cuda()

        rotation_ab = rotation_ab.cuda()
        translation_ab = translation_ab.cuda()

        batch_size = src.size(0)
        num_examples += batch_size

        # UME horn
        num_ws = 3
        pc1_ume = f.ume_matrix(src, f.hard_indicators(f.DFMC(src), num_ws=num_ws), num_ws=num_ws)
        pc2_ume = f.ume_matrix(target, f.hard_indicators(f.DFMC(target), num_ws=num_ws), num_ws=num_ws)
        rotation_ab_pred, translation_ab_pred = f.horn_for_ume(src, pc1_ume, target, pc2_ume)

        chamf, maxmin = chamfer(target, transform_point_cloud(src, rotation_ab_pred, translation_ab_pred))
        R_RMSE = RRMSE(rotation_ab_pred, rotation_ab)
        t_RMSE = torch.sqrt(torch.mean((translation_ab_pred - translation_ab) ** 2))

        total_chamfer += chamf.item() * batch_size
        total_maxmin += maxmin.item() * batch_size
        total_R_RMSE += R_RMSE * batch_size
        total_t_RMSE += t_RMSE.item() * batch_size

    print('==FINAL TEST==')
    print('EPOCH:: %d, chamfer: %f, maxmin: %f, R_RMSE: %f, t_RMSE: %f'
          % (-1, total_chamfer * 1.0 / num_examples,
             total_maxmin * 1.0 / num_examples,
             total_R_RMSE * 1.0 / num_examples,
             total_t_RMSE * 1.0 / num_examples))


if __name__ == '__main__':
    UME(type='Stanford', noise='', batch_size=1)  # noise models: '', 'zero_intersec', 'bernoulli', 'gaussian'
