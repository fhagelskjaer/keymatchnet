#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Frederik Hagelskjaer
@Contact: frhag@mmmi.sdu.dk
@File: pe_gpvn.py
@Time: 2023/3/1 10:00 AM
"""

from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import open3d as o3d
from sklearn.neighbors import KDTree

import copy

import time

def sample_corr_triplets(corr_world: torch.tensor, corr_obj: torch.tensor,
                         sample_factor=1):
    r"""
    From 3D-3D correspondences, sample triplets where the triangle spanned by world coordinates is similar to the
    triangle spanned by object coordinates.
    :returns: corr (2[world,obj], m, 3corr, 3xyz), loss (m,) - sorted by loss
    """
    n, d = corr_world.shape
    assert d == 3 and corr_obj.shape == corr_world.shape

    corr = torch.stack(torch.broadcast_tensors(corr_world, corr_obj)).view(2, n, 3)
    m = n * sample_factor
    corr = corr[:, torch.randint(n, (m, 3), device=corr.device)]  # (2, m, 3corr, 3xyz)

    tri_side = corr[:, :, (1, 2, 2)] - corr[:, :, (0, 0, 1)]  # (2, m, 3(ab, ac, bc), 3xyz)
    tri_side_len = tri_side.norm(dim=-1)  # (2, m, 3)
    tri_side_len_diff = (tri_side_len[0] - tri_side_len[1]).norm(dim=-1)  # (m,)

    tri_area = torch.cross(tri_side[:, :, 0], tri_side[:, :, 1], dim=-1).norm(dim=-1)  # (2, m)
    min_tri_height = (tri_area / (tri_side_len.max(dim=-1)[0] + 1e-9)).min(dim=0)[0]  # (m,)
    loss = tri_side_len_diff / (min_tri_height + 1e-9)

    idx_sort = torch.argsort(loss)
    return corr[:, idx_sort], loss[idx_sort]


def kabsch(P: torch.tensor, Q: torch.tensor):
    """
    Q = R @ P + t
    """
    n, _, m = P.shape  # n sets of m 3D correspondences
    assert m >= 3
    assert P.shape == Q.shape == (n, 3, m), (P.shape, Q.shape)
    mp, mq = P.mean(dim=-1, keepdim=True), Q.mean(dim=-1, keepdim=True)
    P_, Q_ = P - mp, Q - mq
    C = Q_ @ P_.mT  # (n, 3, 3)
    u, _, v = torch.svd(C)  # (n, 3, 3), (n, 3), (n, 3, 3)
    s = torch.ones(n, 3, 1, device=P.device)
    s[:, 2, 0] = torch.det(u @ v.mT)
    R = u @ (s * v.mT)  # (n, 3, 3)
    t = mq - R @ mp  # (n, 3, 1)
    return R, t


def addi(pose, acc_pose, source):
    source_temp = copy.deepcopy(source)
    source_test_temp = copy.deepcopy(source)
    source_temp.transform(acc_pose)
    source_test_temp.transform(pose)
    tree_local = KDTree(np.asarray(source_test_temp.points), leaf_size=2)
    dist, index = tree_local.query(np.asarray(source_temp.points), k=1)
    return np.mean(dist)


def add(pose, acc_pose, source):
    source_temp = copy.deepcopy(source)
    source_test_temp = copy.deepcopy(source)
    source_temp.transform(acc_pose)
    source_test_temp.transform(pose)
    # dist = (np.asarray(source_test_temp.points) - np.asarray(source_temp.points))*(np.asarray(source_test_temp.points) - np.asarray(source_temp.points))
    # return np.mean(dist)


def compute(seg_pred, scene_info, key_pred, obj_model, fpi, num_key, vt, device, mm_dist=16):

    use_normal = True
    triplet_sample_factor = 1
    n_pose_evals = 1000
    size_divider_list = [2, 3, 4, 5]

    seg_pred = seg_pred.permute(0, 2, 1).contiguous()
    key_pred = key_pred.permute(0, 2, 1).contiguous()

    pred = seg_pred.max(dim=2)[1]

    key_pred_max = key_pred.max(dim=2)[1]

    softmax_res = torch.softmax(key_pred, axis=1)

    softmax_max = torch.mul(softmax_res.max(dim=-1)[0], torch.tensor([vt]).to(device))
    batch_size, points = softmax_max.size()
    softmax_max = softmax_max.view(batch_size, points, 1)
    softmax_max = softmax_max.repeat(1, 1, num_key)
    softmax_threshold = softmax_res >= softmax_max

    pred_repeat = pred.view(batch_size, points, 1)
    pred_repeat = pred_repeat.repeat(1, 1, num_key)

    indecies = torch.where(torch.logical_and(softmax_threshold, pred_repeat))

    scene_data = torch.from_numpy(scene_info[0]).to(device)
    obj_data = torch.from_numpy(np.asarray(obj_model[0].points)).to(device)
    obj_normals = torch.from_numpy(np.asarray(obj_model[0].normals)).to(device)

    score_world = key_pred[indecies]

    # create the correspondences
    corr_world = scene_data[indecies[1][indecies[0] == 0], :3]
    corr_obj = obj_data[fpi[0][indecies[2][indecies[0] == 0]], :3]

    corr_world_normals = scene_data[indecies[1][indecies[0] == 0], 3:6]
    corr_obj_normals = obj_normals[fpi[0][indecies[2][indecies[0] == 0]], :3]

    if corr_world.shape[0] == 0:
        return np.eye(4), 0

    triplets, tri_loss = sample_corr_triplets(corr_world, corr_obj, sample_factor=triplet_sample_factor)
    triplets = triplets[:, :n_pose_evals]
    R, t = kabsch(triplets[1].mT, triplets[0].mT)

    corr_obj_proj = R @ corr_obj.permute(1, 0).reshape(1, 3, corr_obj.size(0)).repeat(t.size(0), 1, 1) + t
    inside = torch.sum(
        (corr_world.permute(1, 0).reshape(1, 3, corr_world.size(0)).repeat(t.size(0), 1, 1) - corr_obj_proj) ** 2,
        dim=-2)


    inside = inside < mm_dist

    # TODO
    if use_normal:
        corr_obj_normals_proj = R @ corr_obj_normals.permute(1, 0).reshape(1, 3, corr_obj.size(0)).repeat(t.size(0), 1, 1)
        inside_normal = torch.sum(
            corr_world_normals.permute(1, 0).reshape(1, 3, corr_obj.size(0)).repeat(t.size(0), 1, 1) *
            corr_obj_normals_proj, dim=1)
        inside_normal = torch.acos(inside_normal) < 0.52
        inside = torch.logical_and(inside, inside_normal)

    inside = torch.sum(inside, dim=-1)

    # max_index_list = torch.argsort(-inside, axis=0).detach().cpu().numpy()
    max_index_list = [torch.argmax(inside).detach().cpu().numpy()]

    best_transform = np.eye(4)

    best_transform_score = -np.inf # should be atleast 3

    for i in max_index_list[:10]:
        # Get the transformation of object points
        corr_obj_proj = R[i] @ corr_obj.permute(1, 0) + t[i]
        inside = torch.sum((corr_world - corr_obj_proj.permute(1, 0)) ** 2, dim=-1)
        inside = inside < mm_dist  # represents the distance in mm?

        # use normal vector?
        if use_normal:
            corr_obj_normals_proj = R[i] @ corr_obj_normals.permute(1,0)
            inside_normal = torch.acos(torch.sum(corr_world_normals.permute(1,0) * corr_obj_normals_proj, dim=0)) < 0.52
            inside = torch.logical_and(inside, inside_normal)
        sum_size = inside.sum()

        # refining the pose
        for size_divider in size_divider_list:
            if sum_size.detach().cpu().numpy() > 3:
                corr_obj_new = corr_obj[inside, :].permute(1, 0).view(1, 3, sum_size)
                corr_world_new = corr_world[inside, :].permute(1, 0).view(1, 3, sum_size)
                R_new, t_new = kabsch(corr_obj_new, corr_world_new)

                corr_obj_proj = R_new[0] @ corr_obj.permute(1, 0) + t_new[0]
                inside = torch.sum((corr_world - corr_obj_proj.permute(1, 0)) ** 2, dim=-1)
                inside = inside < mm_dist / size_divider  # Maybe this should decrease slowly
                
                sum_size = inside.sum()

                score_sum = score_world[inside].sum().detach().cpu().numpy()
            else:
                score_sum = -np.inf

        
        if score_sum > best_transform_score and sum_size.detach().cpu().numpy() > 3:
        
            Rnp, tnp = R_new.detach().cpu().numpy(), t_new.detach().cpu().numpy()
            transformation = np.eye(4)
            transformation[:3, :3] = Rnp[0]
            transformation[0, 3] = tnp[0][0]
            transformation[1, 3] = tnp[0][1]
            transformation[2, 3] = tnp[0][2]

            best_transform = transformation
            best_transform_score = score_sum

    return best_transform, best_transform_score

def mm_by_keypoint(keypoints):
    dist_list = []
    for key in keypoints[1:]:
        dist_list.append(np.sqrt(np.sum(np.power(key - keypoints[0], 2))) )
    return np.min(dist_list)

