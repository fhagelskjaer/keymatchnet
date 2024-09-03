#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Frederik Hagelskjaer
@Contact: frhag@mmmi.sdu.dk
@File: pe_gpvn_helsinki.py
@Time: 2023/3/1 10:00 AM
"""


from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import DGCNN_gpvn
from model import DGCNN_gpvn_obj
from model import DGCNN_gpvn_purenet

import numpy as np

from util import get_loss, IOStream
import sklearn.metrics as metrics
import json
import open3d as o3d
import trimesh
from dgl.geometry import farthest_point_sampler
from sklearn.neighbors import KDTree

from data import filterPoints, pc_center2cp, normalize_1d, normalize_2d
import scipy
import copy
from distinctipy import distinctipy

import time

from dataloader_pe import *
from pe_utils import add, addi

def test(args):
    np.set_printoptions(suppress=True)

    torch.autograd.set_detect_anomaly(True)

    pcs = args.num_points

    num_key = args.num_key

    if args.dataset_name == "daen":
        dataloader = GPVN_data_engine(number_of_keypoints=num_key, partition='test')
    elif args.dataset_name == "pickleblu":
        dataloader = PickleData(number_of_keypoints=num_key, cad_string="../../pointposer/cap2-6_remesh.ply", pickle_file="../../pointposer/blu__pose_cloud_pairs.pickle" )
    elif args.dataset_name == "picklecap":
        dataloader = PickleData(number_of_keypoints=num_key, cad_string="../../pointposer/1673308_mm.ply", pickle_file="../../pointposer/cap__pose_cloud_pairs.pickle" )
    else:
        dataloader = GPVN_wrs(number_of_keypoints=num_key, pointcloud_size=pcs, dataset_name=args.dataset_name, single_fpi=args.single_fpi)

    """ new visualization """
    input_colors = [(1, 1, 1), (0, 0, 0)]
    colors_float = distinctipy.get_colors(num_key, input_colors)
    """ end new visualization """

    device = torch.device("cuda" if args.cuda else "cpu")

    if not args.single_fpi:
        model = DGCNN_gpvn(args.k, args.emb_dims, num_key, 0).to(device)
        model = nn.DataParallel(model)
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model.load_state_dict(torch.load(args.model_root))
    else:
        model_obj = DGCNN_gpvn_obj(args.k, args.emb_dims, num_key, 0).to(device)
        model_obj = nn.DataParallel(model_obj)
        model_obj.load_state_dict(torch.load(args.model_root), strict=False)

        model_run = DGCNN_gpvn_purenet(args.k, args.emb_dims, num_key, 0).to(device)
        model_run = nn.DataParallel(model_run)
        model_run.load_state_dict(torch.load(args.model_root), strict=False)

        datareturn = dataloader.get_item(0)
        data, seg, key, obj, fpi, obj_model, scene_info, gt_poses, radius = datareturn
        data, obj, seg, key, fpi = torch.from_numpy(data), torch.from_numpy(obj), torch.from_numpy(
            seg), torch.from_numpy(
            key), torch.from_numpy(fpi)
        obj = obj.permute(0, 2, 1)
        objfeature = model_obj(obj, fpi)

    mean_dist = []

    success = []

    # for _ in range(10):
    for item in range(dataloader.len()):

        datareturn = dataloader.get_item(item)

        if datareturn is None:
            continue
        data, seg, key, obj, fpi, obj_model, scene_info, gt_poses, radius = datareturn

        data, obj, seg, key, fpi = torch.from_numpy(data), torch.from_numpy(obj), torch.from_numpy(
            seg), torch.from_numpy(
            key), torch.from_numpy(fpi)

        data, obj, seg, key, fpi = data.to(device), obj.to(device), seg.to(device), key.to(device), fpi.to(device)
        data = data.permute(0, 2, 1)
        obj = obj.permute(0, 2, 1)
        batch_size = data.size()[0]

        start_time = time.time()

        if args.single_fpi:
            seg_pred, key_pred = model_run(data, objfeature)
        else:
            seg_pred, key_pred = model(data, obj, fpi, device)

        seg_pred = seg_pred.permute(0, 2, 1).contiguous()
        key_pred = key_pred.permute(0, 2, 1).contiguous()

        pred = seg_pred.max(dim=2)[1]
        seg_np = seg.cpu().numpy()
        pred_np = pred.detach().cpu().numpy()

        key_np = key.cpu().numpy()
        key_val = key_pred.detach().cpu().numpy()
        key_pred = key_pred.max(dim=2)[1]
        key_pred_np = key_pred.detach().cpu().numpy()

        fpi_np = fpi.detach().cpu().numpy()

        corres = o3d.utility.Vector2iVector()

        softmax_res = scipy.special.softmax(key_val[0], axis=1)
        softmax_max = np.max(softmax_res, axis=1)

        

        for point_i in range(2048):
            if pred_np[0][point_i] == 1:
                for key_i in range(num_key):
                    if softmax_res[point_i, key_i] > softmax_max[point_i] * args.vt:
                        corres.append(np.array([fpi_np[0][key_i], point_i]))

        
        scene_pc = o3d.geometry.PointCloud()
        scene_pc.points = o3d.utility.Vector3dVector(np.array(scene_info[0])[:, :3])
        # scene_pc.colors = o3d.utility.Vector3dVector(np.array(scene_info[0])[:, 6:])
        scene_pc.normals = o3d.utility.Vector3dVector(np.array(scene_info[0])[:, 3:6])



        if args.visu == 'True' or args.visu_idx == item:
            o3d.visualization.draw_geometries([scene_pc])

        if args.classic:
            # if True:
            radius_feature = 10
            obj_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                obj_model[0],
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
            scene_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                scene_pc,
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

            distance_threshold = 5
            result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                obj_model[0], scene_pc, obj_fpfh, scene_fpfh, True,
                distance_threshold,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                3, [
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                        distance_threshold)
                ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
        else:
            result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
                obj_model[0], scene_pc, corres,
                max_correspondence_distance=5,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                # estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(True),
                ransac_n=3,
                checkers=[
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(5),
                    # o3d.pipelines.registration.CorrespondenceCheckerBasedOnNormal(np.pi/4)
                ],
                # criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
                criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(5000, 0.999))

        if args.icp:
            result = o3d.pipelines.registration.registration_icp(
                obj_model[0], scene_pc, 5, result.transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=10))

        distance = []
        for gt_i in range(len(gt_poses)):
            gt_transform = np.eye(4)
            gt_transform[:3, 3] = gt_poses[gt_i][0]
            gt_transform[:3, :3] = gt_poses[gt_i][1]
            distance.append(addi(result.transformation, gt_transform, obj_model[0]))
        
        mean_dist.append(np.min(distance))
        success.append(np.min(distance) < dataloader.test_radius * 0.10)

        if args.verbose or args.visu_idx == item:
            print("Item: ", item)
            print(result.transformation)
            print(distance)
            print(np.min(distance))

        if args.visu == 'True' or args.visu_idx == item:
            obj_model_show = copy.deepcopy(obj_model[0])

            """ new visualization """
            obj_model_points = np.asarray(obj_model_show.points)
            object_xyz_feature = obj_model_points[fpi_np[0], :]
            treeLocalFeature = KDTree(object_xyz_feature, leaf_size=2)
            dist, indecies = treeLocalFeature.query(obj_model_points[:, :3], k=1)
            obj_colors = []
            for point_i in range(len(indecies)):
                obj_colors.append(colors_float[indecies[point_i][0]])
            obj_model_show.colors = o3d.utility.Vector3dVector(np.array(obj_colors))
            o3d.visualization.draw_geometries([obj_model_show])
            o3d.io.write_point_cloud("pcviz/object_colored.pcd", obj_model_show)
            """ new visualization """
            scene_pc_show = o3d.geometry.PointCloud()
            scene_pc_show.points = scene_pc.points


            seg_colors = []
            key_colors = []
            for point_i in range(2048):
                key_colors.append(colors_float[key_pred_np[0, point_i]])
                if pred_np[0, point_i] == 1:
                    seg_colors.append((1, 0, 0))
                else:
                    # scene_colors.append((0, 0, 0))
                    seg_colors.append((1, 1, 1))

            scene_pc_show.colors = o3d.utility.Vector3dVector(np.array(seg_colors))
            o3d.visualization.draw_geometries([scene_pc_show])
            o3d.io.write_point_cloud("seg_colored.pcd", scene_pc_show) 
            scene_pc_show.colors = o3d.utility.Vector3dVector(np.array(key_colors))
            o3d.visualization.draw_geometries([scene_pc_show])
            o3d.io.write_point_cloud("key_colored.pcd", scene_pc_show)


            scene_colors = []
            for point_i in range(2048):
                if pred_np[0, point_i] == 1:
                    scene_colors.append(colors_float[key_pred_np[0, point_i]])
                else:
                    # scene_colors.append((0, 0, 0))
                    scene_colors.append((1, 1, 1))
            scene_pc_show.colors = o3d.utility.Vector3dVector(np.array(scene_colors))
            o3d.visualization.draw_geometries([scene_pc_show])
            o3d.io.write_point_cloud("pcviz/scene_colored.pcd", scene_pc_show)
            """ new visualization end"""
            """ end new visualization """

            obj_model_show.transform(result.transformation)
            obj_model_show.paint_uniform_color([0, 0.90, 0])
            
            # scene_pc.paint_uniform_color([1, 0.70, 0])

            o3d.visualization.draw_geometries([scene_pc, obj_model_show])
            
            o3d.io.write_point_cloud("pcviz/scene_pose.pcd", scene_pc)
            o3d.io.write_point_cloud("pcviz/object_pose.pcd", obj_model_show)
            # if args.visu == 'True':
            data = data.permute(0, 2, 1)


            for batch_visu in range(batch_size):
                target = o3d.geometry.PointCloud()

                if batch_visu == 0:
                    target.points = scene_pc.points
                else:
                    continue
                    target.points = o3d.utility.Vector3dVector(data.cpu().numpy()[batch_visu, :, :3])
                # target.normals = o3d.utility.Vector3dVector(np.array(point_list)[:,3:6])
                color = np.zeros((data.shape[1], 3), np.float64)
                # print( key_pred_np.shape )
                cutoff = [0, 0]
                seg_cutoff = [0, 0]
                for point_i in range(2048):
                    if pred_np[batch_visu, point_i] == 1 and seg_np[batch_visu, point_i] == 1:
                        seg_cutoff[0] += 1
                        # print( key_np[batch_visu,point_i], key_pred_np[batch_visu,point_i], seg_np[batch_visu,point_i] )

                        if key_np[batch_visu, point_i] == key_pred_np[batch_visu, point_i]:
                            # if( key_val[batch_visu,point_i,key_np[batch_visu,point_i]] > key_val[batch_visu,point_i,key_pred_np[batch_visu,point_i]]*0.9 ):
                            color[point_i, :] = [0, 255.0, 0]  # Green
                            cutoff[0] += 1
                        else:
                            color[point_i, :] = [0, 0, 255.0]  # Blue
                            cutoff[1] += 1
                            # print( key_val[batch_visu,point_i,key_np[batch_visu,point_i]], key_val[batch_visu,point_i,key_pred_np[batch_visu,point_i]] )
                            # print( key_val[batch_visu,point_i,:] )
                    elif pred_np[batch_visu, point_i] == 0 and seg_np[batch_visu, point_i] == 0:
                        seg_cutoff[0] += 1
                        # color[point_i, :] = [0, 0, 0]  # Black
                        color[point_i, :] = [255, 255, 255]  # Black
                    else:
                        seg_cutoff[1] += 1
                        if seg_np[batch_visu, point_i] == 1:  # if it is real object but missed
                            color[point_i, :] = [255.0, 0, 0]  # Red
                        else:
                            color[point_i, :] = [255, 165, 0]

                print("Red: sfn, Yellow: sfp, Black: stn, Blue: stp, Green: stp&correctvote")
                print("Segmentation percent correct", seg_cutoff[0] / (seg_cutoff[0] + seg_cutoff[1]))
                if (cutoff[0] + cutoff[1] == 0):
                    print("None")
                else:
                    print("percentage correct", cutoff[0] / (cutoff[0] + cutoff[1]))

                target.colors = o3d.utility.Vector3dVector(color)

                o3d.visualization.draw_geometries([target])
                o3d.io.write_point_cloud("pcviz/scene_match.pcd", target)

    print(args.dataset_name)
    print("Num key", num_key)
    print(radius)
    print(dataloader.test_radius)
    print(np.sum(success),len(success))
    print(np.sum(success)/len(success))
    print(np.histogram(mean_dist))


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Part Segmentation')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='num of points to use')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_root', type=str, default='', metavar='N',
                        help='Pretrained model root')
    parser.add_argument('--visu', type=str, default='',
                        help='visualize the model')
    parser.add_argument('--visu_idx', type=int, default=-1, metavar='N',
                        help='The idx to visualize, if negative all are selected')
    parser.add_argument('--dataset_name', type=str, default='wrs24',
                        help='Name of the dataset to test')
    parser.add_argument('--single_fpi', type=bool, default=False,
                        help='Compute only a single fpi')
    parser.add_argument('--icp', type=bool, default=False,
                        help='Use ICP')
    parser.add_argument('--classic', type=bool, default=False,
                        help='Use the classic fpfh pose estimation method')
    parser.add_argument('--verbose', type=bool, default=False,
                        help='Verbose mode')
    parser.add_argument('--num_key', type=int, default=20, metavar='num_key',
                        help='Num of key points for the model')
    parser.add_argument('--vt', type=float, default=0.7, metavar='vt',
                        help='Voting threshold')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    test(args)

