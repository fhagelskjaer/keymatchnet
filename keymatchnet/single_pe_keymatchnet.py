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

from model import DGCNN_gpvn
from model import DGCNN_gpvn_obj
from model import DGCNN_gpvn_purenet

import numpy as np

from util import get_loss, IOStream
import sklearn.metrics as metrics
import json
import open3d as o3d
from sklearn.neighbors import KDTree

from data import filterPoints, pc_center2cp, normalize_1d, normalize_2d
import copy
from distinctipy import distinctipy

import time
import sys

from dataloader_pe import *

from pe_utils import *

def select_point(pcd):
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(
        window_name="Open3D -- Select object",
        width=1920 // 2,
        height=1080,
        left=1920 // 2,
        top=0,
        visible=True,
    )
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    return vis.get_picked_points()

def test(args):
    np.set_printoptions(suppress=True)

    torch.autograd.set_detect_anomaly(True)

    num_point = args.num_points

    num_key = args.num_key
    
    cad_string = args.obj

    cloud_name = args.scene

    """ new visualization """
    input_colors = [(1, 1, 1), (0, 0, 0)]
    colors_float = distinctipy.get_colors(num_key, input_colors)
    """ end new visualization """

    ### Load data model
    device = torch.device("cuda" if args.cuda else "cpu")

    model = DGCNN_gpvn(args.k, args.emb_dims, num_key, 0).to(device)
    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model.load_state_dict(torch.load(args.model_root))
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
   

    ### Create object data
    candidate = trimesh.load_mesh(cad_string)
    candidate.export('stuff.stl')
    candidate_extents_max = np.max(candidate.extents)

    radius = np.max(candidate_extents_max)

    obj_cad = o3d.io.read_triangle_mesh('stuff.stl')

    obj_pc = obj_cad.sample_points_poisson_disk(2048)

    obj_pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30), False)

    object_xyz = np.asarray(obj_pc.points)

    # TODO: compute the point cloud
    x = torch.FloatTensor(object_xyz)
    x = np.reshape(x, (1, -1, 3))
    fpi = farthest_point_sampler(x, num_key)

    object_xyz_feature = object_xyz[fpi[0], :]

    obj_pc_temp = copy.deepcopy(obj_pc)
    # R = obj_pc_temp.get_rotation_matrix_from_xyz(
    #     (np.random.uniform(0, np.pi * 2), np.random.uniform(0, np.pi * 2), np.random.uniform(0, np.pi * 2)))
    # obj_pc_temp.rotate(R, center=(0, 0, 0))
    model_pc_out = np.concatenate([np.asarray(obj_pc_temp.points), np.asarray(obj_pc_temp.normals)], axis=1)

    model_pc_out = normalize_2d(model_pc_out)

    obj_model = obj_pc
    obj = model_pc_out.astype('float32')
    fpi = fpi[0].cpu().numpy()

    mm_dist = mm_by_keypoint(np.asarray(obj_model.points)[fpi, :3])*4

    ### Create scene data
    scene_cloud = o3d.io.read_point_cloud(cloud_name)
    scene_cloud = scene_cloud.voxel_down_sample(0.5)
    scene_cloud.estimate_normals( o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=30))
    scene_cloud.orient_normals_to_align_with_direction(orientation_reference=np.array([0.0, 0.0, -1.0]))
    
    scene_pointcloud = np.concatenate([np.asarray(scene_cloud.points), np.asarray(scene_cloud.normals)], axis=1)

    # point_check = [ scene_pointcloud[2000, :3]]
    try:
        point_xyz = args.point_xyz
        # print( point_xyz.split(",") )
        x, y, z = point_xyz.split(",")
        # print( x, y, z )
        x, y, z = float(x), float(y), float(z)
        # print( x, y, z )
        point_check = [[x, y, z ]]
    except:
        point_check = [scene_pointcloud[ select_point(scene_cloud)[0], :3] ]
    print( point_check )


    centerTreeFilter = KDTree([point_check[0]], leaf_size=2)
    distFilter, indeciesFilter = centerTreeFilter.query(scene_pointcloud[:, :3], k=1)
    pointlist = scene_pointcloud[distFilter.flatten() < radius, :]

    if len(pointlist) < 200:
        return

    while len(pointlist) < num_point:
        pointlist = np.array(list(pointlist) + list(pointlist))
    np.random.shuffle(pointlist)

    point_list = pointlist

    data = pc_center2cp(np.array(point_list[:num_point])[:, :6], point_check[0])  # point_array[:NUM_POINT,:6]
    data, _ = normalize_1d(data)
    data = data.astype('float32')
    scene_info = np.array(point_list[:num_point])[:, :6]


    scene_pc = o3d.geometry.PointCloud()
    scene_pc.points = o3d.utility.Vector3dVector(np.array(scene_info)[:, :3])
    scene_pc.normals = o3d.utility.Vector3dVector(np.array(scene_info)[:, 3:6])

    data, obj, fpi = torch.from_numpy(np.array([data])), torch.from_numpy(np.array([obj])), torch.from_numpy(np.array([fpi]))
    data, obj, fpi = data.to(device), obj.to(device), fpi.to(device)

    data = data.permute(0, 2, 1)
    obj = obj.permute(0, 2, 1)

    seg_pred, key_pred = model(data, obj, fpi, device)

    ### Prepare data for network and run through

    result, score = compute(seg_pred, [scene_info], key_pred, [obj_model], fpi, num_key, args.vt, device, mm_dist=mm_dist)

    print( "Score:", score)
    print( "Pose estimation:" )

    print( result )

    if args.visu == 'True':
            seg_pred_np = seg_pred.detach().cpu().numpy()
            key_pred_np = key_pred.cpu().numpy()
            fpi_np = fpi.detach().cpu().numpy()
            
            seg_pred_max_np = np.argmax(seg_pred_np,axis=1)
            key_pred_max_np = np.argmax(key_pred_np,axis=1)

            obj_model_show = copy.deepcopy(obj_model)

            obj_model_points = np.asarray(obj_model_show.points)
            object_xyz_feature = obj_model_points[fpi_np[0], :]
            treeLocalFeature = KDTree(object_xyz_feature, leaf_size=2)
            dist, indecies = treeLocalFeature.query(obj_model_points[:, :3], k=1)
            obj_colors = []
            for point_i in range(len(indecies)):
                obj_colors.append(colors_float[indecies[point_i][0]])
            obj_model_show.colors = o3d.utility.Vector3dVector(np.array(obj_colors))
            
            o3d.visualization.draw_geometries([obj_model_show])
            # o3d.io.write_point_cloud("pcviz/object_colored.pcd", obj_model_show)
            
            scene_pc_show = o3d.geometry.PointCloud()
            scene_pc_show.points = o3d.utility.Vector3dVector(np.array(scene_info)[:, :3])
            # scene_pc_show.colors = o3d.utility.Vector3dVector(np.array(scene_info)[:, 6:])
            scene_pc_show.normals = o3d.utility.Vector3dVector(np.array(scene_info)[:, 3:6])
             
            # o3d.io.write_point_cloud("pcviz/scene_pc_color.pcd", scene_pc_show)

            scene_colors = []
            for point_i in range(2048):
                if seg_pred_max_np[0, point_i] == 1:
                    scene_colors.append(colors_float[key_pred_max_np[0, point_i]])
                else:
                    scene_colors.append((0, 0, 0))
            scene_pc_show.colors = o3d.utility.Vector3dVector(np.array(scene_colors))
            
            o3d.visualization.draw_geometries([scene_pc_show])
            # o3d.io.write_point_cloud("pcviz/scene_colored.pcd", scene_pc_show)

            obj_model_show.transform(result)
            obj_model_show.paint_uniform_color([0, 0.90, 0])

            # scene_pc.paint_uniform_color([1, 0.70, 0])
            # o3d.io.write_point_cloud("pcviz/scene_pose.pcd", scene_pc_show)
            # o3d.io.write_point_cloud("pcviz/object_pose.pcd", obj_model_show)
            o3d.visualization.draw_geometries([scene_pc_show, obj_model_show])



if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Part Segmentation')
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
    parser.add_argument('--verbose', type=bool, default=False,
                        help='Verbose mode')
    parser.add_argument('--num_key', type=int, default=20, metavar='num_key',
                        help='Num of key points for the model')
    parser.add_argument('--vt', type=float, default=0.7, metavar='vt',
                        help='Voting threshold')
    parser.add_argument('--point_xyz', type=str, default='', metavar='N',
                        help='position of point to find, e.g. 5.1,52.5,490 ')
    parser.add_argument('--obj', type=str, default='testdata/09_BGPSL6-9-L30-F7.stl', metavar='N',
                        help='object model')
    parser.add_argument('--scene', type=str, default='testdata/point_cloud.pcd', metavar='N',
                        help='scene point cloud')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    test(args)

