#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Frederik Hagelskjaer
@Contact: frhag@mmmi.sdu.dk
@File: pe_gpvn.py
@Time: 2023/3/1 10:00 AM
"""
import argparse
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

import open3d as o3d
from sklearn.neighbors import KDTree

import copy
from distinctipy import distinctipy

from . import (
    model,
    dataloader_pe,
    data,
    pe_utils,
)


def test(args):
    np.set_printoptions(suppress=True)

    torch.autograd.set_detect_anomaly(True)

    mm_dist = 62

    pcs = args.num_points

    num_key = args.num_key
    
    if args.dataset_name == "pickleblu":
        dataloader = dataloader_pe.PickleData(number_of_keypoints=num_key, cad_string="MFE/cap2-6_remesh.ply", pickle_file="MFE/blu__pose_cloud_pairs.pickle" )
    elif args.dataset_name == "picklecap":
        dataloader = dataloader_pe.PickleData(number_of_keypoints=num_key, cad_string="MFE/1673308_mm.ply", pickle_file="MFE/cap__pose_cloud_pairs.pickle" )
    else:
        dataloader = dataloader_pe.WrsData(number_of_keypoints=num_key, pointcloud_size=pcs, dataset_name=args.dataset_name, single_fpi=args.single_fpi)

    """ new visualization """
    input_colors = [(1, 1, 1), (0, 0, 0)]
    colors_float = distinctipy.get_colors(num_key, input_colors)
    """ end new visualization """

    device = torch.device("cuda" if args.cuda else "cpu")

    if not args.single_fpi:
        network_model = model.DGCNN_gpvn(args.k, args.emb_dims, num_key, 0).to(device)
        network_model = nn.DataParallel(network_model)
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        network_model.load_state_dict(torch.load(args.model_root))
        network_model.eval()
        for param in network_model.parameters():
            param.requires_grad = False
    else:
        datareturn = dataloader.get_item(0)
        _, _, _, obj, fpi, obj_model, _, _, radius = datareturn
       
        mm_dist = pe_utils.mm_by_keypoint(np.asarray(obj_model[0].points)[fpi[0], :3])*4

        obj, fpi = torch.from_numpy(obj), torch.from_numpy(fpi)

        obj, fpi = obj.to(device), fpi.to(device)
        
        model_run = model.DGCNN_gpvn_purenet(args.k, args.emb_dims, num_key, 0).to(device)
        model_run = nn.DataParallel(model_run)
        model_run.load_state_dict(torch.load(args.model_root), strict=False)

        model_obj = model.DGCNN_gpvn_obj(args.k, args.emb_dims, num_key, 0).to(device)
        model_obj = nn.DataParallel(model_obj)
        model_obj.load_state_dict(torch.load(args.model_root), strict=False)

        obj = obj.permute(0, 2, 1)
        
        objfeature = model_obj(obj, fpi, device)

        objfeature.to(device)
        del model_obj
        torch.cuda.synchronize()

        model_run.eval()
        for param in model_run.parameters():
            param.requires_grad = False

    mean_dist = []
    success = []
    angle_dist = []

    one_success = 0
    success_list = []
    for item in range(dataloader.len()):
        success_list.append(0)

    for item in range(dataloader.len()):
      
      start_time = time.time()

      temp_success = 0
        
      if args.verbose:
          print( "Dataloading lasted", time.time() - start_time )

      for _ in range(args.repeat):

        if args.single_fpi:
              datareturn = dataloader.get_item_fast(item)
        else:
              datareturn = dataloader.get_item(item)
        if datareturn is None:
          continue
        start_time = time.time()

        if args.single_fpi:
            input_data, scene_info, gt_poses, radius = datareturn
           
            input_data = np.array(input_data)
            scene_info = np.array(scene_info)

            input_data = torch.from_numpy(input_data)
            input_data = input_data.to(device)
            input_data = input_data.permute(0, 2, 1)
            seg_pred, key_pred = model_run(input_data, objfeature, device)
        else:
            input_data, seg, key, obj, fpi, obj_model, scene_info, gt_poses, radius = datareturn
            mm_dist = pe_utils.mm_by_keypoint(np.asarray(obj_model[0].points)[fpi[0], :3])*4

            input_data, obj, seg, key, fpi = torch.from_numpy(input_data), torch.from_numpy(obj), torch.from_numpy(
                seg), torch.from_numpy(
               key), torch.from_numpy(fpi)
            input_data, obj, seg, key, fpi = input_data.to(device), obj.to(device), seg.to(device), key.to(device), fpi.to(device)
            input_data = input_data.permute(0, 2, 1)
            obj = obj.permute(0, 2, 1)

            seg_pred, key_pred = network_model(input_data, obj, fpi, device)

        batch_size = input_data.size()[0]

        if args.verbose:
            print( "Network lasted", time.time() - start_time )
        result, score = pe_utils.compute(seg_pred, scene_info, key_pred, obj_model, fpi, num_key, args.vt, device, mm_dist=mm_dist)

        if args.verbose:
            print("Pose estimation lasted", time.time() - start_time)

        start_time = time.time()
        distance = []
        distance_z = []
        for gt_i in range(len(gt_poses)):
            gt_transform = np.eye(4)
            
            # import pdb; pdb.set_trace()
        
            gt_transform[:3, 3] = gt_poses[gt_i][0]
            gt_transform[:3, :3] = gt_poses[gt_i][1]
            
            distance.append(pe_utils.addi(result, gt_transform, obj_model[0]))

        mean_dist.append(np.min(np.array(distance)))
        
        success.append(np.min(np.array(distance)) < dataloader.test_radius * 0.10)

        if np.min(distance) < dataloader.test_radius * 0.10:
            temp_success += 1
            success_list[item] += 1

        if args.verbose:
            print("Score estimation lasted", time.time() - start_time)

        if args.verbose or args.visu_idx == item:
            print("Item: ", item)
            print("T:", result, "d:", np.min(distance), "score:", score)
            print(distance)

        if args.visu == 'True' or args.visu_idx == item:
        # if np.min(distance) < dataloader.test_radius * 0.10 and distance_z[0] > 80:

            print(distance,distance_z)

            seg_np = seg.cpu().numpy()
            seg_pred_np = seg_pred.detach().cpu().numpy()
            key_pred_np = key_pred.cpu().numpy()
            key_val = key_pred.detach().cpu().numpy()
            fpi_np = fpi.detach().cpu().numpy()
            
            seg_pred_max_np = np.argmax(seg_pred_np,axis=1)
            key_pred_max_np = np.argmax(key_pred_np,axis=1)

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
            
            if False:
                o3d.visualization.draw_geometries([obj_model_show])
            o3d.io.write_point_cloud("pcviz/object_colored.pcd", obj_model_show)
            
            """ new visualization """
            scene_pc_show = o3d.geometry.PointCloud()
            scene_pc_show.points = o3d.utility.Vector3dVector(np.array(scene_info[0])[:, :3])
            # scene_pc_show.colors = o3d.utility.Vector3dVector(np.array(scene_info[0])[:, 6:])
            scene_pc_show.normals = o3d.utility.Vector3dVector(np.array(scene_info[0])[:, 3:6])
             
            o3d.io.write_point_cloud("pcviz/scene_pc_color.pcd", scene_pc_show)

            scene_colors = []
            for point_i in range(2048):
                if seg_pred_max_np[0, point_i] == 1:
                    scene_colors.append(colors_float[key_pred_max_np[0, point_i]])
                else:
                    scene_colors.append((1, 1, 1))
            scene_pc_show.colors = o3d.utility.Vector3dVector(np.array(scene_colors))
            
            if False:
                o3d.visualization.draw_geometries([scene_pc_show])
                o3d.io.write_point_cloud("pcviz/scene_colored.pcd", scene_pc_show)
            
            """ new visualization end"""
            """ end new visualization """

            obj_model_show.transform(result)
            obj_model_show.paint_uniform_color([0, 0.90, 0])

            # scene_pc.paint_uniform_color([1, 0.70, 0])
            o3d.io.write_point_cloud("pcviz/scene_pose.pcd", scene_pc_show)
            o3d.io.write_point_cloud("pcviz/object_pose.pcd", obj_model_show)
            o3d.visualization.draw_geometries([scene_pc_show, obj_model_show])


      if temp_success > 0:
          one_success += 1          
    print(args.dataset_name)
    print("Num key", num_key)
    print(radius)
    print(dataloader.test_radius)
    
    print(np.sum(success) / len(success), np.sum(success), len(success))


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
    parser.add_argument('--repeat', type=int, default=1,
                        help='Number of repeat')
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

