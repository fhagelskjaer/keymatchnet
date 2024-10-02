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
import trimesh
from dgl.geometry import farthest_point_sampler
from sklearn.neighbors import KDTree

from data import filterPoints, pc_center2cp, normalize_1d, normalize_2d
import scipy
import copy
from distinctipy import distinctipy

import time
import sys

import pandas as pd
import pickle
import glob

class PickleData:
    def __init__(self, number_of_keypoints, cad_string, pickle_file, radius=-1):
        self.full_dataset = []
        self.number_of_keypoints = number_of_keypoints
        self.object_set_set = {}
        candidate = trimesh.load_mesh(cad_string)
        candidate.export('stuff.stl')
        candidate_extents_max = np.max(candidate.extents)

        radius = np.max(candidate_extents_max)

        obj_cad = o3d.io.read_triangle_mesh('stuff.stl')

        obj_pc = obj_cad.sample_points_poisson_disk(2048)

        obj_pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30), False)


        model_pc = np.concatenate([np.asarray(obj_pc.points), np.asarray(obj_pc.normals)], axis=1)

        model_center = obj_pc.get_center()

        self.test_radius = np.sqrt(candidate.extents[0] ** 2 +
            candidate.extents[1] ** 2 +
            candidate.extents[2] ** 2)
        
        if radius < 0:
            radius = np.max(candidate_extents_max)
        else:
            pass
            # self.test_radius = radius
        


        label = "0"
        self.object_set_set[label] = {"obj_pc": obj_pc, "model_pc": model_pc, "radius": radius, "model_center": model_center}

        with open(pickle_file, 'rb') as f:
            object_poses = pickle.load(f)		
	
        for posepair in object_poses:
              scene = "MFE/" + posepair["scene"]
              transform = posepair["transform"]
              model_info = "0"
              # import pdb; pdb.set_trace()
              self.full_dataset.append({"cloud": scene, "label": model_info, "transform": transform})

    
    def get_item(self, item):
        NUM_POINT = 2048
        GOOD_DIST = 2
        BAD_DIST = 2
        while(True):

            object_set = self.object_set_set[self.full_dataset[item]["label"]]

            # scene_folder = object_set["scene_folder"]
            obj_pc = object_set["obj_pc"]
            model_pc = object_set["model_pc"]
            radius = object_set["radius"]
            model_center = object_set["model_center"]

            object_xyz = np.asarray(obj_pc.points)
            x = torch.FloatTensor(object_xyz)
            x = np.reshape(x, (1, -1, 3))
            fpi = farthest_point_sampler(x, self.number_of_keypoints)

            object_xyz_feature = object_xyz[fpi[0], :]

            cloud_name = self.full_dataset[item]["cloud"]

            # print(scene_folder, cloud_index)

            modelCloud = o3d.io.read_point_cloud(cloud_name)
            modelCloud = modelCloud.voxel_down_sample(0.5)
            modelCloud.estimate_normals( o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=30))
            modelCloud.orient_normals_to_align_with_direction(orientation_reference=np.array([0.0, 0.0, -1.0]))

            # print( self.full_dataset[item]["transform"] )

            # o3d.visualization.draw_geometries([modelCloud])

            # input_tree = KDTree(np.asarray(modelCloud.points), leaf_size=2)


            gt_poses = [self.full_dataset[item]["transform"]]

            all_newPL = []
            all_newPLFeature = []
            all_new_center = []
            for gt_pose in gt_poses:
                newPL = np.matmul(object_xyz, gt_pose[:3,:3].transpose())
                newPL = newPL + gt_pose[:3,3]

                newPLFeature = np.matmul(object_xyz_feature, gt_pose[:3,:3].transpose())
                newPLFeature = newPLFeature + gt_pose[:3,3]

                all_newPL.append(newPL)
                all_newPLFeature.append(newPLFeature)

                new_center = np.matmul(model_center, gt_pose[:3,:3].transpose())
                new_center = new_center + gt_pose[:3,3]
                all_new_center.append(new_center)

            newPL_2 = np.concatenate(all_newPL)
            newPLFeature_2 = np.concatenate(all_newPLFeature)

            treeLocal = KDTree(newPL_2, leaf_size=2)
            treeLocalFeature = KDTree(newPLFeature_2, leaf_size=2)

            input_pointcloud = np.concatenate([np.asarray(modelCloud.points), np.asarray(modelCloud.normals)], axis=1)
            input_pointcloud = input_pointcloud[input_pointcloud[:,2] < all_new_center[0][2] + radius*1.5]
            input_pointcloud = input_pointcloud[input_pointcloud[:,2] > all_new_center[0][2] - radius*1.5]
            input_pointcloud = input_pointcloud[input_pointcloud[:,1] < all_new_center[0][1] + radius*1.5]
            input_pointcloud = input_pointcloud[input_pointcloud[:,1] > all_new_center[0][1] - radius*1.5]
            input_pointcloud = input_pointcloud[input_pointcloud[:,0] < all_new_center[0][0] + radius*1.5]
            input_pointcloud = input_pointcloud[input_pointcloud[:,0] > all_new_center[0][0] - radius*1.5]

            # centerTree = KDTree(all_new_center, leaf_size=2)
            # dist, indecies = centerTree.query(input_pointcloud[:, :3], k=1)
            # input_pointcloud = input_pointcloud[dist.flatten() < radius*1.5, :]

            dist, indecies = treeLocal.query(input_pointcloud[:, :3], k=1)

            object_points = np.flatnonzero(dist < GOOD_DIST)

            # a = [treeLocal, treeLocalFeature, dist, indecies, object_points, object_xyz, input_pointcloud]
            
            """    
            color = np.zeros((len(input_pointcloud),3), float)
            color[:,2] = (dist < GOOD_DIST)[:,0] * 255
                # import pdb; pdb.set_trace()
            target = o3d.geometry.PointCloud()        
            target.points = o3d.utility.Vector3dVector(input_pointcloud[:,:3])
            target.colors = o3d.utility.Vector3dVector(color)
            o3d.visualization.draw_geometries([target]) 
            # """

            if (len(object_points) < 30):
                # print(item, self.full_dataset[item]["label"], self.full_dataset[item]["cloud"], self.full_dataset[item]["transform"]) 
                # print("return none")
                # obj_trans = copy.deepcopy(obj_pc).transform(self.full_dataset[item]["transform"])
                # o3d.visualization.draw_geometries([modelCloud, obj_trans])
                # return None

                item = np.random.randint(len(self.full_dataset))
            else:
                break


        # print( object_points.shape, dist.shape, indecies.shape )
        np.random.shuffle(object_points)
        # print(dist, ind,  point_check)
        # list_of_data = []

        l_data = []
        l_seg = []
        l_key = []
        l_mo = []
        l_fpi = []
        l_center = []

        l_obj = []
        l_pointcloud = []
        l_gt = []

        for point_sorted_index in range(1):
            ind = indecies[object_points[point_sorted_index]]

            # print( ind )
            normalized_point_ind = int(ind / len(object_xyz))
            point_check = [input_pointcloud[object_points[point_sorted_index], :3]]
            # print(dist, ind, normalized_point_ind, point_check)

            """
            color = np.zeros((len(input_pointcloud),3), np.float)
            color[:,2] = (dist < GOOD_DIST)[:,0] * 255
            # import pdb; pdb.set_trace()

            target = o3d.geometry.PointCloud()        
            target.points = o3d.utility.Vector3dVector(input_pointcloud[:,:3])
            target.colors = o3d.utility.Vector3dVector(color)
            o3d.visualization.draw_geometries([target]) 
            # """

            # pointlist = filterPoints(input_pointcloud, point_check[0], radius, gt_poses, input_tree)
            
            centerTreeFilter = KDTree([point_check[0]], leaf_size=2)
            distFilter, indeciesFilter = centerTreeFilter.query(input_pointcloud[:, :3], k=1)
            pointlist = input_pointcloud[distFilter.flatten() < radius, :]


            while (len(pointlist) < NUM_POINT):
                pointlist = np.array(list(pointlist) + list(pointlist))
            np.random.shuffle(pointlist)

            point_list = []
            cat_list = []
            feat_list = []

            pp = []
            pf = []

            point_array = np.array(pointlist)
            dist_list, ind_list = treeLocal.query(point_array[:, :3], k=1)
            dist_feature_list, index_feature_list = treeLocalFeature.query(point_array[:, :3], k=1)

            feat_list = index_feature_list % len(object_xyz_feature)
            pt_list = np.array(np.array(ind_list / len(object_xyz), int) == normalized_point_ind, int)
            cat_list = np.array(dist_list < GOOD_DIST, int).flatten() * pt_list.flatten()
            center_point_list = np.array(pointlist)[:NUM_POINT, :3] - all_new_center[normalized_point_ind]
            point_list = pointlist

            # print( point_check[0] )
            data = pc_center2cp(np.array(point_list[:NUM_POINT])[:, :6], point_check[0])  # point_array[:NUM_POINT,:6]
            seg = np.array(cat_list)[:NUM_POINT]
            key = np.array(feat_list)[:NUM_POINT]

            # seg = np.reshape(seg,(-1,1))

            seg = np.reshape(seg, (-1))
            key = np.reshape(key, (-1))

            # cur_zeros_placeholder = np.zeros_like(self.seg)
            # self.seg = np.where( self.seg <= 0, cur_zeros_placeholder, self.seg )

            cur_neg_placeholder = -np.ones_like(seg)
            key = np.where(seg == 0, cur_neg_placeholder, key)

            # object_set = self.object_set_list[ np.random.randint(len(self.object_set_list)) ]
            # obj_pc = object_set["obj_pc"]

            obj_pc_temp = copy.deepcopy(obj_pc)
            R = obj_pc_temp.get_rotation_matrix_from_xyz(
                (np.random.uniform(0, np.pi * 2), np.random.uniform(0, np.pi * 2), np.random.uniform(0, np.pi * 2)))
            obj_pc_temp.rotate(R, center=(0, 0, 0))
            model_pc_out = np.concatenate([np.asarray(obj_pc_temp.points), np.asarray(obj_pc_temp.normals)], axis=1)

            # center_point_cloud  # TODO
            model_pc_out = normalize_2d(model_pc_out)

            data, _ = normalize_1d(data)

            l_data.append(np.copy(data.astype('float32')))
            l_seg.append(np.copy(seg))
            l_key.append(np.copy(key))
            l_mo.append(np.copy(model_pc_out.astype('float32')))
            l_fpi.append(np.copy(fpi[0].cpu().numpy()))
            l_obj.append(obj_pc)
            l_pointcloud.append(np.array(point_list[:NUM_POINT])[:, :9])

     
            l_gt.append( [gt_poses[normalized_point_ind][:3,3], gt_poses[normalized_point_ind][:3,:3]] )

                        
            # print( gt_poses[normalized_point_ind] )

        # return data.astype('float32'), seg, key, model_pc_out.astype('float32'), fpi[0].cpu().numpy()
        # return l_data, l_seg, l_key, l_mo, l_fpi
        return np.array(l_data), np.array(l_seg), np.array(l_key), np.array(l_mo), np.array(
            l_fpi), l_obj, l_pointcloud, [[gt_poses[normalized_point_ind][:3,3], gt_poses[normalized_point_ind][:3,:3]]], radius

    def len(self):
        return len(self.full_dataset)

class WrsData:
    def __init__(self, number_of_keypoints, pointcloud_size, dataset_name, single_fpi):

        self.pointcloud_size = pointcloud_size
        self.train_index = 1

        self.partition = "val"

        self.object_set_list = []
        self.full_dataset = []

        self.number_of_keypoints = number_of_keypoints

        self.dataset_index = 0
        test_range = 1000
        self.good_dist = 2
        if dataset_name == "wrs21":
            cad_string = "/home/fhagelskjaer/workspace/bin-picking-para-pose/data/models/21_MBRAC60-2-10.stl"
            scene_folder = "/home/fhagelskjaer/workspace/BlenderBin/data/21_MBRAC60-2-10_1/train_pbr/"
        elif dataset_name == "wrs24":
            cad_string = "/home/fhagelskjaer/workspace/bin-picking-para-pose/data/models/24_37D-GEARMOTOR-50-70.roty90.stl"
            scene_folder = "/home/fhagelskjaer/workspace/BlenderBin/data/24_37D-GEARMOTOR-50-70.roty90_1/train_pbr/"
        elif dataset_name == "wrs12":
            cad_string = "/home/fhagelskjaer/workspace/bin-picking-para-pose/data/models/12_MBRFA30-2-P6.stl"
            scene_folder = "/home/fhagelskjaer/workspace/BlenderBin/data/12_MBRFA30-2-P6_1/train_pbr/"
        elif dataset_name == "wrs11":
            cad_string = "/home/fhagelskjaer/workspace/bin-picking-para-pose/data/models/11_KZAF1075NA4WA55GA20AA0.stl"
            scene_folder = "/home/fhagelskjaer/workspace/BlenderBin/data/11_KZAF1075NA4WA55GA20AA0_1/train_pbr/"
        elif dataset_name == "wrs20":
            cad_string = "/home/fhagelskjaer/workspace/bin-picking-para-pose/data/models/20_MBGA30-3.stl"
            scene_folder = "/home/fhagelskjaer/workspace/BlenderBin/data/20_MBGA30-3_1/train_pbr/"
        elif dataset_name == "wrs09":
            cad_string = "/home/fhagelskjaer/workspace/bin-picking-para-pose/data/models/09_BGPSL6-9-L30-F7.stl"
            scene_folder = "/home/fhagelskjaer/workspace/BlenderBin/data_old/09_BGPSL6-9-L30-F7_1/train_pbr/"
        elif dataset_name == "wrs15":
            cad_string = "/home/fhagelskjaer/workspace/bin-picking-para-pose/data/models/15_SBARB6200ZZ_30.stl"
            scene_folder = "/home/fhagelskjaer/workspace/BlenderBin/data_old/15_SBARB6200ZZ_30_1/train_pbr/"
        elif dataset_name[:-1] == "erf0":
            cad_string = "../test_cad_models/electrical_test/testcomp_00000" + dataset_name[-1] + ".ply"
            scene_folder = "../test_data/testcomp_00000" + dataset_name[
                -1] + "_1/train_pbr/"
            test_range = 80
        else:
            print("Unknown dataset!")
            sys.exit()

        # load cad model of object
        obj_cad = o3d.io.read_triangle_mesh(cad_string)
        # subsample 2048 points
        obj_pc = obj_cad.sample_points_poisson_disk(pointcloud_size)
        obj_pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30), False)

        # o3d.visualization.draw_geometries([obj_pc],point_show_normal=True)

        model_pc = np.concatenate([np.asarray(obj_pc.points), np.asarray(obj_pc.normals)], axis=1)

        self.single_fpi = single_fpi
        if self.single_fpi:
            object_xyz = np.asarray(obj_pc.points)
            x = torch.FloatTensor(object_xyz)
            x = np.reshape(x, (1, -1, 3))
            fpi = farthest_point_sampler(x, self.number_of_keypoints)
            self.fpi = fpi

        candidate = trimesh.load_mesh(cad_string)
        candidate_extents_max = np.max(candidate.extents)

        self.test_radius = np.sqrt(candidate.extents[0] ** 2 +
                                   candidate.extents[1] ** 2 +
                                   candidate.extents[2] ** 2)

        # self.radius = 4 * np.max( candidate_extents_max )
        # self.radius = 0.6 * np.max( candidate_extents_max )
        radius = np.max(candidate_extents_max)

        self.object_set_list.append(
            {"scene_folder": scene_folder, "obj_pc": obj_pc, "model_pc": model_pc, "radius": radius})

        for i in range(len(self.object_set_list)):
            for j in range(test_range):
                self.full_dataset.append([i, j])

    def get_item(self, item):
        # NUM_POINT = 2048
        NUM_POINT = self.pointcloud_size
        GOOD_DIST = self.good_dist
        BAD_DIST = 3

        object_set = self.object_set_list[self.full_dataset[item][0]]
        scene_folder = object_set["scene_folder"]
        obj_pc = object_set["obj_pc"]
        model_pc = object_set["model_pc"]
        radius = object_set["radius"]

        if self.single_fpi:
            fpi = self.fpi
        else:
            object_xyz = np.asarray(obj_pc.points)
            x = torch.FloatTensor(object_xyz)
            x = np.reshape(x, (1, -1, 3))
            fpi = farthest_point_sampler(x, self.number_of_keypoints)

        object_xyz = np.asarray(obj_pc.points)
        object_xyz_feature = object_xyz[fpi[0], :]

        dataset_index = self.dataset_index

        cloud_index = self.full_dataset[item][1]

        modelCloud = o3d.io.read_point_cloud(
            scene_folder + str(dataset_index).zfill(6) + "/cloud/" + str(cloud_index).zfill(6) + ".pcd")

        input_tree = KDTree(np.asarray(modelCloud.points), leaf_size=2)

        # input_pointcloud = np.concatenate([np.asarray(modelCloud.points), np.asarray(modelCloud.points)], axis=1)
        input_pointcloud = np.concatenate(
            [np.asarray(modelCloud.points), np.asarray(modelCloud.normals), np.asarray(modelCloud.colors)], axis=1)

        gt_file = scene_folder + str(dataset_index).zfill(6) + "/scene_gt.json"
        gty = json.load(open(gt_file))

        gt_poses = []

        for key in gty.keys():
            if (key != str(cloud_index)):
                continue
            # print( gty[key])
            # print( len(gty[key]) )
            for i in range(len(gty[key])):
                point = gty[key][i]['cam_t_m2c']
                rot = np.array(gty[key][i]['cam_R_m2c']).reshape((3, 3))
                if (gty[key][i]['obj_id'] == int(self.train_index)):
                    gt_poses.append((point, rot, 1.0))

        if len(gt_poses) == 0:
            return None

        all_newPL = []
        all_newPLFeature = []
        for gt_pose in gt_poses:
            # print(gt_pose[0])
            # print(gt_pose[1])
            newPL = np.matmul(object_xyz, gt_pose[1].transpose())
            newPL = newPL + gt_pose[0]

            newPLFeature = np.matmul(object_xyz_feature, gt_pose[1].transpose())
            newPLFeature = newPLFeature + gt_pose[0]

            all_newPL.append(newPL)
            all_newPLFeature.append(newPLFeature)

        newPL_2 = np.concatenate(all_newPL)
        newPLFeature_2 = np.concatenate(all_newPLFeature)

        treeLocal = KDTree(newPL_2, leaf_size=2)
        treeLocalFeature = KDTree(newPLFeature_2, leaf_size=2)

        dist, indecies = treeLocal.query(input_pointcloud[:, :3], k=1)
        object_points = np.flatnonzero(dist < GOOD_DIST)

        """
        color = np.zeros((len(input_pointcloud),3), np.float)
        color[:,2] = (dist < GOOD_DIST)[:,0] * 255
        # import pdb; pdb.set_trace()

        target = o3d.geometry.PointCloud()        
        target.points = o3d.utility.Vector3dVector(input_pointcloud[:,:3])
        target.colors = o3d.utility.Vector3dVector(color)
        o3d.visualization.draw_geometries([target]) 
        # """

        if (len(object_points) == 0):
            print("return none")
            return None

        np.random.shuffle(object_points)

        l_data = []
        l_seg = []
        l_key = []
        l_mo = []
        l_fpi = []

        l_obj = []
        l_pointcloud = []
        l_gt = []

        for point_sorted_index in range(1):
            ind = indecies[object_points[point_sorted_index]]
            # print( ind )
            normalized_point_ind = int(ind / len(object_xyz))
            point_check = [input_pointcloud[object_points[point_sorted_index], :3]]
            # print(dist, ind, normalized_point_ind, point_check)

            pointlist = filterPoints(input_pointcloud, point_check[0], radius, gt_poses, input_tree)

            while len(pointlist) < NUM_POINT:
                pointlist = np.array(list(pointlist) + list(pointlist))
            np.random.shuffle(pointlist)

            if False:
            # if True:
                pointlist = np.array(pointlist)
                # sigma = 0.4
                sigma = 2
                clip = sigma * 3
                gaussian_noise = np.clip(sigma * np.random.randn(pointlist.shape[0], 3), -clip, clip)
                pointlist[:, :3] += gaussian_noise
                # sigma = 0.01
                sigma = 0.05
                clip = sigma * 3
                gaussian_noise = np.clip(sigma * np.random.randn(pointlist.shape[0], 3), -clip, clip)
                pointlist[:, 3:6] += gaussian_noise

            point_array = np.array(pointlist)
            dist_list, ind_list = treeLocal.query(point_array[:, :3], k=1)
            dist_feature_list, index_feature_list = treeLocalFeature.query(point_array[:, :3], k=1)

            feat_list = index_feature_list % len(object_xyz_feature)
            pt_list = np.array(np.array(ind_list / len(object_xyz), int) == normalized_point_ind, int)
            cat_list = np.array(dist_list < GOOD_DIST, int).flatten() * pt_list.flatten()
            point_list = pointlist

            # print( point_check[0] )
            data = pc_center2cp(np.array(point_list[:NUM_POINT])[:, :6], point_check[0])  # point_array[:NUM_POINT,:6]
            seg = np.array(cat_list)[:NUM_POINT]
            key = np.array(feat_list)[:NUM_POINT]

            # seg = np.reshape(seg,(-1,1))

            seg = np.reshape(seg, (-1))
            key = np.reshape(key, (-1))

            cur_neg_placeholder = -np.ones_like(seg)
            key = np.where(seg == 0, cur_neg_placeholder, key)

            obj_pc_temp = copy.deepcopy(obj_pc)
            R = obj_pc_temp.get_rotation_matrix_from_xyz(
                (np.random.uniform(0, np.pi * 2), np.random.uniform(0, np.pi * 2), np.random.uniform(0, np.pi * 2)))
            obj_pc_temp.rotate(R, center=(0, 0, 0))
            model_pc_out = np.concatenate([np.asarray(obj_pc_temp.points), np.asarray(obj_pc_temp.normals)], axis=1)

            model_pc_out = normalize_2d(model_pc_out)

            data, _ = normalize_1d(data)

            l_data.append(np.copy(data.astype('float32')))
            l_seg.append(np.copy(seg))
            l_key.append(np.copy(key))
            l_mo.append(np.copy(model_pc_out.astype('float32')))
            l_fpi.append(np.copy(fpi[0].cpu().numpy()))
            l_obj.append(obj_pc)
            l_pointcloud.append(np.array(point_list[:NUM_POINT])[:, :9])
            l_gt.append(gt_poses[normalized_point_ind])

        # return data.astype('float32'), seg, key, model_pc_out.astype('float32'), fpi[0].cpu().numpy()
        # return l_data, l_seg, l_key, l_mo, l_fpi
        return np.array(l_data), np.array(l_seg), np.array(l_key), np.array(l_mo), np.array(
            l_fpi), l_obj, l_pointcloud, gt_poses, radius
        # return data.astype('float32'), seg, key, model_pc_out.astype('float32'), fpi[0]

    def get_item_fast(self, item):
        # NUM_POINT = 2048
        NUM_POINT = self.pointcloud_size
        GOOD_DIST = self.good_dist
        # BAD_DIST = 3

        object_set = self.object_set_list[self.full_dataset[item][0]]
        scene_folder = object_set["scene_folder"]
        obj_pc = object_set["obj_pc"]
        # model_pc = object_set["model_pc"]
        radius = object_set["radius"]

        if self.single_fpi:
            fpi = self.fpi
        else:
            object_xyz = np.asarray(obj_pc.points)
            x = torch.FloatTensor(object_xyz)
            x = np.reshape(x, (1, -1, 3))
            fpi = farthest_point_sampler(x, self.number_of_keypoints)

        object_xyz = np.asarray(obj_pc.points)
        # object_xyz_feature = object_xyz[fpi[0], :]

        dataset_index = self.dataset_index

        cloud_index = self.full_dataset[item][1]

        modelCloud = o3d.io.read_point_cloud(
            scene_folder + str(dataset_index).zfill(6) + "/cloud/" + str(cloud_index).zfill(6) + ".pcd")

        input_tree = KDTree(np.asarray(modelCloud.points), leaf_size=2)

        # input_pointcloud = np.concatenate([np.asarray(modelCloud.points), np.asarray(modelCloud.points)], axis=1)
        input_pointcloud = np.concatenate(
            [np.asarray(modelCloud.points), np.asarray(modelCloud.normals), np.asarray(modelCloud.colors)], axis=1)

        gt_file = scene_folder + str(dataset_index).zfill(6) + "/scene_gt.json"
        gty = json.load(open(gt_file))

        gt_poses = []

        for key in gty.keys():
            if (key != str(cloud_index)):
                continue
            # print( gty[key])
            # print( len(gty[key]) )
            for i in range(len(gty[key])):
                point = gty[key][i]['cam_t_m2c']
                rot = np.array(gty[key][i]['cam_R_m2c']).reshape((3, 3))
                if (gty[key][i]['obj_id'] == int(self.train_index)):
                    gt_poses.append((point, rot, 1.0))

        if len(gt_poses) == 0:
            return None

        all_newPL = []
        # all_newPLFeature = []
        for gt_pose in gt_poses:
            # print(gt_pose[0])
            # print(gt_pose[1])
            newPL = np.matmul(object_xyz, gt_pose[1].transpose())
            newPL = newPL + gt_pose[0]

            # newPLFeature = np.matmul(object_xyz_feature, gt_pose[1].transpose())
            # newPLFeature = newPLFeature + gt_pose[0]

            all_newPL.append(newPL)
            # all_newPLFeature.append(newPLFeature)

        newPL_2 = np.concatenate(all_newPL)
        # newPLFeature_2 = np.concatenate(all_newPLFeature)

        treeLocal = KDTree(newPL_2, leaf_size=2)
        # treeLocalFeature = KDTree(newPLFeature_2, leaf_size=2)

        dist, indecies = treeLocal.query(input_pointcloud[:, :3], k=1)
        object_points = np.flatnonzero(dist < GOOD_DIST)

        """
        color = np.zeros((len(input_pointcloud),3), np.float)
        color[:,2] = (dist < GOOD_DIST)[:,0] * 255
        # import pdb; pdb.set_trace()

        target = o3d.geometry.PointCloud()        
        target.points = o3d.utility.Vector3dVector(input_pointcloud[:,:3])
        target.colors = o3d.utility.Vector3dVector(color)
        o3d.visualization.draw_geometries([target]) 
        # """

        if (len(object_points) == 0):
            print("return none")
            return None

        np.random.shuffle(object_points)

        l_data = []
        l_pointcloud = []
        l_gt = []

        for point_sorted_index in range(1):
            ind = indecies[object_points[point_sorted_index]]
            # print( ind )
            normalized_point_ind = int(ind / len(object_xyz))
            point_check = [input_pointcloud[object_points[point_sorted_index], :3]]
            # print(dist, ind, normalized_point_ind, point_check)

            pointlist = filterPoints(input_pointcloud, point_check[0], radius, gt_poses, input_tree)

            while len(pointlist) < NUM_POINT:
                pointlist = np.array(list(pointlist) + list(pointlist))
            np.random.shuffle(pointlist)

            if False:
            # if True:
                pointlist = np.array(pointlist)
                sigma = 0.4
                clip = sigma * 3
                gaussian_noise = np.clip(sigma * np.random.randn(pointlist.shape[0], 3), -clip, clip)
                pointlist[:, :3] += gaussian_noise
                sigma = 0.01
                clip = sigma * 3
                gaussian_noise = np.clip(sigma * np.random.randn(pointlist.shape[0], 3), -clip, clip)
                pointlist[:, 3:6] += gaussian_noise
            elif False:
            # elif True:
                pointlist = np.array(pointlist)
                sigma = 2
                clip = sigma * 3
                gaussian_noise = np.clip(sigma * np.random.randn(pointlist.shape[0], 3), -clip, clip)
                pointlist[:, :3] += gaussian_noise
                sigma = 0.05
                clip = sigma * 3
                gaussian_noise = np.clip(sigma * np.random.randn(pointlist.shape[0], 3), -clip, clip)
                pointlist[:, 3:6] += gaussian_noise

            point_list = pointlist

            data = pc_center2cp(np.array(point_list[:NUM_POINT])[:, :6], point_check[0])  # point_array[:NUM_POINT,:6]

            data, _ = normalize_1d(data)

            l_data.append(np.copy(data.astype('float32')))
            l_pointcloud.append(np.array(point_list[:NUM_POINT])[:, :9])
            l_gt.append(gt_poses[normalized_point_ind])

        return np.array(l_data), l_pointcloud, gt_poses, radius

    def len(self):
        return len(self.full_dataset)
