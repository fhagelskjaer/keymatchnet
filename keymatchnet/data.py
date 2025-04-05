#!/usr/bin/env python
# -*- coding: utf-8 -*-
import open3d as o3d
import os
import sys
import glob
import h5py
import numpy as np
import torch
import json
import copy
from torch.utils.data import Dataset

from sklearn.neighbors import KDTree

import trimesh

import pickle

from . import (
    fps
)

def load_data_h5(data_dir):
    # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # DATA_DIR = os.path.join(BASE_DIR, 'data')
    # DATA_DIR = BASE_DIR
    all_data = []
    all_label = []
    all_seg = []
    all_key = []
    # print( os.path.join(DATA_DIR, dataset, '*%s*.h5'%partition) )
    # for h5_name in glob.glob(os.path.join(DATA_DIR, 'cuneiform_32k', '*%s*.h5'%partition)):
    # for h5_name in glob.glob(os.path.join(DATA_DIR, dataset, '*%s*.h5'%partition)):
    for h5_name in glob.glob(data_dir):
        f = h5py.File(h5_name, 'r+')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        seg = f['cat'][:].astype('int64')
        key = f['feat'][:].astype('int64')

        f.close()
        all_data.append(data)
        all_label.append(label)
        all_seg.append(seg)
        all_key.append(key)

    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    all_seg = np.concatenate(all_seg, axis=0)
    all_key = np.concatenate(all_key, axis=0)

    return all_data, all_label, all_seg, all_key


class PVN(Dataset):
    # def __init__(self, num_points, partition='train'):
    def __init__(self, dataset, partition='train', num_points=0, noise_model=None):
        # self.data, self.label, self.seg, self.key = load_data_h5(partition, dataset)
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(BASE_DIR, dataset, '*synth*.h5')

        self.num_points = num_points

        self.partition = partition
        print(data_dir)

        self.data, self.label, self.seg, self.key = load_data_h5(data_dir)

        dlen = self.data.shape[0]
        slen = int(dlen * 0.9)
        if self.partition == 'train':
            self.data, self.label, self.seg, self.key = self.data[:slen, ...], self.label[:slen, ...], self.seg[:slen,
                                                                                                       ...], self.key[
                                                                                                             :slen, ...]
        else:
            # self.data, self.label, self.seg, self.key = self.data[:slen,...], self.label[:slen,...], self.seg[:slen,...], self.key[:slen,...]
            self.data, self.label, self.seg, self.key = self.data[slen:, ...], self.label[slen:, ...], self.seg[slen:,
                                                                                                       ...], self.key[
                                                                                                             slen:, ...]

        cur_zeros_placeholder = np.zeros_like(self.seg)
        self.seg = np.where(self.seg <= 0, cur_zeros_placeholder, self.seg)

        cur_neg_placeholder = -np.ones_like(self.seg)
        self.key = np.where(self.seg == 0, cur_neg_placeholder, self.key)

        # self.all_keys = json.load(open(os.path.join(BASE_DIR, dataset, 'overallid_to_catid_partid.json'), 'r'))
        # self.num_key = len(self.all_keys)

        self.noise_model = noise_model

    def __getitem__(self, item):
        data = self.data[item]
        label = self.label[item]
        seg = self.seg[item]
        key = self.key[item]

        indexes = list(range(data.shape[0]))
        np.random.shuffle(indexes)

        data = data[indexes[:self.num_points]]
        seg = seg[indexes[:self.num_points]]
        key = key[indexes[:self.num_points]]

        # print( np.mean(data, axis=0), np.max(data, axis=0), np.std(data, axis=0))

        if self.noise_model is not None:
            data = data[:, :6]
            N, C = data.shape
            data = np.reshape(data, (1, N, C))
            data = self.noise_model.compute(data)
            data = np.reshape(data, (N, C))

        return data[:, :6], seg, key

    def __len__(self):
        return self.data.shape[0]
        ## return 100


def filterPoints(pointList, p, distance, groundtruth, tree):
    point = np.array([p[:3]])
    point.reshape(3, 1)

    filteredList = tree.query_radius(np.array(point), r=distance)
    filteredList = pointList[filteredList[0]]

    return filteredList

def pc_center(pc):
    pc_out = pc.copy()
    centroid = np.mean(pc[:, :3], axis=0)
    pc_out[:, :3] = pc[:, :3] - centroid
    return pc_out


def pc_center2cp(pc, centroid):
    pc_out = pc.copy()
    pc_out[:, :3] = pc[:, :3] - centroid
    return pc_out


class GPVN(Dataset):
    # def __init__(self, num_points, partition='train'):
    def __init__(self, partition='train'):
        # self.data, self.label, self.seg, self.key = load_data_h5(partition, dataset)
        # cad_string = "/home/fhagelskjaer/workspace/bin-picking-para-pose/data/models/10_EDCS10.stl"
        # self.scene_folder = "/home/fhagelskjaer/workspace/BlenderBin/data/10_EDCS10_1/train_pbr/"
        # cad_string = "/home/fhagelskjaer/workspace/mfe/TO127P1524X482-7N_JEDEC_TO-263CB.ply"
        # self.scene_folder = "/home/fhagelskjaer/workspace/BlenderBin/data/TO127P1524X482-7N_JEDEC_TO-263CB_1/train_pbr/"
        cad_string = "/home/fhagelskjaer/workspace/hats/big/hat_000000.ply"
        self.scene_folder = "/home/fhagelskjaer/workspace/BlenderBin/data/hat_000000_1/train_pbr/"
        # load cad model of object
        obj_cad = o3d.io.read_triangle_mesh(cad_string)
        # subsample 2048 points
        self.obj_pc = obj_cad.sample_points_poisson_disk(2048)
        self.obj_pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30), False)

        self.model_pc = np.concatenate([np.asarray(self.obj_pc.points), np.asarray(self.obj_pc.normals)], axis=1)

        # object_xyz = np.asarray(self.obj_pc.points)
        # x = torch.FloatTensor(object_xyz)
        # x = np.reshape(x,(1,-1, 3))
        # self.fpi = farthest_point_sampler(x,20,start_idx=0)

        self.partition = partition

        self.train_index = 1

        candidate = trimesh.load_mesh(cad_string)
        candidate_extents_max = np.max(candidate.extents)

        # self.radius = 4 * np.max( candidate_extents_max )
        # self.radius = 0.6 * np.max( candidate_extents_max )
        self.radius = np.max(candidate_extents_max)

    def __getitem__(self, item):

        if self.partition == "test":
            item += (1000 - 24)

        NUM_POINT = 2048
        GOOD_DIST = 2
        BAD_DIST = 3

        # print( item )

        object_xyz = np.asarray(self.obj_pc.points)
        fpi = fps.farthest_point_sampler(object_xyz, 20)


        # fpi = self.fpi[0,:]
        # np.random.shuffle(fpi)

        object_xyz_feature = object_xyz[fpi[0], :]

        # get index TODO
        dataset_index = 0

        # CAMERA_PARAM = self.scene_folder + str(dataset_index).zfill(6) + "/" + "scene_camera.json"

        # camera_param = json.load(open(CAMERA_PARAM))

        # first_image_camera_key = list(camera_param.keys())[item]

        # cameraMat = camera_param[first_image_camera_key]['cam_K']
        # cameraMat = np.reshape(cameraMat, (3,3))

        # depth_scale = camera_param[first_image_camera_key]['depth_scale']; image_ext = ".jpg"

        # print( "So far1" )

        # intrinsic = o3d.camera.PinholeCameraIntrinsic()
        # intrinsic.set_intrinsics(1944, 1200, cameraMat[0,0], cameraMat[1,1], cameraMat[0,2], cameraMat[1,2])

        # color_raw = o3d.io.read_image(self.scene_folder + str(dataset_index).zfill(6) + "/rgb/" + str(item).zfill(6) + image_ext)
        # depth_raw = o3d.io.read_image(self.scene_folder + str(dataset_index).zfill(6) + "/depth/" + str(item).zfill(6) + ".png")
        # rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, depth_scale=1/depth_scale, depth_trunc=500.0, convert_rgb_to_intensity=False)
        # modelCloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

        # depth_raw = o3d.io.read_image(self.scene_folder + str(dataset_index).zfill(6) + "/depth/" + str(item).zfill(6) + ".png")
        # modelCloud = o3d.geometry.PointCloud.create_from_depth_image(depth_raw, intrinsic, depth_scale=1/depth_scale, depth_trunc=500.0)

        # print( modelCloud )
        # print( "So far1-3" )
        # modelCloud = modelCloud.voxel_down_sample(1)
        # print( "So far1-4" )
        # modelCloud.estimate_normals( o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=30))
        # modelCloud.estimate_normals( o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30), False)
        # modelCloud.estimate_normals( o3d.geometry.KDTreeSearchParamKNN(knn=20), False)
        # modelCloud.estimate_normals( o3d.geometry.KDTreeSearchParamRadius( radius = 10), True) # False)

        # print( "So far1-6" )
        # modelCloud.orient_normals_to_align_with_direction(orientation_reference=np.array([0.0, 0.0, -1.0]))
        # print( "So far1-7" )

        # print( "So far1-8" )

        modelCloud = o3d.io.read_point_cloud(
            self.scene_folder + str(dataset_index).zfill(6) + "/cloud/" + str(item).zfill(6) + ".pcd")

        # import copy
        # newtarget = copy.deepcopy(modelCloud)
        # R = newtarget.get_rotation_matrix_from_xyz((0,  np.pi, 0))
        # newtarget.rotate(R, center=(0, 0, 0))
        # o3d.visualization.draw_geometries([newtarget, modelCloud])

        input_tree = KDTree(np.asarray(modelCloud.points), leaf_size=2)

        # input_pointcloud = np.concatenate( [ np.asarray(modelCloud.points), np.asarray(modelCloud.normals), np.asarray(modelCloud.colors) ], axis = 1 )
        # input_pointcloud = np.concatenate( [ np.asarray(modelCloud.points), np.asarray(modelCloud.points), np.asarray(modelCloud.colors) ], axis = 1 )
        input_pointcloud = np.concatenate([np.asarray(modelCloud.points), np.asarray(modelCloud.normals)], axis=1)

        gt_file = self.scene_folder + str(dataset_index).zfill(6) + "/scene_gt.json"
        gty = json.load(open(gt_file))

        gt_poses = []

        for key in gty.keys():
            if (key != str(item)):
                continue
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

        # print( "So far2" )

        """
        dist, ind = treeLocal.query(input_pointcloud[:,:3], k=1)
        color = np.zeros((len(input_pointcloud),3), np.float)
        color[:,2] = (dist < GOOD_DIST)[:,0] * 255
        # import pdb; pdb.set_trace()

        target = o3d.geometry.PointCloud()        
        target.points = o3d.utility.Vector3dVector(input_pointcloud[:,:3])
        target.colors = o3d.utility.Vector3dVector(color)
        o3d.visualization.draw_geometries([target]) 
        # """

        cnt = 0
        while (True):
            cnt += 1

            if (cnt > 15000):  # Object could not be found
                print("return none")
                return None

            randomIndex = np.random.randint(0, len(input_pointcloud))
            # print(randomIndex)
            point_check = input_pointcloud[randomIndex, :3]

            point_check = np.array([point_check])
            point_check.reshape(3, 1)
            dist, ind = treeLocal.query(point_check, k=1)
            # print( dist )

            # print("length xyz", len(object_xyz))

            normalized_point_ind = int(ind / len(object_xyz))

            if (dist[0][0] > 3):
                continue

            pointlist = filterPoints(input_pointcloud, point_check[0], self.radius, gt_poses, input_tree)

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

            for i in range(len(pointlist)):

                p = pointlist[i]
                dist = dist_list[i]
                ind = ind_list[i]

                pt = int(ind / len(object_xyz)) == normalized_point_ind

                dist_feature = dist_feature_list[i]
                index_feature = index_feature_list[i] % len(object_xyz_feature)

                point_list.append(p)
                feat_list.append(index_feature)

                if (dist < GOOD_DIST and pt):
                    cat_list.append(1)
                    # pp.append( p )
                    # pf.append( index_feature )
                elif (dist > BAD_DIST or not pt):
                    cat_list.append(0)
                else:
                    # cat_list.append( -1 )
                    cat_list.append(0)
            break

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

        return data.astype('float32'), seg, key, self.model_pc.astype('float32'), fpi[0]

    def __len__(self):
        if self.partition == "test":
            # return 1000
            # return 20
            return 24
        else:
            return 1000 - 24
            # return 1000
            # return 40


def normalize_2d(matrix):
    # Only this is changed to use 2-norm put 2 instead of 1
    mean = np.mean(matrix, axis=0)
    matrix = matrix - mean
    # std = np.std(matrix, axis=0)
    std = np.std(matrix)
    matrix = matrix / std
    return matrix


def normalize_1d(matrix):
    # Only this is changed to use 2-norm put 2 instead of 1
    # std = np.std(matrix, axis=0)
    std = np.std(matrix)
    matrix = matrix / std
    return matrix, std


class GPVN_set_knocpose(Dataset):
    # def __init__(self, num_points, partition='train'):
    def __init__(self, number_of_keypoints, partition='train'):

        self.partition = partition
        self.train_index = 1

        self.object_set_list = []
        self.full_dataset = []

        self.number_of_keypoints = number_of_keypoints

        if self.partition == "train":
            # used_range = range(298)
            # used_range= list(range(580))
            used_range = list(range(1450))
            used_range.remove(74)
            used_range.remove(309)
            used_range = list(range(0, 1450))
        elif self.partition == "val":
            # used_range = range(298,311)
            # used_range = range(580,601)
            used_range = range(1450, 1500)
            # used_range = range(1,2)
        elif self.partition == "test":
            # used_range = range(0,10)
            used_range = range(0, 7)
            # used_range = [6]
        else:
            print("Unknown partition")
            return

        for hat_index in used_range:
            # if False:
            if self.partition == "train" or self.partition == "val":
                # cad_string = "/home/fhagelskjaer/workspace/gpvn/big/hat_" + str(hat_index).zfill(6) + ".ply"
                # scene_folder = "/home/fhagelskjaer/workspace/BlenderBin/data/hat_" + str(hat_index).zfill(6) + "_1/train_pbr/"
                cad_string = "/home/fhagelskjaer/workspace/gpvn/electrical_big/comp_" + str(hat_index).zfill(6) + ".ply"
                scene_folder = "/home/fhagelskjaer/workspace/BlenderBin/data/comp_" + str(hat_index).zfill(
                    6) + "_1/train_pbr/"
            elif self.partition == "test":
                # cad_string = "/home/fhagelskjaer/workspace/gpvn/test_data/test_hat_" + str(hat_index).zfill(6) + ".ply"
                # scene_folder = "/home/fhagelskjaer/workspace/BlenderBin/data/test_hat_" + str(hat_index).zfill(6) + "_1/train_pbr/"
                cad_string = "/home/fhagelskjaer/workspace/gpvn/electrical_test/testcomp_" + str(hat_index).zfill(
                    6) + ".ply"
                scene_folder = "/home/fhagelskjaer/workspace/BlenderBin/data/testcomp_" + str(hat_index).zfill(
                    6) + "_1/train_pbr/"

            # load cad model of object
            obj_cad = o3d.io.read_triangle_mesh(cad_string)
            # subsample 2048 points
            obj_pc = obj_cad.sample_points_poisson_disk(2048)
            obj_pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30), False)

            # o3d.visualization.draw_geometries([obj_pc],point_show_normal=True)

            model_pc = np.concatenate([np.asarray(obj_pc.points), np.asarray(obj_pc.normals)], axis=1)

            model_center = obj_pc.get_center()

            candidate = trimesh.load_mesh(cad_string)
            candidate_extents_max = np.max(candidate.extents)

            # self.radius = 4 * np.max( candidate_extents_max )
            # self.radius = 0.6 * np.max( candidate_extents_max )
            radius = np.max(candidate_extents_max)

            self.object_set_list.append(
                {"scene_folder": scene_folder, "obj_pc": obj_pc, "model_pc": model_pc, "radius": radius,
                 "model_center": model_center})

        for i in range(len(self.object_set_list)):
            gt_file = "knocpose/logs/comp_" + str(i).zfill(6) + ".json"
            gty = json.load(open(gt_file))
            keys = gty.keys()
            cnt = 0
            while not cnt == 80:
                for key in keys:
                    # if len(gty[key]) and (int(key) + 1)%4== 0:
                    if len(gty[key]):
                        self.full_dataset.append([i, int(key)])
                        cnt += 1
                    if cnt == 80:
                        break
                if cnt == 0:
                    break
            print(gt_file, cnt)
            # print(i,int(key),len(gty[key]))

    def __getitem__(self, item):
        NUM_POINT = 2048
        GOOD_DIST = 2
        BAD_DIST = 3

        object_set = self.object_set_list[self.full_dataset[item][0]]
        scene_folder = object_set["scene_folder"]
        obj_pc = object_set["obj_pc"]
        model_pc = object_set["model_pc"]
        radius = object_set["radius"]
        model_center = object_set["model_center"]

        object_xyz = np.asarray(obj_pc.points)
        fpi = fps.farthest_point_sampler(object_xyz, self.number_of_keypoints)

        object_xyz_feature = object_xyz[fpi[0], :]

        # get index TODO
        dataset_index = 0

        # item = item%1000
        # item = item%80

        cloud_index = self.full_dataset[item][1]

        # print(scene_folder, cloud_index)

        modelCloud = o3d.io.read_point_cloud(
            scene_folder + str(dataset_index).zfill(6) + "/cloud/" + str(cloud_index).zfill(6) + ".pcd")

        input_tree = KDTree(np.asarray(modelCloud.points), leaf_size=2)

        input_pointcloud = np.concatenate([np.asarray(modelCloud.points), np.asarray(modelCloud.normals)], axis=1)

        # gt_file = scene_folder + str(dataset_index).zfill(6) + "/scene_gt.json"
        # gty = json.load(open(gt_file))

        gt_poses = []

        gt_file = "knocpose/logs/comp_" + str(self.full_dataset[item][0]).zfill(6) + ".json"
        gty = json.load(open(gt_file))

        # print( gty )
        # print( cloud_index )

        for pose_as_list in gty[str(cloud_index)]:
            pose = np.reshape(np.array(pose_as_list), (4, 4))
            gt_poses.append((pose[:3, 3], pose[:3, :3], 1.0))

        # print( gt_poses )

        if len(gt_poses) == 0:
            print("No poses found")
            return None

        all_newPL = []
        all_newPLFeature = []
        all_new_center = []
        for gt_pose in gt_poses:
            newPL = np.matmul(object_xyz, gt_pose[1].transpose())
            newPL = newPL + gt_pose[0]

            newPLFeature = np.matmul(object_xyz_feature, gt_pose[1].transpose())
            newPLFeature = newPLFeature + gt_pose[0]

            all_newPL.append(newPL)
            all_newPLFeature.append(newPLFeature)

            new_center = np.matmul(model_center, gt_pose[1].transpose())
            new_center = new_center + gt_pose[0]
            # all_new_center.append(newPL - new_center)
            all_new_center.append(new_center)

        newPL_2 = np.concatenate(all_newPL)
        newPLFeature_2 = np.concatenate(all_newPLFeature)

        treeLocal = KDTree(newPL_2, leaf_size=2)
        treeLocalFeature = KDTree(newPLFeature_2, leaf_size=2)

        # print( "So far2" )

        dist, indecies = treeLocal.query(input_pointcloud[:, :3], k=1)
        object_points = np.flatnonzero(dist < GOOD_DIST)

        a = [treeLocal, treeLocalFeature, dist, indecies, object_points, object_xyz, input_pointcloud]

        # with open('filename.pickle', 'wb') as handle:
        #     pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # print( "First lasted time: ", time.time() - start_time )
        # start_time = time.time()
        # with open('filename.pickle', 'rb') as handle:
        #     treeLocal, treeLocalFeature, dist, indecies, object_points, object_xyz, input_pointcloud = pickle.load(handle)

        # for each scene save
        # treeLocal
        # treeLocalFeature
        # objectpoints
        # input_pointcloud
        # gt_poses

        if (len(object_points) == 0):
            print(self.full_dataset[item][0], self.full_dataset[item][1])
            print("No points found")
            return None

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

        for point_sorted_index in range(2):
            ind = indecies[object_points[point_sorted_index]]
            # print( ind )
            normalized_point_ind = int(ind / len(object_xyz))
            point_check = [input_pointcloud[object_points[point_sorted_index], :3]]
            # print(dist, ind, normalized_point_ind, point_check)

            """
            color = np.zeros((len(input_pointcloud),3), float)
            color[:,2] = (dist < GOOD_DIST)[:,0] * 255
            # import pdb; pdb.set_trace()

            target = o3d.geometry.PointCloud()        
            target.points = o3d.utility.Vector3dVector(input_pointcloud[:,:3])
            target.colors = o3d.utility.Vector3dVector(color)
            o3d.visualization.draw_geometries([target]) 
            # """

            pointlist = filterPoints(input_pointcloud, point_check[0], radius, gt_poses, input_tree)

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

            data, std_data = normalize_1d(data)

            # if False: # if True:
            if self.partition == "train":
                #  jitter on the model
                shift_range = 0.10
                shift = np.random.uniform(-shift_range, shift_range, (3))
                model_pc_out[:, :3] += shift

                #  noise on the model
                # sigma = 0.005
                sigma = np.random.uniform(0.0001, 0.0075)
                clip = sigma * 3
                gaussian_noise = np.clip(sigma * np.random.randn(2048, 6), -clip, clip)
                model_pc_out += gaussian_noise

                #  noise on the scene
                # sigma = 0.005
                sigma = np.random.uniform(0.0001, 0.0075)
                clip = sigma * 3
                gaussian_noise = np.clip(sigma * np.random.randn(2048, 6), -clip, clip)
                data += gaussian_noise

                # print( data.astype('float32').shape, seg.shape, key.shape, model_pc_out.astype('float32').shape, fpi[0].shape)

            # print( "Lasted time: ", time.time() - start_time )
            # list_of_data.append( [data.astype('float32'), seg, key, model_pc_out.astype('float32'), fpi[0]])
            l_data.append(np.copy(data.astype('float32')))
            l_seg.append(np.copy(seg))
            l_key.append(np.copy(key))
            l_mo.append(np.copy(model_pc_out.astype('float32')))
            l_fpi.append(np.copy(fpi[0].cpu().numpy()))
            l_center.append(np.copy(center_point_list / std_data))

        # return data.astype('float32'), seg, key, model_pc_out.astype('float32'), fpi[0].cpu().numpy()
        # return l_data, l_seg, l_key, l_mo, l_fpi
        # return np.array(l_data), np.array(l_seg), np.array(l_key), np.array(l_mo), np.array(l_fpi), np.array(l_center)
        return np.array(l_data), np.array(l_seg), np.array(l_key), np.array(l_mo), np.array(l_fpi)
        # return data.astype('float32'), seg, key, model_pc_out.astype('float32'), fpi[0]

    def __len__(self):
        return len(self.full_dataset)


class GPVN_set(Dataset):
    # def __init__(self, num_points, partition='train'):
    def __init__(self, number_of_keypoints, max_data=-1,  partition='train'):

        self.max_data = max_data
        self.partition = partition
        self.train_index = 1

        self.object_set_list = []
        self.full_dataset = []

        self.number_of_keypoints = number_of_keypoints

        if self.partition == "train":
            # used_range = range(298)
            # used_range= list(range(580))
            used_range = list(range(1450))
            used_range.remove(74)
            used_range.remove(309)
            # used_range= list(range(0,20))
            # np.random.seed(0) # TODO
            # np.random.shuffle(used_range) # TODO
            # used_range = used_range[:145] # TODO
        elif self.partition == "val": 
            # used_range = range(298,311)
            # used_range = range(580,601)
            used_range = range(1450, 1500)
            # used_range = range(1,2)
        elif self.partition == "test":
            # used_range = range(0,10)
            used_range = range(0, 7)
            # used_range = [6]
        else:
            print("Unknown partition")
            return

        for hat_index in used_range:
            if self.partition == "train" or self.partition == "val":
                # cad_string = "/home/fhagelskjaer/workspace/gpvn/big/hat_" + str(hat_index).zfill(6) + ".ply"
                # scene_folder = "/home/fhagelskjaer/workspace/BlenderBin/data/hat_" + str(hat_index).zfill(6) + "_1/train_pbr/"
                cad_string = "/home/fhagelskjaer/workspace/gpvn/electrical_big/comp_" + str(hat_index).zfill(6) + ".ply"
                scene_folder = "/home/fhagelskjaer/workspace/BlenderBin/data/comp_" + str(hat_index).zfill(
                    6) + "_1/train_pbr/"
            elif self.partition == "test":
                # cad_string = "/home/fhagelskjaer/workspace/gpvn/test_data/test_hat_" + str(hat_index).zfill(6) + ".ply"
                # scene_folder = "/home/fhagelskjaer/workspace/BlenderBin/data/test_hat_" + str(hat_index).zfill(6) + "_1/train_pbr/"
                cad_string = "/home/fhagelskjaer/workspace/gpvn/electrical_test/testcomp_" + str(hat_index).zfill(
                    6) + ".ply"
                scene_folder = "/home/fhagelskjaer/workspace/BlenderBin/data/testcomp_" + str(hat_index).zfill(
                    6) + "_1/train_pbr/"

            # load cad model of object
            obj_cad = o3d.io.read_triangle_mesh(cad_string)
            # subsample 2048 points
            obj_pc = obj_cad.sample_points_poisson_disk(2048)
            obj_pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30), False)

            # o3d.visualization.draw_geometries([obj_pc],point_show_normal=True)

            model_pc = np.concatenate([np.asarray(obj_pc.points), np.asarray(obj_pc.normals)], axis=1)

            model_center = obj_pc.get_center()

            candidate = trimesh.load_mesh(cad_string)
            candidate_extents_max = np.max(candidate.extents)

            # self.radius = 4 * np.max( candidate_extents_max )
            # self.radius = 0.6 * np.max( candidate_extents_max )
            radius = np.max(candidate_extents_max)

            self.object_set_list.append(
                {"scene_folder": scene_folder, "obj_pc": None, "model_pc": model_pc, "radius": radius,
                 "model_center": model_center})

        for i in range(len(self.object_set_list)):
            for j in range(80):
                for _ in range(10):
                    self.full_dataset.append([i, j])

    def __getitem__(self, item):
        NUM_POINT = 2048
        GOOD_DIST = 2
        BAD_DIST = 3

        object_set = self.object_set_list[self.full_dataset[item][0]]
        scene_folder = object_set["scene_folder"]
        obj_pc = object_set["obj_pc"]
        model_pc = object_set["model_pc"]
        radius = object_set["radius"]
        model_center = object_set["model_center"]

        # TODO subsample randomly from pc

        object_xyz = model_pc[:,:3]
        x = torch.FloatTensor(object_xyz)
        x = np.reshape(x, (1, -1, 3))

        # TODO 
        if self.partition == "train" and np.random.rand() < 0.6:
            fpi_idx = list(range(object_xyz.shape[0])) 
            np.random.shuffle(fpi_idx)
            fpi = np.array(fpi_idx[:self.number_of_keypoints])
        else:
            fpi = fps.farthest_point_sampler(object_xyz, self.number_of_keypoints)

         

        object_xyz_feature = object_xyz[fpi, :]

        # get index TODO
        dataset_index = 0

        # item = item%1000
        # item = item%80

        cloud_index = self.full_dataset[item][1]

        # print(scene_folder, cloud_index)

        modelCloud = o3d.io.read_point_cloud(
            scene_folder + str(dataset_index).zfill(6) + "/cloud/" + str(cloud_index).zfill(6) + ".pcd")

        input_tree = KDTree(np.asarray(modelCloud.points), leaf_size=2)

        input_pointcloud = np.concatenate([np.asarray(modelCloud.points), np.asarray(modelCloud.normals)], axis=1)

        gt_file = scene_folder + str(dataset_index).zfill(6) + "/scene_gt.json"
        gty = json.load(open(gt_file))

        gt_poses = []

        for key in gty.keys():
            if (key != str(cloud_index)):
                continue
            for i in range(len(gty[key])):
                point = gty[key][i]['cam_t_m2c']
                rot = np.array(gty[key][i]['cam_R_m2c']).reshape((3, 3))
                if (gty[key][i]['obj_id'] == int(self.train_index)):
                    gt_poses.append((point, rot, 1.0))

        if len(gt_poses) == 0:
            return None

        all_newPL = []
        all_newPLFeature = []
        all_new_center = []
        for gt_pose in gt_poses:
            newPL = np.matmul(object_xyz, gt_pose[1].transpose())
            newPL = newPL + gt_pose[0]

            newPLFeature = np.matmul(object_xyz_feature, gt_pose[1].transpose())
            newPLFeature = newPLFeature + gt_pose[0]

            all_newPL.append(newPL)
            all_newPLFeature.append(newPLFeature)

            new_center = np.matmul(model_center, gt_pose[1].transpose())
            new_center = new_center + gt_pose[0]
            # all_new_center.append(newPL - new_center)
            all_new_center.append(new_center)

        newPL_2 = np.concatenate(all_newPL)
        newPLFeature_2 = np.concatenate(all_newPLFeature)

        treeLocal = KDTree(newPL_2, leaf_size=2)
        treeLocalFeature = KDTree(newPLFeature_2, leaf_size=2)

        # print( "So far2" )

        dist, indecies = treeLocal.query(input_pointcloud[:, :3], k=1)
        object_points = np.flatnonzero(dist < GOOD_DIST)

        a = [treeLocal, treeLocalFeature, dist, indecies, object_points, object_xyz, input_pointcloud]

        # with open('filename.pickle', 'wb') as handle:
        #     pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # print( "First lasted time: ", time.time() - start_time )
        # start_time = time.time()
        # with open('filename.pickle', 'rb') as handle:
        #     treeLocal, treeLocalFeature, dist, indecies, object_points, object_xyz, input_pointcloud = pickle.load(handle)

        # for each scene save
        # treeLocal
        # treeLocalFeature
        # objectpoints
        # input_pointcloud
        # gt_poses

        if (len(object_points) == 0):
            print("return none")
            return None

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

        for point_sorted_index in range(2):
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

            pointlist = filterPoints(input_pointcloud, point_check[0], radius, gt_poses, input_tree)

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

            # obj_pc_temp = copy.deepcopy(obj_pc)
            obj_pc_temp = o3d.geometry.PointCloud()        
            obj_pc_temp.points = o3d.utility.Vector3dVector(model_pc[:,:3])
            obj_pc_temp.normals = o3d.utility.Vector3dVector(model_pc[:,3:6])
            R = obj_pc_temp.get_rotation_matrix_from_xyz(
                (np.random.uniform(0, np.pi * 2), np.random.uniform(0, np.pi * 2), np.random.uniform(0, np.pi * 2)))
            obj_pc_temp.rotate(R, center=(0, 0, 0))
            model_pc_out = np.concatenate([np.asarray(obj_pc_temp.points), np.asarray(obj_pc_temp.normals)], axis=1)

            # center_point_cloud  # TODO
            model_pc_out = normalize_2d(model_pc_out)

            data, std_data = normalize_1d(data)

            # if False: # if True:
            if self.partition == "train":
                #  jitter on the model
                shift_range = 0.10
                shift = np.random.uniform(-shift_range, shift_range, (3))
                model_pc_out[:, :3] += shift

                #  noise on the model
                # sigma = 0.005
                sigma = np.random.uniform(0.0001, 0.0075)
                clip = sigma * 3
                gaussian_noise = np.clip(sigma * np.random.randn(2048, 6), -clip, clip)
                model_pc_out += gaussian_noise

                #  noise on the scene
                # sigma = 0.005
                sigma = np.random.uniform(0.0001, 0.0075)
                clip = sigma * 3
                gaussian_noise = np.clip(sigma * np.random.randn(2048, 6), -clip, clip)
                data += gaussian_noise

                # print( data.astype('float32').shape, seg.shape, key.shape, model_pc_out.astype('float32').shape, fpi[0].shape)

            # print( "Lasted time: ", time.time() - start_time )
            # list_of_data.append( [data.astype('float32'), seg, key, model_pc_out.astype('float32'), fpi[0]])
            l_data.append(np.copy(data.astype('float32')))
            l_seg.append(np.copy(seg))
            l_key.append(np.copy(key))
            l_mo.append(np.copy(model_pc_out.astype('float32')))
            l_fpi.append(np.copy(fpi[0]))
            l_center.append(np.copy(center_point_list / std_data))

        # return data.astype('float32'), seg, key, model_pc_out.astype('float32'), fpi[0].cpu().numpy()
        # return l_data, l_seg, l_key, l_mo, l_fpi
        # return np.array(l_data), np.array(l_seg), np.array(l_key), np.array(l_mo), np.array(l_fpi), np.array(l_center)
        return np.array(l_data), np.array(l_seg), np.array(l_key), np.array(l_mo), np.array(
            l_fpi)  # , np.array(l_center)
        # return data.astype('float32'), seg, key, model_pc_out.astype('float32'), fpi[0]

    def __len__(self):
        if self.max_data == -1:
            return len(self.full_dataset)
        else:
            return self.max_data

class GPVN_wrs(Dataset):
    # def __init__(self, num_points, partition='train'):
    def __init__(self, number_of_keypoints, partition='train'):

        self.partition = partition
        self.train_index = 1

        self.object_set_list = []
        self.full_dataset = []

        self.number_of_keypoints = number_of_keypoints

        cad_string = "/home/fhagelskjaer/workspace/bin-picking-para-pose/data/models/24_37D-GEARMOTOR-50-70.roty90.stl"
        scene_folder = "/home/fhagelskjaer/workspace/BlenderBin/data/24_37D-GEARMOTOR-50-70.roty90_1/train_pbr/"

        cad_string = "/home/fhagelskjaer/workspace/bin-picking-para-pose/data/models/21_MBRAC60-2-10.stl"
        scene_folder = "/home/fhagelskjaer/workspace/BlenderBin/data/21_MBRAC60-2-10_1/train_pbr/"

        # cad_string = "/home/fhagelskjaer/workspace/df_spindleBase.stl"
        # scene_folder = "/home/fhagelskjaer/workspace/BlenderBin/data_old/df_spindleBase_1/train_pbr/"

        # cad_string = "/home/fhagelskjaer/Desktop/objects_for_tests/le_yellowBrick.stl"
        # scene_folder = "/home/fhagelskjaer/workspace/BlenderBin/data_old/le_yellowBrick_1/train_pbr/"

        # load cad model of object
        obj_cad = o3d.io.read_triangle_mesh(cad_string)
        # subsample 2048 points
        obj_pc = obj_cad.sample_points_poisson_disk(2048)
        obj_pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30), False)

        # o3d.visualization.draw_geometries([obj_pc],point_show_normal=True)

        model_pc = np.concatenate([np.asarray(obj_pc.points), np.asarray(obj_pc.normals)], axis=1)

        candidate = trimesh.load_mesh(cad_string)
        candidate_extents_max = np.max(candidate.extents)

        # self.radius = 4 * np.max( candidate_extents_max )
        # self.radius = 0.6 * np.max( candidate_extents_max )
        radius = np.max(candidate_extents_max)

        self.object_set_list.append(
            {"scene_folder": scene_folder, "obj_pc": obj_pc, "model_pc": model_pc, "radius": radius})

        for i in range(len(self.object_set_list)):
            for j in range(1000):
                self.full_dataset.append([i, j])

    def __getitem__(self, item):
        NUM_POINT = 2048
        GOOD_DIST = 2
        BAD_DIST = 3

        object_set = self.object_set_list[self.full_dataset[item][0]]
        scene_folder = object_set["scene_folder"]
        obj_pc = object_set["obj_pc"]
        model_pc = object_set["model_pc"]
        radius = object_set["radius"]

        object_xyz = np.asarray(obj_pc.points)
        fpi = fps.farthest_point_sampler(object_xyz, self.number_of_keypoints)


        object_xyz_feature = object_xyz[fpi[0], :]

        # get index TODO
        dataset_index = 0

        # item = item%1000
        # item = item%80

        cloud_index = self.full_dataset[item][1]

        # print(scene_folder, cloud_index)

        modelCloud = o3d.io.read_point_cloud(
            scene_folder + str(dataset_index).zfill(6) + "/cloud/" + str(cloud_index).zfill(6) + ".pcd")

        input_tree = KDTree(np.asarray(modelCloud.points), leaf_size=2)

        input_pointcloud = np.concatenate([np.asarray(modelCloud.points), np.asarray(modelCloud.points)], axis=1)

        gt_file = scene_folder + str(dataset_index).zfill(6) + "/scene_gt.json"
        gty = json.load(open(gt_file))

        gt_poses = []

        for key in gty.keys():
            if (key != str(cloud_index)):
                continue
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

        # print( "So far2" )

        dist, indecies = treeLocal.query(input_pointcloud[:, :3], k=1)
        object_points = np.flatnonzero(dist < GOOD_DIST)

        a = [treeLocal, treeLocalFeature, dist, indecies, object_points, object_xyz, input_pointcloud]

        # with open('filename.pickle', 'wb') as handle:
        #     pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # print( "First lasted time: ", time.time() - start_time )
        # start_time = time.time()
        # with open('filename.pickle', 'rb') as handle:
        #     treeLocal, treeLocalFeature, dist, indecies, object_points, object_xyz, input_pointcloud = pickle.load(handle)

        # for each scene save
        # treeLocal
        # treeLocalFeature
        # objectpoints
        # input_pointcloud
        # gt_poses

        if (len(object_points) == 0):
            print("return none")
            return None

        # print( object_points.shape, dist.shape, indecies.shape )
        np.random.shuffle(object_points)
        # print(dist, ind,  point_check)
        # list_of_data = []

        l_data = []
        l_seg = []
        l_key = []
        l_mo = []
        l_fpi = []

        for point_sorted_index in range(2):
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

            pointlist = filterPoints(input_pointcloud, point_check[0], radius, gt_poses, input_tree)

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

            data = normalize_1d(data)

            # if False: # self.partition == "train":
            # if True: # self.partition == "train":
            if self.partition == "train":
                #  jitter on the model
                shift_range = 0.10
                shift = np.random.uniform(-shift_range, shift_range, (3))
                model_pc_out[:, :3] += shift

                #  noise on the model
                # sigma = 0.005
                sigma = np.random.uniform(0.0001, 0.0075)
                clip = sigma * 3
                gaussian_noise = np.clip(sigma * np.random.randn(2048, 6), -clip, clip)
                model_pc_out += gaussian_noise

                #  noise on the scene
                # sigma = 0.005
                sigma = np.random.uniform(0.0001, 0.0075)
                clip = sigma * 3
                gaussian_noise = np.clip(sigma * np.random.randn(2048, 6), -clip, clip)
                data += gaussian_noise

                # print( data.astype('float32').shape, seg.shape, key.shape, model_pc_out.astype('float32').shape, fpi[0].shape)

            # print( "Lasted time: ", time.time() - start_time )
            # list_of_data.append( [data.astype('float32'), seg, key, model_pc_out.astype('float32'), fpi[0]])
            l_data.append(np.copy(data.astype('float32')))
            l_seg.append(np.copy(seg))
            l_key.append(np.copy(key))
            l_mo.append(np.copy(model_pc_out.astype('float32')))
            l_fpi.append(np.copy(fpi[0].cpu().numpy()))

        # return data.astype('float32'), seg, key, model_pc_out.astype('float32'), fpi[0].cpu().numpy()
        # return l_data, l_seg, l_key, l_mo, l_fpi
        return np.array(l_data), np.array(l_seg), np.array(l_key), np.array(l_mo), np.array(l_fpi)
        # return data.astype('float32'), seg, key, model_pc_out.astype('float32'), fpi[0]

    def __len__(self):
        return len(self.full_dataset)


class GPVN_icbin(Dataset):
    # def __init__(self, num_points, partition='train'):
    def __init__(self, number_of_keypoints, obj_id, scene_id, partition='train'):

        self.num_key = number_of_keypoints
        self.partition = partition
        self.train_index = obj_id

        self.object_set_list = []
        self.full_dataset = []

        if self.partition == "test":
            # used_range = range(1,2)
            used_range = [scene_id]
        elif self.partition == "train":
            used_range = [0, 1]
        else:
            print("Unknown partition")
            return

        for hat_index in used_range:
            if self.partition == "train" or self.partition == "val":
                cad_string = "/home/fhagelskjaer/workspace/bop/icbin/models/obj_" + str(hat_index).zfill(6) + ".ply"
                scene_folder = "/home/fhagelskjaer/workspace/bop/icbin/test/" + str(hat_index).zfill(6) + "/"

            elif self.partition == "test":

                # cad_string = "/home/fhagelskjaer/workspace/bop/icbin/models/obj_" + str(hat_index).zfill(6) + ".ply"
                cad_string = "/home/fhagelskjaer/workspace/bop/icbin/models/obj_" + str(obj_id).zfill(6) + ".ply"
                scene_folder = "/home/fhagelskjaer/workspace/bop/icbin/test/" + str(hat_index).zfill(6) + "/"

            # load cad model of object
            obj_cad = o3d.io.read_triangle_mesh(cad_string)
            # subsample 2048 points
            obj_pc = obj_cad.sample_points_poisson_disk(2048)
            obj_pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30), False)

            # o3d.visualization.draw_geometries([obj_pc],point_show_normal=True)

            model_pc = np.concatenate([np.asarray(obj_pc.points), np.asarray(obj_pc.normals)], axis=1)

            candidate = trimesh.load_mesh(cad_string)
            candidate_extents_max = np.max(candidate.extents)

            # self.radius = 4 * np.max( candidate_extents_max )
            # self.radius = 0.6 * np.max( candidate_extents_max )
            radius = np.max(candidate_extents_max)

            self.object_set_list.append(
                {"scene_folder": scene_folder, "obj_pc": obj_pc, "model_pc": model_pc, "radius": radius})

        for i in range(len(self.object_set_list)):
            scene_folder = self.object_set_list[i]["scene_folder"]
            CAMERA_PARAM = scene_folder + "scene_camera.json"
            camera_param = json.load(open(CAMERA_PARAM))
            number_of_samples = len(camera_param)
            for key in list(camera_param.keys()):
                self.full_dataset.append([i, key])

    def __getitem__(self, item):

        # if self.partition == "test":
        #     item += (len(self.full_dataset)-self.test_offset)

        NUM_POINT = 2048
        GOOD_DIST = 10
        BAD_DIST = 12

        object_set = self.object_set_list[self.full_dataset[item][0]]
        scene_folder = object_set["scene_folder"]
        obj_pc = object_set["obj_pc"]
        model_pc = object_set["model_pc"]
        radius = object_set["radius"]

        object_xyz = np.asarray(obj_pc.points)
        fpi = fps.farthest_point_sampler(object_xyz, self.num_key)


        object_xyz_feature = object_xyz[fpi[0], :]

        # get index TODO

        # item = item%1000
        # item = item%80

        cloud_index = self.full_dataset[item][1]

        # print(scene_folder, cloud_index)

        modelCloud = o3d.io.read_point_cloud(scene_folder + "/cloud/" + str(cloud_index).zfill(6) + ".pcd")

        input_tree = KDTree(np.asarray(modelCloud.points), leaf_size=2)

        input_pointcloud = np.concatenate([np.asarray(modelCloud.points), np.asarray(modelCloud.normals)], axis=1)

        gt_file = scene_folder + "/scene_gt.json"
        gty = json.load(open(gt_file))

        gt_poses = []

        if self.partition == "test":
            object_index = self.train_index
        elif self.partition == "train":
            object_index = object_set["train_index"]

        for key in gty.keys():
            if (key != str(cloud_index)):
                continue
            for i in range(len(gty[key])):
                point = gty[key][i]['cam_t_m2c']
                rot = np.array(gty[key][i]['cam_R_m2c']).reshape((3, 3))
                if (gty[key][i]['obj_id'] == int(object_index)):
                    gt_poses.append((point, rot, 1.0))

        if len(gt_poses) == 0:
            print("No visible poses")
            return None

        all_newPL = []
        all_newPLFeature = []
        for gt_pose in gt_poses:
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

        # print( "So far2" )

        """
        dist, ind = treeLocal.query(input_pointcloud[:,:3], k=1)
        color = np.zeros((len(input_pointcloud),3), float)
        color[:,2] = (dist < GOOD_DIST)[:,0] * 255
        # import pdb; pdb.set_trace()

        target = o3d.geometry.PointCloud()        
        target.points = o3d.utility.Vector3dVector(input_pointcloud[:,:3])
        target.colors = o3d.utility.Vector3dVector(color)
        o3d.visualization.draw_geometries([target]) 
        # """

        l_data = []
        l_seg = []
        l_key = []
        l_mo = []
        l_fpi = []

        for _ in range(2):
            # print( "So far3" )
            cnt = 0
            while (True):
                cnt += 1

                if (cnt > 15000):  # Object could not be found
                    print("return none")
                    return None

                randomIndex = np.random.randint(0, len(input_pointcloud))
                # print(randomIndex)
                point_check = input_pointcloud[randomIndex, :3]

                point_check = np.array([point_check])
                point_check.reshape(3, 1)
                dist, ind = treeLocal.query(point_check, k=1)
                # print( dist )

                # print("length xyz", len(object_xyz))

                normalized_point_ind = int(ind / len(object_xyz))

                if (dist[0][0] > 3):
                    continue

                pointlist = filterPoints(input_pointcloud, point_check[0], radius, gt_poses, input_tree)

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

                for i in range(len(pointlist)):

                    p = pointlist[i]
                    dist = dist_list[i]
                    ind = ind_list[i]

                    pt = int(ind / len(object_xyz)) == normalized_point_ind

                    dist_feature = dist_feature_list[i]
                    index_feature = index_feature_list[i] % len(object_xyz_feature)

                    point_list.append(p)
                    feat_list.append(index_feature)

                    if (dist < GOOD_DIST and pt):
                        cat_list.append(1)
                        # pp.append( p )
                        # pf.append( index_feature )
                    elif (dist > BAD_DIST or not pt):
                        cat_list.append(0)
                    else:
                        # cat_list.append( -1 )
                        cat_list.append(0)
                break

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

            if self.partition == "train":
                #  jitter on the model
                shift_range = 0.1
                shift = np.random.uniform(-shift_range, shift_range, (3))
                model_pc_out[:, :3] += shift

                #  noise on the model
                sigma = 0.02
                clip = sigma * 3
                gaussian_noise = np.clip(sigma * np.random.randn(2048, 6), -clip, clip)
                model_pc_out += gaussian_noise

                #  noise on the scene
                sigma = 0.02
                clip = sigma * 3
                gaussian_noise = np.clip(sigma * np.random.randn(2048, 6), -clip, clip)
                data += gaussian_noise

            l_data.append(np.copy(data.astype('float32')))
            l_seg.append(np.copy(seg))
            l_key.append(np.copy(key))
            l_mo.append(np.copy(model_pc_out.astype('float32')))
            l_fpi.append(np.copy(fpi[0].cpu().numpy()))

        # return data.astype('float32'), seg, key, model_pc_out.astype('float32'), fpi[0].cpu().numpy()
        # return l_data, l_seg, l_key, l_mo, l_fpi
        return np.array(l_data), np.array(l_seg), np.array(l_key), np.array(l_mo), np.array(l_fpi)
        # print( data.astype('float32').shape, seg.shape, key.shape, model_pc_out.astype('float32').shape, fpi[0].shape)
        # return data.astype('float32'), seg, key, model_pc_out.astype('float32'), fpi[0]

    def __len__(self):
        return len(self.full_dataset)



class GPVN_tless(Dataset):
    # def __init__(self, num_points, partition='train'):
    def __init__(self, number_of_keypoints, max_data=-1, partition='train'):

        self.num_key = number_of_keypoints
        self.max_data = max_data
        self.partition = partition

        self.object_set_list = []
        self.full_dataset = []

        if self.partition == "test":
            # used_range = range(1,2)
            used_range = [scene_id]
        elif self.partition == "train":
            model_used_range =  range(1, 31)
            scene_used_range = range(1, 21)
        else:
            print("Unknown partition")
            return

        for hat_index in model_used_range:
            if self.partition == "train" or self.partition == "val":
                cad_string = "/home/fhagelskjaer/workspace/bop/tless/models_cad/obj_" + str(hat_index).zfill(6) + ".ply"
                # scene_folder = "/home/fhagelskjaer/workspace/bop/tless/test_primesense/" + str(hat_index).zfill(6) + "/"

            # load cad model of object
            obj_cad = o3d.io.read_triangle_mesh(cad_string)
            # subsample 2048 points
            obj_pc = obj_cad.sample_points_poisson_disk(2048)
            obj_pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30), False)

            # o3d.visualization.draw_geometries([obj_pc],point_show_normal=True)

            model_pc = np.concatenate([np.asarray(obj_pc.points), np.asarray(obj_pc.normals)], axis=1)

            candidate = trimesh.load_mesh(cad_string)
            candidate_extents_max = np.max(candidate.extents)

            # self.radius = 4 * np.max( candidate_extents_max )
            # self.radius = 0.6 * np.max( candidate_extents_max )
            radius = np.max(candidate_extents_max)

            model_center = obj_pc.get_center()

            self.object_set_list.append(
                {"scene_folder": None, "obj_pc": obj_pc, "model_pc": model_pc, "radius": radius, "obj_id": hat_index, "model_center": model_center})

        for i, object_set in enumerate(self.object_set_list):

            for scene_index in scene_used_range:
                scene_folder = "/home/fhagelskjaer/workspace/bop/tless/test_primesense/" + str(scene_index).zfill(6) + "/"

                gt_file = scene_folder + "scene_gt.json"
                gty = json.load(open(gt_file))
                keys = gty.keys()

                gt_file_info = scene_folder + "scene_gt_info.json"
                gty_info = json.load(open(gt_file_info))

                
                for key in keys:
                    for instance_idx, instance in enumerate(gty[key]):
                        if object_set["obj_id"] == instance["obj_id"] and gty_info[key][instance_idx]["visib_fract"] > 0.50:
                            # \
                            # and not(object_set["obj_id"] == 2 and scene_index == 1 and int(key) == 474) \
                            # and not(object_set["obj_id"] == 3 and scene_index == 7 and int(key) == 487) \
                            # and not(object_set["obj_id"] == 1 and scene_index == 7 and int(key) == 478):
                            self.full_dataset.append([i, int(key), scene_folder])
                            # print(i, int(key), scene_folder)
                            break

    def __getitem__(self, item):
        returning = self.getitem(item)
        while returning is None:
            rand_item = np.random.randint(self.__len__())
            returning = self.getitem(rand_item)
        return returning

    def getitem(self, item):

        # if self.partition == "test":
        #     item += (len(self.full_dataset)-self.test_offset)

        NUM_POINT = 2048
        GOOD_DIST = 10
        BAD_DIST = 12

        object_set = self.object_set_list[self.full_dataset[item][0]]
        # scene_folder = object_set["scene_folder"]
        scene_folder = self.full_dataset[item][2]
        obj_pc = object_set["obj_pc"]
        model_pc = object_set["model_pc"]
        radius = object_set["radius"]
        model_center = object_set["model_center"]

        object_xyz = np.asarray(obj_pc.points)
        fpi = fps.farthest_point_sampler(object_xyz, self.num_key)


        object_xyz_feature = object_xyz[fpi[0], :]

        # get index TODO

        # item = item%1000
        # item = item%80

        cloud_index = self.full_dataset[item][1]

        # print(scene_folder, cloud_index)

        modelCloud = o3d.io.read_point_cloud(scene_folder + "/cloud/" + str(cloud_index).zfill(6) + ".pcd")

        input_tree = KDTree(np.asarray(modelCloud.points), leaf_size=2)

        input_pointcloud = np.concatenate([np.asarray(modelCloud.points), np.asarray(modelCloud.normals)], axis=1)

        gt_file = scene_folder + "/scene_gt.json"
        gty = json.load(open(gt_file))

        gt_poses = []

        object_index = object_set["obj_id"]

        for key in gty.keys():
            if (key != str(cloud_index)):
                continue
            for i in range(len(gty[key])):
                point = gty[key][i]['cam_t_m2c']
                rot = np.array(gty[key][i]['cam_R_m2c']).reshape((3, 3))
                if (gty[key][i]['obj_id'] == int(object_index)):
                    gt_poses.append((point, rot, 1.0))

        if len(gt_poses) == 0:
            print("No visible poses")
            return None

        all_newPL = []
        all_newPLFeature = []
        all_newCenter = []
        for gt_pose in gt_poses:
            newPL = np.matmul(object_xyz, gt_pose[1].transpose())
            newPL = newPL + gt_pose[0]

            newPLFeature = np.matmul(object_xyz_feature, gt_pose[1].transpose())
            newPLFeature = newPLFeature + gt_pose[0]

            all_newPL.append(newPL)
            all_newPLFeature.append(newPLFeature)

            new_center = np.matmul(model_center, gt_pose[1].transpose() )
            new_center = new_center + gt_pose[0]
            # all_new_center.append(newPL - new_center) 
            all_newCenter.append([new_center]) 



        newPL_2 = np.concatenate(all_newPL)
        newPLFeature_2 = np.concatenate(all_newPLFeature)


        treeLocal = KDTree(newPL_2, leaf_size=2)
        treeLocalFeature = KDTree(newPLFeature_2, leaf_size=2)



        

        input_pointcloud = np.concatenate([np.asarray(modelCloud.points), np.asarray(modelCloud.normals)], axis=1)
        
        all_newCenter_2 = np.concatenate(all_newCenter)
        centerTree = KDTree(all_newCenter_2, leaf_size=2)
        dist, indecies = centerTree.query(input_pointcloud[:, :3], k=1)
        # print( dist.shape )
        input_pointcloud = input_pointcloud[dist.flatten() < radius*1.5, :]
        
        if (len(input_pointcloud) == 0):
            # print( object_set["obj_id"], scene_folder + "/cloud/" + str(cloud_index).zfill(6), gt_poses)
            return None
        
        # input_tree = KDTree(np.asarray(modelCloud.points), leaf_size=2)
        input_tree = KDTree(input_pointcloud[:,:3], leaf_size=2)



        dist, indecies = treeLocal.query(input_pointcloud[:, :3], k=1)
        object_points = np.flatnonzero(dist < GOOD_DIST)
        # with open('filename.pickle', 'wb') as handle:
        #     pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # print( "First lasted time: ", time.time() - start_time )
        # start_time = time.time()
        # with open('filename.pickle', 'rb') as handle:
        #     treeLocal, treeLocalFeature, dist, indecies, object_points, object_xyz, input_pointcloud = pickle.load(handle)

        # for each scene save
        # treeLocal
        # treeLocalFeature
        # objectpoints
        # input_pointcloud
        # gt_poses

        if (len(object_points) == 0):
            # print( object_set["obj_id"], scene_folder + "/cloud/" + str(cloud_index).zfill(6), gt_poses)
            return None

        # print( object_points.shape, dist.shape, indecies.shape )
        np.random.shuffle(object_points)
        # print(dist, ind,  point_check)
        # list_of_data = []

        l_data = []
        l_seg = []
        l_key = []
        l_mo = []
        l_fpi = []

        for point_sorted_index in range(2):
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

            pointlist = filterPoints(input_pointcloud, point_check[0], radius, gt_poses, input_tree)

            while (len(pointlist) < NUM_POINT):
                pointlist = np.array(list(pointlist) + list(pointlist))
            np.random.shuffle(pointlist)

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

            data, std_data = normalize_1d(data)

            if self.partition == "train":
                #  jitter on the model
                shift_range = 0.10
                shift = np.random.uniform(-shift_range, shift_range, (3))
                model_pc_out[:, :3] += shift

                #  noise on the model
                # sigma = 0.005
                sigma = np.random.uniform(0.0001, 0.0075)
                clip = sigma * 3
                gaussian_noise = np.clip(sigma * np.random.randn(2048, 6), -clip, clip)
                model_pc_out += gaussian_noise

                #  noise on the scene
                # sigma = 0.005
                sigma = np.random.uniform(0.0001, 0.0075)
                clip = sigma * 3
                gaussian_noise = np.clip(sigma * np.random.randn(2048, 6), -clip, clip)
                data += gaussian_noise
            # print( "Lasted time: ", time.time() - start_time )
            # list_of_data.append( [data.astype('float32'), seg, key, model_pc_out.astype('float32'), fpi[0]])
            l_data.append(np.copy(data.astype('float32')))
            l_seg.append(np.copy(seg))
            l_key.append(np.copy(key))
            l_mo.append(np.copy(model_pc_out.astype('float32')))
            l_fpi.append(np.copy(fpi[0].cpu().numpy()))

        return np.array(l_data), np.array(l_seg), np.array(l_key), np.array(l_mo), np.array(l_fpi)

    def __len__(self):
        if self.max_data == -1:
            return len(self.full_dataset)
        else:
            return self.max_data


class GPVN_lmo(Dataset):
    # def __init__(self, num_points, partition='train'):
    def __init__(self, number_of_keypoints, max_data=-1, partition='train'):

        self.num_key = number_of_keypoints
        self.max_data = max_data
        self.partition = partition
    

        self.object_set_list = []
        self.full_dataset = []

        if self.partition == "test":
            # used_range = range(1,2)
            used_range = [scene_id]
        elif self.partition == "train":
            model_used_range = [1,5,6,8,9,10,11,12]
            scene_used_range = [2]
        else:
            print("Unknown partition")
            return

        for hat_index in model_used_range:
            if self.partition == "train" or self.partition == "val":
                cad_string = "/home/fhagelskjaer/workspace/bop/lmo/models/obj_" + str(hat_index).zfill(6) + ".ply"
                # scene_folder = "/home/fhagelskjaer/workspace/bop/tless/test_primesense/" + str(hat_index).zfill(6) + "/"

            # load cad model of object
            obj_cad = o3d.io.read_triangle_mesh(cad_string)
            # subsample 2048 points
            obj_pc = obj_cad.sample_points_poisson_disk(2048)
            obj_pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30), False)

            # o3d.visualization.draw_geometries([obj_pc],point_show_normal=True)

            model_pc = np.concatenate([np.asarray(obj_pc.points), np.asarray(obj_pc.normals)], axis=1)

            candidate = trimesh.load_mesh(cad_string)
            candidate_extents_max = np.max(candidate.extents)

            # self.radius = 4 * np.max( candidate_extents_max )
            # self.radius = 0.6 * np.max( candidate_extents_max )
            radius = np.max(candidate_extents_max)

            model_center = obj_pc.get_center()

            self.object_set_list.append(
                {"scene_folder": None, "obj_pc": obj_pc, "model_pc": model_pc, "radius": radius, "obj_id": hat_index, "model_center": model_center})

        for i, object_set in enumerate(self.object_set_list):

            for scene_index in scene_used_range:
                scene_folder = "/home/fhagelskjaer/workspace/bop/lmo/test/" + str(scene_index).zfill(6) + "/"

                gt_file = scene_folder + "scene_gt.json"
                gty = json.load(open(gt_file))
                keys = gty.keys()

                gt_file_info = scene_folder + "scene_gt_info.json"
                gty_info = json.load(open(gt_file_info))

                
                for key in keys:
                    for instance_idx, instance in enumerate(gty[key]):
                        if object_set["obj_id"] == instance["obj_id"] and gty_info[key][instance_idx]["visib_fract"] > 0.50:
                            self.full_dataset.append([i, int(key), scene_folder])
                            # print(i, int(key), scene_folder)
                            break

    def __getitem__(self, item):
        returning = self.getitem(item)
        while returning is None:
            rand_item = np.random.randint(self.__len__())
            returning = self.getitem(rand_item)
        return returning

    def getitem(self, item):

        # if self.partition == "test":
        #     item += (len(self.full_dataset)-self.test_offset)

        NUM_POINT = 2048
        GOOD_DIST = 10
        BAD_DIST = 12

        object_set = self.object_set_list[self.full_dataset[item][0]]
        # scene_folder = object_set["scene_folder"]
        scene_folder = self.full_dataset[item][2]
        obj_pc = object_set["obj_pc"]
        model_pc = object_set["model_pc"]
        radius = object_set["radius"]
        model_center = object_set["model_center"]


        object_xyz = np.asarray(obj_pc.points)
        fpi = fps.farthest_point_sampler(object_xyz, self.number_of_keypoints)


        object_xyz_feature = object_xyz[fpi[0], :]

        # get index TODO

        cloud_index = self.full_dataset[item][1]

        # print(scene_folder, cloud_index)

        modelCloud = o3d.io.read_point_cloud(scene_folder + "/cloud/" + str(cloud_index).zfill(6) + ".pcd")
        

        gt_file = scene_folder + "/scene_gt.json"
        gty = json.load(open(gt_file))

        gt_poses = []

        object_index = object_set["obj_id"]

        for key in gty.keys():
            if (key != str(cloud_index)):
                continue
            for i in range(len(gty[key])):
                point = gty[key][i]['cam_t_m2c']
                rot = np.array(gty[key][i]['cam_R_m2c']).reshape((3, 3))
                if (gty[key][i]['obj_id'] == int(object_index)):
                    gt_poses.append((point, rot, 1.0))

        if len(gt_poses) == 0:
            print("No visible poses")
            return None

        all_newPL = []
        all_newPLFeature = []
        all_newCenter = []
        
        for gt_pose in gt_poses:
            newPL = np.matmul(object_xyz, gt_pose[1].transpose())
            newPL = newPL + gt_pose[0]

            newPLFeature = np.matmul(object_xyz_feature, gt_pose[1].transpose())
            newPLFeature = newPLFeature + gt_pose[0]

            all_newPL.append(newPL)
            all_newPLFeature.append(newPLFeature)

            new_center = np.matmul(model_center, gt_pose[1].transpose() )
            new_center = new_center + gt_pose[0]
            # all_new_center.append(newPL - new_center) 
            all_newCenter.append([new_center]) 



        newPL_2 = np.concatenate(all_newPL)
        newPLFeature_2 = np.concatenate(all_newPLFeature)


        treeLocal = KDTree(newPL_2, leaf_size=2)
        treeLocalFeature = KDTree(newPLFeature_2, leaf_size=2)



        

        input_pointcloud = np.concatenate([np.asarray(modelCloud.points), np.asarray(modelCloud.normals)], axis=1)
        
        all_newCenter_2 = np.concatenate(all_newCenter)
        centerTree = KDTree(all_newCenter_2, leaf_size=2)
        dist, indecies = centerTree.query(input_pointcloud[:, :3], k=1)
        # print( dist.shape )
        input_pointcloud = input_pointcloud[dist.flatten() < radius*1.5, :]
        
        # input_tree = KDTree(np.asarray(modelCloud.points), leaf_size=2)
        input_tree = KDTree(input_pointcloud[:,:3], leaf_size=2)



        dist, indecies = treeLocal.query(input_pointcloud[:, :3], k=1)
        object_points = np.flatnonzero(dist < GOOD_DIST)

        # a = [treeLocal, treeLocalFeature, dist, indecies, object_points, object_xyz, input_pointcloud]
        # with open('filename.pickle', 'wb') as handle:
        #     pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # print( "First lasted time: ", time.time() - start_time )
        # start_time = time.time()
        # with open('filename.pickle', 'rb') as handle:
        #     treeLocal, treeLocalFeature, dist, indecies, object_points, object_xyz, input_pointcloud = pickle.load(handle)

        # for each scene save
        # treeLocal
        # treeLocalFeature
        # objectpoints
        # input_pointcloud
        # gt_poses

        if (len(object_points) == 0):
            # print( object_set["obj_id"], scene_folder + "/cloud/" + str(cloud_index).zfill(6), gt_poses)
            return None

        # print( object_points.shape, dist.shape, indecies.shape )
        np.random.shuffle(object_points)
        # print(dist, ind,  point_check)
        # list_of_data = []

        l_data = []
        l_seg = []
        l_key = []
        l_mo = []
        l_fpi = []

        for point_sorted_index in range(2):
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

            pointlist = filterPoints(input_pointcloud, point_check[0], radius, gt_poses, input_tree)

            while (len(pointlist) < NUM_POINT):
                pointlist = np.array(list(pointlist) + list(pointlist))
            np.random.shuffle(pointlist)

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

            data, std_data = normalize_1d(data)
            
            if self.partition == "train":
                #  jitter on the model
                shift_range = 0.10
                shift = np.random.uniform(-shift_range, shift_range, (3))
                model_pc_out[:, :3] += shift

                #  noise on the model
                # sigma = 0.005
                sigma = np.random.uniform(0.0001, 0.0075)
                clip = sigma * 3
                gaussian_noise = np.clip(sigma * np.random.randn(2048, 6), -clip, clip)
                model_pc_out += gaussian_noise

                #  noise on the scene
                # sigma = 0.005
                sigma = np.random.uniform(0.0001, 0.0075)
                clip = sigma * 3
                gaussian_noise = np.clip(sigma * np.random.randn(2048, 6), -clip, clip)
                data += gaussian_noise

            # print( "Lasted time: ", time.time() - start_time )
            # list_of_data.append( [data.astype('float32'), seg, key, model_pc_out.astype('float32'), fpi[0]])
            l_data.append(np.copy(data.astype('float32')))
            l_seg.append(np.copy(seg))
            l_key.append(np.copy(key))
            l_mo.append(np.copy(model_pc_out.astype('float32')))
            l_fpi.append(np.copy(fpi[0].cpu().numpy()))

        return np.array(l_data), np.array(l_seg), np.array(l_key), np.array(l_mo), np.array(l_fpi)

    def __len__(self):
        if self.max_data == -1:
            return len(self.full_dataset)
        else:
            return self.max_data



from scipy.spatial.transform import Rotation as R


def filterPoints(pointList, p, distance, groundtruth, tree):
    point = np.array([p[:3]])
    point.reshape(3, 1)

    filteredList = tree.query_radius(np.array(point), r=distance)
    filteredList = pointList[filteredList[0]]

    return filteredList

class GPVN_megapose(Dataset):
    # def __init__(self, num_points, partition='train'):
    def __init__(self, number_of_keypoints, max_data=-1, partition='train'):

        self.max_data = max_data
        self.partition = partition
        self.train_index = 1

        # self.object_set_list = []
        self.full_dataset = []

        self.number_of_keypoints = number_of_keypoints

        # training_models = glob.glob("../megapose/google_obj/google_scanned_objects/models_bop-renderer_scale=0.1/*")
        
        training_models = json.load(open("../megapose/google_obj/google_scanned_objects/valid_meshes.json", 'r')) 

        # print( training_models )

        self.object_set_set = {}

        for file_string in training_models:
        # for file_string in training_models[:10]:

            # print( file_string )

            cad_string = "../megapose/google_obj/google_scanned_objects/models_bop-renderer_scale=0.1/" +  file_string + "/meshes/model.ply"
            
            # print( cad_string )

            # load cad model of object
            
            candidate = trimesh.load_mesh(cad_string)
            candidate.export('stuff.stl')
            candidate_extents_max = np.max(candidate.extents)

            # print( candidate.extents )

            # self.radius = 4 * np.max( candidate_extents_max )
            # self.radius = 0.6 * np.max( candidate_extents_max )
            radius = np.max(candidate_extents_max)
            
            # obj_cad = o3d.io.read_triangle_mesh(cad_string)
            obj_cad = o3d.io.read_triangle_mesh('stuff.stl')

            # subsample 2048 points
            obj_pc = obj_cad.sample_points_poisson_disk(2048)

            obj_pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30), False)

            # o3d.visualization.draw_geometries([obj_pc],point_show_normal=True)

            model_pc = np.concatenate([np.asarray(obj_pc.points), np.asarray(obj_pc.normals)], axis=1)

            model_center = obj_pc.get_center()

            label = file_string # .split('/')[-1]
            # print( label )

            self.object_set_set[label] = {"obj_pc": obj_pc, "model_pc": model_pc, "radius": radius, "model_center": model_center}
        
        object_set_set_keys = list(self.object_set_set.keys())

        # print( "object_set_set_keys" )
        # print( object_set_set_keys )

        self.scene_list = glob.glob("../megapose/scenes/*object_datas.json")
        for scene in self.scene_list:
            # print( scene )
            models_in_scene = json.load(open(scene, 'r'))
            # print(models_in_scene)
            for model_info in models_in_scene:
                # print(model_info["label"][4:], model_info["visib_fract"], model_info["TWO"])
                if model_info["visib_fract"] > 0.20 and model_info["label"][4:] in object_set_set_keys:
                    # print(model_info["label"], model_info["visib_fract"], model_info["TWO"])
                    transform = np.eye(4)
                    r = R.from_quat(model_info["TWO"][0]) 
                    temp_tcp = np.eye(4)     
                    transform[:3, :3] = r.as_matrix()     
                    transform[:3, 3] = model_info["TWO"][1]
                    transform[:3, 3] *= 1000.0

                    self.full_dataset.append({"cloud": scene[:-18]+".pcd", "label": model_info["label"][4:], "transform": transform})

    def __getitem__(self, item):
        NUM_POINT = 2048
        GOOD_DIST = 5
        BAD_DIST = 3
        while(True):

            if self.max_data != -1:
                item = np.random.randint(len(self.full_dataset))

            object_set = self.object_set_set[self.full_dataset[item]["label"]]

            # scene_folder = object_set["scene_folder"]
            obj_pc = object_set["obj_pc"]
            model_pc = object_set["model_pc"]
            radius = object_set["radius"]
            model_center = object_set["model_center"]

            object_xyz = np.asarray(obj_pc.points)
            fpi = fps.farthest_point_sampler(object_xyz, self.number_of_keypoints)


            object_xyz_feature = object_xyz[fpi[0], :]

            cloud_name = self.full_dataset[item]["cloud"]

            # print(scene_folder, cloud_index)



            modelCloud = o3d.io.read_point_cloud(cloud_name)
            


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

            if (len(object_points) < 50):
                print(item, self.full_dataset[item]["label"], self.full_dataset[item]["cloud"], self.full_dataset[item]["transform"]) 
                print("return none")
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

        for point_sorted_index in range(6):
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

            data, std_data = normalize_1d(data)

            # if False: # if True:
            if self.partition == "train":
                #  jitter on the model
                shift_range = 0.10
                shift = np.random.uniform(-shift_range, shift_range, (3))
                model_pc_out[:, :3] += shift

                #  noise on the model
                # sigma = 0.005
                sigma = np.random.uniform(0.0001, 0.0075)
                clip = sigma * 3
                gaussian_noise = np.clip(sigma * np.random.randn(2048, 6), -clip, clip)
                model_pc_out += gaussian_noise

                #  noise on the scene
                # sigma = 0.005
                sigma = np.random.uniform(0.0001, 0.0075)
                clip = sigma * 3
                gaussian_noise = np.clip(sigma * np.random.randn(2048, 6), -clip, clip)
                data += gaussian_noise

                # print( data.astype('float32').shape, seg.shape, key.shape, model_pc_out.astype('float32').shape, fpi[0].shape)

            # print( "Lasted time: ", time.time() - start_time )
            # list_of_data.append( [data.astype('float32'), seg, key, model_pc_out.astype('float32'), fpi[0]])
            l_data.append(np.copy(data.astype('float32')))
            l_seg.append(np.copy(seg))
            l_key.append(np.copy(key))
            l_mo.append(np.copy(model_pc_out.astype('float32')))
            l_fpi.append(np.copy(fpi[0].cpu().numpy()))
            l_center.append(np.copy(center_point_list / std_data))

        # return data.astype('float32'), seg, key, model_pc_out.astype('float32'), fpi[0].cpu().numpy()
        # return l_data, l_seg, l_key, l_mo, l_fpi
        # return np.array(l_data), np.array(l_seg), np.array(l_key), np.array(l_mo), np.array(l_fpi), np.array(l_center)
        return np.array(l_data), np.array(l_seg), np.array(l_key), np.array(l_mo), np.array(l_fpi)  # , np.array(l_center)
        # return data.astype('float32'), seg, key, model_pc_out.astype('float32'), fpi[0]

    def __len__(self):
        if self.max_data == -1:
            return len(self.full_dataset)
        else:
            return self.max_data


import pandas as pd

class GPVN_data_engine(Dataset):
    # def __init__(self, num_points, partition='train'):
    def __init__(self, number_of_keypoints, max_data=-1, partition='train'):

        self.max_data = max_data
        self.partition = partition
        self.train_index = 1

        # self.object_set_list = []
        self.full_dataset = []

        self.number_of_keypoints = number_of_keypoints


        self.object_set_set = {}


        training_models = [["8","../novo/asply/216M003.ply"]]

        for label, file_string in training_models:
        # for file_string in training_models[:10]:

            cad_string = file_string 
            
            # print( cad_string )

            # load cad model of object
            
            candidate = trimesh.load_mesh(cad_string)
            candidate.export('stuff.stl')
            candidate_extents_max = np.max(candidate.extents)

            # print( candidate.extents )

            # self.radius = 4 * np.max( candidate_extents_max )
            # self.radius = 0.6 * np.max( candidate_extents_max )
            radius = np.max(candidate_extents_max)
            
            # obj_cad = o3d.io.read_triangle_mesh(cad_string)
            obj_cad = o3d.io.read_triangle_mesh('stuff.stl')

            # subsample 2048 points
            obj_pc = obj_cad.sample_points_poisson_disk(2048)

            obj_pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30), False)

            # o3d.visualization.draw_geometries([obj_pc],point_show_normal=True)

            model_pc = np.concatenate([np.asarray(obj_pc.points), np.asarray(obj_pc.normals)], axis=1)

            model_center = obj_pc.get_center()

            # label = file_string # .split('/')[-1]
            # print( label )

            self.object_set_set[label] = {"obj_pc": obj_pc, "model_pc": model_pc, "radius": radius, "model_center": model_center}
        
        object_set_set_keys = list(self.object_set_set.keys())

        # print( "object_set_set_keys" )
        # print( object_set_set_keys )



        df = pd.read_csv("novolog/good_poses.csv")
        df_len = len(df)

        if partition == "train":
            df = df[:int(df_len*0.8)]
        else:
            df = df[int(df_len*0.8):]

        # import pdb; pdb.set_trace()
        for index, pe_set in df.iterrows():
            scene = "novolog/" + pe_set["pc"].strip("\n") + "point_cloud.pcd" 
            model_info = str(pe_set["idx"])
            transform = pe_set["cam2obj"]
            out = transform.replace("\n","").replace("[","").replace("]","").split(" ")
            while "" in out:
                out.remove("")
            arr = np.array(out, float)
            transform = np.reshape(arr,(4,4))
       
            # import pdb; pdb.set_trace()
            self.full_dataset.append({"cloud": scene, "label": model_info, "transform": transform})

    def __getitem__(self, item):
        NUM_POINT = 2048
        GOOD_DIST = 5
        BAD_DIST = 3
        while(True):

            if self.max_data != -1:
                item = np.random.randint(len(self.full_dataset))

            object_set = self.object_set_set[self.full_dataset[item]["label"]]

            # scene_folder = object_set["scene_folder"]
            obj_pc = object_set["obj_pc"]
            model_pc = object_set["model_pc"]
            radius = object_set["radius"]
            model_center = object_set["model_center"]

            object_xyz = np.asarray(obj_pc.points)
            x = torch.FloatTensor(object_xyz)
            x = np.reshape(x, (1, -1, 3))
            if self.partition == "train" and np.random.rand() < 0.6:
                fpi_idx = list(range(object_xyz.shape[0])) 
                np.random.shuffle(fpi_idx)
                fpi = np.array(fpi_idx[:self.number_of_keypoints])
            else:
                fpi = fps.farthest_point_sampler(object_xyz, self.number_of_keypoints)


            object_xyz_feature = object_xyz[fpi, :]

            cloud_name = self.full_dataset[item]["cloud"]

            # print(scene_folder, cloud_index)

            modelCloud = o3d.io.read_point_cloud(cloud_name)


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
                print(item, self.full_dataset[item]["label"], self.full_dataset[item]["cloud"], self.full_dataset[item]["transform"]) 
                print("return none")
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

        for point_sorted_index in range(2):
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

            data, std_data = normalize_1d(data)

            # if False: # if True:
            if self.partition == "train":
                #  jitter on the model
                shift_range = 0.10
                shift = np.random.uniform(-shift_range, shift_range, (3))
                model_pc_out[:, :3] += shift

                #  noise on the model
                # sigma = 0.005
                sigma = np.random.uniform(0.0001, 0.0075)
                clip = sigma * 3
                gaussian_noise = np.clip(sigma * np.random.randn(2048, 6), -clip, clip)
                model_pc_out += gaussian_noise

                #  noise on the scene
                # sigma = 0.005
                sigma = np.random.uniform(0.0001, 0.0075)
                clip = sigma * 3
                gaussian_noise = np.clip(sigma * np.random.randn(2048, 6), -clip, clip)
                data += gaussian_noise

                # print( data.astype('float32').shape, seg.shape, key.shape, model_pc_out.astype('float32').shape, fpi[0].shape)

            # print( "Lasted time: ", time.time() - start_time )
            # list_of_data.append( [data.astype('float32'), seg, key, model_pc_out.astype('float32'), fpi[0]])
            l_data.append(np.copy(data.astype('float32')))
            l_seg.append(np.copy(seg))
            l_key.append(np.copy(key))
            l_mo.append(np.copy(model_pc_out.astype('float32')))
            l_fpi.append(np.copy(fpi[0]))
            l_center.append(np.copy(center_point_list / std_data))

        # return data.astype('float32'), seg, key, model_pc_out.astype('float32'), fpi[0].cpu().numpy()
        # return l_data, l_seg, l_key, l_mo, l_fpi
        # return np.array(l_data), np.array(l_seg), np.array(l_key), np.array(l_mo), np.array(l_fpi), np.array(l_center)
        return np.array(l_data), np.array(l_seg), np.array(l_key), np.array(l_mo), np.array(l_fpi)  # , np.array(l_center)
        # return data.astype('float32'), seg, key, model_pc_out.astype('float32'), fpi[0]

    def __len__(self):
        if self.max_data == -1:
            return len(self.full_dataset)
        else:
            return self.max_data


class GPVN_data_engine2(Dataset):
    # def __init__(self, num_points, partition='train'):
    def __init__(self, number_of_keypoints, datafolder, training_models, max_data=-1, partition='train'):

        self.max_data = max_data
        self.partition = partition
        self.train_index = 1

        # self.object_set_list = []
        self.full_dataset = []

        self.number_of_keypoints = number_of_keypoints

        self.object_set_set = {}

        for label, file_string in training_models:
        # for file_string in training_models[:10]:

            cad_string = file_string 
            
            # print( cad_string )

            # load cad model of object
            
            candidate = trimesh.load_mesh(cad_string)
            candidate.export('stuff.stl')
            candidate_extents_max = np.max(candidate.extents)

            # print( candidate.extents )

            # self.radius = 4 * np.max( candidate_extents_max )
            # self.radius = 0.6 * np.max( candidate_extents_max )
            radius = np.max(candidate_extents_max)
            
            # obj_cad = o3d.io.read_triangle_mesh(cad_string)
            obj_cad = o3d.io.read_triangle_mesh('stuff.stl')

            # subsample 2048 points
            obj_pc = obj_cad.sample_points_poisson_disk(2048)

            obj_pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30), False)

            # o3d.visualization.draw_geometries([obj_pc],point_show_normal=True)

            model_pc = np.concatenate([np.asarray(obj_pc.points), np.asarray(obj_pc.normals)], axis=1)

            model_center = obj_pc.get_center()

            # label = file_string # .split('/')[-1]
            # print( label )

            self.object_set_set[label] = {"obj_pc": obj_pc, "model_pc": model_pc, "radius": radius, "model_center": model_center}
        
        object_set_set_keys = list(self.object_set_set.keys())

        # print( "object_set_set_keys" )
        # print( object_set_set_keys )
        
        #datafolder = "novolog4"
        #datafolder = "novolog3dprint"
        #datafolder = "../novolog_obj3"

        df = pd.read_pickle(datafolder+"/good_poses.pkl")

        if partition == "train2000":
            df = df[200:2200]
        elif partition == "train1000":
            df = df[200:1200]
        elif partition == "train500":
            df = df[200:700]
        elif partition == "train200":
            df = df[200:400]
        elif partition == "train20":
            df = df[200:220]
        elif partition == "train10":
            df = df[200:210]
        elif partition == "train1":
            df = df[200:201]
        elif partition == "test":
            df = df[:200]
        elif partition == "train":
            df = df[200:]
        else:
            print("Illigal partion")
            sys.exit()
            # f = df[200:]

        # import pdb; pdb.set_trace()
        for index, pe_set in df.iterrows():
            scene = datafolder + "/" + pe_set["pc"].strip("\n") + "point_cloud.pcd" 
            model_info = str(pe_set["idx"])
            transform = pe_set["cam2obj"]
       
            # import pdb; pdb.set_trace()
            self.full_dataset.append({"cloud": scene, "label": model_info, "transform": transform})
        print( "length of the dataset is:", len(self.full_dataset))


    def __getitem__(self, item):
        NUM_POINT = 2048
        GOOD_DIST = 5
        BAD_DIST = 3
        while(True):

            # TODO: this should not be true
            if self.max_data != -1:
                item = np.random.randint(len(self.full_dataset))

            object_set = self.object_set_set[self.full_dataset[item]["label"]]

            # scene_folder = object_set["scene_folder"]
            obj_pc = object_set["obj_pc"]
            model_pc = object_set["model_pc"]
            radius = object_set["radius"]
            model_center = object_set["model_center"]

            object_xyz = np.asarray(obj_pc.points)
            x = torch.FloatTensor(object_xyz)
            x = np.reshape(x, (1, -1, 3))
            if self.partition == "train" and np.random.rand() < 0.6:
                fpi_idx = list(range(object_xyz.shape[0])) 
                np.random.shuffle(fpi_idx)
                fpi = np.array(fpi_idx[:self.number_of_keypoints])
            else:
                fpi = fps.farthest_point_sampler(object_xyz, self.number_of_keypoints)


            object_xyz_feature = object_xyz[fpi, :]

            cloud_name = self.full_dataset[item]["cloud"]

            # print(scene_folder, cloud_index)

            modelCloud = o3d.io.read_point_cloud(cloud_name)


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
                print(item, self.full_dataset[item]["label"], self.full_dataset[item]["cloud"], self.full_dataset[item]["transform"]) 
                print("return none")
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

        for point_sorted_index in range(2):
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

            data, std_data = normalize_1d(data)

            # if False: # if True:
            if self.partition == "train":
                #  jitter on the model
                shift_range = 0.10
                shift = np.random.uniform(-shift_range, shift_range, (3))
                model_pc_out[:, :3] += shift

                #  noise on the model
                # sigma = 0.005
                sigma = np.random.uniform(0.0001, 0.0075)
                clip = sigma * 3
                gaussian_noise = np.clip(sigma * np.random.randn(2048, 6), -clip, clip)
                model_pc_out += gaussian_noise

                #  noise on the scene
                # sigma = 0.005
                sigma = np.random.uniform(0.0001, 0.0075)
                clip = sigma * 3
                gaussian_noise = np.clip(sigma * np.random.randn(2048, 6), -clip, clip)
                data += gaussian_noise

                # print( data.astype('float32').shape, seg.shape, key.shape, model_pc_out.astype('float32').shape, fpi[0].shape)

            # print( "Lasted time: ", time.time() - start_time )
            # list_of_data.append( [data.astype('float32'), seg, key, model_pc_out.astype('float32'), fpi[0]])
            l_data.append(np.copy(data.astype('float32')))
            l_seg.append(np.copy(seg))
            l_key.append(np.copy(key))
            l_mo.append(np.copy(model_pc_out.astype('float32')))
            l_fpi.append(np.copy(fpi[0]))
            l_center.append(np.copy(center_point_list / std_data))

        # return data.astype('float32'), seg, key, model_pc_out.astype('float32'), fpi[0].cpu().numpy()
        # return l_data, l_seg, l_key, l_mo, l_fpi
        # return np.array(l_data), np.array(l_seg), np.array(l_key), np.array(l_mo), np.array(l_fpi), np.array(l_center)
        return np.array(l_data), np.array(l_seg), np.array(l_key), np.array(l_mo), np.array(l_fpi)  # , np.array(l_center)
        # return data.astype('float32'), seg, key, model_pc_out.astype('float32'), fpi[0]

    def __len__(self):
        if self.max_data == -1:
            return len(self.full_dataset)
        else:
            return self.max_data
           
def center_point_cloud_in_z(target, min_dist_z, max_dist_z):
    points = np.asarray(target.points)
    target = target.select_by_index(np.where(points[:, 2] < max_dist_z)[0])
    points = np.asarray(target.points)
    target = target.select_by_index(np.where(points[:, 2] > min_dist_z)[0])
    return target

def center_point_cloud_in_x(target, min_dist_x, max_dist_x):
    points = np.asarray(target.points)
    target = target.select_by_index(np.where(points[:, 0] < max_dist_x)[0])
    points = np.asarray(target.points)
    target = target.select_by_index(np.where(points[:, 0] > min_dist_x)[0])
    return target

def center_point_cloud_in_y(target, min_dist_y, max_dist_y):
    points = np.asarray(target.points)
    target = target.select_by_index(np.where(points[:, 1] < max_dist_y)[0])
    points = np.asarray(target.points)
    target = target.select_by_index(np.where(points[:, 1] > min_dist_y)[0])
    return target

class GPVN_pickle(Dataset):
    # def __init__(self, num_points, partition='train'):
    # def __init__(self, number_of_keypoints, max_data=-1, partition='train'):
    def __init__(self, number_of_keypoints, cad_string, pickle_file, max_data=-1, partition='train', radius=-1):
        self.full_dataset = []
        self.number_of_keypoints = number_of_keypoints
        self.object_set_set = {}
        
        self.max_data = max_data
        self.partition = partition
        
        candidate = trimesh.load_mesh(cad_string)
        candidate.export('stuff.stl')
        candidate_extents_max = np.max(candidate.extents)

        radius = np.max(candidate_extents_max)

        obj_cad = o3d.io.read_triangle_mesh('stuff.stl')

        obj_pc = obj_cad.sample_points_poisson_disk(2048)

        obj_pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30), False)

        model_pc = np.concatenate([np.asarray(obj_pc.points), np.asarray(obj_pc.normals)], axis=1)

        model_center = obj_pc.get_center()

        if radius < 0:
            self.test_radius = np.sqrt(candidate.extents[0] ** 2 +
                candidate.extents[1] ** 2 +
                candidate.extents[2] ** 2)
        else:
            self.test_radius = radius
        label = "0"
        self.object_set_set[label] = {"model_pc": model_pc, "radius": radius, "model_center": model_center}

        with open(pickle_file, 'rb') as f:
            object_poses = pickle.load(f)		
	
        for posepair in object_poses:
              scene = "/home/fhagelskjaer/workspace/pointposer/" + posepair["scene"]
              transform = posepair["transform"]
              model_info = "0"
              # import pdb; pdb.set_trace()
              self.full_dataset.append({"cloud": scene, "label": model_info, "transform": transform})


    def __getitem__(self, item):
        NUM_POINT = 2048
        GOOD_DIST = 2
        BAD_DIST = 2
        while(True):

            # TODO: this should not be true
            if self.max_data != -1:
                item = np.random.randint(len(self.full_dataset))

            object_set = self.object_set_set[self.full_dataset[item]["label"]]

            # scene_folder = object_set["scene_folder"]
            model_pc = object_set["model_pc"]
            radius = object_set["radius"]
            model_center = object_set["model_center"]

            object_xyz = model_pc[:,:3] #  np.asarray(obj_pc.points)
            x = torch.FloatTensor(object_xyz)
            x = np.reshape(x, (1, -1, 3))
            if self.partition == "train" and np.random.rand() < 0.6:
                fpi_idx = list(range(object_xyz.shape[0])) 
                np.random.shuffle(fpi_idx)
                fpi = np.array(fpi_idx[:self.number_of_keypoints])
            else:
                fpi = fps.farthest_point_sampler(object_xyz, self.number_of_keypoints)


            object_xyz_feature = object_xyz[fpi, :]

            cloud_name = self.full_dataset[item]["cloud"]

            # print(scene_folder, cloud_index)


            

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

            

            modelCloud = o3d.io.read_point_cloud(cloud_name)
           

            modelCloud = center_point_cloud_in_y(modelCloud, all_new_center[0][1] - radius*1.5, all_new_center[0][1] + radius*1.5)
            modelCloud = center_point_cloud_in_z(modelCloud, all_new_center[0][2] - radius*1.5, all_new_center[0][2] + radius*1.5)
            modelCloud = center_point_cloud_in_x(modelCloud, all_new_center[0][0] - radius*1.5, all_new_center[0][0] + radius*1.5)

            
            modelCloud = modelCloud.voxel_down_sample(0.5)
            modelCloud.estimate_normals( o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30))
            modelCloud.orient_normals_to_align_with_direction(orientation_reference=np.array([0.0, 0.0, -1.0]))            

            # o3d.visualization.draw_geometries([modelCloud])
            
            input_pointcloud = np.concatenate([np.asarray(modelCloud.points), np.asarray(modelCloud.normals)], axis=1)

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
                print(item, self.full_dataset[item]["label"], self.full_dataset[item]["cloud"], self.full_dataset[item]["transform"]) 
                print("return none")
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

        for point_sorted_index in range(2):
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
        
            # obj_pc_temp = copy.deepcopy(obj_pc)
            obj_pc_temp = o3d.geometry.PointCloud()        
            obj_pc_temp.points = o3d.utility.Vector3dVector(model_pc[:,:3])
            obj_pc_temp.normals = o3d.utility.Vector3dVector(model_pc[:,3:6])
            
            R = obj_pc_temp.get_rotation_matrix_from_xyz(
                (np.random.uniform(0, np.pi * 2), np.random.uniform(0, np.pi * 2), np.random.uniform(0, np.pi * 2)))
            obj_pc_temp.rotate(R, center=(0, 0, 0))
            model_pc_out = np.concatenate([np.asarray(obj_pc_temp.points), np.asarray(obj_pc_temp.normals)], axis=1)

            # center_point_cloud  # TODO
            model_pc_out = normalize_2d(model_pc_out)

            data, std_data = normalize_1d(data)

            # if False: # if True:
            if self.partition == "train":
                #  jitter on the model
                shift_range = 0.10
                shift = np.random.uniform(-shift_range, shift_range, (3))
                model_pc_out[:, :3] += shift

                #  noise on the model
                # sigma = 0.005
                sigma = np.random.uniform(0.0001, 0.0075)
                clip = sigma * 3
                gaussian_noise = np.clip(sigma * np.random.randn(2048, 6), -clip, clip)
                model_pc_out += gaussian_noise

                #  noise on the scene
                # sigma = 0.005
                sigma = np.random.uniform(0.0001, 0.0075)
                clip = sigma * 3
                gaussian_noise = np.clip(sigma * np.random.randn(2048, 6), -clip, clip)
                data += gaussian_noise

                # print( data.astype('float32').shape, seg.shape, key.shape, model_pc_out.astype('float32').shape, fpi[0].shape)

            # print( "Lasted time: ", time.time() - start_time )
            # list_of_data.append( [data.astype('float32'), seg, key, model_pc_out.astype('float32'), fpi[0]])
            l_data.append(np.copy(data.astype('float32')))
            l_seg.append(np.copy(seg))
            l_key.append(np.copy(key))
            l_mo.append(np.copy(model_pc_out.astype('float32')))
            l_fpi.append(np.copy(fpi[0]))
            l_center.append(np.copy(center_point_list / std_data))

        # return data.astype('float32'), seg, key, model_pc_out.astype('float32'), fpi[0].cpu().numpy()
        # return l_data, l_seg, l_key, l_mo, l_fpi
        # return np.array(l_data), np.array(l_seg), np.array(l_key), np.array(l_mo), np.array(l_fpi), np.array(l_center)
        return np.array(l_data), np.array(l_seg), np.array(l_key), np.array(l_mo), np.array(l_fpi)  # , np.array(l_center)
        # return data.astype('float32'), seg, key, model_pc_out.astype('float32'), fpi[0]

    def __len__(self):
        if self.max_data == -1:
            return len(self.full_dataset)
        else:
            return self.max_data

