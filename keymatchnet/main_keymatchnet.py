#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: An Tao, Pengliang Ji
@Contact: ta19@mails.tsinghua.edu.cn, jpl1723@buaa.edu.cn
@File: main_semseg_s3dis.py
@Time: 2021/7/20 7:49 PM
"""


from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from data import * # GPVN, GPVN_set, GPVN_icbin, GPVN_wrs, GPVN_set_knocpose, GPVN_tless, GPVN_lmo, GPVN_megapose, GPVN_data_engine, GPVN_data_engine2
# import data as dataf
from model import DGCNN_gpvn
import numpy as np
from torch.utils.data import DataLoader
from util import get_loss, IOStream
import sklearn.metrics as metrics
import json
import open3d as o3d

import time
import multiprocessing

def _init_():
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    if not os.path.exists('outputs/'+args.exp_name):
        os.makedirs('outputs/'+args.exp_name)
    if not os.path.exists('outputs/'+args.exp_name+'/'+'models'):
        os.makedirs('outputs/'+args.exp_name+'/'+'models')
    os.system('cp main_gpvn.py outputs'+'/'+args.exp_name+'/'+'main_gpvn.py.backup')
    os.system('cp model.py outputs' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py outputs' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py outputs' + '/' + args.exp_name + '/' + 'data.py.backup')

def getFullElectronicLoaders(num_key, args):
    train_loader = DataLoader(GPVN_set(number_of_keypoints=num_key, partition='train'),
                              num_workers=18, batch_size=args.batch_size//2, 
                              shuffle=True, drop_last=True,
                              worker_init_fn=worker_init_fn,
                              persistent_workers=True)
    test_loader = DataLoader(GPVN_set(number_of_keypoints=num_key, partition='val'), 
                        num_workers=10, batch_size=args.test_batch_size//2, 
                        shuffle=False, drop_last=False,
                        persistent_workers=True)
    return train_loader, test_loader


def train(args, io):
    num_key = 20
   
    train_loader, test_loader = getFullElectronicLoadersSpawnTest(num_key, args)

    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    model = DGCNN_gpvn(args.k, args.emb_dims, num_key, args.dropout).to(device)
    print(str(model))

    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=1e-3)
    elif args.scheduler == 'step':
        # scheduler = StepLR(opt, 20, 0.5, args.epochs)
        scheduler = StepLR(opt, step_size=20, gamma=0.7)

    criterion = get_loss

    if(args.model_root != ''):
        model.load_state_dict(torch.load(args.model_root))

    for epoch in range(args.epochs):
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_true_cls = []
        train_pred_cls = []
        train_true_key = []
        train_pred_key = []
        # train_true_seg = []
        # train_pred_seg = []
        # train_label_seg = []
        for data, seg, key, obj, fpi in train_loader:

            data = torch.flatten(data, start_dim=0, end_dim=1)
            seg = torch.flatten(seg, start_dim=0, end_dim=1)
            key = torch.flatten(key, start_dim=0, end_dim=1)
            obj = torch.flatten(obj, start_dim=0, end_dim=1)
            fpi = torch.flatten(fpi, start_dim=0, end_dim=1)
            
            # data = noise_model.compute(data, epoch)
            data, obj, seg, key, fpi = data.to(device), obj.to(device), seg.to(device), key.to(device), fpi.to(device)
            data = data.permute(0, 2, 1)
            obj = obj.permute(0, 2, 1)
            batch_size = data.size()[0]
            
            # for _ in range(2):
            opt.zero_grad()
            model.train()
            seg_pred, key_pred = model(data, obj, fpi, device)
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            key_pred = key_pred.permute(0, 2, 1).contiguous()
            loss = criterion(seg_pred.view(-1, 2), seg.view(-1,1).squeeze(), key_pred.view(-1, num_key), key.view(-1,1).squeeze(), 0.8, -1)
            loss.backward()
            opt.step()
            
            pred = seg_pred.max(dim=2)[1]               # (batch_size, num_points)
            count += batch_size
            train_loss += loss.item() * batch_size

            seg_np = seg.cpu().numpy()                  # (batch_size, num_points)
            pred_np = pred.detach().cpu().numpy()       # (batch_size, num_points)
        
            train_true_cls.append(seg_np.reshape(-1))       # (batch_size * num_points)
            train_pred_cls.append(pred_np.reshape(-1))      # (batch_size * num_points)

            key_np = key.cpu().numpy()                  # (batch_size, num_points)
            key_pred = key_pred.max(dim=2)[1]               # (batch_size, num_points)
            key_pred_np = key_pred.detach().cpu().numpy()       # (batch_size, num_points)
        
            train_true_key.append(key_np.reshape(-1))       # (batch_size * num_points)
            train_pred_key.append(key_pred_np.reshape(-1))      # (batch_size * num_points)

        if args.scheduler == 'cos':
            scheduler.step()
        elif args.scheduler == 'step':
            if opt.param_groups[0]['lr'] > 1e-5:
                scheduler.step()
            if opt.param_groups[0]['lr'] < 1e-5:
                for param_group in opt.param_groups:
                    param_group['lr'] = 1e-5
        train_true_cls = np.concatenate(train_true_cls)
        train_pred_cls = np.concatenate(train_pred_cls)
        train_true_key = np.concatenate(train_true_key)
        train_pred_key = np.concatenate(train_pred_key)
        train_acc = metrics.accuracy_score(train_true_cls, train_pred_cls)
        train_key_acc = metrics.accuracy_score(train_true_key, train_pred_key, sample_weight=train_true_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(train_true_cls, train_pred_cls)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f, train key acc: %.6f' % (epoch, 
                                                                                                  train_loss*1.0/count,
                                                                                                  train_acc,
                                                                                                  avg_per_class_acc,
                                                                                                  train_key_acc)
        # print( metrics.classification_report(train_true_cls, train_pred_cls) )                                                                                          
        io.cprint(outstr)

        # noise_model.compute(epoch, train_loss* 1.0/count)

        torch.save(model.state_dict(), 'outputs/%s/models/model.t7' % (args.exp_name))
        # if epoch != 59: 
        #     continue

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        # model.train()
        test_true_cls = []
        test_pred_cls = []
        test_true_key = []
        test_pred_key = []
        
        for data, seg, key, obj, fpi in test_loader:
         
            data = torch.flatten(data, start_dim=0, end_dim=1)
            seg = torch.flatten(seg, start_dim=0, end_dim=1)
            key = torch.flatten(key, start_dim=0, end_dim=1)
            obj = torch.flatten(obj, start_dim=0, end_dim=1)
            fpi = torch.flatten(fpi, start_dim=0, end_dim=1)

            # data = noise_model.compute(data, epoch)
            data, obj, seg, key, fpi = data.to(device), obj.to(device), seg.to(device), key.to(device), fpi.to(device)
            data = data.permute(0, 2, 1)
            obj = obj.permute(0, 2, 1)
            batch_size = data.size()[0]
            # with torch.no_grad():
            seg_pred, key_pred = model(data, obj, fpi, device)
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            key_pred = key_pred.permute(0, 2, 1).contiguous()
            loss = criterion(seg_pred.view(-1, 2), seg.view(-1,1).squeeze(), key_pred.view(-1, num_key), key.view(-1,1).squeeze(), 0.8, -1)

            test_loss += loss.item() * batch_size
            count += batch_size

            pred = seg_pred.max(dim=2)[1]               # (batch_size, num_points)
            seg_np = seg.cpu().numpy()                  # (batch_size, num_points)
            pred_np = pred.detach().cpu().numpy()       # (batch_size, num_points)

            test_true_cls.append(seg_np.reshape(-1))       # (batch_size * num_points)
            test_pred_cls.append(pred_np.reshape(-1))      # (batch_size * num_points)

            key_np = key.cpu().numpy()                  # (batch_size, num_points)
            key_pred = key_pred.max(dim=2)[1]               # (batch_size, num_points)
            key_pred_np = key_pred.detach().cpu().numpy()       # (batch_size, num_points)

            test_true_key.append(key_np.reshape(-1))       # (batch_size * num_points)
            test_pred_key.append(key_pred_np.reshape(-1))      # (batch_size * num_points)

        test_true_cls = np.concatenate(test_true_cls)
        test_pred_cls = np.concatenate(test_pred_cls)
        test_true_key = np.concatenate(test_true_key)
        test_pred_key = np.concatenate(test_pred_key)
        test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
        test_key_acc = metrics.accuracy_score(test_true_key, test_pred_key, sample_weight=test_true_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f, test key acc: %.6f' % (epoch,
                                                                                                  test_loss*1.0/count,
                                                                                                  test_acc,
                                                                                                  avg_per_class_acc,
                                                                                                  test_key_acc)
        # print( metrics.classification_report(test_true_cls, test_pred_cls) )                                                                                          
        io.cprint(outstr)

        # if np.mean(test_ious) >= best_test_iou:
        #     best_test_iou = np.mean(test_ious)


def test(args, io):
    torch.autograd.set_detect_anomaly(True)

    num_key = 20

    test_loader = DataLoader(GPVN_set(number_of_keypoints=num_key, partition='test'), 
        worker_init_fn=worker_init_fn,                        
        num_workers=10, 
        batch_size=args.test_batch_size//2, 
        shuffle=False, 
        drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    model = DGCNN_gpvn(args.k, args.emb_dims, num_key, args.dropout).to(device)

    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    model.load_state_dict(torch.load(args.model_root))

    criterion = get_loss
    
    test_loss = 0.0
    count = 0.0
    model.eval()
    # model.train()
    test_true_cls = []
    test_pred_cls = []
    test_true_key = []
    test_pred_key = []

    visu_idx = 0

    for data, seg, key, obj, fpi in test_loader:
        # print( data.size(), seg.size() )

        data = torch.flatten(data, start_dim=0, end_dim=1)
        seg = torch.flatten(seg, start_dim=0, end_dim=1)
        key = torch.flatten(key, start_dim=0, end_dim=1)
        obj = torch.flatten(obj, start_dim=0, end_dim=1)
        fpi = torch.flatten(fpi, start_dim=0, end_dim=1)


        # data = noise_model.compute(data, epoch)
        data, obj, seg, key, fpi = data.to(device), obj.to(device), seg.to(device), key.to(device), fpi.to(device)
        data = data.permute(0, 2, 1)
        obj = obj.permute(0, 2, 1)
        batch_size = data.size()[0]
        # with torch.no_grad():
        seg_pred, key_pred = model(data, obj, fpi, device)
        seg_pred = seg_pred.permute(0, 2, 1).contiguous()
        key_pred = key_pred.permute(0, 2, 1).contiguous()
        loss = criterion(seg_pred.view(-1, 2), seg.view(-1,1).squeeze(), key_pred.view(-1, num_key), key.view(-1,1).squeeze(), 0.8, -1)

        test_loss += loss.item() * batch_size
        count += batch_size

        pred = seg_pred.max(dim=2)[1]               # (batch_size, num_points)
        seg_np = seg.cpu().numpy()                  # (batch_size, num_points)
        pred_np = pred.detach().cpu().numpy()       # (batch_size, num_points)

        test_true_cls.append(seg_np.reshape(-1))       # (batch_size * num_points)
        test_pred_cls.append(pred_np.reshape(-1))      # (batch_size * num_points)

        key_np = key.cpu().numpy()                  # (batch_size, num_points)
        key_val = key_pred.detach().cpu().numpy()
        key_pred = key_pred.max(dim=2)[1]               # (batch_size, num_points)
        key_pred_np = key_pred.detach().cpu().numpy()       # (batch_size, num_points)
        
        test_true_key.append(key_np.reshape(-1))       # (batch_size * num_points)
        test_pred_key.append(key_pred_np.reshape(-1))      # (batch_size * num_points)
        
        if args.visu == 'True':
            data = data.permute(0, 2, 1)
            # print( data.cpu().numpy().shape )
            for batch_visu in range(batch_size):
                if (not visu_idx == args.visu_idx) and (args.visu_idx >= 0):
                    visu_idx += 1
                    continue
                else:
                    visu_idx += 1
                target = o3d.geometry.PointCloud()
                target.points = o3d.utility.Vector3dVector(data.cpu().numpy()[batch_visu,:,:3])
                # target.normals = o3d.utility.Vector3dVector(np.array(point_list)[:,3:6])
                color = np.zeros((data.shape[1],3), np.float64)
                # print( key_pred_np.shape )
                cutoff = [0,0]
                seg_cutoff = [0,0]
                for point_i in range(2048):
                    if pred_np[batch_visu,point_i] == 1 and seg_np[batch_visu,point_i] == 1:
                        seg_cutoff[0] += 1
                        # print( key_np[batch_visu,point_i], key_pred_np[batch_visu,point_i], seg_np[batch_visu,point_i] )

                        if key_np[batch_visu,point_i] == key_pred_np[batch_visu,point_i]:
                        # if( key_val[batch_visu,point_i,key_np[batch_visu,point_i]] > key_val[batch_visu,point_i,key_pred_np[batch_visu,point_i]]*0.9 ):
                            color[point_i,:] = [0,255.0,0] # Green
                            cutoff[0] += 1
                        else:
                            color[point_i,:] = [0,0,255.0] # Blue
                            cutoff[1] += 1
                            # print( key_val[batch_visu,point_i,key_np[batch_visu,point_i]], key_val[batch_visu,point_i,key_pred_np[batch_visu,point_i]] )
                            # print( key_val[batch_visu,point_i,:] )
                    elif pred_np[batch_visu,point_i] == 0 and seg_np[batch_visu,point_i] == 0:
                        seg_cutoff[0] += 1
                        color[point_i,:] = [0,0,0] # Black
                    else:
                        seg_cutoff[1] += 1
                        if seg_np[batch_visu,point_i] == 1: # if it is real object but missed
                            color[point_i,:] = [255.0,0,0] # Red
                        else:
                            color[point_i,:] = [255, 165, 0]

                print( "Red: sfn, Yellow: sfp, Black: stn, Blue: stp, Green: stp&correctvote" )
                print( "seg percent correct", seg_cutoff[0]/(seg_cutoff[0]+seg_cutoff[1]) )
                if( cutoff[0]+cutoff[1] == 0 ):
                    print( "None" )
                else:
                    print( "percentage correct", cutoff[0]/(cutoff[0]+cutoff[1]) )

                #dist = pred_np[0,:]
                # print( dist[0,:] )
                #color[:,1] = dist*255 
                target.colors = o3d.utility.Vector3dVector(color)
            
                import copy
                newtarget = copy.deepcopy(target)
                R = newtarget.get_rotation_matrix_from_xyz((0, np.pi, 0))
                newtarget.rotate(R, center=(0, 0, 0))
                # o3d.visualization.draw_geometries([newtarget, target]) 
                o3d.visualization.draw_geometries([newtarget]) 
		

    test_true_cls = np.concatenate(test_true_cls)
    test_pred_cls = np.concatenate(test_pred_cls)
    test_true_key = np.concatenate(test_true_key)
    test_pred_key = np.concatenate(test_pred_key)
    test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
    test_key_acc = metrics.accuracy_score(test_true_key, test_pred_key, sample_weight=test_true_cls)
    # test_key_acc = metrics.accuracy_score(test_true_key, test_pred_key, sample_weight=test_pred_cls)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
    outstr = 'Test:  loss: %.6f, test acc: %.6f, test avg acc: %.6f, test key acc: %.6f' % (  test_loss*1.0/count,
                                                                                              test_acc,
                                                                                              avg_per_class_acc,
                                                                                              test_key_acc)
    io.cprint(outstr)




if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Part Segmentation')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--scheduler', type=str, default='cos', metavar='N',
                        choices=['cos', 'step'],
                        help='Scheduler to use, [cos, step]')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
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
    parser.add_argument('--partion', type=str, default='train', metavar='N',
                        help='If only part is to be used, e.g. 1, 10, 20, 2000')
    args = parser.parse_args()

    _init_()

    io = IOStream('outputs/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)
