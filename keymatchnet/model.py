#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from huggingface_hub import PyTorchModelHubMixin

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None, dim9=False, device=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)   # (batch_size, num_points, k)
        else:
            idx = knn(x[:, :3, :], k=k)
    # device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature      # (batch_size, 2*num_dims, num_points, k)


class DGCNN_pvn(nn.Module):
    def __init__(self, k, emb_dims, num_key, dropout=0.0):
        super(DGCNN_pvn, self).__init__()
        self.k = k

        self.conv1 = nn.Sequential(nn.Conv2d(12, 64, kernel_size=1, bias=False),
                                   nn.GroupNorm(32, 64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   nn.GroupNorm(32, 64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   nn.GroupNorm(32, 64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   nn.GroupNorm(32, 64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   nn.GroupNorm(32, 64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, emb_dims, kernel_size=1, bias=False),
                                   nn.GroupNorm(32, emb_dims),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(1216, 512, kernel_size=1, bias=False),
                                   nn.GroupNorm(32, 512),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                   nn.GroupNorm(32, 256),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=dropout)
        self.conv9 = nn.Sequential(nn.Conv1d(256, 2, kernel_size=1, bias=False))
        self.conv10 = nn.Sequential(nn.Conv1d(256, num_key, kernel_size=1, bias=False))

    def forward(self, x, device):
        batch_size = x.size(0)
        num_points = x.size(2)

        x = get_graph_feature(x, k=self.k, dim9=True, device=device)   # (batch_size, 9, num_points) -> (batch_size, 9*2, num_points, k)
        x = self.conv1(x)                       # (batch_size, 9*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k, device=device)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k, device=device)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = torch.cat((x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)

        x = self.conv6(x)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        x = x.repeat(1, 1, num_points)          # (batch_size, 1024, num_points)
        x = torch.cat((x, x1, x2, x3), dim=1)   # (batch_size, 1024+64*3, num_points)

        x = self.conv7(x)                       # (batch_size, 1024+64*3, num_points) -> (batch_size, 512, num_points)
        x = self.conv8(x)                       # (batch_size, 512, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        x_seg = self.conv9(x)                       # (batch_size, 256, num_points) -> (batch_size, 13, num_points)
        x_key = self.conv10(x)                       # (batch_size, 256, num_points) -> (batch_size, 13, num_points)
        
        return x_seg, x_key


class DGCNN_gpvn(nn.Module,
        PyTorchModelHubMixin,
        library_name="keymatchnet",
        repo_url="https://github.com/fhagelskjaer/keymatchnet",
        paper_url="https://arxiv.org/abs/2303.16102",
        docs_url="https://keymatchnet.github.io/",
    ):
    def __init__(self, k, emb_dims, num_key, dropout=0.0):
        super(DGCNN_gpvn, self).__init__()
        self.k = k
        self.num_key = num_key

        self.conv_obj1 = nn.Sequential(nn.Conv2d(12, 64, kernel_size=1, bias=False),
                                   nn.GroupNorm(32, 64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv_obj2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   nn.GroupNorm(32, 64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv_obj3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   nn.GroupNorm(32, 64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv_obj4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   nn.GroupNorm(32, 64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv_obj5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   nn.GroupNorm(32, 64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv_obj6 = nn.Sequential(nn.Conv1d(192, emb_dims, kernel_size=1, bias=False),
                                   nn.GroupNorm(32, emb_dims),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv_obj7 = nn.Sequential(nn.Conv1d(1216, 512, kernel_size=1, bias=False),
                                   nn.GroupNorm(32, 512),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv_obj8 = nn.Sequential(nn.Conv1d(512, 128, kernel_size=1, bias=False),
                                   nn.GroupNorm(32, 128),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.obj_dp1 = nn.Dropout(p=dropout)
        self.obj_dp2 = nn.Dropout(p=dropout)


        self.conv1 = nn.Sequential(nn.Conv2d(12, 64, kernel_size=1, bias=False),
                                   nn.GroupNorm(32, 64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   nn.GroupNorm(32, 64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   nn.GroupNorm(32, 64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   nn.GroupNorm(32, 64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   nn.GroupNorm(32, 64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, emb_dims, kernel_size=1, bias=False),
                                   nn.GroupNorm(32, emb_dims),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(1024+64*3, 512, kernel_size=1, bias=False),
        # self.conv7 = nn.Sequential(nn.Conv1d(1856, 512, kernel_size=1, bias=False),
                                   nn.GroupNorm(32, 512),
                                   nn.LeakyReLU(negative_slope=0.2))
        # self.conv8 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
        self.conv8 = nn.Sequential(nn.Conv2d(512+128, 256, kernel_size=1, bias=False),
                                   nn.GroupNorm(32, 256),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8r = nn.Sequential(nn.Conv2d(256+128, 256, kernel_size=1, bias=False),
                                   nn.GroupNorm(32, 256),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=dropout)
        self.dp2 = nn.Dropout(p=dropout)
        # self.conv9 = nn.Sequential(nn.Conv1d(256, 2, kernel_size=1, bias=False))
        # self.conv9 = nn.Sequential(nn.Conv1d(64*20, 2, kernel_size=1, bias=False))
        self.conv9 = nn.Sequential(nn.Conv1d(256, 2, kernel_size=1, bias=False))
        # self.conv10 = nn.Sequential(nn.Conv1d(256, num_key, kernel_size=1, bias=False))
        # self.conv10 = nn.Sequential(nn.Conv1d(64*20, num_key, kernel_size=1, bias=False))
        self.conv10 = nn.Sequential(nn.Conv2d(256, 1, kernel_size=1, bias=False))

        
        # self.dist1 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=1, bias=True),
        #                            nn.ReLU())
        # self.dist2 = nn.Sequential(nn.Conv1d(128, 64, kernel_size=1, bias=False),
        #                            nn.ReLU())
        # self.dist3 = nn.Conv1d(64, 3, kernel_size=1, bias=False)



    def forward(self, x, objpc, fpi, device):

        obj = objpc
        
        batch_size = obj.size(0)
        num_points = obj.size(2)

        new_num_points = fpi.size(1)
        # device = torch.device('cuda')
        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1)*num_points
        fpi = fpi + idx_base
        fpi = fpi.view(-1)
        

        obj = get_graph_feature(obj, k=self.k, dim9=True, device=device)   # (batch_size, 9, num_points) -> (batch_size, 9*2, num_points, k)
        obj = self.conv_obj1(obj)                       # (batch_size, 9*2, num_points, k) -> (batch_size, 64, num_points, k)
        obj = self.conv_obj2(obj)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        obj1 = obj.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        obj = get_graph_feature(obj1, k=self.k, device=device)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)

        _, num_dims, _ = obj1.size() # TODO
        obj1 = obj1.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
        obj1 = obj1.view(batch_size*num_points, -1)[fpi, :]
        obj1 = obj1.view(batch_size, new_num_points, num_dims)
        obj1 = obj1.permute(0, 2, 1)

        _, num_dims, _, k = obj.size() # TODO
        obj = obj.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
        obj = obj.view(batch_size*num_points, -1)[fpi, :]
        obj = obj.view(batch_size, new_num_points, num_dims, k)
        obj = obj.permute(0, 2, 1, 3)        

        obj = self.conv_obj3(obj)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        obj = self.conv_obj4(obj)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        obj2 = obj.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        
        # objpc = objpc.permute(0, 2, 1)
        # fpi = farthest_point_sampler(objpc[:,:,:3], 20, start_idx=0)

        # _, num_dims, _ = obj2.size() # TODO
        # obj2 = obj2.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
        # obj2 = obj2.view(batch_size*num_points, -1)[fpi, :]
        # obj2 = obj2.view(batch_size, new_num_points, num_dims)
        # obj2 = obj2.permute(0, 2, 1)

        obj = get_graph_feature(obj2, k=self.k, device=device)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        obj = self.conv_obj5(obj)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        obj3 = obj.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        obj = torch.cat((obj1, obj2, obj3), dim=1)      # (batch_size, 64*3, num_points)
        
        obj = self.conv_obj6(obj)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        obj = obj.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        obj = obj.repeat(1, 1, self.num_key)          # (batch_size, 1024, num_points)
        obj = torch.cat((obj, obj1, obj2, obj3), dim=1)   # (batch_size, 1024+64*3, num_points)
        
        obj = self.conv_obj7(obj)                       # (batch_size, 1024+64*3, num_points) -> (batch_size, 512, num_points)
        obj = self.obj_dp1(obj)
        obj = self.conv_obj8(obj)                       # (batch_size, 512, num_points) -> (batch_size, 32, num_points)
        obj = self.obj_dp2(obj)
        
        obs, embs, kp = obj.size()
    
        obj = obj.view(batch_size, embs, 1, kp)
        obj = obj.repeat(1, 1, num_points, 1)          # (batch_size, 1024, num_points)
        

        num_points = x.size(2)

        x = get_graph_feature(x, k=self.k, dim9=True, device=device)   # (batch_size, 9, num_points) -> (batch_size, 9*2, num_points, k)
        x = self.conv1(x)                       # (batch_size, 9*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k, device=device)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k, device=device)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = torch.cat((x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)

        x = self.conv6(x)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        # x = torch.cat((x, obj), dim=1)          # (batch_size, 1088, 1)
        
        x = x.repeat(1, 1, num_points)          # (batch_size, 1024, num_points)
        x = torch.cat((x, x1, x2, x3), dim=1)   # (batch_size, 1024+64*3, num_points)

        # print( x.size() )

        x = self.conv7(x)                       # (batch_size, 1024+64*3, num_points) -> (batch_size, 512, num_points)


        x = x.view(batch_size, -1, num_points, 1)
        x = x.repeat(1, 1, 1, kp)
        x = torch.cat((x, obj), dim=1)
        x = self.conv8(x)                       # (batch_size, 512, num_points) -> (batch_size, 256, num_points)
        x = x.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        x = self.dp1(x)

        x = x.view(batch_size, -1, num_points, 1)
        x = x.repeat(1, 1, 1, kp)
        x = torch.cat((x, obj), dim=1)
        x = self.conv8r(x)                       # (batch_size, 512, num_points) -> (batch_size, 256, num_points)

        # print( "size", x.size() )

        # x = torch((x, obj_flat), dim=1)
        # x = x.permute(0, 1, 3, 2)
        # x = x.reshape(batch_size, -1, num_points)

        x = self.dp2(x)
        
        x_seg = x.max(dim=-1, keepdim=False)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)
        x_seg = self.conv9(x_seg)                       # (batch_size, 256, num_points) -> (batch_size, 13, num_points)
        # x_seg = x_seg.permute(0, 1, 3, 2)
        # x_seg = x_seg.reshape(batch_size, -1, num_points)

        
        x_key = self.conv10(x)                       # (batch_size, 256, num_points) -> (batch_size, 13, num_points)
        x_key = x_key.permute(0, 1, 3, 2)
        x_key = x_key.reshape(batch_size, -1, num_points)


        # x_center = self.dist1(x)
        # x_center = x_center.max(dim=-1, keepdim=False)[0]
        # x_center = self.dist2(x_center)
        #xrc = self.dpc1(xrc)
        # x_center = self.dist3(x_center)
        #xrc = self.dpc2(xrc)

        return x_seg, x_key #, x_center


class DGCNN_gpvn_obj(nn.Module):
    def __init__(self, k, emb_dims, num_key, dropout=0.0):
        super(DGCNN_gpvn_obj, self).__init__()
        self.k = k
        self.num_key = num_key

        self.conv_obj1 = nn.Sequential(nn.Conv2d(12, 64, kernel_size=1, bias=False),
                                   nn.GroupNorm(32, 64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv_obj2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   nn.GroupNorm(32, 64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv_obj3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   nn.GroupNorm(32, 64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv_obj4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   nn.GroupNorm(32, 64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv_obj5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   nn.GroupNorm(32, 64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv_obj6 = nn.Sequential(nn.Conv1d(192, emb_dims, kernel_size=1, bias=False),
                                   nn.GroupNorm(32, emb_dims),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv_obj7 = nn.Sequential(nn.Conv1d(1216, 512, kernel_size=1, bias=False),
                                   nn.GroupNorm(32, 512),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv_obj8 = nn.Sequential(nn.Conv1d(512, 128, kernel_size=1, bias=False),
                                   nn.GroupNorm(32, 128),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.obj_dp1 = nn.Dropout(p=dropout)
        self.obj_dp2 = nn.Dropout(p=dropout)

    def forward(self, objpc, fpi, device):

        obj = objpc
        
        batch_size = obj.size(0)
        num_points = obj.size(2)

        new_num_points = fpi.size(1)
        device = torch.device('cuda')
        # device = torch.device('cpu')
        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1)*num_points
        fpi = fpi + idx_base
        fpi = fpi.view(-1)
        

        obj = get_graph_feature(obj, k=self.k, dim9=True, device=device)   # (batch_size, 9, num_points) -> (batch_size, 9*2, num_points, k)
        obj = self.conv_obj1(obj)                       # (batch_size, 9*2, num_points, k) -> (batch_size, 64, num_points, k)
        obj = self.conv_obj2(obj)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        obj1 = obj.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        obj = get_graph_feature(obj1, k=self.k, device=device)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)

        _, num_dims, _ = obj1.size() # TODO
        obj1 = obj1.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
        obj1 = obj1.view(batch_size*num_points, -1)[fpi, :]
        obj1 = obj1.view(batch_size, new_num_points, num_dims)
        obj1 = obj1.permute(0, 2, 1)

        _, num_dims, _, k = obj.size() # TODO
        obj = obj.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
        obj = obj.view(batch_size*num_points, -1)[fpi, :]
        obj = obj.view(batch_size, new_num_points, num_dims, k)
        obj = obj.permute(0, 2, 1, 3)        

        obj = self.conv_obj3(obj)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        obj = self.conv_obj4(obj)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        obj2 = obj.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        
        # objpc = objpc.permute(0, 2, 1)
        # fpi = farthest_point_sampler(objpc[:,:,:3], 20, start_idx=0)

        # _, num_dims, _ = obj2.size() # TODO
        # obj2 = obj2.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
        # obj2 = obj2.view(batch_size*num_points, -1)[fpi, :]
        # obj2 = obj2.view(batch_size, new_num_points, num_dims)
        # obj2 = obj2.permute(0, 2, 1)

        obj = get_graph_feature(obj2, k=self.k, device=device)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        obj = self.conv_obj5(obj)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        obj3 = obj.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        obj = torch.cat((obj1, obj2, obj3), dim=1)      # (batch_size, 64*3, num_points)
        
        obj = self.conv_obj6(obj)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        obj = obj.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        obj = obj.repeat(1, 1, self.num_key)          # (batch_size, 1024, num_points)
        obj = torch.cat((obj, obj1, obj2, obj3), dim=1)   # (batch_size, 1024+64*3, num_points)
        
        obj = self.conv_obj7(obj)                       # (batch_size, 1024+64*3, num_points) -> (batch_size, 512, num_points)
        obj = self.conv_obj8(obj)                       # (batch_size, 512, num_points) -> (batch_size, 32, num_points)
        
        return obj


class DGCNN_gpvn_scene(nn.Module):
    def __init__(self, k, emb_dims):
        super(DGCNN_gpvn_scene, self).__init__()
        self.k = k


        self.conv1 = nn.Sequential(nn.Conv2d(12, 64, kernel_size=1, bias=False),
                                   nn.GroupNorm(32, 64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   nn.GroupNorm(32, 64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   nn.GroupNorm(32, 64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   nn.GroupNorm(32, 64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   nn.GroupNorm(32, 64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, emb_dims, kernel_size=1, bias=False),
                                   nn.GroupNorm(32, emb_dims),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(1024+64*3, 512, kernel_size=1, bias=False),
        # self.conv7 = nn.Sequential(nn.Conv1d(1856, 512, kernel_size=1, bias=False),
                                   nn.GroupNorm(32, 512),
                                   nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x, device):

        num_points = x.size(2)

        x = get_graph_feature(x, k=self.k, dim9=True, device=device)   # (batch_size, 9, num_points) -> (batch_size, 9*2, num_points, k)
        x = self.conv1(x)                       # (batch_size, 9*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k, device=device)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k, device=device)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = torch.cat((x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)

        x = self.conv6(x)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        # x = torch.cat((x, obj), dim=1)          # (batch_size, 1088, 1)
        
        x = x.repeat(1, 1, num_points)          # (batch_size, 1024, num_points)
        x = torch.cat((x, x1, x2, x3), dim=1)   # (batch_size, 1024+64*3, num_points)

        # print( x.size() )

        x = self.conv7(x)                       # (batch_size, 1024+64*3, num_points) -> (batch_size, 512, num_points)

        return x


class DGCNN_gpvn_purenet(nn.Module):
    def __init__(self, k, emb_dims, num_key, dropout=0.0):
        super(DGCNN_gpvn_purenet, self).__init__()
        self.k = k
        self.num_key = num_key


        self.conv1 = nn.Sequential(nn.Conv2d(12, 64, kernel_size=1, bias=False),
                                   nn.GroupNorm(32, 64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   nn.GroupNorm(32, 64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   nn.GroupNorm(32, 64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   nn.GroupNorm(32, 64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   nn.GroupNorm(32, 64),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, emb_dims, kernel_size=1, bias=False),
                                   nn.GroupNorm(32, emb_dims),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(1024+64*3, 512, kernel_size=1, bias=False),
        # self.conv7 = nn.Sequential(nn.Conv1d(1856, 512, kernel_size=1, bias=False),
                                   nn.GroupNorm(32, 512),
                                   nn.LeakyReLU(negative_slope=0.2))
        # self.conv8 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
        self.conv8 = nn.Sequential(nn.Conv2d(512+128, 256, kernel_size=1, bias=False),
                                   nn.GroupNorm(32, 256),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8r = nn.Sequential(nn.Conv2d(256+128, 256, kernel_size=1, bias=False),
                                   nn.GroupNorm(32, 256),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=dropout)
        self.dp2 = nn.Dropout(p=dropout)
        # self.conv9 = nn.Sequential(nn.Conv1d(256, 2, kernel_size=1, bias=False))
        # self.conv9 = nn.Sequential(nn.Conv1d(64*20, 2, kernel_size=1, bias=False))
        self.conv9 = nn.Sequential(nn.Conv1d(256, 2, kernel_size=1, bias=False))
        # self.conv10 = nn.Sequential(nn.Conv1d(256, num_key, kernel_size=1, bias=False))
        # self.conv10 = nn.Sequential(nn.Conv1d(64*20, num_key, kernel_size=1, bias=False))
        self.conv10 = nn.Sequential(nn.Conv2d(256, 1, kernel_size=1, bias=False))

    def forward(self, x, obj, device):

        batch_size = x.size(0)
        num_points = x.size(2)

        obs, embs, kp = obj.size()
    
        obj = obj.view(batch_size, embs, 1, kp)
        obj = obj.repeat(1, 1, num_points, 1)          # (batch_size, 1024, num_points)


        x = get_graph_feature(x, k=self.k, dim9=True, device=device)   # (batch_size, 9, num_points) -> (batch_size, 9*2, num_points, k)
        x = self.conv1(x)                       # (batch_size, 9*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k, device=device)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k, device=device)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = torch.cat((x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)

        x = self.conv6(x)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        # x = torch.cat((x, obj), dim=1)          # (batch_size, 1088, 1)
        
        x = x.repeat(1, 1, num_points)          # (batch_size, 1024, num_points)
        x = torch.cat((x, x1, x2, x3), dim=1)   # (batch_size, 1024+64*3, num_points)

        # print( x.size() )

        x = self.conv7(x)                       # (batch_size, 1024+64*3, num_points) -> (batch_size, 512, num_points)


        x = x.view(batch_size, -1, num_points, 1)
        x = x.repeat(1, 1, 1, kp)
        x = torch.cat((x, obj), dim=1)
        x = self.conv8(x)                       # (batch_size, 512, num_points) -> (batch_size, 256, num_points)
        x = x.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        x = self.dp1(x)

        x = x.view(batch_size, -1, num_points, 1)
        x = x.repeat(1, 1, 1, kp)
        x = torch.cat((x, obj), dim=1)
        x = self.conv8r(x)                       # (batch_size, 512, num_points) -> (batch_size, 256, num_points)

        # print( "size", x.size() )

        # x = torch((x, obj_flat), dim=1)
        # x = x.permute(0, 1, 3, 2)
        # x = x.reshape(batch_size, -1, num_points)

        x = self.dp2(x)
        
        x_seg = x.max(dim=-1, keepdim=False)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)
        x_seg = self.conv9(x_seg)                       # (batch_size, 256, num_points) -> (batch_size, 13, num_points)
        # x_seg = x_seg.permute(0, 1, 3, 2)
        # x_seg = x_seg.reshape(batch_size, -1, num_points)
        
        
        x_key = self.conv10(x)                       # (batch_size, 256, num_points) -> (batch_size, 13, num_points)

        x_key = x_key.permute(0, 1, 3, 2)
        x_key = x_key.reshape(batch_size, -1, num_points)

        return x_seg, x_key
