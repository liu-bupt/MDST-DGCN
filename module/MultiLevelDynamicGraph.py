import os
import math
import sys

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as Func
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class DGCN_LSTM(nn.Module):

    r"""The basic module for applying a graph convolution.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        seq_len (int): number of time step
        aggregate_type (int): aggregate function type. 1 - max pooling;
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 seq_len,
                 bias=True,
                 align=1,
                 aggregate_type=1):# 1:max pooling; 2
        super(DGCN_LSTM,self).__init__()
#         print("outch",out_channels)
        self.out_channels = out_channels
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, 1),
            padding=(0, 0),
            stride=(1, 1),
            dilation=(1, 1),
            bias=bias)
        # print("aggregate_type",aggregate_type)
        self.align = align
        if aggregate_type == 1:
            self.aggre = nn.AdaptiveMaxPool2d((seq_len,1))
        self.lstm = nn.LSTM(out_channels,out_channels)
    def init_hidden(self, batch):
        return (
            torch.zeros(1, batch, self.out_channels).cuda(),
            torch.zeros(1, batch, self.out_channels).cuda()
        )
    def forward(self, x,A,state_tuple=None):
        # x: (N, T_{in}, in_channels)     N: pedestrian number in the scene, V: k-nearest neighbors 
        # A: Adjacency list   (T,N,V) N: pedestrian number in the scene, V: k-nearest neighbors 
        x_rel = torch.zeros_like(x)
        x_rel[:,1:,:] = x[:,1:,:]-x[:,0:-1,:]
        if self.align == 1:
            new_x = []
            for _ in range(A.shape[0]):
                tmp = x[:,_,:][A[_]] - x[:,_:_+1,:]# N,V,in_channels
                new_x.append(tmp)
            new_x = torch.stack(new_x,dim=2)#  N, V, T_{in}, in_channels
        elif self.align == 0:
            new_x = []
            for _ in range(A.shape[0]):
                tmp = x[:,_,:][A[_]]# N,V,in_channels
                new_x.append(tmp)
            new_x = torch.stack(new_x,dim=2)#  N, V, T_{in}, in_channels
        elif self.align == 2:
            new_x = []
            for _ in range(A.shape[0]):
                tmp = x[:,_,:][A[_]] - x[:,_:_+1,:]# N,V,in_channels
                # print("before",tmp)
                for i in range(tmp.shape[0]):
                    pos_x = tmp[i,:,0]
                    pos_y = tmp[i,:,1]
                    pos_x[pos_x==0] = x_rel[i,_,0]
                    pos_y[pos_y==0] = x_rel[i,_,1]
                    tmp[i,:,0] = pos_x
                    tmp[i,:,1] = pos_y
                # print("after",tmp)
                new_x.append(tmp)
            # assert(False)
            new_x = torch.stack(new_x,dim=2)#  N, V, T_{in}, in_channels
        elif self.align == 3:
            new_x = []
            for _ in range(A.shape[0]):
                tmp = x[:,_,:][A[_]] - x[:,_:_+1,:]# N,V,in_channels
                # print("before",tmp)
                index = tmp.sum(dim=1).sum(dim=1)==0
                # print(tmp[index].shape)
                # print(x_rel[:,_,:][index].shape)
                tmp[index] = x_rel[:,_,:][index].unsqueeze(1).repeat(1,tmp.shape[1],1)
                new_x.append(tmp)
            # assert(False)
            new_x = torch.stack(new_x,dim=2)#  N, V, T_{in}, in_channels

        x = new_x.permute(0,3,2,1).contiguous()#  N, in_channels, T_{in}, V
        x = self.conv(x)
        x = self.aggre(x).squeeze(-1)#N, in_channels, T_{in}

#         print("xinner:",x.shape)
        x = x.permute(2,0,1).contiguous()#T_{in}, N, in_channels
        if state_tuple == None:
            state_tuple = self.init_hidden(x.shape[1])
        x,state_tuple = self.lstm(x,state_tuple)
        x = x.permute(1,0,2).contiguous()
        state_tuple = [
            state_tuple[0].permute(1,0,2).contiguous(),
            state_tuple[1].permute(1,0,2).contiguous()
        ]
        # print("x.shape",x.shape)
        return x, A, state_tuple
class MultiLevel_DGCN_LSTM(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 align = 1,
                 level_num = 1,
                 seq_len=8,
                 bias=True,
                 node_aggregate_type=1,
                 level_aggregate_type=1,
                 ):
        super(MultiLevel_DGCN_LSTM,self).__init__()
        self.level_num=level_num
        self.level_aggregate_type = level_aggregate_type
        if self.level_aggregate_type == 1:
            self.level_aggre = nn.AdaptiveMaxPool2d((out_channels,1))
        elif self.level_aggregate_type == 2:
            self.level_aggre = make_mlp([level_num,level_num,1],batch_norm=False)
        elif self.level_aggregate_type == 3:
            self.level_aggre = Aggregation(shape=(level_num),equation='ntcl,l->ntc')
        elif self.level_aggregate_type == 4:
            self.level_aggre = Aggregation(shape=(out_channels,level_num),equation='ntcl,cl->ntc')
        elif self.level_aggregate_type == 5:##7
            self.level_aggre = Aggregation_Attn()
        # self.shared_dgcn_lstm = DGCN_LSTM(in_channels,out_channels,seq_len,align=align,aggregate_type=node_aggregate_type)
        
        self.dgcn_lstms = nn.ModuleList()
        for j in range(level_num):
            self.dgcn_lstms.append(DGCN_LSTM(in_channels,out_channels,seq_len,align=align,aggregate_type=node_aggregate_type))
        
    def forward(self,X,As,key_out,key_states,return_scores=False):
        # As : (level_num,T,N,V) 
        state_tuple_h = []
        state_tuple_c = []
        newXs = []
        for level in range(self.level_num):
            A = As[level]
            X_new,A_new,state_tuple = self.dgcn_lstms[level](X,A)
            newXs.append(X_new)
            state_tuple_h.append(state_tuple[0])
            state_tuple_c.append(state_tuple[1])
        state_tuple_h = torch.stack(state_tuple_h) # level_num * N * T * output_channel
        state_tuple_h = state_tuple_h.permute(1,2,3,0).contiguous() # N * T * output_channel * level_num
        if self.level_aggregate_type <= 4:
            state_tuple_h = self.level_aggre(state_tuple_h).squeeze(-1) # N * T * output_channel
        else:
            if return_scores:
                state_tuple_h,scores_h = self.level_aggre(state_tuple_h,key_states[0],return_scores)
            else:
                state_tuple_h = self.level_aggre(state_tuple_h,key_states[0])
        state_tuple_c = torch.stack(state_tuple_c) # level_num * N * T * output_channel
        state_tuple_c = state_tuple_c.permute(1,2,3,0).contiguous() # N * T * output_channel * level_num
        if self.level_aggregate_type <= 4:
            state_tuple_c = self.level_aggre(state_tuple_c).squeeze(-1) # N * T * output_channel
        else:
            if return_scores:
                state_tuple_c,scores_c = self.level_aggre(state_tuple_c,key_states[1],return_scores)
            else:
                state_tuple_c = self.level_aggre(state_tuple_c,key_states[1])

        newXs = torch.stack(newXs)# level_num * N * T * output_channel
        newXs = newXs.permute(1,2,3,0).contiguous() # N * T * output_channel * level_num
        if self.level_aggregate_type <= 4:
            X_new = self.level_aggre(newXs).squeeze(-1) # N * T * output_channel
        else:
            if return_scores:
                X,scores_x = self.level_aggre(newXs,key_out,return_scores)
            else:
                X = self.level_aggre(newXs,key_out)# T*N*C
            X = X.permute(1,0,2).contiguous()
        if return_scores:
            return X,As,(state_tuple_h,state_tuple_c),(scores_h,scores_c,scores_x)
        return X,As,(state_tuple_h,state_tuple_c)

class MultiLevelDynamicGraphCnnLstmEncoder(nn.Module):
    def __init__(self,
                 channels_list = [2,32],
                 align_list = [1],
                 level_num = 1,
                 seq_len=8,
                 bias=True,
                 node_aggregate_type=1,
                 level_aggregate_type=1,
                 ):
        super(MultiLevelDynamicGraphCnnLstmEncoder,self).__init__()
        self.n_mulLevel_dgcn_lstm=len(channels_list)-1
        self.mulLevel_dgcn_lstm = nn.ModuleList()
        self.mulLevel_dgcn_lstm.append(
            MultiLevel_DGCN_LSTM(
                in_channels = channels_list[0],
                out_channels = channels_list[1],
                level_num=level_num,
                seq_len = seq_len,
                align=align_list[0],
                node_aggregate_type=node_aggregate_type,
                level_aggregate_type=level_aggregate_type
                )
            )
        for j in range(1,self.n_mulLevel_dgcn_lstm):
            self.mulLevel_dgcn_lstm.append(
                MultiLevel_DGCN_LSTM(
                    in_channels = channels_list[j],
                    out_channels = channels_list[j+1],
                    level_num=level_num,
                    seq_len = seq_len,
                    align=align_list[j],
                    node_aggregate_type=node_aggregate_type,
                    level_aggregate_type=level_aggregate_type
                    )
                )
    def forward(self,X,As,key_out,key_states,return_scores=False):
        # As : (level_num,T,N,V) 
        for j in range(self.n_mulLevel_dgcn_lstm):
            if return_scores:
                X,As,state_tuple,scores_tuple = self.mulLevel_dgcn_lstm[j](X,As,key_out,key_states,return_scores)
            else:
                X,As,state_tuple = self.mulLevel_dgcn_lstm[j](X,As,key_out,key_states,return_scores)
        if return_scores:
            return X,As,state_tuple,scores_tuple
        return X,As,state_tuple

class MultiLevelDynamicGraphEncoder(nn.Module):
    def __init__(self,
                 channels_list = [2,32],
                 align_list = [1],
                 level_num = 1,
                 seq_len=8,
                 bias=True,
                 node_aggregate_type=1,
                 level_aggregate_type=1,
                 ):
        super(MultiLevelDynamicGraphEncoder,self).__init__()
        self.n_gcn_lstm=len(channels_list)-1
        self.level_num=level_num
        self.level_aggregate_type = level_aggregate_type
        if self.level_aggregate_type == 1:
            self.level_aggre = nn.AdaptiveMaxPool2d((channels_list[-1],1))
        self.gcn_lstms = nn.ModuleList()
        self.gcn_lstms.append(DGCN_LSTM(channels_list[0],channels_list[1],seq_len,align=align_list[0],aggregate_type=node_aggregate_type))
        for j in range(1,self.n_gcn_lstm):
            self.gcn_lstms.append(DGCN_LSTM(channels_list[j],channels_list[j+1],seq_len,align=align_list[j],aggregate_type=node_aggregate_type))
        
    def forward(self,X,As):
        # As : (level_num,T,N,V) 
        for j in range(self.n_gcn_lstm):
            state_tuple_h = []
            state_tuple_c = []
            newXs = []
            for level in range(self.level_num):
                A = As[level]
                X_new,A_new,state_tuple = self.gcn_lstms[j](X,A)
                newXs.append(X_new)
                state_tuple_h.append(state_tuple[0])
                state_tuple_c.append(state_tuple[1])
            state_tuple_h = torch.stack(state_tuple_h) # level_num * N * T * output_channel
            state_tuple_h = state_tuple_h.permute(1,2,3,0).contiguous() # N * T * output_channel * level_num
            state_tuple_h = self.level_aggre(state_tuple_h).squeeze(-1) # N * T * output_channel

            state_tuple_c = torch.stack(state_tuple_c) # level_num * N * T * output_channel
            state_tuple_c = state_tuple_c.permute(1,2,3,0).contiguous() # N * T * output_channel * level_num
            state_tuple_c = self.level_aggre(state_tuple_c).squeeze(-1) # N * T * output_channel

            newXs = torch.stack(newXs)# level_num * N * T * output_channel
            newXs = newXs.permute(1,2,3,0).contiguous() # N * T * output_channel * level_num
            X = self.level_aggre(newXs).squeeze(-1) # N * T * output_channel
        return X,As,(state_tuple_h,state_tuple_c)

class Encoder(nn.Module):
    """Encoder is part of both TrajectoryGenerator and
    TrajectoryDiscriminator"""
    def __init__(
        self, embedding_dim=64, h_dim=64, num_layers=1, dropout=0.0,batchnorm=False
    ):
        super(Encoder, self).__init__()

        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.batchnorm = batchnorm
        self.encoder = nn.LSTM(
            embedding_dim, h_dim, num_layers, dropout=dropout
        )
        if batchnorm:
            self.bn = nn.BatchNorm1d(2)
        self.spatial_embedding = nn.Linear(2, embedding_dim)

    def init_hidden(self, batch):
        return (
            torch.zeros(self.num_layers, batch, self.h_dim).cuda(),
            torch.zeros(self.num_layers, batch, self.h_dim).cuda()
        )

    def forward(self, obs_traj_rel):
        """
        Inputs:
        - obs_traj_rel: Tensor of shape (obs_len, batch, 2)
        Output:
        - final_h: Tensor of shape (self.num_layers, batch, self.h_dim)
        """
        # Encode observed Trajectory
        if self.batchnorm:
            obs_traj_rel = self.bn(obs_traj_rel.permute(1,2,0).contiguous()).permute(2,0,1).contiguous()
            
        batch = obs_traj_rel.size(1)
        obs_traj_embedding = self.spatial_embedding(obs_traj_rel.view(-1, 2))
        obs_traj_embedding = F.dropout(obs_traj_embedding,p=self.dropout,training=self.training)
        obs_traj_embedding = obs_traj_embedding.view(
            -1, batch, self.embedding_dim
        )
        state_tuple = self.init_hidden(batch)
        output, state_tuple = self.encoder(obs_traj_embedding, state_tuple)
        return output, state_tuple
class Decoder_noise(nn.Module):
    """Decoder is part of both TrajectoryGenerator and
    TrajectoryDiscriminator"""
    def __init__(
        self, pred_len=12, embedding_dim=64, h_dim=64, num_layers=1, dropout=0.0,batchnorm=False
    ):
        super(Decoder_noise, self).__init__()

        self.pred_len = pred_len
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.batchnorm = batchnorm
        self.decoder = nn.LSTM(
            embedding_dim, h_dim, num_layers, dropout=dropout
        )
        if batchnorm:
            self.bn=nn.BatchNorm1d(2)
        self.spatial_embedding = nn.Linear(2, embedding_dim)
        self.h_embedding = nn.Linear(h_dim+h_dim//2,h_dim)
        self.hidden2pos = nn.Linear(h_dim, 2)
    def init_hidden(self, batch):
        return (
            torch.zeros(self.num_layers, batch, self.h_dim).cuda(),
            torch.zeros(self.num_layers, batch, self.h_dim).cuda()
        )
    def get_noise(self,shape, noise_type):
        if noise_type == 'gaussian':
            return torch.randn(*shape).cuda()
        elif noise_type == 'uniform':
            return torch.rand(*shape).sub_(0.5).mul_(2.0).cuda()
    def forward(self, obs_traj_rel , obs_traj, state_tuple):
        """
        Inputs:
        - obs_traj_rel: Tensor of shape (obs_len, batch, 2)
        Output:
        - final_h: Tensor of shape (self.num_layers, batch, self.h_dim)
        """
        last_pos_rel = obs_traj_rel[-1]
        last_pos = obs_traj[-1]
        pred_traj_fake_rel = []
        pred_traj_fake = []

        sp1,sp2,sp3 = state_tuple[0].shape
        noise_h = self.get_noise((sp1,sp2,sp3//2),'gaussian')
        noise_c = self.get_noise((sp1,sp2,sp3//2),'gaussian')
        state_tuple = [self.h_embedding(torch.cat([noise_h,state_tuple[0]],dim=2)),
                        self.h_embedding(torch.cat([noise_c,state_tuple[1]],dim=2))]
        for _ in range(self.pred_len):
            if self.batchnorm:
                last_pos_rel = self.bn(last_pos_rel)
            embedding = self.spatial_embedding(last_pos_rel)
            embedding = F.dropout(embedding,self.dropout,self.training)
            output,state_tuple = self.decoder(embedding.unsqueeze(0),state_tuple)
            output = F.dropout(output.squeeze(),self.dropout,self.training)
            pos_rel = self.hidden2pos(output)
            last_pos = last_pos+pos_rel
            pred_traj_fake_rel.append(pos_rel)
            pred_traj_fake.append(last_pos)
            last_pos_rel = pos_rel
        pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=0)
        pred_traj_fake = torch.stack(pred_traj_fake, dim=0)
        return pred_traj_fake_rel,pred_traj_fake


class Decoder_RF(nn.Module):
    """Decoder is part of both TrajectoryGenerator and
    TrajectoryDiscriminator"""
    def __init__(
        self, pred_len=12, embedding_dim=64, h_dim=64, num_layers=1, dropout=0.0,noise=True
    ):
        super(Decoder_RF, self).__init__()

        self.pred_len = pred_len
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.noise = noise
        self.decoder = nn.LSTM(
            embedding_dim, h_dim, num_layers, dropout=dropout
        )
        if self.noise:
            self.h_embedding = nn.Linear(h_dim+h_dim+h_dim//2, h_dim)
        else:
            self.h_embedding = nn.Linear(h_dim+h_dim, h_dim)
        self.spatial_embedding = nn.Linear(2, embedding_dim)
        self.hidden2pos = nn.Linear(h_dim, 2)
    def init_hidden(self, batch):
        return (
            torch.zeros(self.num_layers, batch, self.h_dim).cuda(),
            torch.zeros(self.num_layers, batch, self.h_dim).cuda()
        )
    def get_noise(self,shape, noise_type):
        if noise_type == 'gaussian':
            return torch.randn(*shape).cuda()
        elif noise_type == 'uniform':
            return torch.rand(*shape).sub_(0.5).mul_(2.0).cuda()
    def forward(self, obs_traj_rel , obs_traj, state_tuple_en, state_tuple_rf):
        """
        Inputs:
        - obs_traj_rel: Tensor of shape (obs_len, batch, 2)
        Output:
        - final_h: Tensor of shape (self.num_layers, batch, self.h_dim)
        """
        last_pos_rel = obs_traj_rel[-1]
        last_pos = obs_traj[-1]
        pred_traj_fake_rel = []
        pred_traj_fake = []
        
        if self.noise:
            sp1,sp2,sp3 = state_tuple_en[0].shape
            noise_h = self.get_noise((sp1,sp2,sp3//2),'gaussian')
            noise_c = self.get_noise((sp1,sp2,sp3//2),'gaussian')
            state_tuple = [self.h_embedding(torch.cat([noise_h,state_tuple_en[0],state_tuple_rf[0]],dim=2)),
                            self.h_embedding(torch.cat([noise_c,state_tuple_en[1],state_tuple_rf[1]],dim=2))]
        else:
            state_tuple = [self.h_embedding(torch.cat([state_tuple_en[0],state_tuple_rf[0]],dim=2)),
                            self.h_embedding(torch.cat([state_tuple_en[1],state_tuple_rf[1]],dim=2))]
        
        for _ in range(self.pred_len):
            embedding = self.spatial_embedding(last_pos_rel)
            embedding = F.dropout(embedding,self.dropout,self.training)
            output,state_tuple = self.decoder(embedding.unsqueeze(0),state_tuple)
            output = F.dropout(output.squeeze(),self.dropout,self.training)
            pos_rel = self.hidden2pos(output)
            last_pos = last_pos+pos_rel
            pred_traj_fake_rel.append(pos_rel)
            pred_traj_fake.append(last_pos)
            last_pos_rel = pos_rel
        pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=0)
        pred_traj_fake = torch.stack(pred_traj_fake, dim=0)
        return pred_traj_fake_rel,pred_traj_fake
class MultiLevelSocialModel(nn.Module):
    def __init__(self,
                 obs_len=8,
                 pred_len=12,
                 channels_list=[2,32],
                 align_list=[1],
                 level_num=1,
                 node_aggregate_type=1,
                 level_aggregate_type=1,
                 embedding_dim = 32
                 ):
        super(MultiLevelSocialModel,self).__init__()
        self.MGencoder = MultiLevelDynamicGraphEncoder(
                                channels_list = channels_list,
                                align_list=align_list,
                                level_num = level_num,
                                seq_len = obs_len,
                                bias=True,
                                node_aggregate_type=node_aggregate_type,
                                level_aggregate_type=level_aggregate_type
                                )
        self.encoder = Encoder( embedding_dim=embedding_dim,h_dim=channels_list[-1],num_layers=1,dropout=0.0)
        self.decoder = Decoder_RF(pred_len=pred_len,embedding_dim=embedding_dim,h_dim=channels_list[-1],num_layers=1,dropout=0.0)
    def forward(self,X,As,obs_traj_rel,obs_traj):
        X,As,state_tuple_mg = self.MGencoder(X,As)
        output, state_tuple_en = self.encoder(obs_traj_rel)
        pred_traj_fake_rel,pred_traj_fake = self.decoder(obs_traj_rel , obs_traj, state_tuple_en, state_tuple_mg)
        return pred_traj_fake_rel,pred_traj_fake

class MultiLevelDynamicSocialModel(nn.Module):
    def __init__(self,
                 obs_len=8,
                 pred_len=12,
                 channels_list=[2,32],
                 align_list=[1],
                 level_num=1,
                 node_aggregate_type=1,
                 level_aggregate_type=1,
                 embedding_dim = 32,
                 noise=True
                 ):
        super(MultiLevelDynamicSocialModel,self).__init__()
        self.MLDGencoder = MultiLevelDynamicGraphCnnLstmEncoder(
                                channels_list = channels_list,
                                align_list=align_list,
                                level_num = level_num,
                                seq_len = obs_len,
                                bias=True,
                                node_aggregate_type=node_aggregate_type,
                                level_aggregate_type=level_aggregate_type
                                )
        self.encoder = Encoder( embedding_dim=embedding_dim,h_dim=channels_list[-1],num_layers=1,dropout=0.0)
        self.decoder = Decoder_RF(pred_len=pred_len,embedding_dim=embedding_dim,h_dim=channels_list[-1],num_layers=1,dropout=0.0,noise=noise)
    def forward(self,X,As,obs_traj_rel,obs_traj,return_scores=False):
        output, state_tuple_en = self.encoder(obs_traj_rel)
        if return_scores:
            X,As,state_tuple_mg,scores_tuple = self.MLDGencoder(X,As,output,state_tuple_en,return_scores)
        else:
            X,As,state_tuple_mg = self.MLDGencoder(X,As,output,state_tuple_en,return_scores)
        
        pred_traj_fake_rel,pred_traj_fake = self.decoder(obs_traj_rel , obs_traj, state_tuple_en, state_tuple_mg)
        if return_scores:
            return pred_traj_fake_rel,pred_traj_fake,scores_tuple
        return pred_traj_fake_rel,pred_traj_fake

def make_mlp(dim_list, activation='relu', batch_norm=True, dropout=0):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-2], dim_list[1:-1]):
        # layers.append(nn.utils.spectral_norm(nn.Linear(dim_in, dim_out)))
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    # layers.append(nn.utils.spectral_norm(nn.Linear(dim_list[-2], dim_list[-1])))
    layers.append(nn.Linear(dim_list[-2], dim_list[-1]))
    return nn.Sequential(*layers) 

class Aggregation(nn.Module):
    def __init__(self,shape=10,equation='ntcl,l->ntc'):
        super(Aggregation,self).__init__()
        self.equation = equation
        if type(shape) == int:
            shape = [shape]
        # self.weights = nn.Parameter(torch.ones(shape)/shape[-1])
        init_weights = torch.zeros(shape)
        init_weights[:,0]=1
        self.weights = nn.Parameter(init_weights)
    def forward(self,x):
        # x.shape -> N * T * output_channel * level_num
        x = torch.einsum(self.equation, (x, self.weights))
        return x


"""
按轨迹求层次汇聚的score
"""
class Aggregation_Attn(nn.Module):
    def __init__(self):
        super(Aggregation_Attn_3,self).__init__()
    def forward(self,query,key,return_scores=False):
        # x.shape -> N * T * output_channel * level_num
        # print("query:",query.shape)
        # print("key:",key.shape)
        scores = torch.einsum('ntcl,tnc->nl', (query, key))
        scores = torch.softmax(scores,dim=1)
        # print(scores.shape)
        # print(scores)
        out = torch.einsum('ntcl,nl->tnc', (query, scores))
        if return_scores:
            return out,scores
        else:
            return out


class MultiLevel_Shared_DGCN_LSTM(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 align = 1,
                 level_num = 1,
                 seq_len=8,
                 bias=True,
                 node_aggregate_type=1,
                 level_aggregate_type=1,
                 ):
        super(MultiLevel_Shared_DGCN_LSTM,self).__init__()
        self.level_num=level_num
        self.level_aggregate_type = level_aggregate_type
        if self.level_aggregate_type == 1:
            self.level_aggre = nn.AdaptiveMaxPool2d((out_channels,1))
        elif self.level_aggregate_type == 2:
            self.level_aggre = make_mlp([level_num,level_num,1],batch_norm=False)
        elif self.level_aggregate_type == 3:
            self.level_aggre = Aggregation(shape=(level_num),equation='ntcl,l->ntc')
        elif self.level_aggregate_type == 4:
            self.level_aggre = Aggregation(shape=(out_channels,level_num),equation='ntcl,cl->ntc')
        elif self.level_aggregate_type == 5:
            self.level_aggre = Aggregation_Attn()
        
        self.shared_dgcn_lstm = DGCN_LSTM(in_channels,out_channels,seq_len,align=align,aggregate_type=node_aggregate_type)
        
    def forward(self,X,As,key_out,key_states,return_scores=False):
        # As : (level_num,T,N,V) 
        state_tuple_h = []
        state_tuple_c = []
        newXs = []
        for level in range(self.level_num):
            A = As[level]
            X_new,A_new,state_tuple = self.shared_dgcn_lstm(X,A)
            newXs.append(X_new)
            state_tuple_h.append(state_tuple[0])
            state_tuple_c.append(state_tuple[1])
        
        state_tuple_h = torch.stack(state_tuple_h) # level_num * N * T * output_channel
        # print("state_tuple_h:",state_tuple_h.shape)
        state_tuple_h = state_tuple_h.permute(1,2,3,0).contiguous() # N * T * output_channel * level_num
        if self.level_aggregate_type <= 4:
            state_tuple_h = self.level_aggre(state_tuple_h).squeeze(-1) # N * T * output_channel
        else:
            if return_scores:
                state_tuple_h,scores_h = self.level_aggre(state_tuple_h,key_states[0],return_scores)
            else:
                state_tuple_h = self.level_aggre(state_tuple_h,key_states[0])
        state_tuple_c = torch.stack(state_tuple_c) # level_num * N * T * output_channel
        state_tuple_c = state_tuple_c.permute(1,2,3,0).contiguous() # N * T * output_channel * level_num
        if self.level_aggregate_type <= 4:
            state_tuple_c = self.level_aggre(state_tuple_c).squeeze(-1) # N * T * output_channel
        else:
            if return_scores:
                state_tuple_c,scores_c = self.level_aggre(state_tuple_c,key_states[1],return_scores)
            else:
                state_tuple_c = self.level_aggre(state_tuple_c,key_states[1])
        newXs = torch.stack(newXs)# level_num * N * T * output_channel
        
        newXs = newXs.permute(1,2,3,0).contiguous() # N * T * output_channel * level_num
        if self.level_aggregate_type <= 4:
            X = self.level_aggre(newXs).squeeze(-1) # N * T * output_channel
        else:
            if return_scores:
                X,scores_x = self.level_aggre(newXs,key_out,return_scores)
            else:
                X = self.level_aggre(newXs,key_out)# T*N*C
            X = X.permute(1,0,2).contiguous()
        if return_scores:
            return X,As,(state_tuple_h,state_tuple_c),(scores_h,scores_c,scores_x)
        return X,As,(state_tuple_h,state_tuple_c)

class MultiLevelDynamicSharedGraphCnnLstmEncoder(nn.Module):
    def __init__(self,
                 channels_list = [2,32],
                 align_list = [1],
                 level_num = 1,
                 seq_len=8,
                 bias=True,
                 node_aggregate_type=1,
                 level_aggregate_type=1,
                 ):
        super(MultiLevelDynamicSharedGraphCnnLstmEncoder,self).__init__()
        self.n_mulLevel_dgcn_lstm=len(channels_list)-1
        self.mulLevel_dgcn_lstm = nn.ModuleList()
        self.mulLevel_dgcn_lstm.append(
            MultiLevel_Shared_DGCN_LSTM(
                in_channels = channels_list[0],
                out_channels = channels_list[1],
                level_num=level_num,
                seq_len = seq_len,
                align=align_list[0],
                node_aggregate_type=node_aggregate_type,
                level_aggregate_type=level_aggregate_type
                )
            )
        for j in range(1,self.n_mulLevel_dgcn_lstm):
            self.mulLevel_dgcn_lstm.append(
                MultiLevel_Shared_DGCN_LSTM(
                    in_channels = channels_list[j],
                    out_channels = channels_list[j+1],
                    level_num=level_num,
                    seq_len = seq_len,
                    align=align_list[j],
                    node_aggregate_type=node_aggregate_type,
                    level_aggregate_type=level_aggregate_type
                    )
                )
    def forward(self,X,As,key_out,key_states,return_scores=False):
        # As : (level_num,T,N,V) 
        for j in range(self.n_mulLevel_dgcn_lstm):
            if return_scores:
                X,As,state_tuple,scores_tuple = self.mulLevel_dgcn_lstm[j](X,As,key_out,key_states,return_scores)
            else:
                X,As,state_tuple = self.mulLevel_dgcn_lstm[j](X,As,key_out,key_states,return_scores)
        if return_scores:
            return X,As,state_tuple,scores_tuple
        return X,As,state_tuple

class MultiLevelDynamicSocialModel_SharedGCNLSTM(nn.Module):
    def __init__(self,
                 obs_len=8,
                 pred_len=12,
                 channels_list=[2,32],
                 align_list=[1],
                 level_num=1,
                 node_aggregate_type=1,
                 level_aggregate_type=1,
                 embedding_dim = 32,
                 noise=True
                 ):
        super(MultiLevelDynamicSocialModel_SharedGCNLSTM,self).__init__()
        self.MLDGencoder = MultiLevelDynamicSharedGraphCnnLstmEncoder(
                                channels_list = channels_list,
                                align_list=align_list,
                                level_num = level_num,
                                seq_len = obs_len,
                                bias=True,
                                node_aggregate_type=node_aggregate_type,
                                level_aggregate_type=level_aggregate_type
                                )
        self.encoder = Encoder( embedding_dim=embedding_dim,h_dim=channels_list[-1],num_layers=1,dropout=0.0)
        self.decoder = Decoder_RF(pred_len=pred_len,embedding_dim=embedding_dim,h_dim=channels_list[-1],num_layers=1,dropout=0.0,noise=noise)
    def forward(self,X,As,obs_traj_rel,obs_traj,return_scores=False):
        output, state_tuple_en = self.encoder(obs_traj_rel)
        if return_scores:
            X,As,state_tuple_mg,scores_tuple = self.MLDGencoder(X,As,output,state_tuple_en,return_scores)
        else:
            X,As,state_tuple_mg = self.MLDGencoder(X,As,output,state_tuple_en,return_scores)
        
        pred_traj_fake_rel,pred_traj_fake = self.decoder(obs_traj_rel , obs_traj, state_tuple_en, state_tuple_mg)
        if return_scores:
            return pred_traj_fake_rel,pred_traj_fake,scores_tuple
        return pred_traj_fake_rel,pred_traj_fake