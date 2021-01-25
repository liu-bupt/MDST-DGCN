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
from joblib import Parallel, delayed
import torch.optim as optim

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from numpy import linalg as LA
import networkx as nx
from tqdm import tqdm
import time
from sklearn.neighbors import KDTree


def seq_collate(data):
    (obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list,
     non_linear_ped_list, loss_mask_list, X_obs_list,X_obs_rel_list,As_obs_list) = zip(*data)

    _len = [len(seq) for seq in obs_seq_list]
    
    obs_traj = torch.cat(obs_seq_list,dim=1)
    pred_traj = torch.cat(pred_seq_list,dim=1)
    obs_traj_rel = torch.cat(obs_seq_rel_list,dim=1)
    pred_traj_rel = torch.cat(pred_seq_rel_list,dim=1)
    non_linear_ped = torch.cat(non_linear_ped_list)
    loss_mask = torch.cat(loss_mask_list)
    X_obs = torch.cat(X_obs_list,dim=0)
    X_obs_rel = torch.cat(X_obs_rel_list,dim=0)

    # A_obs: level_num * t * num_ped * k
    count = 0
    countlist = [count]
    # print("len(As_obs_list)",len(As_obs_list))
    As_obs = []
    for _ in range(len(As_obs_list)): 
        # print("type(As_obs_list[_])",type(As_obs_list[_]))
        # print(As_obs_list[_][0].shape)
        # print("count:",count)
        # print("As_obs_list[_]:",As_obs_list[_].shape)
        # print("As_obs_list[_]",As_obs_list[_])
        As_obs.append(As_obs_list[_]+count)
        count = count+As_obs_list[_].shape[2]
        countlist.append(count)
    As_obs = torch.cat(As_obs,dim=2)
    
    seq_start_end = [
            [start, end]
            for start, end in zip(countlist[:-1], countlist[1:])
        ]
    
    seq_start_end = torch.LongTensor(seq_start_end)
    out = [
        obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, non_linear_ped,
        loss_mask,seq_start_end, X_obs, X_obs_rel, As_obs
    ]

    return out

def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0
def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)


class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, data_dir, obs_len=8, pred_len=12, skip=1, threshold=0.002,
        min_ped=10,max_ped=np.inf,dis_list=[0],k=30, delim='\t'):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        -  
        in the scene are considered as its neighbors.
        """
        super(TrajectoryDataset, self).__init__()

        self.max_peds_in_frame = 0
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim
        
        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_ped = []
        for path in all_files:
            data = read_file(path, delim)
            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            num_sequences = int(
                math.ceil((len(frames) - self.seq_len + 1) / skip))

            for idx in range(0, num_sequences * self.skip + 1, skip):
                curr_seq_data = np.concatenate(
                    frame_data[idx:idx + self.seq_len], axis=0)
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                self.max_peds_in_frame = max(self.max_peds_in_frame,len(peds_in_curr_seq))
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2,
                                         self.seq_len))
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_loss_mask = np.zeros((len(peds_in_curr_seq),
                                           self.seq_len))
                num_peds_considered = 0
                _non_linear_ped = []
                for _, ped_id in enumerate(peds_in_curr_seq):
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] ==
                                                 ped_id, :]
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                    if pad_end - pad_front != self.seq_len:
                        continue
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                    curr_ped_seq = curr_ped_seq
                    # Make coordinates relative
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    rel_curr_ped_seq[:, 1:] = \
                        curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                    _idx = num_peds_considered
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                    # Linear vs Non-Linear Trajectory
                    _non_linear_ped.append(
                        poly_fit(curr_ped_seq, pred_len, threshold))
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    num_peds_considered += 1

                if num_peds_considered >= min_ped and num_peds_considered <= max_ped:
                    non_linear_ped += _non_linear_ped
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)

        # Convert numpy -> Torch Tensor
        self.obs_traj = seq_list[:, :, :self.obs_len]
        self.pred_traj = seq_list[:, :, self.obs_len:]
        self.obs_traj_rel = seq_list_rel[:, :, :self.obs_len]
        self.pred_traj_rel = seq_list_rel[:, :, self.obs_len:]
        self.loss_mask = loss_mask_list
        self.non_linear_ped = non_linear_ped
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]
        #Convert to Graphs 
        
        self.As_obs = [] 
        
        print("Processing Data .....")
        np.random.seed(0)
        pbar = tqdm(total=len(self.seq_start_end)) 
        for ss in range(len(self.seq_start_end)):
            pbar.update(1)
            A_obs_tmp = []
            start, end = self.seq_start_end[ss]
            seq_obs_traj = self.obs_traj[start:end,:] # b 2 t
            

            for t in range(obs_len):

                A_obs_tmp_t = []

                X = seq_obs_traj[:,:,t]
                
                tree = KDTree(X)
                index = np.zeros((end-start),dtype=int)
                for dis_thr in dis_list:
                    querys,diss = tree.query_radius(X,dis_thr,return_distance=True,sort_results=True)
                    # print("dis_thr",dis_thr)
                    # print("diss",diss)
                    querys_new = []
                    for _,query_ped in enumerate(querys):
                        # index[_] = np.sum(diss[_]<=0.5)
                        tmp = query_ped
                        query_ped = query_ped[index[_]:]
                        tmp_index = index[_]
                        index[_] = index[_]+query_ped.shape[0]
                        if _ not in query_ped:
                            query_ped = np.append(query_ped,_)
                        if (query_ped.shape[0] > k):
                            print('tmp',tmp.shape)
                            print('tmp_index',tmp_index)
                            print("query_ped.shape[0]:",query_ped.shape[0])
                            print("k:",k)
                            print("dis_thr:",dis_thr)
                        #     print("index[_]:",index[_])
                        #     print(sum([np.unique(query[_]).shape[0] for query in A_obs_tmp_t]))
                        assert(query_ped.shape[0] <= k)
                        # if query_ped.shape[0] > 60:
                        #     print("query_ped.shape[0]:",query_ped.shape[0],"dis_thr:",dis_thr)
                        # try:
                        #     query_ped.shape[0] <= k
                        # except:
                        #     print("query_ped.shape[0]:",query_ped.shape[0],'k:',k)
                        if query_ped.shape[0] < k:
                            randselectIndex = np.random.randint(0,query_ped.shape[0],(k-query_ped.shape[0]))
                            # print("query_ped",query_ped.shape)
                            # print("randselectIndex",randselectIndex)
                            # print(randselectIndex.shape)
                            # print('query_ped[randselectIndex]',query_ped[randselectIndex])
                            # print(query_ped[randselectIndex].shape)
                            query_ped = np.concatenate([query_ped,query_ped[randselectIndex]]) # k
                        querys_new.append(query_ped)# num_ped * k
                    A_obs_tmp_t.append(querys_new)# level_num * num_ped * k
                A_obs_tmp.append(A_obs_tmp_t)# t * level_num * num_ped * k
                    
            A_obs_tmp = torch.tensor(A_obs_tmp)# t * level_num * num_ped * k
            A_obs_tmp = A_obs_tmp.permute(1,0,2,3).contiguous()# level_num * t * num_ped * k
            self.As_obs.append(A_obs_tmp)
            
        pbar.close()
        self.obs_traj = torch.from_numpy(self.obs_traj).type(torch.float)
        self.pred_traj = torch.from_numpy(self.pred_traj).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(self.obs_traj_rel).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(self.pred_traj_rel).type(torch.float)
        self.loss_mask = torch.from_numpy(self.loss_mask).type(torch.float)
        self.non_linear_ped = torch.from_numpy(self.non_linear_ped).type(torch.float)
        
    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]

        
        out = [
            self.obs_traj[start:end, :].permute(2, 0, 1).contiguous(), self.pred_traj[start:end, :].permute(2, 0, 1).contiguous(),
            self.obs_traj_rel[start:end, :].permute(2, 0, 1).contiguous(), self.pred_traj_rel[start:end, :].permute(2, 0, 1).contiguous(),
            self.non_linear_ped[start:end], self.loss_mask[start:end, :],
            self.obs_traj[start:end, :].permute(0, 2, 1).contiguous(),
            self.obs_traj_rel[start:end, :].permute(0, 2, 1).contiguous(),
            self.As_obs[index]
        ]
        
        return out
    


def data_loader(args, path, shuffle, max_ped=np.inf):
    dset = TrajectoryDataset(
        path,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        skip=args.skip,
        delim=args.delim,
        min_ped=args.min_ped,
        max_ped=max_ped,
        dis_list=args.dis_list,
        k = args.sample_k
        )

    loader = DataLoader(
        dset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.loader_num_workers,
        collate_fn=seq_collate)
    return dset, loader


import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug',default=0,type=int)
    # Dataset options
    parser.add_argument('--dataset_name', default='zara1', type=str)
    parser.add_argument('--delim', default='\t')
    parser.add_argument('--loader_num_workers', default=0, type=int)
    parser.add_argument('--obs_len', default=8, type=int)
    parser.add_argument('--pred_len', default=12, type=int)
    parser.add_argument('--skip', default=1, type=int)
    parser.add_argument("--min_ped",default=2,type=int)
    parser.add_argument("--batch_size",default=4,type=int)
    parser.add_argument("--sample_k",default=120,type=int)
    parser.add_argument("--dis_list",default=[0.3, 0.6, 2],nargs='+', type=float)
    args = parser.parse_args()

    data_set = '/mnt/sharedisk/liuhaibo/code/pythoncode/RF/datasets/pedWalkNorm/test'
    dset, train_loader = data_loader(args,data_set,shuffle=False)
    # dset, train_loader2 = data_loader2(args,data_set,shuffle=False)
    # data_set = '/mnt/sharedisk/liuhaibo/code/pythoncode/RF/datasets/zara1/test'
    # dset, loader = data_loader(args,data_set,shuffle=True)
    sample_num = len(train_loader)
    iterations_per_epoch = sample_num//32
    print(
        'There are {} iterations per epoch'.format(iterations_per_epoch)
    )
    # dset = TrajectoryDataset(
    #     data_set,
    #     obs_len=args,
    #     pred_len=12,
    #     skip=1,
    #     delim='\t',
    #     min_ped=2,
    #     num_nearest_neighbors_list=[0]
    #     )

    # loader = DataLoader(
    #     dset,
    #     batch_size=1,
    #     shuffle=True,
    #     num_workers=4,
    #     collate_fn=seq_collate)
    
    # X_obs1 = []
    # for cnt,batch in enumerate(train_loader): 
    #     #Get data
    #     # batch[:8] = [tensor.cuda() for tensor in batch[:8]]
    #     obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,\
    #     loss_mask,X_obs,X_obs_rel,As_obs,As_obs_rel = batch
    #     X_obs1.append(X_obs[As_obs])
    #     print("X_obs1",As_obs[0][0])
    #     print(X_obs1[-1].shape)
    #     if cnt == args.batch_size-1:
    #         break
    # X_obs1 = torch.cat(X_obs1)
    for _,batch in enumerate(train_loader):
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,\
        loss_mask,seq_start_end,X_obs,X_obs_rel,As_obs = batch
        print(As_obs.shape)
        for start,end in seq_start_end:
            # print("start:{}   end:{}".format(start,end))
            # print("As_obs[0,0,start]",As_obs[0,0,start])
            print("As_obs[0,0,start]",torch.unique(As_obs[0,0,start]).shape)
            # print('As_obs[1,0,start]',As_obs[1,0,start])
            print("As_obs[1,0,start]",torch.unique(As_obs[1,0,start]).shape)
            # print('As_obs[2,0,start]',As_obs[2,0,start])
            print("As_obs[2,0,start]",torch.unique(As_obs[2,0,start]).shape)
        if _+1%2 == 0:
            break
    # print(As_obs[seq_start_end[:,0]])
    # print(X_obs1)
    # print(X_obs2)
    
    # print(X_obs1.shape)
    # print(X_obs2.shape)
    # print(X_obs2.shape[0]*X_obs2.shape[1]*X_obs2.shape[2]*X_obs2.shape[3])
    # print((X_obs1 == X_obs2).sum())
        # for _,tensor in enumerate(batch[:8]):
        #     print(_,type(tensor),tensor.device,tensor.shape)
        # print("obs_traj",obs_traj.shape)
        # print("pred_traj_gt",pred_traj_gt.shape)
        # print("obs_traj_rel",obs_traj_rel.shape)
        # print("pred_traj_gt_rel",pred_traj_gt_rel.shape)
        # for A in As_obs:
        #     print("A.shape",A.shape)
            # print(A)
        
        # print(len(V_obs_levels))
        # print(type(V_obs_levels))
        # for i in range(len(V_obs_levels)):
        #     print(len(V_obs_levels[i]))
        #     print(type(V_obs_levels[i]))
        #     print(V_obs_levels[i].shape)

        # print("------------------------------------------")
        # print(len(A_obs_levels))
        # print(type(A_obs_levels))
        # for i in range(len(A_obs_levels)):
        #     print(len(A_obs_levels[i]))
        #     print(type(A_obs_levels[i]))
        #     print("A_obs_levels[i]",A_obs_levels[i].shape)
        
        # print(type(cluster_label_levels))
        # print((cluster_label_levels.shape))
        # cluster_label_levels = cluster_label_levels.cuda().squeeze() 
        # print(cluster_label_levels)
        # n_clusters = V_obs_levels[0].shape[2]
        # for i in range(2):
        #     n_clusters = round(n_clusters/2+0.5)
        #     for j in range(n_clusters):
        #         index = cluster_label_levels[i] == j
        #         print("V_obs_levels[{}][0,:,index,:]".format(i),index,'\n',V_obs_levels[i][0,:,index,:])
        # break