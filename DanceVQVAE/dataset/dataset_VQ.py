

import codecs as cs
import os
import random
from os.path import join as pjoin

import numpy as np
import torch
from torch.utils import data
from tqdm import tqdm

import DanceVQVAE.options.option_vq as option_vq

args = option_vq.get_args_parser()
class VQMotionDataset(data.Dataset):
    def __init__(self, dataset_name,sfile, window_size = 64, unit_length = 4):
        self.window_size = window_size
        self.unit_length = unit_length
        self.dataset_name = dataset_name

        if dataset_name == 'all':

            self.data_root=args.data_root
            self.motion_dir = pjoin(self.data_root, 'alldata_new_joint_vecs264')
            if not os.path.exists(self.motion_dir):
                self.motion_dir = pjoin(self.data_root, 'new_joint_vecs264')

            self.joints_num = 22
            self.max_motion_length = 196
            
            if args.meta_dir!=None:
                self.meta_dir = args.meta_dir
            else:
                self.meta_dir =pjoin(self.data_root, 'meta')

                    
        mean = np.load(pjoin(self.meta_dir, 'Mean.npy'))
        std = np.load(pjoin(self.meta_dir, 'Std.npy'))
        if not os.path.exists(pjoin(self.meta_dir, 'Mean.npy')):
            raise FileNotFoundError(f"File not found: {mean}")
        if not os.path.exists(pjoin(self.meta_dir, 'Std.npy')):
            raise FileNotFoundError(f"File not found: {std}")

        split_file = sfile

        self.data = []
        self.lengths = []
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(self.motion_dir, name + '.npy'))

                
                
                
                if motion.shape[0] < self.window_size:
                    continue
                self.lengths.append(motion.shape[0] - self.window_size)
                self.data.append(motion)
            except:

                pass
        self.mean = mean
        self.std = std

        print("Total number of motions {}".format(len(self.data)))

    def inv_transform(self, data):
        return data * self.std + self.mean
    
    def compute_sampling_prob(self) : 
        
        prob = np.array(self.lengths, dtype=np.float32)
        prob /= np.sum(prob)
        return prob
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        motion = self.data[item]
        
        idx = random.randint(0, len(motion) - self.window_size)

        motion = motion[idx:idx+self.window_size]
        "Z Normalization"
        motion = (motion - self.mean) / self.std

        return motion

def DATALoader(dataset_name,
               batch_size,
               sfile,
               num_workers = 8,
               window_size = 64,
               unit_length = 4
               ):
    
    trainSet = VQMotionDataset(dataset_name,sfile=sfile, window_size=window_size, unit_length=unit_length)
    prob = trainSet.compute_sampling_prob()
    sampler = torch.utils.data.WeightedRandomSampler(prob, num_samples = len(trainSet) * 1000, replacement=True)
    train_loader = torch.utils.data.DataLoader(trainSet,
                                              batch_size,
                                              shuffle=True,
                                              #sampler=sampler,
                                              num_workers=num_workers,
                                              #collate_fn=collate_fn,
                                              drop_last = True)
    
    return train_loader,trainSet



def DATALoader_val(dataset_name,
               batch_size,
               sfile,
               num_workers = 8,
               window_size = 64,
               unit_length = 4
               ):
    
    testSet = VQMotionDataset(dataset_name,sfile=sfile, window_size=window_size, unit_length=unit_length)
    prob = testSet.compute_sampling_prob()
    sampler = torch.utils.data.WeightedRandomSampler(prob, num_samples = len(testSet) * 1000, replacement=True)
    test_loader = torch.utils.data.DataLoader(testSet,
                                              batch_size,
                                              shuffle=False,
                                              #sampler=sampler,
                                              num_workers=num_workers,
                                              #collate_fn=collate_fn,
                                              drop_last = True)
    
    return test_loader,testSet
def cycle(iterable):
    while True:
        for x in iter(iterable):  # 注意这里的 iter()
            yield x
# class MotionDataset(data.Dataset):
#     def __init__(self, opt, split_file,data_root):
#         self.opt = opt
#         joints_num = opt.nb_joints

        
#         motion_dir = pjoin(data_root, 'alldata_new_joint_vecs264')
#         if not os.path.exists(motion_dir):
#             motion_dir = pjoin(data_root, 'new_joint_vecs264')
#         print("motion_dir",motion_dir)
#         mean = np.load(pjoin(data_root, 'Mean.npy'))
#         std = np.load(pjoin(data_root, 'Std.npy'))
#         self.data = []
#         self.lengths = []
#         id_list = []
#         with open(split_file, 'r') as f:
#             for line in f.readlines():
#                 if line.strip():  
#                     id_list.append(line.strip())

#         for name in tqdm(id_list):
#             try:
#                 motion = np.load(pjoin(motion_dir, name + '.npy'))
#                 # motion = np.delete(motion, [0, 2], axis=1)
#                 #delete
#                 #print(motion.shape)
#                 if motion.shape[0] < opt.window_size:
#                     # print("name",name)
#                     # print("motion_shape",motion.shape)
#                     #print(opt.window_size)
#                     #print(motion.shape[0])
#                     continue
#                 self.lengths.append(motion.shape[0] - opt.window_size)
#                 self.data.append(motion)
#             except Exception as e:
#                 # Some motion may not exist in KIT dataset
#                 print("name",name)
#                 print(e)
#                 # pass

#         self.cumsum = np.cumsum([0] + self.lengths)

     

#         self.mean = mean
#         self.std = std
#         # #264
#         # self.mean = np.delete(mean, [0, 2])
#         # self.std = np.delete(std, [0, 2])
#         # breakpoint()
#         print("Total number of motions {}, snippets {}".format(len(self.data), self.cumsum[-1]))
        

#     def inv_transform(self, data):
#         return data * self.std + self.mean

#     def __len__(self):
#         return self.cumsum[-1]

#     def __getitem__(self, item):
#         if item != 0:
#             motion_id = np.searchsorted(self.cumsum, item) - 1
#             idx = item - self.cumsum[motion_id] - 1
#         else:
#             motion_id = 0
#             idx = 0
#         motion = self.data[motion_id][idx:idx + self.opt.window_size]
#         "Z Normalization"
#         motion = (motion - self.mean) / self.std

#         return motion