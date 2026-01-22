import random
import sys
from os.path import join as pjoin

import h5py
import numpy as np
import torch
from myutils.moutils.process_fd_2 import get_cont6d_params
from torch.utils import data
from tqdm import tqdm
from utils.plot_script import *
from utils.preprocess import *
from utils.utils import *
from utils.utils import skipprocess_motion_np


def get_713_from3388(data):
    res = data[..., :709]
    if isinstance(data, np.ndarray):
        res = np.concatenate([res, data[..., 2674:2678]], axis=-1  )
    elif torch.is_tensor(data):
        res = torch.cat([res, data[..., 2674:2678]], dim=-1  )
    return res

def recover_root_rot_pos(data):
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    '''Get Y-axis rotation from rotation velocity'''
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    '''Add Y-axis rotation to root position'''
    r_pos = qrot(qinv(r_rot_quat), r_pos)

    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos

def recover_from_3388mofeas_noinithands(data):
    # assert data.shape[-1] == 669
    assert data.shape[-1] in [669,709,713,3388]
    if data.shape[-1] == 669:
        r_rot_quat, r_pos = recover_root_rot_pos(data)
    elif data.shape[-1] in [709,713,3388]:
        r_pos = data[..., [670,671,672]]
        r_rot_ang = data[..., 669]
        # print('r_rot_ang', r_rot_ang[:40])
        # print("torch.cos(r_rot_ang)", torch.cos(r_rot_ang).shape)
        r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
        # print("r_rot_quat", r_rot_quat.shape)
        r_rot_quat[..., 0] = torch.cos(r_rot_ang)
        r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    else:
        print('data shape not in 669 or 3388', data.shape)
        raise
    
    if data.shape[-1] == 669:
        positions = data[..., 4:51 * 3 + 4]
    if data.shape[-1] in [709,713]:
        positions = torch.cat( [data[..., 4:21 * 3 + 4], data[..., 673 : 673+9], data[..., 21 * 3 + 4:51 * 3 + 4]], dim=-1 )
    elif data.shape[-1] in [3388]:
        positions = torch.cat( [data[..., 4:21 * 3 + 4], data[..., 673 : 673+9], data[..., 21 * 3 + 4:51 * 3 + 4],data[..., 669:2634]], dim=-1 )
    new_positions = positions.clone().view(positions.shape[:-1] + (-1, 3))
    
    '''Add Y-axis rotation to local joints'''
    new_positions = qrot(qinv(r_rot_quat[..., None, :]).expand(new_positions.shape[:-1] + (4,)), new_positions)

    '''Add root XZ to local joints'''
    new_positions_ro = new_positions.clone()
    new_positions_ro[..., 0] = new_positions[..., 0] + r_pos[..., 0:1]
    new_positions_ro[..., 2] = new_positions[..., 2] + r_pos[..., 2:3]

    '''Concate root and joints'''
    positions = torch.cat([r_pos.unsqueeze(-2), new_positions_ro], dim=-2)

    return positions

def get262_fromsmplfea709(data):
    assert data.shape[-1] == 709
    assert len(data.shape) == 2
    T = data.shape[0]
    # data = torch.cat( [data, data[-1:]],dim=0 )
    rotations22 = data[:-1,157:157+21*6].clone().detach().cpu().numpy()
    positions = recover_from_3388mofeas_noinithands(data)
    positions = positions.detach().cpu().numpy()

    positions22 = positions[:,:22]
    joint_vels = positions22[1:] - positions22[:-1]
    joint_vels = joint_vels.reshape(len(joint_vels), -1)

    fid_l = [7,10]
    fid_r = [8,11]
    def foot_detect(positions, thres):
        velfactor, heightfactor = np.array([thres, thres]), np.array([0.12, 0.05])

        feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
        feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
        feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
        feet_l_h = positions[:-1,fid_l,1]
        feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float32)

        feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
        feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
        feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
        feet_r_h = positions[:-1,fid_r,1]
        feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float32)
        return feet_l, feet_r
    #
    feet_l, feet_r = foot_detect(positions22, 0.001)

    positions22 = positions22.reshape(len(positions22), 66)[1:]
    
    motion = np.concatenate([positions22, joint_vels, rotations22, feet_l, feet_r], axis=1)
    return motion

class InterHumanDataset_h5(data.Dataset):
    def __init__(self, opt: dict):
        self.opt = opt
        MODE = opt.MODE
        init_mode = opt.init_mode
        genmode = opt.genmode
        normdir = opt.normdir
        represent = opt.represent
        useh5 = opt.useh5
        donorm = opt.donorm
        self.rawtext_path = opt.text_dir
        self.text_dir = opt.clip_text
        self.split_dir = opt.split

        assert init_mode in ['leader', 'follower']    # init leader or follower
        assert genmode in ['double', 'reactive']
        self.db = None
        self.represent = represent
        self.donorm = donorm
        print('normdir', normdir)

        if normdir is not None:
            self.mean = np.load(os.path.join(normdir, 'Mean.npy'))
            self.std = np.load(os.path.join(normdir, 'Std.npy'))  
        else:
            self.mean = np.load(os.path.join(opt['norm_' + init_mode], 'Mean.npy'))
            self.std = np.load(os.path.join(opt['norm_' + init_mode], 'Std.npy'))

        self.useh5 = useh5
        self.MODE = MODE
        self.task_name = 't2m'

        self.max_cond_length = 1
        self.min_cond_length = 1
        self.max_gt_length = 300
        self.min_gt_length = 15

        self.max_length = self.max_cond_length + self.max_gt_length -1
        self.min_length = self.min_cond_length + self.min_gt_length -1

        # self.motion_rep = opt.MOTION_REP
        # self.data_list = []
        # self.motion_dict = {}

        
        if useh5:
            self.h5path = opt[genmode]['inited' + init_mode + '_h5']
            # self.h5 = h5py.File(self.h5path, 'r', driver='core') 
        else:
            self.motion_dir = opt[genmode]['inited' + init_mode + '_fea']  # initedleader_fea
        self.longmo_dir = opt[genmode]['inited' + init_mode + '_long_fea']  #.initedleader_long_fea
        

        self.ignore_list = []
        try:
            self.ignore_list = open(os.path.join(self.split_dir, "ignore_list.txt"), "r").readlines()
        except Exception as e:
            print(e)
        data_list = []
        if self.MODE == "train":
            try:
                data_list = open(os.path.join(self.split_dir, "train.txt"), "r").readlines()
            except Exception as e:
                print(e)
        elif self.MODE == "val":
            try:
                data_list = open(os.path.join(self.split_dir, "val.txt"), "r").readlines()
            except Exception as e:
                print(e)
        elif self.MODE == "test":
            try:
                data_list = open(os.path.join(self.split_dir, "test.txt"), "r").readlines()
            except Exception as e:
                print(e)
        elif self.MODE == "all":
            try:
                trainlist = open(os.path.join(self.split_dir, "train.txt"), "r").readlines()
                val_list = open(os.path.join(self.split_dir, "val.txt"), "r").readlines()
                test_list = open(os.path.join(self.split_dir, "test.txt"), "r").readlines()
                data_list = trainlist + val_list + test_list
            except Exception as e:
                print(e)
        # data_list = [line.strip() for line in data_list]

        random.shuffle(data_list)

        print('len data_list', len(data_list))
        print('self.h5path', self.h5path)
        # sys.exit(0)
        if self.useh5:
            # h5keys = list()
            with h5py.File(self.h5path, 'r') as db:
                h5keys = list(db.keys())
            data_dict, name_list = self.load_from_h5(h5keys, data_list)
        else:
            file_list = os.listdir(self.motion_dir)
            data_dict, name_list = self.load_from_dir(file_list, data_list)

        self.data_dict = data_dict  # data_dict
        self.name_list = name_list

        print("total dataset is: ", len(self.name_list))



    def load_from_h5(self, h5keys, data_list):
        name_list = []
        data_dict = {}

        num = 0
        for filel in tqdm(h5keys):
            if filel.split(".")[0]+"\n" in self.ignore_list: # or int(motion_name)>1000
                print("ignore: ", filel)
                continue
            if not filel.split('#')[-1].split('person')[0]+'\n' in data_list:
                continue
            if '#' not in filel:
                continue
            with h5py.File(self.h5path, 'r') as db:
                filef = db[filel].attrs["fo"]
                leng = db[filel]["mo"].shape[0] 
                if leng < self.min_gt_length:
                    continue
                # elif leng > self.max_gt_length:
                #     continue

            # num += 1
            # if num == 2000:
                # break

            name = filel.split('#')[-1].split('.')[0]
            
            leader = filel
            follower = filef
          

            text_clipfea = np.load(os.path.join(self.text_dir, name.split('person')[0] + '.npy'),   allow_pickle=True )
            raw_text = [item.replace("\n", "") for item in open(os.path.join(self.rawtext_path, name.split('person')[0] + '.txt'), "r").readlines()]
            name_list.append(filel)

            data_dict[filel] = {'anno': text_clipfea, 'text': raw_text,
                                'leader': leader, 'follower': follower,
                                'le_name': filel,
                                'fo_name': filef,
                                }
            

        return data_dict, name_list

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        data = self.data_dict[self.name_list[idx]]

        le_name = data["le_name"]
        fo_name = data["fo_name"]

        if self.useh5:
            if not os.path.exists(data["leader"]):
                if self.db is None:
                    self.db = h5py.File(self.h5path, 'r')
                    print('load db')
                fullmotion = np.array( self.db[ data["leader"] ]['mo'] )
                dims = fullmotion.shape[-1]//2
                if self.represent == 709:
                    le_fullmotion = fullmotion[1:, :self.represent]
                    fo_fullmotion = fullmotion[1:, dims:dims+self.represent]
                    le_fullmotion = get262_fromsmplfea709(torch.from_numpy(le_fullmotion))
                    fo_fullmotion = get262_fromsmplfea709(torch.from_numpy(fo_fullmotion))
                elif self.represent == 713:
                    le_fullmotion = get_713_from3388(fullmotion[1:, :dims])
                    fo_fullmotion = get_713_from3388(fullmotion[1:, dims:])
            else:
                print('data[leader] not in h5:', data["leader"])
                sys.exit(0)
                le_list = os.listdir(data["leader"])
                fo_list = os.listdir(data["follower"])
                filel_ =  random.choice(le_list)
                filef_ =  random.choice(fo_list)
                le_fullmotion = np.load( os.path.join(self.longmo_dir, filel_) )
                fo_fullmotion = np.load( os.path.join(self.longmo_dir, filef_ ) )
        else:
            sys.exit(0)
            if data["leader"].endswith('.npy'):
                le_fullmotion = np.load(data["leader"])
                fo_fullmotion = np.load(data["follower"])
            else:
                le_list = os.listdir(data["leader"])
                fo_list = os.listdir(data["follower"])
                filel_ =  random.choice(le_list)
                filef_ =  random.choice(fo_list)
                le_fullmotion = np.load( os.path.join(self.longmo_dir, filel_) )
                fo_fullmotion = np.load( os.path.join(self.longmo_dir, filef_ ) )

        # print('---origin', np.concatenate([le_fullmotion[0,0:4],le_fullmotion[0,669:673]],axis=0 ) )
        if self.donorm:
            if self.represent == 3388:
                le_motion = (le_fullmotion - self.mean) / self.std
                fo_motion = (fo_fullmotion - self.mean) / self.std
            elif self.represent == 709:
                # print('709 norm')
                le_motion = (le_fullmotion - self.mean[...,:709]) / self.std[...,:709]
                fo_motion = (fo_fullmotion - self.mean[...,:709]) / self.std[...,:709]
            elif self.represent == 713:
                # print('713 norm')
                le_motion = (le_fullmotion - self.mean) / self.std
                fo_motion = (fo_fullmotion - self.mean) / self.std
            elif self.represent == 669:
                raise
                # le_fullmotion = get_3345from5308(le_fullmotion)[...,:669]
                # fo_fullmotion = get_3345from5308(fo_fullmotion)[...,:669]
                # le_motion = (le_fullmotion - self.mean[...,:669]) / self.std[...,:669]
                # fo_motion = (fo_fullmotion - self.mean[...,:669]) / self.std[...,:669]
            print('-------------------normalized!!')
        else:
            le_motion = le_fullmotion
            fo_motion = fo_fullmotion
            # print('-------------------not normalized!!')

        random_int = np.random.randint(low=0, high=data['anno'].shape[0])
        textfea =  data['anno'][random_int]
        textfea = np.squeeze(textfea, axis=0) 
        rawtext = data['text'][random_int]

        # 当motion长度大于300时使用
        length = le_fullmotion.shape[0]
        if length > self.max_length:

            idx = random.choice(list(range(0, length - self.max_gt_length, 1)))
            gt_length = self.max_gt_length
            le_motion = le_motion[idx:idx + gt_length]
            fo_motion = fo_motion[idx:idx + gt_length]
        else:
            idx = 0
            gt_length = min(length - idx, self.max_gt_length )
            le_motion = le_motion[idx:idx + gt_length]
            fo_motion = fo_motion[idx:idx + gt_length]

        gt_length = len(le_motion)
        if gt_length < self.max_gt_length:
            padding_len = self.max_gt_length - gt_length
            D = fo_motion.shape[1]
            padding_zeros = np.zeros((padding_len, D))
            gt_motion_le = np.concatenate((le_motion, padding_zeros), axis=0).astype(np.float32)
            gt_motion_fo = np.concatenate((fo_motion, padding_zeros), axis=0).astype(np.float32)
        else:
            gt_motion_le = le_motion.astype(np.float32)
            gt_motion_fo = fo_motion.astype(np.float32)
        assert len(gt_motion_le) == self.max_gt_length
        assert len(gt_motion_fo) == self.max_gt_length

 
        return le_name, rawtext, gt_motion_le, gt_motion_fo, gt_length