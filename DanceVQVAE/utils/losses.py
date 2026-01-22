# import torch
# import numpy as np 
# import torch.nn as nn
# from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_axis_angle
# import sys,os
# # print('(__file__)', __file__)
# # print('os.path.abspath(__file__)', os.path.abspath(__file__))
# # print('dirname dirname', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# # sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# # print('sys.path', sys.path)
# from .motion_process import recover_from_ric264

# def get_joints(data):
#     return data[..., 5 : (22 - 1) * 3 + 5]#对应局部位置

# def compute_rotation_loss(motion_pred, motion_gt, rot_indices=[1, 2, *range(68, 194)]):
#     batch_size, seq_len, _ = motion_pred.shape
    
#     # 提取特征
#     pred_rot = motion_pred[:, :, rot_indices]  
#     gt_rot = motion_gt[:, :, rot_indices]
    
#     # 提取根部旋转角度Y和旋转速度（标量）
#     root_rot_angle_pred = pred_rot[:, :, 0]
#     root_rot_angle_gt = gt_rot[:, :, 0]
#     root_rot_vel_pred = pred_rot[:, :, 1]
#     root_rot_vel_gt = gt_rot[:, :, 1]
    
#     # 提取局部旋转 (6D) 并转换为旋转矩阵
#     local_rot_pred = pred_rot[:, :, 2:128].view(batch_size, seq_len, 21, 6)
#     local_rot_gt = gt_rot[:, :, 2:128].view(batch_size, seq_len, 21, 6)
#     local_rot_matrix_pred = rotation_6d_to_matrix(local_rot_pred)  # (batch_size, seq_len, 21, 3, 3)
#     local_rot_matrix_gt = rotation_6d_to_matrix(local_rot_gt)
    
#     # rot_loss
#     local_rot_loss = torch.mean((local_rot_matrix_pred - local_rot_matrix_gt) ** 2)
#     root_rot_angle_loss = torch.mean((root_rot_angle_pred - root_rot_angle_gt) ** 2)
#     root_rot_vel_loss = torch.mean((root_rot_vel_pred - root_rot_vel_gt) ** 2)
#     rot_loss = local_rot_loss + root_rot_angle_loss + root_rot_vel_loss
    
#     # rot_acc_loss
#     pred_rot_diff = pred_rot[:, 1:, :] - pred_rot[:, :-1, :]
#     gt_rot_diff = gt_rot[:, 1:, :] - gt_rot[:, :-1, :]
    
#     # 局部旋转差值
#     local_rot_pred_t1 = pred_rot[:, 1:, 2:128].view(batch_size, seq_len-1, 21, 6)
#     local_rot_pred_t0 = pred_rot[:, :-1, 2:128].view(batch_size, seq_len-1, 21, 6)
#     local_rot_gt_t1 = gt_rot[:, 1:, 2:128].view(batch_size, seq_len-1, 21, 6)
#     local_rot_gt_t0 = gt_rot[:, :-1, 2:128].view(batch_size, seq_len-1, 21, 6)
#     local_rot_matrix_pred_t1 = rotation_6d_to_matrix(local_rot_pred_t1)
#     local_rot_matrix_pred_t0 = rotation_6d_to_matrix(local_rot_pred_t0)
#     local_rot_matrix_gt_t1 = rotation_6d_to_matrix(local_rot_gt_t1)
#     local_rot_matrix_gt_t0 = rotation_6d_to_matrix(local_rot_gt_t0)
    
#     local_rot_diff_pred = torch.matmul(local_rot_matrix_pred_t1, local_rot_matrix_pred_t0.transpose(-1, -2))
#     local_rot_diff_gt = torch.matmul(local_rot_matrix_gt_t1, local_rot_matrix_gt_t0.transpose(-1, -2))
#     local_rot_diff_axis_angle_pred = matrix_to_axis_angle(local_rot_diff_pred)
#     local_rot_diff_axis_angle_gt = matrix_to_axis_angle(local_rot_diff_gt)
#     local_rot_acc_loss = torch.mean((local_rot_diff_axis_angle_pred - local_rot_diff_axis_angle_gt) ** 2)
    
#     # 根部旋转角度和速度差值
#     root_rot_angle_diff_pred = pred_rot_diff[:, :, 0]
#     root_rot_angle_diff_gt = gt_rot_diff[:, :, 0]
#     root_rot_angle_acc_loss = torch.mean((root_rot_angle_diff_pred - root_rot_angle_diff_gt) ** 2)
    
#     root_rot_vel_diff_pred = pred_rot_diff[:, :, 1]
#     root_rot_vel_diff_gt = gt_rot_diff[:, :, 1]
#     root_rot_vel_acc_loss = torch.mean((root_rot_vel_diff_pred - root_rot_vel_diff_gt) ** 2)
    
#     # 总差值损失
#     rot_acc_loss = local_rot_acc_loss + root_rot_angle_acc_loss + root_rot_vel_acc_loss
    
#     return rot_loss, rot_acc_loss


# class MotionNormalizerTorch():
#     def __init__(self, meanfile, stdfile):
#         mean = np.load(meanfile)
#         std = np.load(stdfile)
#         self.motion_mean = torch.from_numpy(mean).float()
#         self.motion_std = torch.from_numpy(std).float()

#     def forward(self, x):
#         device = x.device
#         x = x.clone()
#         x = (x - self.motion_mean.to(device)) / self.motion_std.to(device)
#         return x

#     def backward(self, x, global_rt=False):
#         device = x.device
#         x = x.clone()
#         x = x * self.motion_std.to(device) + self.motion_mean.to(device)
#         return x

# class ReConsLoss(nn.Module):
#     def __init__(self, args, recons_loss, nb_joints, w_motion=1, w_joints=0.05, w_vel=0.06, w_acc=0.06, w_foot=300):
#         super(ReConsLoss, self).__init__()
        
#         if recons_loss == 'l1': 
#             self.Loss = torch.nn.L1Loss()
#             self.Loss_ = torch.nn.L1Loss(reduction='none')
#         elif recons_loss == 'l2' : 
#             self.Loss = torch.nn.MSELoss()
#             self.Loss_ = torch.nn.MSELoss(reduction='none')
#         elif recons_loss == 'l1_smooth' : 
#             self.Loss = torch.nn.SmoothL1Loss()
#             self.Loss_ = torch.nn.SmoothL1Loss(reduction='none')
            
#         self.w_motion = w_motion
#         self.w_joints = w_joints
#         self.w_vel = w_vel
#         self.w_acc = w_acc
#         self.w_foot = w_foot
#         self.nb_joints = 22
#         self.motion_dim = 264
#         if (args.vel_decoder):
#             self.xz_vel = 1
#             self.xz_acc = 1
#             self.rot_vel = 0
#             self.rot_acc = 0
#         elif (args.rot_decoder):
#             self.rot_vel = 1
#             self.rot_acc = 1
#             self.xz_vel = 0
#             self.xz_acc = 0
#         elif not (args.vel_decoder) and not (args.rot_decoder):
#             self.xz_vel = 0
#             self.xz_acc = 0
#             self.rot_vel = 0
#             self.rot_acc = 0
        
#         self.normalizer = MotionNormalizerTorch(os.path.join(args.data_root, 'Mean.npy'), os.path.join(args.data_root, 'Std.npy'))
        
#     def forward(self, motion_pred, motion_gt) : 
#         loss = self.Loss(motion_pred, motion_gt)
#         return loss
    
#     def forward_joints(self, motion_pred, motion_gt) : 
#         loss = self.Loss(motion_pred[..., 5 : (self.nb_joints - 1) * 3 + 5], motion_gt[..., 5 : (self.nb_joints - 1) * 3 + 5])
#         return loss

#     def foot_detect(self, feet_vel, feet_h, thres=0.001):
#         velfactor = torch.tensor([thres, thres, thres, thres], device=feet_vel.device)
#         heightfactor = torch.tensor([0.12, 0.05, 0.12, 0.05], device=feet_vel.device)
        
#         feet_x = (feet_vel[..., 0]) ** 2
#         feet_y = (feet_vel[..., 1]) ** 2
#         feet_z = (feet_vel[..., 2]) ** 2
        
#         contact = (((feet_x + feet_y + feet_z) < velfactor) & (feet_h < heightfactor)).float()
#         return contact
    
#     def mix_masked_mse(self, prediction, target, mask, batch_mask, contact_mask=None):
#         # contact_mask shape is [b,t,4]， mask shape is [b,t], batch_mask shape is [b]
#         loss = self.Loss_(prediction, target).mean(dim=-1, keepdim=True)  # [b,t,4,1]
#         if contact_mask is not None:
#             loss = (loss[..., 0] * contact_mask).sum(dim=-1, keepdim=False) / (contact_mask.sum(dim=-1, keepdim=False) + 1.e-7)
#         loss = (loss * mask).sum(dim=(-1, -2)) / (mask.sum(dim=(-1, -2)) + 1.e-7)  # [b]
#         loss = (loss * batch_mask).sum(dim=0) / (batch_mask.sum(dim=0) + 1.e-7)
        
#         return loss

#     def forward_all(self, motion_pred, motion_gt):
#         motion_pred_un = motion_pred.clone()
#         pred_joints = get_joints(motion_pred)  # [batch_size, seq_len, 63]
#         gt_joints = get_joints(motion_gt)
        
#         motion_loss = self.Loss(motion_pred, motion_gt)
#         joints_loss = self.Loss(pred_joints, gt_joints)
        
#         pred_vel = pred_joints[:, 1:] - pred_joints[:, :-1]  # [batch_size, seq_len-1, 63]
#         gt_vel = gt_joints[:, 1:] - gt_joints[:, :-1]
#         vel_loss = self.Loss(pred_vel, gt_vel)
        
#         pred_acc = pred_vel[:, 1:] - pred_vel[:, :-1]  # [batch_size, seq_len-2, 63]
#         gt_acc = gt_vel[:, 1:] - gt_vel[:, :-1]
#         acc_loss = self.Loss(pred_acc, gt_acc)
        
#         xz_vel_loss = self.Loss(motion_pred[:, :, 3:5], motion_gt[:, :, 3:5])
#         pred_xz_vel_diff = motion_pred[:, 1:, 3:5] - motion_pred[:, :-1, 3:5]  # [batch_size, seq_len-1, 2]
#         gt_xz_vel_diff = motion_gt[:, 1:, 3:5] - motion_gt[:, :-1, 3:5]
#         xz_acc_loss = self.Loss(pred_xz_vel_diff, gt_xz_vel_diff)
        
#         rot_loss, rot_acc_loss = compute_rotation_loss(motion_pred, motion_gt)
        
        
#         # New contact loss
#         '''新加的足部接触损失函数'''
#         foot_idx = [7,8,10,11]
#         motion_pred_un = self.normalizer.backward(motion_pred_un)
#         global_joints = recover_from_ric264(motion_pred_un, 22)
#         B,T,J,C = global_joints.shape
#         assert len(global_joints.shape) == 4
#         assert global_joints.shape[2] == 22
#         assert global_joints.shape[3] == 3
#         pred_foot_vel = global_joints[:, 1:, foot_idx, :] - global_joints[:, :-1, foot_idx, :]  #[batch_size, seq_len-1, 4, 3]
#         pred_foot_h = global_joints[:, :-1, foot_idx, 1]  # [batch_size, seq_len-1, 4]
#         contact = self.foot_detect(pred_foot_vel, pred_foot_h, 0.001)
#         temporal_mask = torch.ones(B, T, device=motion_pred.device)
#         batch_mask = torch.ones(B, device=motion_pred.device)
#         foot_loss = self.mix_masked_mse(pred_foot_vel, torch.zeros_like(pred_foot_vel), temporal_mask[:, :-1],
#                                                             batch_mask,
#                                                             contact)
        
        
#         total_loss = (
#             self.w_motion * motion_loss +
#             self.w_joints * joints_loss +
#             self.w_vel * vel_loss +
#             self.w_acc * acc_loss +
#             self.w_foot * foot_loss +
#             self.xz_vel * xz_vel_loss +
#             self.xz_acc * xz_acc_loss +
#             self.rot_vel * rot_loss +
#             self.rot_acc * rot_acc_loss
#         )
        

#         losses = {
#             'motion': motion_loss,
#             'joints': joints_loss,
#             'vel': vel_loss,
#             'acc': acc_loss,
#             'foot': foot_loss,
#             'total': total_loss
#         }
#         # print('losses', losses) # [batch_size]
        
#         return total_loss, motion_loss, xz_vel_loss, xz_acc_loss, rot_loss, rot_acc_loss, foot_loss
    


# # if __name__ == "__main__":
# #     # 测试代码 可以删掉了
# #     pred_mofea = torch.randn(2, 10, 264) # B,T,J,3
# #     pred_joints = torch.randn(2, 10, 22, 3) # B,T,J,3
# #     B,T,J,C = pred_joints.shape
# #     assert len(pred_joints.shape) == 4
# #     assert pred_joints.shape[2] == 22
# #     assert pred_joints.shape[3] == 3
# #     Loss = torch.nn.MSELoss()
# #     def foot_detect(feet_vel, feet_h, thres=0.001):
# #         velfactor = torch.tensor([thres, thres, thres, thres], device=feet_vel.device)
# #         heightfactor = torch.tensor([0.12, 0.05, 0.12, 0.05], device=feet_vel.device)
        
# #         feet_x = (feet_vel[..., 0]) ** 2
# #         feet_y = (feet_vel[..., 1]) ** 2
# #         feet_z = (feet_vel[..., 2]) ** 2
        
# #         contact = (((feet_x + feet_y + feet_z) < velfactor) & (feet_h < heightfactor)).float()
# #         return contact
    
# #     def mix_masked_mse(prediction, target, mask, batch_mask, contact_mask=None):
# #         # contact_mask shape is [b,t,4]， mask shape is [b,t], batch_mask shape is [b]
# #         loss = self.Loss_(prediction, target).mean(dim=-1, keepdim=True)  # [b,t,4,1]
# #         if contact_mask is not None:
# #             loss = (loss[..., 0] * contact_mask).sum(dim=-1, keepdim=False) / (contact_mask.sum(dim=-1, keepdim=False) + 1.e-7)
# #         # print('loss', loss.shape) 
        
# #         # mask = mask.unsqueeze(-1)
# #         # print('mask', mask.shape)
# #         loss = (loss * mask).sum(dim=(-1, -2)) / (mask.sum(dim=(-1, -2)) + 1.e-7)  # [b]
# #         loss = (loss * batch_mask).sum(dim=0) / (batch_mask.sum(dim=0) + 1.e-7)

# #         return loss

# #     # import ipdb; ipdb.set_trace()
# #     foot_idx = [7, 8, 10, 11]
    
# #     pred_foot_vel = pred_joints[:, 1:, foot_idx, :] - pred_joints[:, :-1, foot_idx, :]  # [batch_size, seq_len-1, 4, 3]
# #     pred_foot_h = pred_joints[:, :-1, foot_idx, 1]  # [batch_size, seq_len-1, 4]
# #     contact = foot_detect(pred_foot_vel, pred_foot_h, 0.001)
    
# #     temporal_mask = torch.ones(B, T, device=pred_foot_vel.device)
# #     batch_mask = torch.ones(B, device=pred_foot_vel.device)
# #     foot_loss = mix_masked_mse(pred_foot_vel, torch.zeros_like(pred_foot_vel), temporal_mask[:, :-1],
# #                                                           batch_mask,
# #                                                           contact) * self.weights["FC"]
    
    
    
#     # print("foot_loss Loss:", foot_loss.item())


import sys

import torch
import torch.nn as nn


def get_joints(data):
    return data[..., 5 : (22 - 1) * 3 + 5]


from pytorch3d.transforms import matrix_to_axis_angle, rotation_6d_to_matrix


def compute_rotation_loss(motion_pred, motion_gt, rot_indices=[1, 2, *range(68, 194)]):
    batch_size, seq_len, _ = motion_pred.shape

    # 提取旋转特征
    pred_rot = motion_pred[:, :, rot_indices]  # 形状: (batch_size, seq_len, 128)
    gt_rot = motion_gt[:, :, rot_indices]

    # 提取根部旋转角度Y和旋转速度（标量）
    root_rot_angle_pred = pred_rot[:, :, 0]  # (batch_size, seq_len)
    root_rot_angle_gt = gt_rot[:, :, 0]
    root_rot_vel_pred = pred_rot[:, :, 1]  # (batch_size, seq_len)
    root_rot_vel_gt = gt_rot[:, :, 1]

    # 提取局部旋转 (6D)
    local_rot_pred = pred_rot[:, :, 2:128].view(batch_size, seq_len, 21, 6)  # (batch_size, seq_len, 21, 6)
    local_rot_gt = gt_rot[:, :, 2:128].view(batch_size, seq_len, 21, 6)

    # rot_loss：直接比较 6D 表示的 MSE
    local_rot_loss = torch.mean((local_rot_pred - local_rot_gt) ** 2)
    root_rot_angle_loss = torch.mean((root_rot_angle_pred - root_rot_angle_gt) ** 2)
    root_rot_vel_loss = torch.mean((root_rot_vel_pred - root_rot_vel_gt) ** 2)
    rot_loss = local_rot_loss + root_rot_angle_loss + root_rot_vel_loss

    # rot_acc_loss：比较 6D 表示的时间差值
    pred_rot_diff = pred_rot[:, 1:, :] - pred_rot[:, :-1, :]  # (batch_size, seq_len-1, 128)
    gt_rot_diff = gt_rot[:, 1:, :] - gt_rot[:, :-1, :]  # (batch_size, seq_len-1, 128)

    # 局部旋转差值
    local_rot_pred_diff = pred_rot_diff[:, :, 2:128].view(batch_size, seq_len-1, 21, 6)  # (batch_size, seq_len-1, 21, 6)
    local_rot_gt_diff = gt_rot_diff[:, :, 2:128].view(batch_size, seq_len-1, 21, 6)
    local_rot_acc_loss = torch.mean((local_rot_pred_diff - local_rot_gt_diff) ** 2)

    # 根部旋转角度和速度差值
    root_rot_angle_diff_pred = pred_rot_diff[:, :, 0]  # (batch_size, seq_len-1)
    root_rot_angle_diff_gt = gt_rot_diff[:, :, 0]
    root_rot_angle_acc_loss = torch.mean((root_rot_angle_diff_pred - root_rot_angle_diff_gt) ** 2)

    root_rot_vel_diff_pred = pred_rot_diff[:, :, 1]  # (batch_size, seq_len-1)
    root_rot_vel_diff_gt = gt_rot_diff[:, :, 1]
    root_rot_vel_acc_loss = torch.mean((root_rot_vel_diff_pred - root_rot_vel_diff_gt) ** 2)

    # 总差值损失
    rot_acc_loss = local_rot_acc_loss + root_rot_angle_acc_loss + root_rot_vel_acc_loss

    return rot_loss, rot_acc_loss
# def compute_rotation_loss(motion_pred, motion_gt, rot_indices=[1, 2, *range(68, 194)]):
#     batch_size, seq_len, _ = motion_pred.shape
    
#     # 提取特征
#     pred_rot = motion_pred[:, :, rot_indices]  # 形状: (batch_size, seq_len, 128)
#     gt_rot = motion_gt[:, :, rot_indices]
    
#     # 提取根部旋转角度Y和旋转速度（标量）
#     root_rot_angle_pred = pred_rot[:, :, 0]
#     root_rot_angle_gt = gt_rot[:, :, 0]
#     root_rot_vel_pred = pred_rot[:, :, 1]
#     root_rot_vel_gt = gt_rot[:, :, 1]
    
#     # 提取局部旋转 (6D) 并转换为旋转矩阵
#     local_rot_pred = pred_rot[:, :, 2:128].view(batch_size, seq_len, 21, 6)
#     local_rot_gt = gt_rot[:, :, 2:128].view(batch_size, seq_len, 21, 6)
#     local_rot_matrix_pred = rotation_6d_to_matrix(local_rot_pred)  # (batch_size, seq_len, 21, 3, 3)
#     local_rot_matrix_gt = rotation_6d_to_matrix(local_rot_gt)
    
#     # rot_loss
#     local_rot_loss = torch.mean((local_rot_matrix_pred - local_rot_matrix_gt) ** 2)
#     root_rot_angle_loss = torch.mean((root_rot_angle_pred - root_rot_angle_gt) ** 2)
#     root_rot_vel_loss = torch.mean((root_rot_vel_pred - root_rot_vel_gt) ** 2)
#     rot_loss = local_rot_loss + root_rot_angle_loss + root_rot_vel_loss
    
#     # rot_acc_loss
#     pred_rot_diff = pred_rot[:, 1:, :] - pred_rot[:, :-1, :]
#     gt_rot_diff = gt_rot[:, 1:, :] - gt_rot[:, :-1, :]
    
#     # 局部旋转差值
#     local_rot_pred_t1 = pred_rot[:, 1:, 2:128].view(batch_size, seq_len-1, 21, 6)
#     local_rot_pred_t0 = pred_rot[:, :-1, 2:128].view(batch_size, seq_len-1, 21, 6)
#     local_rot_gt_t1 = gt_rot[:, 1:, 2:128].view(batch_size, seq_len-1, 21, 6)
#     local_rot_gt_t0 = gt_rot[:, :-1, 2:128].view(batch_size, seq_len-1, 21, 6)
#     local_rot_matrix_pred_t1 = rotation_6d_to_matrix(local_rot_pred_t1)
#     local_rot_matrix_pred_t0 = rotation_6d_to_matrix(local_rot_pred_t0)
#     local_rot_matrix_gt_t1 = rotation_6d_to_matrix(local_rot_gt_t1)
#     local_rot_matrix_gt_t0 = rotation_6d_to_matrix(local_rot_gt_t0)
    
#     local_rot_diff_pred = torch.matmul(local_rot_matrix_pred_t1, local_rot_matrix_pred_t0.transpose(-1, -2))
#     local_rot_diff_gt = torch.matmul(local_rot_matrix_gt_t1, local_rot_matrix_gt_t0.transpose(-1, -2))
#     local_rot_diff_axis_angle_pred = matrix_to_axis_angle(local_rot_diff_pred)
#     local_rot_diff_axis_angle_gt = matrix_to_axis_angle(local_rot_diff_gt)
#     local_rot_acc_loss = torch.mean((local_rot_diff_axis_angle_pred - local_rot_diff_axis_angle_gt) ** 2)
    
#     # 根部旋转角度和速度差值
#     root_rot_angle_diff_pred = pred_rot_diff[:, :, 0]
#     root_rot_angle_diff_gt = gt_rot_diff[:, :, 0]
#     root_rot_angle_acc_loss = torch.mean((root_rot_angle_diff_pred - root_rot_angle_diff_gt) ** 2)
    
#     root_rot_vel_diff_pred = pred_rot_diff[:, :, 1]
#     root_rot_vel_diff_gt = gt_rot_diff[:, :, 1]
#     root_rot_vel_acc_loss = torch.mean((root_rot_vel_diff_pred - root_rot_vel_diff_gt) ** 2)
    
#     # 总差值损失
#     rot_acc_loss = local_rot_acc_loss + root_rot_angle_acc_loss + root_rot_vel_acc_loss
    
#     return rot_loss, rot_acc_loss
class ReConsLoss(nn.Module):
    def __init__(self,args, recons_loss, nb_joints, w_motion=1, w_joints=0.05, w_vel=0.06, w_acc=0.06, w_foot=0.04):
        super(ReConsLoss, self).__init__()
        
        if recons_loss == 'l1': 
            self.Loss = torch.nn.L1Loss()
        elif recons_loss == 'l2' : 
            self.Loss = torch.nn.MSELoss()
        elif recons_loss == 'l1_smooth' : 
            self.Loss = torch.nn.SmoothL1Loss()
        
        # 4 global motion associated to root
        # 12 local motion (3 local xyz, 3 vel xyz, 6 rot6d)
        # 3 global vel xyz
        # 4 foot contact
        self.w_motion=w_motion
        self.w_joints=w_joints
        self.w_vel=w_vel
        self.w_acc=w_acc
        self.w_foot=w_foot
        # self.w_joints=0.02
        # self.w_vel=0.02
        # self.w_acc=0.02
        # self.w_foot=0.02
        self.nb_joints = 22
        self.motion_dim = 264
        if (args.vel_decoder):
            self.xz_vel=1
            self.xz_acc=1
            self.rot_vel=0
            self.rot_acc=0
        elif (args.rot_decoder):
            self.rot_vel=1
            self.rot_acc=1
            self.xz_vel=0
            self.xz_acc=0
            self.w_motion=0
            self.w_joints=0
            self.w_vel=0
            self.w_acc=0
            self.w_foot=0
        elif not (args.vel_decoder) and not (args.rot_decoder):
            self.xz_vel=0
            self.xz_acc=0
            self.rot_vel=0
            self.rot_acc=0
        
    def forward(self, motion_pred, motion_gt) : 
        loss = self.Loss(motion_pred, motion_gt)
        return loss
    
    def forward_joints(self, motion_pred, motion_gt) : 
        loss = self.Loss(motion_pred[..., 5 : (self.nb_joints - 1) * 3 + 5], motion_gt[..., 5 : (self.nb_joints - 1) * 3 + 5])
        return loss


    def forward_all(self, motion_pred, motion_gt):
        pred_joints = get_joints(motion_pred)  # [batch_size, seq_len, 63]
        gt_joints = get_joints(motion_gt)
        
        motion_loss = self.Loss(motion_pred, motion_gt)
        joints_loss = self.Loss(pred_joints, gt_joints)
        
        pred_vel = pred_joints[:, 1:] - pred_joints[:, :-1]  # [batch_size, seq_len-1, 63]
        gt_vel = gt_joints[:, 1:] - gt_joints[:, :-1]
        vel_loss = self.Loss(pred_vel, gt_vel)
        
        pred_acc = pred_vel[:, 1:] - pred_vel[:, :-1]  # [batch_size, seq_len-2, 63]
        gt_acc = gt_vel[:, 1:] - gt_vel[:, :-1]
        acc_loss = self.Loss(pred_acc, gt_acc)
        
        foot_mask = torch.zeros(self.nb_joints-1, dtype=torch.bool, device=motion_pred.device)
        foot_idx = [7, 8, 10, 11]
        foot_mask[[idx - 1 for idx in foot_idx]] = True
        foot_mask = foot_mask.repeat(3)
        pred_foot_joints = pred_joints[..., foot_mask]  # [batch_size, seq_len, 12]
        gt_foot_joints = gt_joints[..., foot_mask]
        pred_foot_vel = pred_foot_joints[:, 1:] - pred_foot_joints[:, :-1]  # [batch_size, seq_len-1, 12]
        
        pred_contact = motion_pred[..., -4:]
        pred_contact = pred_contact[:, :-1]
        static_idx = pred_contact > 0.95
        static_idx = static_idx.repeat_interleave(3, dim=-1)
        xz_vel_loss = self.Loss(motion_pred[:, :, 3:5], motion_gt[:, :, 3:5])
        pred_xz_vel_diff = motion_pred[:, 1:, 3:5] - motion_pred[:, :-1, 3:5]  # [batch_size, seq_len-1, 2]
        gt_xz_vel_diff = motion_gt[:, 1:, 3:5] - motion_gt[:, :-1, 3:5]
        # pred_xz_acc = pred_xz_vel_diff[:, 1:] - pred_xz_vel_diff[:, :-1]  # [batch_size, seq_len-2, 2]
        # gt_xz_acc = gt_xz_vel_diff[:, 1:] - gt_xz_vel_diff[:, :-1]
        xz_acc_loss = self.Loss(pred_xz_vel_diff, gt_xz_vel_diff)
        
        # rot_indices = [1, 2, *range(68, 194)]  # 总计 1 + 1 + 63 = 65维
        # rot_loss = self.Loss(motion_pred[:, :,rot_indices], motion_gt[:, :, rot_indices])
        # pred_rot_diff = motion_pred[:, 1:, rot_indices] - motion_pred[:, :-1, rot_indices]  # [batch_size, seq_len-1, 2]
        # gt_rot_diff = motion_gt[:, 1:, rot_indices] - motion_gt[:, :-1, rot_indices]
        # rot_acc_loss = self.Loss(pred_rot_diff, gt_rot_diff)
        rot_loss,rot_acc_loss=compute_rotation_loss(motion_pred,motion_gt)
        
        pred_foot_vel_static = pred_foot_vel.clone()
        pred_foot_vel_static[~static_idx] = 0
        target_foot_vel = torch.zeros_like(pred_foot_vel)
        foot_loss = self.Loss(pred_foot_vel_static, target_foot_vel)
        
        total_loss = (
            self.w_motion * motion_loss +
            self.w_joints * joints_loss +
            self.w_vel * vel_loss +
            self.w_acc * acc_loss +
            self.w_foot * foot_loss+
            self.xz_vel*xz_vel_loss+self.xz_acc*xz_acc_loss
            +self.rot_vel*rot_loss +self.rot_acc*rot_acc_loss

        )
        
        losses = {
            'motion': motion_loss,
            'joints': joints_loss,
            'vel': vel_loss,
            'acc': acc_loss,
            'foot': foot_loss,
            'total': total_loss
        }
        
        return total_loss,motion_loss,xz_vel_loss,xz_acc_loss,rot_loss,rot_acc_loss,foot_loss
    
    