import sys

import numpy as np
import torch

sys.path.append('/data1/hzy/HumanMotion/LDmofea266')
from common.quaternion import qinv, qrot


def recover_from_ric264(data,joints_num=22):
      r_pos_y=data[:,0]
      rot_vel = data[..., 2]
      r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
      '''Get Y-axis rotation from rotation velocity'''
      r_rot_ang[..., 1:] = rot_vel[..., :-1]
      r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

      r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
      r_rot_quat[..., 0] = torch.cos(r_rot_ang)
      r_rot_quat[..., 2] = torch.sin(r_rot_ang)

      r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
      r_pos[..., 1:, [0, 2]] = data[..., :-1, 3:5]
      '''Add Y-axis rotation to root position'''
      r_pos = qrot(qinv(r_rot_quat), r_pos)

      r_pos = torch.cumsum(r_pos, dim=-2)

      r_pos[..., 1] = data[..., 0]

      positions = data[..., 5:(joints_num - 1) * 3 + 5]
      positions = positions.view(positions.shape[:-1] + (-1, 3))

      '''Add Y-axis rotation to local joints'''
      positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

      '''Add root XZ to joints'''
      positions[..., 0] += r_pos[..., 0:1]
      positions[..., 2] += r_pos[..., 2:3]
      '''Concate root and joints'''
      positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

      return positions