# 注意这个转的是四元数
# -*- coding: UTF-8 -*-
import argparse
# from tqdm import tqdm
import json
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Quaternion, Vector

## 需要更改的参数 在 L338
## 以下不需要变化
# from: https://github.com/vchoutas/smplx/blob/main/smplx/joint_names.py
smplx_bones= {
    1: 'pelvis',
    2: 'left_hip',
    3: 'right_hip',
    4: 'spine1',
    5: 'left_knee',
    6: 'right_knee',
    7: 'spine2',
    8: 'left_ankle',
    9: 'right_ankle',
    10: 'spine3',
    11: 'left_foot',
    12: 'right_foot',
    13: 'neck',
    14: 'left_collar',
    15: 'right_collar',
    16: 'head',
    17: 'left_shoulder',
    18: 'right_shoulder',
    19: 'left_elbow',
    20: 'right_elbow',
    21: 'left_wrist',
    22: 'right_wrist',
    23: 'jaw',
    24: 'left_eye_smplhf',
    25: 'right_eye_smplhf',
    26: 'left_index1',
    27: 'left_index2',
    28: 'left_index3',
    29: 'left_middle1',
    30: 'left_middle2',
    31: 'left_middle3',
    32: 'left_pinky1',
    33: 'left_pinky2',
    34: 'left_pinky3',
    35: 'left_ring1',
    36: 'left_ring2',
    37: 'left_ring3',
    38: 'left_thumb1',
    39: 'left_thumb2',
    40: 'left_thumb3',
    41: 'right_index1',
    42: 'right_index2',
    43: 'right_index3',
    44: 'right_middle1',
    45: 'right_middle2',
    46: 'right_middle3',
    47: 'right_pinky1',
    48: 'right_pinky2',
    49: 'right_pinky3',
    50: 'right_ring1',
    51: 'right_ring2',
    52: 'right_ring3',
    53: 'right_thumb1',
    54: 'right_thumb2',
    55: 'right_thumb3',
    ## 下面是人脸的部分了, 这个插件没有提供.
    # 56: 'nose',
    # 57: 'right_eye',
    # 58: 'left_eye',
    # 59: 'right_ear',
    # 60: 'left_ear',
    # 61: 'left_big_toe',
    # 62: 'left_small_toe',
    # 63: 'left_heel',
    # 64: 'right_big_toe',
    # 65: 'right_small_toe',
    # 66: 'right_heel',
    # 67: 'left_thumb',
    # 68: 'left_index',
    # 69: 'left_middle',
    # 70: 'left_ring',
    # 71: 'left_pinky',
    # 72: 'right_thumb',
    # 73: 'right_index',
    # 74: 'right_middle',
    # 75: 'right_ring',
    # 76: 'right_pinky',
    # 77: 'right_eye_brow1',
    # 78: 'right_eye_brow2',
    # 79: 'right_eye_brow3',
    # 80: 'right_eye_brow4',
    # 81: 'right_eye_brow5',
    # 82: 'left_eye_brow5',
    # 83: 'left_eye_brow4',
    # 84: 'left_eye_brow3',
    # 85: 'left_eye_brow2',
    # 86: 'left_eye_brow1',
    # 87: 'nose1',
    # 88: 'nose2',
    # 89: 'nose3',
    # 90: 'nose4',
    # 91: 'right_nose_2',
    # 92: 'right_nose_1',
    # 93: 'nose_middle',
    # 94: 'left_nose_1',
    # 95: 'left_nose_2',
    # 96: 'right_eye1',
    # 97: 'right_eye2',
    # 98: 'right_eye3',
    # 99: 'right_eye4',
    # 100: 'right_eye5',
    # 101: 'right_eye6',
    # 102: 'left_eye4',
    # 103: 'left_eye3',
    # 104: 'left_eye2',
    # 105: 'left_eye1',
    # 106: 'left_eye6',
    # 107: 'left_eye5',
    # 108: 'right_mouth_1',
    # 109: 'right_mouth_2',
    # 110: 'right_mouth_3',
    # 111: 'mouth_top',
    # 112: 'left_mouth_3',
    # 113: 'left_mouth_2',
    # 114: 'left_mouth_1',
    # 115: 'left_mouth_5',  
    # 116: 'left_mouth_4',  
    # 117: 'mouth_bottom',
    # 118: 'right_mouth_4',
    # 119: 'right_mouth_5',
    # 120: 'right_lip_1',
    # 121: 'right_lip_2',
    # 122: 'lip_top',
    # 123: 'left_lip_2',
    # 124: 'left_lip_1',
    # 125: 'left_lip_3',
    # 126: 'lip_bottom',
    # 127: 'right_lip_3',
    # Face contour
    # 128: 'right_contour_1',
    # 129: 'right_contour_2',
    # 130: 'right_contour_3',
    # 131: 'right_contour_4',
    # 132: 'right_contour_5',
    # 133: 'right_contour_6',
    # 134: 'right_contour_7',
    # 135: 'right_contour_8',
    # 136: 'contour_middle',
    # 137: 'left_contour_8',
    # 138: 'left_contour_7',
    # 139: 'left_contour_6',
    # 140: 'left_contour_5',
    # 141: 'left_contour_4',
    # 142: 'left_contour_3',
    # 143: 'left_contour_2',
    # 144: 'left_contour_1'
}



smplh_bones= {
    0: "Hips",  # 0
    1: "LeftUpLeg",  # 1
    2: "RightUpLeg",  # 2
    3: "Spine", # 3
    4: "LeftLeg", # 4
    5: "RightLeg", # 5
    6: "Spine1", # 6
    7: "LeftFoot",# 7
    8: "RightFoot",# 8
    9: "Spine2", # 9
    10: "LeftToeBase", # 10
    11: "RightToeBase", # 11
    12: "Neck",  # 12
    13: "LeftShoulder", # 13
    14: "RightShoulder", # 14
    15: "Head", # 15
    16: "LeftArm", # 16
    17: "RightArm",  # 17
    18: "LeftForeArm", # 18
    19: "RightForeArm",  # 19
    20: "LeftHand", # 20
    21: "RightHand", # 21
    22: "LeftHandIndex1",
    23: "LeftHandIndex2",
    24: "LeftHandIndex3",
    25: "LeftHandMiddle1",
    26: "LeftHandMiddle2",
    27: "LeftHandMiddle3",
    28: "LeftHandPinky1",
    29: "LeftHandPinky2",
    30: "LeftHandPinky3",
    31: "LeftHandRing1",
    32: "LeftHandRing2",
    33: "LeftHandRing3",
    34: "LeftHandThumb1",
    35: "LeftHandThumb2",
    36: "LeftHandThumb3",
    37: "RightHandIndex1",
    38: "RightHandIndex2",
    39: "RightHandIndex3",
    40: "RightHandMiddle1",
    41: "RightHandMiddle2",
    42: "RightHandMiddle3",
    43: "RightHandPinky1",
    44: "RightHandPinky2",
    45: "RightHandPinky3",
    46: "RightHandRing1",
    47: "RightHandRing2",
    48: "RightHandRing3",
    49: "RightHandThumb1", 
    50: "RightHandThumb2",
    51: "RightHandThumb3",
}

def get_total_animation_frames(obj_name):
    active_object=bpy.data.objects.get(obj_name)
    print(active_object)
    animation_data=active_object.animation_data
    print(animation_data)
    action=animation_data.action
    print(action)
    print(action.frame_range)
    total_frames=int(action.frame_range[1])
    return total_frames

def remove_startup_cube():
    try:
        bpy.ops.object.mode_set(mode='OBJECT')
    except:pass
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    
    for v in bpy.data.objects.values():
        bpy.data.objects.remove(v)
    for v in bpy.data.materials.values():
        bpy.data.materials.remove(v)
    for v in bpy.data.armatures.values():
        bpy.data.armatures.remove(v)
    for v in bpy.data.meshes.values():
        bpy.data.meshes.remove(v)
    for v in bpy.data.actions.values():
        bpy.data.actions.remove(v)
    for v in bpy.data.sounds.values():
        bpy.data.sounds.remove(v)
    for v in bpy.data.collections.values():
        bpy.data.collections.remove(v)

def rodrigues_from_pose(armature, bone_name):
    # Use quaternion mode for all bone rotations
    if armature.pose.bones[bone_name].rotation_mode != 'QUATERNION':
        armature.pose.bones[bone_name].rotation_mode = 'QUATERNION'

    # if(bone_name=="pelvis"):bone_name="root"
    quat = armature.pose.bones[bone_name].rotation_quaternion
    (axis, angle) = quat.to_axis_angle()
    rodrigues = axis
    rodrigues.normalize()
    rodrigues = rodrigues * angle
    return rodrigues

def qua_rodrigues_from_pose(armature, bone_name):
    # Use quaternion mode for all bone rotations
    if armature.pose.bones[bone_name].rotation_mode != 'QUATERNION':
        armature.pose.bones[bone_name].rotation_mode = 'QUATERNION'

    # if(bone_name=="pelvis"):bone_name="root"
    quat = armature.pose.bones[bone_name].rotation_quaternion
    return quat

def update_corrective_poseshapes(self, context):
    # if self.smplx_corrective_poseshapes:
    #     bpy.ops.object.smplx_set_poseshapes('EXEC_DEFAULT')
    # else:
    #     bpy.ops.object.smplx_reset_poseshapes('EXEC_DEFAULT')
    pass

def set_pose_from_rodrigues(armature, bone_name, rodrigues, rodrigues_reference=None):
    rod = Vector((rodrigues[0], rodrigues[1], rodrigues[2]))
    angle_rad = rod.length
    axis = rod.normalized()

    if armature.pose.bones[bone_name].rotation_mode != 'QUATERNION':
        armature.pose.bones[bone_name].rotation_mode = 'QUATERNION'

    quat = Quaternion(axis, angle_rad)

    if rodrigues_reference is None:
        armature.pose.bones[bone_name].rotation_quaternion = quat
    else:
        # SMPL-X is adding the reference rodrigues rotation to the relaxed hand rodrigues rotation, so we have to do the same here.
        # This means that pose values for relaxed hand model cannot be interpreted as rotations in the local joint coordinate system of the relaxed hand.
        # https://github.com/vchoutas/smplx/blob/f4206853a4746139f61bdcf58571f2cea0cbebad/smplx/body_models.py#L1190
        #   full_pose += self.pose_mean
        rod_reference = Vector((rodrigues_reference[0], rodrigues_reference[1], rodrigues_reference[2]))
        rod_result = rod + rod_reference
        angle_rad_result = rod_result.length
        axis_result = rod_result.normalized()
        quat_result = Quaternion(axis_result, angle_rad_result)
        armature.pose.bones[bone_name].rotation_quaternion = quat_result

        """
        rod_reference = Vector((rodrigues_reference[0], rodrigues_reference[1], rodrigues_reference[2]))
        angle_rad_reference = rod_reference.length
        axis_reference = rod_reference.normalized()
        quat_reference = Quaternion(axis_reference, angle_rad_reference)

        # Rotate first into reference pose and then add the target pose
        armature.pose.bones[bone_name].rotation_quaternion = quat_reference @ quat
        """
    return



def createKeyFrame():
    ob = bpy.data.objects['Armature']
    bpy.context.view_layer.objects.active = ob #相当于鼠标左键选中
    bpy.ops.object.mode_set(mode='POSE') #切换为pose更改模式
    hip=ob.pose.bones['mixamorig1:Hips'] #选中其中一块骨骼，根据自己模型中骨骼的名称，名称可以在Outliner(大纲)视图找到
    # #对骨骼进行旋转
    hip.rotation_mode = 'XZY'			# x y z
    # select axis in ['X','Y','Z']  <--bone local
    axis = 'Y'
    angle = 60
    hip.rotation_euler.rotate_axis(axis, math.radians(angle))
    # hip.rotation_euler[2] = 1.0472

    bpy.ops.object.mode_set(mode='OBJECT')
    #insert a keyframe
    hip.keyframe_insert(data_path="rotation_euler" ,frame=1)


def main():
    ## 需要更改的参数
    output_dir = r'E:\\QUA\\' # 保存的 dir
    fbxdir = r'E:\\smplx_fbx\\' # 输入的 fbx 路径
    objname = "SMPLX-neutral" # "Armature" ## 在这更改导入的骨架的名字, 注意不是网格, 鼠标点一下就知道了, 要切换到 pose

    # 去重, 之前处理过的 name 就不再重复处理
     # 列出指定目录下所有文件和子目录
    all_entries = os.listdir(output_dir)
    processed_files = [entry for entry in all_entries if os.path.isfile(os.path.join(output_dir, entry))]

    flag=0
    for filename in os.listdir(fbxdir):
        #if(flag>=110):
            #break
        flag+=1
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()

        print("joints num", len(smplx_bones))
        name = filename[:-4]

        if name + ".npy" in processed_files:
             continue

        read_fbx_path = os.path.join(fbxdir, name + '.fbx')
        bpy.ops.import_scene.fbx(filepath=read_fbx_path)  # 导入fbx文件            # 读取的
        bpy.context.scene.tool_settings.lock_object_mode = False  # 关闭 Edit->Lock Objects modes 选项
        print("frame_end: {}".format(bpy.context.scene.frame_end))

        # remove_startup_cube()
        read_ob = bpy.data.objects[objname]  # 读取的物体
        bpy.context.view_layer.objects.active = read_ob  # 相当于鼠标左键选中
        bpy.context.scene.tool_settings.lock_object_mode = False
        bpy.ops.object.mode_set(mode='POSE')  # 切换为pose更改模式

        total_frames = get_total_animation_frames(objname) # 

        pose_data=[]
        for frame in (range(0, int(total_frames+1))):
            bpy.context.scene.frame_set(frame)

            root=read_ob.pose.bones["pelvis"]
            pose_data.append(root.location[0]  )
            pose_data.append(root.location[1] )
            pose_data.append(root.location[2] )
            for bone in smplx_bones:
                # debug!!!
                # try:
                #     bpy.ops.object.mode_set(mode='OBJECT')
                # except:
                #     pass
                joint_name = smplx_bones[bone]
                #print(joint_name)
                joint_pose = qua_rodrigues_from_pose(read_ob, joint_name)
                read_obj = read_ob.pose.bones[smplx_bones[bone]]
                pose_data.append(joint_pose[0])
                pose_data.append(joint_pose[1])
                pose_data.append(joint_pose[2])
                pose_data.append(joint_pose[3])
            bpy.ops.object.mode_set(mode='OBJECT')  # 切换为object更改模式  
            # remove_startup_cube()

        pose_data=np.reshape(pose_data,(total_frames+1, len(smplx_bones)*4+3))
        output_path=os.path.join(output_dir, name + '.npy')      
        np.save(output_path, pose_data)
        print(name +" shape:",pose_data.shape)
#        break
#       清除当前窗口所有文件 | delete current to process next
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()


if __name__== '__main__':

    main()

"""
~/software/blender-3.0.1-linux-x64/blender -P read_one_fbx.py /media/lrh/新加卷/dataset/dance/netease/ChoreoMaster_Dataset/motion_fbx/0071@000.fbx /home/lrh/Documents/DANCE/fbx_samples/ImportScene/json/0071@000.json /home/lrh/Documents/DANCE/my_pose2avatar/data/blender_json/0071@000_blend.json
/Applications/Blender.app/Contents/MacOS/Blender -P read_one_fbx.py /Users/lironghui/Documents/Research/Dataset/ChoreoMaster_Dataset/motion_fbx/0071@000.fbx /Users/lironghui/Documents/Research/Choreography/fbx_samples/ImportScene/json/0071@000.json /Users/lironghui/Documents/Research/Dataset/ChoreoMaster_Dataset/blender_json/0071@000_blend.json

"""