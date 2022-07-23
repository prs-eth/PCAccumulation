from __future__ import print_function
import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial.transform import Rotation as R


def euler2mat(angle):
    """
    convert euler angles [B, 3] to rotation matrix, reference: https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/inverse_warp.py
    :param angle: rx, ry, rz [B, 3]
    :return: rotation matrix [B, 3, 3]
    """
    B = angle.size(0)
    x, y, z = angle[:,0], angle[:,1], angle[:,2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach()*0
    ones = zeros.detach()+1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz,  cosz, zeros,
                        zeros, zeros,  ones], dim=1).reshape(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros,  siny,
                        zeros,  ones, zeros,
                        -siny, zeros,  cosy], dim=1).reshape(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros,  cosx, -sinx,
                        zeros,  sinx,  cosx], dim=1).reshape(B, 3, 3)

    rotMat = xmat @ ymat @ zmat
    return rotMat


def quat2mat(quat):
    """
    convert quaternion to rotation matrix ([x, y, z, w] to follow scipy
    :param quat: four quaternion of rotation
    :return: rotation matrix [B, 3, 3]
    """
    # norm_quat = torch.cat([quat[:, :1].detach()*0 + 1, quat], dim=1)
    # norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    # w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat


def pose_vec2mat(vec, rotation_mode='euler'):
    """
    :param vec: tx, ty, tz, rx, ry, rz [B, 6]
    :param rotation_mode: 'euler' or 'quat'
    :return: rotation matrix [B, 3, 3] and translation matrix [B, 3, 1]
    """
    translation_mat = vec[:, :3].unsqueeze(-1)  # [B, 3, 1]
    rot = vec[:, 3:]
    if rotation_mode == 'euler':
        rot_mat = euler2mat(rot)  # [B, 3, 3]
    elif rotation_mode == 'quat':
        rot_mat = quat2mat(rot)  # [B, 3, 3]
    return rot_mat, translation_mat



def mat2euler(rot_mat, seq='xyz'):
    """
    convert rotation matrix to euler angle
    :param rot_mat: rotation matrix rx*ry*rz [B, 3, 3]
    :param seq: seq is xyz(rotate along z first) or zyx
    :return: three angles, x, y, z
    """
    r11 = rot_mat[:, 0, 0]
    r12 = rot_mat[:, 0, 1]
    r13 = rot_mat[:, 0, 2]
    r21 = rot_mat[:, 1, 0]
    r22 = rot_mat[:, 1, 1]
    r23 = rot_mat[:, 1, 2]
    r31 = rot_mat[:, 2, 0]
    r32 = rot_mat[:, 2, 1]
    r33 = rot_mat[:, 2, 2]
    if seq == 'xyz':
        z = torch.atan2(-r12, r11)
        y = torch.asin(r13)
        x = torch.atan2(-r23, r33)
    else:
        y = torch.asin(-r31)
        x = torch.atan2(r32, r33)
        z = torch.atan2(r21, r11)
    return torch.stack((x, y, z), dim=1)


def mat2quat(rot_mat, seq='xyz'):
    """
    covert rotation matrix to quaternion
    :param rot_mat: rotation matrix [B, 3, 3]
    :param seq: 'xyz'(rotate along z first) or 'zyx'
    :return: quaternion of the first three entries
    """
    pass


def mat2pose_vec(rot_mat, translation_mat, rotation_mode='euler', seq='xyz'):
    """
    Convert rotation matrix and translation matrix to 6DoF
    :param rot_mat: [B, 3, 3]
    :param translation_mat: [B, 3, 1]
    :param rotation_mode: 'euler' or quat'
    :param seq: 'xyz'(rotate along z first) or 'zyx'
    :return: pose_vec  - tx, ty, tz, rx, ry, rz [B, 6]
    """
    pass


def transform_point_cloud(point_cloud, rotation, translation):
    """
    :param point_cloud: [B, 3, N]
    :param rotation: Euler angel [B, 3]
    :param translation: Translation [B, 3]
    :return:
    """
    if len(rotation.size()) == 2:
        rot_mat = quat2mat(rotation)
    else:
        rot_mat = rotation
    return torch.matmul(rot_mat, point_cloud) + translation.unsqueeze(2)


def npmat2euler(mats, seq='zyx'):
    eulers = []
    for i in range(mats.shape[0]):
        r = Rotation.from_dcm(mats[i])
        eulers.append(r.as_euler(seq, degrees=True))
    return np.asarray(eulers, dtype='float32')