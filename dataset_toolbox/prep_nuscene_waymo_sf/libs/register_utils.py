# import hdbscan
from sklearn.cluster import DBSCAN
import pickle, torch, math
from libs.utils import to_o3d_pcd, load_yaml, load_pkl
import open3d as o3d
import numpy as np

defaultConfigs = load_yaml('configs/default.yaml')
distinct_colors_dict = load_pkl(defaultConfigs['path']['color_info'])

R = np.array([7.533745e-03, -9.999714e-01, -6.166020e-04, 1.480249e-02, 7.280733e-04,-9.998902e-01, 9.998621e-01, 7.523790e-03, 1.480755e-02]).reshape(3, 3)
T = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01]).reshape(3, 1)
kitti_velo2cam = np.hstack([R, T])
kitti_velo2cam= np.vstack((kitti_velo2cam, [0, 0, 0, 1])).T
waymo_velo2cam = np.eye(4)
nuscene_velo2cam = np.eye(4)

VELO2CAM= {
    'kitti': kitti_velo2cam,
    'waymo': waymo_velo2cam,
    'waymo_full': waymo_velo2cam,
    'nuscene': nuscene_velo2cam
}

def rotation_error(R1, R2):
    """
    Torch batch implementation of the rotation error between the estimated and the ground truth rotatiom matrix. 
    Rotation error is defined as r_e = \arccos(\frac{Trace(\mathbf{R}_{ij}^{T}\mathbf{R}_{ij}^{\mathrm{GT}) - 1}{2})
    Args: 
        R1 (torch tensor): Estimated rotation matrices [b,3,3]
        R2 (torch tensor): Ground truth rotation matrices [b,3,3]
    Returns:
        ae (torch tensor): Rotation error in angular degreees [b,1]
    """
    R_ = torch.matmul(R1.transpose(1,2), R2)
    e = torch.stack([(torch.trace(R_[_, :, :]) - 1) / 2 for _ in range(R_.shape[0])], dim=0).unsqueeze(1)
    # Clamp the errors to the valid range (otherwise torch.acos() is nan)
    e = torch.clamp(e, -1, 1, out=None)

    ae = torch.acos(e)
    pi = torch.Tensor([math.pi])
    ae = 180. * ae / pi.to(ae.device).type(ae.dtype)
    # for idx in range(R1.size(0)):
    #     if ae[idx]!=0:
    #         print(ae[idx], torch.equal(R1[idx], R2[idx]))
    #         import pdb
    #         pdb.set_trace()

    return ae


def translation_error(t1, t2):
    """
    Torch batch implementation of the rotation error between the estimated and the ground truth rotatiom matrix. 
    Rotation error is defined as r_e = \arccos(\frac{Trace(\mathbf{R}_{ij}^{T}\mathbf{R}_{ij}^{\mathrm{GT}) - 1}{2})
    Args: 
        t1 (torch tensor): Estimated translation vectors [b,3,1]
        t2 (torch tensor): Ground truth translation vectors [b,3,1]
    Returns:
        te (torch tensor): translation error in meters [b,1]
    """
    return torch.norm(t1-t2, dim=(1, 2))


def ego_motion_compensation(points, time_indice, tsfm):
    """
    Input (torch.Tensor):
        points:         [N, 3]
        time_indice:    [N]
        tsfm:           [n_frames, 4, 4]
    """
    point_tsfm = tsfm[time_indice.long()]
    rot, trans = point_tsfm[:,:3,:3], point_tsfm[:,:3,3][:,:,None]
    rec_points = (torch.matmul(rot, points[:,:,None]) + trans).squeeze(-1)
    return rec_points
    

def reconstruct_sequence(points, time_indice, inst_labels, tsfm, n_frames):
    """
    Reconstruct a sequence of point clouds, this only works for batch_size = 1
    Input:
        points:         [N,3]
        time_indice:    [N]
        inst_labels:    [N]
        tsfm:           [M, n_frames, 4, 4]
        n_frames:       integer
    """
    inst_labels = inst_labels.long()
    assert n_frames == tsfm.size(1)
    tsfm = tsfm.view(-1, 4, 4)
    indice = (inst_labels * n_frames + time_indice).long()
    point_tsfm = tsfm[indice]

    # point_tsfm = tsfm[inst_labels] #[N, n_frame, 4, 4]
    # gather_indice = time_indice[:,None,None,None].repeat(1,1,4,4).long()
    # point_tsfm = torch.gather(point_tsfm, dim=1, index = gather_indice).squeeze(1)        
    rot, trans = point_tsfm[:,:3,:3], point_tsfm[:,:3,3][:,:,None]
    rec_points = (torch.matmul(rot, points[:,:,None]) + trans).squeeze(-1)
    return rec_points
    

def get_rot_matrix_from_yaw_angle_torch(yaw_angle, radian=True, right_hand=True):
    """
    Conver rotation angle along z axis to rotation matrix
    yaw_angle is in radian
    """
    if not radian:
        yaw_angle = yaw_angle / 180 * np.pi
    
    rot_sin  = torch.sin(yaw_angle)
    rot_cos = torch.cos(yaw_angle)

    rot_mat = torch.eye(3).to(rot_sin.device)

    if right_hand:
        rot_mat[0,0], rot_mat[0,1] = rot_cos, rot_sin
        rot_mat[1,0], rot_mat[1,1] = -rot_sin, rot_cos
    else:
        rot_mat[0,0], rot_mat[0,1] = rot_cos, -rot_sin
        rot_mat[1,0], rot_mat[1,1] = rot_sin, rot_cos

    return rot_mat


def get_rot_matrix_from_yaw_angle_np(yaw_angle, radian=True, right_hand=True):
    """
    Conver rotation angle along z axis to rotation matrix
    yaw_angle is in degrees
    """
    if not radian:
        yaw_angle = yaw_angle / 180 * np.pi

    rot_sin = np.sin(yaw_angle)
    rot_cos = np.cos(yaw_angle)
    ones = np.ones_like(rot_cos)
    zeros = np.zeros_like(rot_cos)

    if right_hand:
        rot_mat = np.stack([
            [rot_cos, rot_sin, zeros],
            [-rot_sin, rot_cos, zeros],
            [zeros, zeros, ones]
        ])
    else:
        rot_mat = np.stack([
            [rot_cos, -rot_sin, zeros],
            [rot_sin, rot_cos, zeros],
            [zeros, zeros, ones]
        ])

    return rot_mat


def get_rot_matrix_from_yaw_angle_np(yaw_angle, radian=True, right_hand=True):
    """
    Conver rotation angle along z axis to rotation matrix
    yaw_angle is in degrees
    """
    if not radian:
        yaw_angle = yaw_angle / 180 * np.pi

    rot_sin = np.sin(yaw_angle)
    rot_cos = np.cos(yaw_angle)
    ones = np.ones_like(rot_cos)
    zeros = np.zeros_like(rot_cos)

    if right_hand:
        rot_mat = np.stack([
            [rot_cos, rot_sin, zeros],
            [-rot_sin, rot_cos, zeros],
            [zeros, zeros, ones]
        ])
    else:
        rot_mat = np.stack([
            [rot_cos, -rot_sin, zeros],
            [rot_sin, rot_cos, zeros],
            [zeros, zeros, ones]
        ])

    return rot_mat

def get_relative_pose_np(tsfm_src, tsfm_tgt, dataset):
    """
    Compute relative pose given two transformations
    """
    velo2cam = VELO2CAM[dataset]
    M = (velo2cam @ tsfm_src.T @ np.linalg.inv(tsfm_tgt.T)@ np.linalg.inv(velo2cam)).T 
    return M

def get_relative_pose_torch(tsfm_src, tsfm_tgt, dataset):
    """
    We compute relative transformation matrix from source point cloud to target point cloud
    T_src @ X_src = T_tgt @ X_tgt
    X_tgt = T_tgt^-1 @ T_src @ X_src
    T_relative = T_tgt^-1 @ T_src
    """
    if dataset in ['waymo','nuscene']:
        M = torch.linalg.solve(tsfm_tgt, tsfm_src)
    else:
        velo2cam = VELO2CAM[dataset]
        velo2cam = torch.from_numpy(velo2cam).float().to(tsfm_src.device)
        M = (velo2cam @ tsfm_src.T @ torch.inverse(tsfm_tgt.T)@ torch.inverse(velo2cam)).T 
    return M

def apply_tsfm(src, tsfm):
    """
    tsfm:   [4,4]
    src:    [N,3]
    """
    R, t = tsfm[:3,:3], tsfm[:3,3][:,None]
    src = (R @ src.T + t).T
    return src

def register_odometry_torch(src, tsfm_src, tsfm_tgt, dataset):
    M = get_relative_pose_torch(tsfm_src, tsfm_tgt, dataset)
    src = apply_tsfm(src, M)
    return src

def register_odometry_np(src, tsfm_src, tsfm_tgt, dataset):
    M = get_relative_pose_np(tsfm_src, tsfm_tgt, dataset)
    src = apply_tsfm(src, M)
    return src

def convert_rot_trans_to_tsfm(rot, trans):
    """
    Construct 4x4 transformation matrix given 3x3 rotation matrix and 3x1 translation vector
    """
    tsfm = np.eye(4)
    tsfm[:3,:3] = rot
    tsfm[:3,3] = trans.flatten()
    return tsfm


def transformation_residuals(x1, x2, R, t):
    """
    Computer the pointwise residuals based on the estimated transformation paramaters
    
    Args:
        x1  (torch array): points of the first point cloud [b,n,3]
        x2  (torch array): points of the second point cloud [b,n,3]
        R   (torch array): estimated rotation matrice [b,3,3]
        t   (torch array): estimated translation vectors [b,3,1]
    Returns:
        res (torch array): pointwise residuals (Eucledean distance) [b,n,1]
    """
    x2_reconstruct = torch.matmul(R, x1.transpose(1, 2)) + t 

    res = torch.norm(x2_reconstruct.transpose(1, 2) - x2, dim=2)

    return res


def kabsch_transformation_estimation(x1, x2, weights=None, normalize_w = True, eps = 1e-7, best_k = 0, w_threshold = 0):
    """
    Torch differentiable implementation of the weighted Kabsch algorithm (https://en.wikipedia.org/wiki/Kabsch_algorithm). Based on the correspondences and weights calculates
    the optimal rotation matrix in the sense of the Frobenius norm (RMSD), based on the estimate rotation matrix is then estimates the translation vector hence solving
    the Procrustes problem. This implementation supports batch inputs.
    Args:
        x1            (torch array): points of the first point cloud [b,n,3]
        x2            (torch array): correspondences for the PC1 established in the feature space [b,n,3]
        weights       (torch array): weights denoting if the coorespondence is an inlier (~1) or an outlier (~0) [b,n]
        normalize_w   (bool)       : flag for normalizing the weights to sum to 1
        best_k        (int)        : number of correspondences with highest weights to be used (if 0 all are used)
        w_threshold   (float)      : only use weights higher than this w_threshold (if 0 all are used)
    Returns:
        rot_matrices  (torch array): estimated rotation matrices [b,3,3]
        trans_vectors (torch array): estimated translation vectors [b,3,1]
        res           (torch array): pointwise residuals (Eucledean distance) [b,n]
        valid_gradient (bool): Flag denoting if the SVD computation converged (gradient is valid)
    """
    if weights is None:
        weights = torch.ones(x1.shape[0],x1.shape[1]).type_as(x1).to(x1.device)

    if normalize_w:
        sum_weights = torch.sum(weights,dim=1,keepdim=True) + eps
        weights = (weights/sum_weights)

    weights = weights.unsqueeze(2)

    if best_k > 0:
        indices = np.argpartition(weights.cpu().numpy(), -best_k, axis=1)[0,-best_k:,0]
        weights = weights[:,indices,:]
        x1 = x1[:,indices,:]
        x2 = x2[:,indices,:]

    if w_threshold > 0:
        weights[weights < w_threshold] = 0


    x1_mean = torch.matmul(weights.transpose(1,2), x1) / (torch.sum(weights, dim=1).unsqueeze(1) + eps)
    x2_mean = torch.matmul(weights.transpose(1,2), x2) / (torch.sum(weights, dim=1).unsqueeze(1) + eps)

    x1_centered = x1 - x1_mean
    x2_centered = x2 - x2_mean

    weight_matrix = torch.diag_embed(weights.squeeze(2))

    cov_mat = torch.matmul(x1_centered.transpose(1, 2),
                           torch.matmul(weight_matrix, x2_centered))

    try:
        u, s, v = torch.svd(cov_mat)
    except Exception as e:
        r = torch.eye(3,device=x1.device)
        r = r.repeat(x1_mean.shape[0],1,1)
        t = torch.zeros((x1_mean.shape[0],3,1), device=x1.device)

        res = transformation_residuals(x1, x2, r, t)

        return r, t, res, True

    tm_determinant = torch.det(torch.matmul(v.transpose(1, 2), u.transpose(1, 2)))

    determinant_matrix = torch.diag_embed(torch.cat((torch.ones((tm_determinant.shape[0],2),device=x1.device), tm_determinant.unsqueeze(1)), 1))

    rotation_matrix = torch.matmul(v,torch.matmul(determinant_matrix,u.transpose(1,2)))

    # translation vector
    translation_matrix = x2_mean.transpose(1,2) - torch.matmul(rotation_matrix,x1_mean.transpose(1,2))

    # Residuals
    res = transformation_residuals(x1, x2, rotation_matrix, translation_matrix)

    return rotation_matrix, translation_matrix, res, False