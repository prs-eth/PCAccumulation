"""
We evaluate two ways to remove ground points
Fast Segmentation of 3D Point Clouds: A Paradigm on LiDAR Data for Autonomous Vehicle Applications, ICRA'17
"""
from toolbox.utils import load_pkl, natural_key, to_o3d_pcd, multi_vis
import numpy as np
from glob import glob

def extract_init_seed(pts_sort, n_lpr, th_seed):
    lpr = np.mean(pts_sort[:n_lpr, 2])
    seed = pts_sort[pts_sort[:, 2] < lpr + th_seed, :]
    return seed


def get_non_ground(pts):
    """
    Input:
        pts(np.ndarray):    [N,3]  
    Return:
        is_not_ground:      [N]
    """
    th_seeds_ = 1.2
    num_lpr_ = 20
    n_iter = 10
    th_dist_ = 0.3
    pts_sort = pts[pts[:, 2].argsort(), :]
    pts_g = extract_init_seed(pts_sort, num_lpr_, th_seeds_)
    normal_ = np.zeros(3)
    for i in range(n_iter):
        mean = np.mean(pts_g, axis=0)[:3]
        xx = np.mean((pts_g[:, 0] - mean[0]) * (pts_g[:, 0] - mean[0]))
        xy = np.mean((pts_g[:, 0] - mean[0]) * (pts_g[:, 1] - mean[1]))
        xz = np.mean((pts_g[:, 0] - mean[0]) * (pts_g[:, 2] - mean[2]))
        yy = np.mean((pts_g[:, 1] - mean[1]) * (pts_g[:, 1] - mean[1]))
        yz = np.mean((pts_g[:, 1] - mean[1]) * (pts_g[:, 2] - mean[2]))
        zz = np.mean((pts_g[:, 2] - mean[2]) * (pts_g[:, 2] - mean[2]))
        cov = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
        U, S, V = np.linalg.svd(cov)
        normal_ = U[:, 2]
        d_ = -normal_.dot(mean)
        th_dist_d_ = th_dist_ - d_
        result = pts[:, :3].dot(normal_)
        is_not_ground = result >= th_dist_d_
        pts_n_g = pts[result>th_dist_d_]
        pts_g = pts[result<th_dist_d_]
    return is_not_ground