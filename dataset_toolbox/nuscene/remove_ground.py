"""
We evaluate two ways to remove ground points
Fast Segmentation of 3D Point Clouds: A Paradigm on LiDAR Data for Autonomous Vehicle Applications, ICRA'17
"""
from torchsparse.utils import sparse_quantize
from utils import load_pkl, natural_key, to_o3d_pcd, multi_vis
import numpy as np
from glob import glob
from tqdm import tqdm

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


if __name__=='__main__':
    data_dir = '../datasets/waymo/customise/validation'
    voxel_size = 0.15
    for seq_id in range(20):
        seq_id = str(seq_id).zfill(3)
        lidar_path = sorted(glob(data_dir+f'/{seq_id}/lidar/*.npy'), key=natural_key)
        label_path = sorted(glob(data_dir+f'/{seq_id}/label/*.pkl'), key=natural_key)
        idx = np.arange(len(lidar_path)).tolist()
        np.random.shuffle(idx)
        lidar_path = np.array(lidar_path)[idx].tolist()
        label_path = np.array(label_path)[idx].tolist()

        for idx, lidar in tqdm(enumerate(lidar_path)):
            laser_data = np.load(lidar)
            label_data = load_pkl(label_path[idx])
            veh_to_global = label_data['veh_to_global']
            sel = laser_data[:,3] == 0
            points = laser_data[sel,:3]
            pcd = to_o3d_pcd(points)

            is_not_ground = get_ground(points)
            pcd_remove_icra = to_o3d_pcd(points[is_not_ground])

            sel_ground_points = laser_data[:,3] > 0
            ground_points = laser_data[sel_ground_points,:3]
            dist = np.linalg.norm(ground_points[:,:2], axis=1)
            ground_height = ground_points[dist < 3, 2].mean()

            is_not_ground = points[:,2] > ground_height+0.05
            pcd_remove_thre = to_o3d_pcd(points[is_not_ground])

            multi_vis([pcd, pcd_remove_icra, pcd_remove_thre],['input','GPF','Simple thresholding'])