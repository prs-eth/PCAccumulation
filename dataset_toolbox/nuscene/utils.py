"""
General utility functions

Author: Shengyu Huang
Last modified: 30.11.2020
"""

import os,re,sys,json,yaml,random, argparse, torch, pickle, frnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from scipy.spatial.transform import Rotation

from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import minkowski
import open3d as o3d
import multiprocessing as mp
count_job = 0

_EPS = 1e-7  # To prevent division by zero

def mp_process(func, task):
    p = mp.Pool(processes=mp.cpu_count())
    p.map(func, task)
    p.close()
    p.join()

def dict_to_array(c_dict):
    """
    Convert a dictionary to array, both key and value are numeric values
    Return:
        c_array:    c_array[key] = c_dict[key]
    """
    n_elements = max([ele for ele,_ in c_dict.items()])+1
    c_array = np.zeros([n_elements]).astype(np.int32)
    for key, value in c_dict.items():
        c_array[key] = int(value)
    return c_array
    

def lighter(color, percent):
    '''assumes color is rgb between (0, 0, 0) and (1,1,1)'''
    color = np.array(color)
    white = np.array([1, 1, 1])
    vector = white-color
    return color + vector * percent


def radius_search(query_points, ref_points, radius, max_K):
    """
    https://github.com/lxxue/FRNN
    GPU based efficient radius search
    Input:
        query_points:   [M,3]
        ref_points:     [N,3]
        radius:         float, search radius
        max_K:          maximal neighborhood
    Return:
        idxs:           [M, max_K]   -1 means padding index
    """
    device = torch.device('cuda')
    n_query = torch.tensor([query_points.size(0)]).long().to(device)
    n_ref = torch.tensor([ref_points.size(0)]).long().to(device)
    query_points = query_points[None].float().to(device)
    ref_points = ref_points[None].float().to(device)

    dists, idxs, _,_ = frnn.frnn_grid_points(query_points, ref_points, n_query, n_ref, max_K, radius, grid=None, return_nn=False, return_sorted=False)
    return idxs



def get_persistence_prior(query_points, ref_points, ref_time_indice, radius, query_frames, K):
    """
    https://www.youtube.com/watch?v=COgEQuqTAug
    Static ---> 1, Dynamic ---> 0
    Input:
        query_points:   [M,3]
        ref_points:     [N, 3]
        time_indice:    [N], starts from 1
        radius:         float, search radius
        K               integer, max K
    Return:
        scores:         [M]
    """
    device = torch.device('cuda')
    idxs = radius_search(query_points, ref_points, radius, K)
    idxs = idxs + 1 # map -1 to 0
    n_frames = len(query_frames)
    assert 0 not in query_frames and n_frames > 1

    # 2. gather time indice information
    time_indice = torch.cat((torch.tensor([0]).to(device),ref_time_indice.to(device)))
    gather_time_indice = time_indice[idxs.view(-1)].view(-1, K)
    
    # 3. get frequency representation [N, K], discard indice if indice = 0 
    bin_count = []
    for frame_idx in query_frames:
        count = torch.sum((gather_time_indice==frame_idx), dim=1)
        bin_count.append(count)
    bin_count = torch.stack(bin_count, dim=1)
    freq = bin_count / (torch.sum(bin_count,1,keepdim=True) + _EPS)
    
    # 4. get kl_divergence 
    scores = torch.zeros(freq.size(0)).to(device)
    sel = torch.any(freq, dim=1)
    sel_freq = freq[sel] + _EPS
    uniform_freq = torch.ones(sel_freq.size()).to(device) / n_frames
    kl_div = F.kl_div(uniform_freq.log(), sel_freq,reduction='none').sum(1)
    scores[sel] = 1 - kl_div / torch.tensor(n_frames).log()
    return scores


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

class Logger:
    def __init__(self, path):
        self.path = path
        self.fw = open(self.path+'/log','a')

    def write(self, text):
        self.fw.write(text)
        self.fw.flush()

    def close(self):
        self.fw.close()

def load_pcd(file_name):
    pcd = o3d.io.read_point_cloud(file_name)
    return pcd


def save_pcd(pcd, file_name):
    """
    save a point cloud to ply file
    """
    o3d.io.write_point_cloud(file_name, pcd)


def load_pkl(path):
    """
    Load a .pkl object
    """
    file = open(path ,'rb')
    return pickle.load(file)

def save_pkl(obj, path ):
    """
    save a dictionary to a pickle file
    """
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_yaml(path):
    """
    Loads configs from .yaml file

    Args:
        path (str): path to the config file

    Returns: 
        config (dict): dictionary of the configuration parameters, merge sub_dicts

    """
    with open(path,'r') as f:
        cfg = yaml.safe_load(f)

    return cfg


def setup_seed(seed):
    """
    fix random seed for deterministic training
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def square_distance(src, dst, normalised = False):
    """
    Calculate Euclid distance between each two points.
    Args:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Returns:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    if(normalised):
        dist += 2
    else:
        dist += torch.sum(src ** 2, dim=-1)[:, :, None]
        dist += torch.sum(dst ** 2, dim=-1)[:, None, :]

    dist = torch.clamp(dist, min=1e-12, max=None)
    return dist
    

def validate_gradient(model):
    """
    Confirm all the gradients are non-nan and non-inf
    """
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.any(torch.isnan(param.grad)):
                return False
            if torch.any(torch.isinf(param.grad)):
                return False
    return True


def natural_key(string_):
    """
    Sort strings by numbers in the name
    """
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]


def makedirs(folder):
    if(os.path.exists(folder)):
        return 
    else:
        os.makedirs(folder)


def get_blue():
    """
    Get color blue for rendering
    """
    return [0, 0.651, 0.929]

def get_yellow():
    """
    Get color yellow for rendering
    """
    return [1, 0.706, 0]

def to_tensor(array):
    """
    Convert array to tensor
    """
    if(not isinstance(array,torch.Tensor)):
        return torch.from_numpy(array)
    else:
        return array

def to_array(tensor):
    """
    Conver tensor to array
    """
    if(not isinstance(tensor,np.ndarray)):
        if(tensor.device == torch.device('cpu')):
            return tensor.numpy()
        else:
            return tensor.cpu().numpy()
    else:
        return tensor

def to_tsfm(rot,trans):
    tsfm = np.eye(4)
    tsfm[:3,:3]=rot
    tsfm[:3,3]=trans.flatten()
    return tsfm

def to_o3d_vec(vec):
    """
    Create open3d array objects
    """
    return o3d.utility.Vector3dVector(to_array(vec))
    
def to_o3d_pcd(xyz):
    """
    Convert tensor/array to open3d PointCloud
    xyz:       [N, 3]
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(to_array(xyz))
    return pcd

def to_o3d_feats(embedding):
    """
    Convert tensor/array to open3d features
    embedding:  [N, 3]
    """
    feats = o3d.registration.Feature()
    feats.data = to_array(embedding).T
    return feats

def canonicalise_random_indice(indice):
    """
    Convert randomly scattered indice(unordered) to canonical spcae.
    For example: [1,4,4,6,10] ----> [0,1,1,2,3]
    Input:
        indice: list
    """
    unique_ids = sorted(set(indice))  # make sure -1 is mapped to 0
    mapping_dict = dict()
    for idx,ele in enumerate(unique_ids):
        mapping_dict[ele] = idx
    
    mapped_list = [mapping_dict[ele] for ele in indice]
    return mapped_list


def vis_o3d(pcds, render=True, window_name = None):
    """
    Input:
        pcds:   a list of open3d objects
    """
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    if(render):
        for eachpcd in pcds:
            try:
                eachpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))
            except:
                pass
    pcds.append(mesh_frame)
    if(window_name is not None):
        o3d.visualization.draw_geometries(pcds, window_name=window_name)
    else:
        o3d.visualization.draw_geometries(pcds)

def multi_vis(pcds, names, render = True, width = 960, height = 540, shift = 100):
    """
    Visulise point clouds in multiple windows, we allow at most 4 windows

    Input:
        pcds:   a list of pcds
        names:  a list of window names
    """
    assert len(pcds) == len(names)
    n_windows = len(pcds)

    window_corners = [
        [0,0],
        [0,height+shift],
        [width, 0],
        [width+shift, height+shift]
    ]
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=10, origin=[0, 0, 0])

    # estimate normals for better visualisation
    if(render):
        for each_pcd in pcds:
            each_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=50))

    # initialise the windows
    vis_1 = o3d.visualization.Visualizer()
    vis_1.create_window(window_name=names[0], width=width, height=height, left=window_corners[0][0], top=window_corners[0][1])
    vis_1.add_geometry(pcds[0])
    vis_1.add_geometry(mesh_frame)

    if(n_windows >= 2):
        vis_2 = o3d.visualization.Visualizer()
        vis_2.create_window(window_name=names[1], width=width, height=height, left=window_corners[1][0], top=window_corners[1][1])
        vis_2.add_geometry(pcds[1])
        vis_2.add_geometry(mesh_frame)
        if(n_windows >=3):
            vis_3 = o3d.visualization.Visualizer()
            vis_3.create_window(window_name=names[2], width=width, height=height, left=window_corners[2][0], top=window_corners[2][1])
            vis_3.add_geometry(pcds[2])
            vis_3.add_geometry(mesh_frame)
            if(n_windows>=4):
                vis_4 = o3d.visualization.Visualizer()
                vis_4.create_window(window_name=names[3], width=width, height=height, left=window_corners[3][0], top=window_corners[3][1])
                vis_4.add_geometry(pcds[3])
                vis_4.add_geometry(mesh_frame)

    # start rendering
    while True:
        vis_1.update_geometry(pcds[0])
        vis_1.update_geometry(mesh_frame)
        if not vis_1.poll_events():
            break
        vis_1.update_renderer()

        if(n_windows>=2):
            vis_2.update_geometry(pcds[1])
            vis_2.update_geometry(mesh_frame)
            if not vis_2.poll_events():
                break
            vis_2.update_renderer()

            cam = vis_1.get_view_control().convert_to_pinhole_camera_parameters()
            cam2 = vis_2.get_view_control().convert_to_pinhole_camera_parameters()
            cam2.extrinsic = cam.extrinsic
            vis_2.get_view_control().convert_from_pinhole_camera_parameters(cam2)


            if(n_windows>=3):
                vis_3.update_geometry(pcds[2])
                vis_3.update_geometry(mesh_frame)
                if not vis_3.poll_events():
                    break
                vis_3.update_renderer()

                cam3 = vis_3.get_view_control().convert_to_pinhole_camera_parameters()
                cam3.extrinsic = cam.extrinsic
                vis_3.get_view_control().convert_from_pinhole_camera_parameters(cam3)

                if(n_windows>=4):
                    vis_4.update_geometry(pcds[3])
                    vis_4.update_geometry(mesh_frame)
                    if not vis_4.poll_events():
                        break
                    vis_4.update_renderer()

                    cam4 = vis_4.get_view_control().convert_to_pinhole_camera_parameters()
                    cam4.extrinsic = cam.extrinsic
                    vis_4.get_view_control().convert_from_pinhole_camera_parameters(cam4)
    
    vis_1.destroy_window()
    if(n_windows>=2):
        vis_2.destroy_window()

        if(n_windows>=3):
            vis_3.destroy_window()

            if(n_windows>=4):
                vis_4.destroy_window()