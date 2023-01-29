
from copy import deepcopy
import numpy as np
from scipy.spatial.transform import Rotation as R
from libs.register_utils import register_odometry_np, get_relative_pose, kabsch_transformation_estimation,convert_rot_trans_to_tsfm, get_rot_matrix_from_yaw_angle_np
from libs.bbox_utils import center_to_corner_box3d, points_in_rbbox, corners_to_lines
from libs.utils import to_tensor, to_array


def get_max_rotation_along_z(pose_list):
    """
    This is used to filter out frames with too big ego-car rotation angles
    """
    rot_along_z = []
    for idx in range(pose_list.shape[0]):
        r = R.from_matrix(pose_list[idx][:3,:3])
        euler_angle = r.as_euler('zyx', degrees=True)
        rot_along_z.append(euler_angle[0])
    rot_along_z = max(rot_along_z) - min(rot_along_z)
    return rot_along_z


class InstanceObservations:
    """
    We use this class to simulate dynamic objects from static objects
    In the end, we provide:
        - simulated tubes
        - point-wise time indice
        - relative transformation from each frame to the anchor frame
    """
    def __init__(self, input_dict, n_frames, min_points = 10):
        self.min_points = min_points  
        self.n_frames = n_frames
        self.static_instances = dict()
        self.dynamic_instances = dict()
        self.simulated_tubes = dict()
        self.pose_list = input_dict['pose_list']
        self.extract_instances(input_dict)
        self.simulate_tube_from_static_objects()
        self.get_real_tubes()

    def extract_instances(self, input_dict):
        """
        We reorganise data into instances
        We only consider instances with
        1) at least self.min_points points
        2) points in the last frame(anchor frame)
        """
        points = input_dict['points']
        inst_label = input_dict['inst_label']
        time_indice = input_dict['time_indice']
        meta_bbox = input_dict['meta_bbox']

        for key, value in meta_bbox.items():
            bbox_index = value['bbox_index']
            sd_label = value['sd_label']
            sel = inst_label == bbox_index
            if(sel.sum() > self.min_points): # filter instances with too few points
                meta_bbox[key]['points'] = points[sel]
                meta_bbox[key]['points_ts_indice'] = time_indice[sel]

                if (time_indice[sel] == 0).sum():  # we only consider instances with observations in the last frame
                    if(sd_label == 1):
                        self.dynamic_instances[key] = meta_bbox[key]
                    else:
                        self.static_instances[key] = meta_bbox[key]
            else:
                meta_bbox[key]['points'] = None
                meta_bbox[key]['points_ts_indice'] = None


    def simulate_tube_from_static_objects(self):
        """
        Here we provide
        1) points and associated time_indice [N, 3],[N]
        2) relative poses   [n_frames, 4, 4]
        """
        self.simulated_tubes = dict()

        for key, value in self.static_instances.items():
            # determine orientation: clock-wise angle along z axis from positive y direction
            bbox = value['bbox'][0]  
            points = deepcopy(value['points'])
            points_ts_indice = value['points_ts_indice']
            yaw_angle = np.pi / 2 - bbox[-1]   

            # align static objects to positive x axis
            rot_mat = get_rot_matrix_from_yaw_angle_np( - yaw_angle - np.pi / 2)
            tsfm_mat = np.eye(4)
            tsfm_mat[:3,:3] = rot_mat
            points = np.dot(rot_mat,points.T).T
            
            # apply ego-motion to static object
            relative_pose_list = []
            n_points_list = []
            for idx in range(self.n_frames):
                sel = points_ts_indice == idx
                relative_pose = get_relative_pose(self.pose_list[0], self.pose_list[idx],'waymo')
                relative_pose_list.append(tsfm_mat.T @ np.linalg.inv(relative_pose) @ tsfm_mat)
                n_points_list.append(sel.sum())
                if(sel.sum()): 
                    points[sel] = register_odometry_np(points[sel],self.pose_list[0], self.pose_list[idx], 'waymo')

            # transform back the point clouds by original bbox orientation
            points = np.dot(rot_mat.T, points.T).T
            dist_to_sensor = np.linalg.norm(points.mean(0))
            self.simulated_tubes[key]={
                'points': points,
                'time_indice': points_ts_indice,
                'relative_poses': np.array(relative_pose_list),
                'len_points': np.array(n_points_list),
                'dist_to_sensor': dist_to_sensor,
                'sem_label': value['sem_label'],
                'speed': max(value['speed'])
            }

    def get_real_tubes(self):
        self.real_tubes = dict()
        if(len(self.dynamic_instances.keys())):
            for key, value in self.dynamic_instances.items():
                points = deepcopy(value['points'])
                points_ts_indice = value['points_ts_indice']
                dist_to_sensor = np.linalg.norm(points.mean(0))
                # get ground truth relative pose by aligning bbox, here we rely on SVD
                # remember to transform bbox using ego-motion first
                bbox = np.array(value['bbox'])
                corners_idx = value['time_indice']
                corners = center_to_corner_box3d(bbox[:,:3], bbox[:,3:6],-bbox[:,-1])  # [N, 8, 3]
                anchor_corners = corners[0]

                relative_pose_list = []
                n_points_list = []
                for idx in range(self.n_frames):
                    sel = points_ts_indice == idx
                    n_points_list.append(sel.sum())
                    if(idx in corners_idx):
                        c_corners = corners[corners_idx.index(idx)]
                        c_corners = register_odometry_np(c_corners,self.pose_list[corners_idx.index(idx)], self.pose_list[0],'waymo').astype(np.float32)
                        rotation_matrix, translation_matrix, res, _ = kabsch_transformation_estimation(to_tensor(c_corners)[None],to_tensor(anchor_corners)[None])
                        c_tsfm = convert_rot_trans_to_tsfm(to_array(rotation_matrix[0]), to_array(translation_matrix[0]))
                        relative_pose_list.append(c_tsfm)
                    else:
                        relative_pose_list.append(np.eye(4))
                self.real_tubes[key] ={
                    'points': points, 
                    'time_indice': points_ts_indice,
                    'relative_poses': np.array(relative_pose_list),
                    'len_points': np.array(n_points_list),
                    'dist_to_sensor': dist_to_sensor,
                    'sem_label': value['sem_label'],
                    'speed': max(value['speed'])
                }