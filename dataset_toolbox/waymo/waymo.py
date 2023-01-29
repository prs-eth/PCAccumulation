import os
from tqdm import tqdm
from glob import glob
from termcolor import colored
from libs.waymo_decoder import extract_objects, extract_points
import multiprocessing as mp
from libs.utils import makedirs, save_pkl
import tensorflow.compat.v1 as tf
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import frame_utils
import numpy as np
from libs.box_np_ops import center_to_corner_box3d, points_in_rbbox


class WaymoOpenDataset:
    """
    A class to:
    1. download Waymo Open Dataset
    2. extract points and associated bbox annotations
    """
    def __init__(self, base_folder, output_dir):
        self.base_folder = base_folder
        self.output_dir = output_dir

        self.sequences = {
            'training': glob(base_folder + '/training/*.tfrecord'),
            'validation': glob(base_folder + '/validation/*.tfrecord')
        }
        n_train_seqs = len(self.sequences['training'])
        n_val_seqs = len(self.sequences['validation'])
        print(colored(f'We have in total {n_train_seqs} sequences for training'))
        print(colored(f'We have in total {n_val_seqs} sequences for validation'))

    def batch_download(self, split, output_dir):
        """
        Script for batch downloading the waymo dataset and uncompress the raw data
        """
        assert split in ['training','validation']
        url_template = 'gs://waymo_open_dataset_v_1_2_0/{split}/{split}_%04d.tar'.format(split=split)
        output_dir = os.path.join(output_dir, split)
        if split == 'training':
            num_segs = 32
        elif split == 'validation':
            num_segs = 8
        
        for seg_id in range(0, num_segs):
            flag = os.system('gsutil cp ' + url_template % seg_id + ' ' + output_dir)
            assert flag == 0, 'Failed to download segment %d. Make sure gsutil is installed'%seg_id
            os.system('cd %s; tar xf %s_%04d.tar'%(output_dir, split, seg_id))
            print('Segment %d done' % seg_id)
        
        print('Dataset download finished')

    def extract_lidar_bbox_label(self, c_sequence):
        dump_dir = os.path.join(self.output_dir, c_sequence.split('/')[-2], c_sequence.split('/')[-1].split('.')[0])
        makedirs(dump_dir)
        makedirs(dump_dir+'/lidar')
        makedirs(dump_dir+'/label')

        dataset = tf.data.TFRecordDataset(c_sequence, compression_type='')
        for frame_id, data in enumerate(dataset):
            frame_id = str(frame_id).zfill(4)
            data_path = f'{dump_dir}/lidar/{frame_id}'
            anno_path = f'{dump_dir}/label/{frame_id}.pkl'

            # 1. parse raw data
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            (range_images, camera_projections,range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)

            # 2. extract points
            points, _ = frame_utils.convert_range_image_to_point_cloud(frame,range_images,camera_projections,range_image_top_pose)
            points_all = np.concatenate(points, axis=0).astype(np.float)
            laser_indice = []
            for idx, pts in enumerate(points):
                laser_indice.append(np.ones(pts.shape[0]) * idx)
            laser_indice = np.concatenate(laser_indice, axis = 0).astype(np.int32)

            # 3. extract bbox 
            pose = np.reshape(np.array(frame.pose.transform), [4, 4])
            pose_rot = pose[:3, :3] 
            objects = extract_objects(frame.laser_labels, pose_rot)
            bboxes = []
            names = []
            labels = []
            for eachbox in objects:
                bboxes.append(eachbox['box'])
                names.append(eachbox['name'])
                labels.append(eachbox['label'])
            bboxes = np.array(bboxes)
            if len(bboxes):
                bboxes = np.array(bboxes)
                bboxes[:,-1] = - bboxes[:,-1] # center, size, ref_velocity, yaw_angle

                # 4. points to bbox indices, -1 means background points
                indices = points_in_rbbox(points_all, bboxes).astype(np.int)  
                indices = np.hstack([np.ones((indices.shape[0],1)) * 0.5,indices])
                ind_bbox = indices.argmax(1) 
                ind_bbox -= 1  
                ind_bbox = ind_bbox.astype(np.int)
                assert ind_bbox.min() == -1
            else:
                ind_bbox = np.zeros_like(laser_indice) - 1

            # 5. save points, laser_indice, ind_bbox
            laser_indice = laser_indice[:,None]
            ind_bbox = ind_bbox[:,None]
            assert  ind_bbox.shape[0] == laser_indice.shape[0] == points_all.shape[0]
            laser_data = np.hstack([points_all, laser_indice, ind_bbox])

            np.save(data_path,laser_data)

            # 6. save annotations
            frame_name = '{scene_name}_{location}_{time_of_day}_{timestamp}'.format(
                scene_name=frame.context.name,
                location=frame.context.stats.location,
                time_of_day=frame.context.stats.time_of_day,
                timestamp=frame.timestamp_micros)
            annotations = {
                'scene_name': frame.context.name,
                'frame_name': frame_name,
                'frame_id': frame_id,
                'veh_to_global': pose,  
                'objects': objects,
            }
            save_pkl(annotations, anno_path)


def ss1():
    """
    Generate lidar points and bbox annotations
    """
    wod = WaymoOpenDataset('/scratch3/waymo/waymo_format', '/scratch3/waymo/our_format')
    seqs = []
    seqs.extend(wod.sequences['training'])
    seqs.extend(wod.sequences['validation'])
    p = mp.Pool(processes=mp.cpu_count())
    p.map(wod.extract_lidar_bbox_label, seqs)
    p.close()
    p.join()    


if __name__=='__main__':
    wod = WaymoOpenDataset('/scratch3/waymo/waymo_format', '/scratch3/waymo/our_format')