import os, sys, math, itertools
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from termcolor import colored
from tqdm import tqdm

import tensorflow.compat.v1 as tf
from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
from utils import vis_o3d, to_o3d_pcd, get_blue, multi_vis, to_o3d_vec, get_yellow
from bbox_utils import center_to_corner_box3d, points_in_rbbox, corners_to_lines
from waymo_decoder import extract_objects, extract_points
import open3d as o3d
tf.enable_eager_execution()
MY_CMAP = plt.cm.get_cmap('Set1')(np.arange(10))[:,:3]


from utils import load_pkl
color_dict = load_pkl('distinct_colors.pkl')[99][1:]
color_dict = np.clip(color_dict, 0,1)


def show_camera_image(camera_image, camera_labels, layout, cmap=None):
    """Show a camera image and the given camera labels."""
    plt.figure(figsize=(25, 20))
    ax = plt.subplot(*layout)

    # Draw the camera labels.
    for camera_labels in frame.camera_labels:
        # Ignore camera labels that do not correspond to this camera.
        if camera_labels.name != camera_image.name:
            continue

        # Iterate over the individual labels.
        for label in camera_labels.labels:
            # Draw the object bounding box.
            ax.add_patch(patches.Rectangle(
                xy=(label.box.center_x - 0.5 * label.box.length,
                    label.box.center_y - 0.5 * label.box.width),
                width=label.box.length,
                height=label.box.width,
                linewidth=1,
                edgecolor='red',
                facecolor='none'))

    # Show the camera image.
    ax.imshow(tf.image.decode_jpeg(camera_image.image), cmap=cmap)
    ax.set_title(open_dataset.CameraName.Name.Name(camera_image.name))
    ax.grid(False)
    # plt.show()

def plot_range_image_helper(data, name, layout, vmin = 0, vmax=1, cmap='gray'):
    """Plots range image.

    Args:
        data: range image data
        name: the image title
        layout: plt layout
        vmin: minimum value of the passed data
        vmax: maximum value of the passed data
        cmap: color map
    """
    plt.figure(figsize=(64, 20))
    plt.subplot(*layout)
    plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title(name)
    plt.grid(False)
    plt.axis('off')
    plt.show()

def get_range_image(range_images, laser_name, return_index):
    """
    Returns range image given a laser name and its return index.
    """
    return range_images[laser_name][return_index]

def show_range_image(range_image, layout_index_start = 1):
    """
    Shows range image.
    Args:
        range_image: the range image data from a given lidar of type MatrixFloat.
        layout_index_start: layout offset
    """
    range_image_tensor = tf.convert_to_tensor(range_image.data)
    range_image_tensor = tf.reshape(range_image_tensor, range_image.shape.dims)
    lidar_image_mask = tf.greater_equal(range_image_tensor, 0)
    range_image_tensor = tf.where(lidar_image_mask, range_image_tensor,
                                    tf.ones_like(range_image_tensor) * 1e10)
    range_image_range = range_image_tensor[...,0] 
    range_image_intensity = range_image_tensor[...,1]
    range_image_elongation = range_image_tensor[...,2]
    plot_range_image_helper(range_image_range.numpy(), 'range',
                    [8, 1, layout_index_start], vmax=75, cmap='gray')
    plot_range_image_helper(range_image_intensity.numpy(), 'intensity',
                    [8, 1, layout_index_start + 1], vmax=1.5, cmap='gray')
    plot_range_image_helper(range_image_elongation.numpy(), 'elongation',
                    [8, 1, layout_index_start + 2], vmax=1.5, cmap='gray')

def rgba(r):
    """
    Generates a color based on range.
    Args:
        r: the range value of a given point.
    Returns:
        The color for a given range
    """
    c = plt.get_cmap('jet')((r % 20.0) / 20.0)
    c = list(c)
    c[-1] = 0.5  # alpha
    return c

def plot_image(camera_image):
    """
    Plot a cmaera image.
    """
    plt.figure(figsize=(20, 12))
    plt.imshow(tf.image.decode_jpeg(camera_image.image))
    plt.grid("off")

def plot_points_on_image(projected_points, camera_image, rgba_func,point_size=5.0):
    """
    Plots points on a camera image.
    Args:
        projected_points: [N, 3] numpy array. The inner dims are
        [camera_x, camera_y, range].
        camera_image: jpeg encoded camera image.
        rgba_func: a function that generates a color from a range value.
        point_size: the point size.
    """
    plot_image(camera_image)

    xs = []
    ys = []
    colors = []

    for point in projected_points:
        xs.append(point[0])  # width, col
        ys.append(point[1])  # height, row
        colors.append(rgba_func(point[2]))

    plt.scatter(xs, ys, c=colors, s=point_size, edgecolors="none")
    plt.show()


def label2color(label):
    colors = [[204/255, 0, 0], [52/255, 101/255, 164/255],
    [245/255, 121/255, 0], [115/255, 210/255, 22/255]]

    return colors[label]

def plot_boxes(boxes):
    visuals =[] 
    for eachbox in boxes:
        box = eachbox['box'][None,:]
        label = eachbox['label']-1  # [0,1,2,3]
        inst_id = eachbox['name']
        corner = center_to_corner_box3d(box[:, :3], box[:, 3:6], -box[:, -1])[0].tolist()
        color = label2color(label)
        visuals.append(corners_to_lines(corner, color))
    return visuals

def vis_camera_image(frame):
    print(colored(f'Found {len(frame.images)} images','green'))
    for index, image in enumerate(frame.images):
        show_camera_image(image, frame.camera_labels, [3, 3, index+1])

def vis_range_image(frame, range_images):
    frame.lasers.sort(key=lambda laser: laser.name)
    show_range_image(get_range_image(range_images,open_dataset.LaserName.TOP, 0), 1)
    show_range_image(get_range_image(range_images,open_dataset.LaserName.TOP, 1), 4)

def vis_lidar(sequences):
    for c_sequence in sequences:
        dataset = tf.data.TFRecordDataset(c_sequence, compression_type='')

        for data in dataset:
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))

            (range_images, camera_projections,range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)

            points, cp_points = frame_utils.convert_range_image_to_point_cloud(frame,range_images,camera_projections,range_image_top_pose)
            points_all = np.concatenate(points, axis=0)
            pos_indice = []
            for idx, pts in enumerate(points):
                pos_indice.append(np.ones(pts.shape[0]) * idx)
            pos_indice = np.concatenate(pos_indice, axis = 0).astype(np.int32)

            pcd_all = to_o3d_pcd(points_all)
            pcd_all.colors = to_o3d_vec(MY_CMAP[pos_indice])

            pcd_top = to_o3d_pcd(points[0])
            pcd_top.paint_uniform_color(get_blue())

            multi_vis([pcd_all, pcd_top],['all points', 'Top LiDAR'], render=True)
            break

def vis_bbox_pcd(sequences):
    for c_sequence in sequences:
        dataset = tf.data.TFRecordDataset(c_sequence, compression_type='')

        for data in dataset:
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))

            (range_images, camera_projections,range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)

            #extract points
            points, _ = frame_utils.convert_range_image_to_point_cloud(frame,range_images,camera_projections,range_image_top_pose)
            points_all = np.concatenate(points, axis=0)
            pos_indice = []
            for idx, pts in enumerate(points):
                pos_indice.append(np.ones(pts.shape[0]) * idx)
            points_indice = np.concatenate(pos_indice, axis = 0).astype(np.int32)
            points_top = points[0]

            pcd_all = to_o3d_pcd(points_all)
            pcd_all.colors = to_o3d_vec(MY_CMAP[points_indice])

            pcd_top = to_o3d_pcd(points[0])
            pcd_top.paint_uniform_color(get_blue())

            # lidars = extract_points(frame.lasers, frame.context.laser_calibrations, frame.pose)
            # points = lidars['points_xyz']
            # pcd_top = to_o3d_pcd(points)
            # pcd_top.paint_uniform_color(get_blue())


            # extract bbox 
            veh_to_global = np.array(frame.pose.transform)
            ref_pose = np.reshape(np.array(frame.pose.transform), [4, 4])
            global_from_ref_rotation = ref_pose[:3, :3] 
            objects = extract_objects(frame.laser_labels, global_from_ref_rotation)

            # plot bbox and point cloud, here be careful with the yaw angle
            bbox = plot_boxes(objects)
            visuals = [pcd_top]
            visuals+=bbox
            vis_o3d(visuals,render=True)

            break

def get_ground_height(sequences):
    for c_sequence in sequences:
        dataset = tf.data.TFRecordDataset(c_sequence, compression_type='')

        for data in dataset:
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))

            (range_images, camera_projections,range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)

            #extract points
            points, _ = frame_utils.convert_range_image_to_point_cloud(frame,range_images,camera_projections,range_image_top_pose)
            points_ground = np.concatenate(points[1:], axis=0)
            # heights = points_ground[:,2]
            # ground_height = np.quantile(heights, 0.5)
            dist = np.linalg.norm(points_ground[:,:2], axis=1)
            ground_height = points_ground[dist < 3, 2].mean()

            pcd = to_o3d_pcd(points_ground[points_ground[:,2] < ground_height])
            pcd.paint_uniform_color(get_blue())
            vis_o3d([pcd],render=True, window_name=f'Ground points, height: {round(ground_height, 4)}')
            break

def get_circle_radius(sequences):
    for c_sequence in sequences:
        dataset = tf.data.TFRecordDataset(c_sequence, compression_type='')

        for data in dataset:
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))

            (range_images, camera_projections,range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)

            #extract points
            points, _ = frame_utils.convert_range_image_to_point_cloud(frame,range_images,camera_projections,range_image_top_pose)
            points_top = points[0]
            radius = np.linalg.norm(points_top[:,:2], axis=1)
            print(radius.min())


def analyse_bbox_velocity(sequences):
    for c_sequence in sequences:
        dataset = tf.data.TFRecordDataset(c_sequence, compression_type='')

        boundingboxes = dict()

        for data in dataset:
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))

            (range_images, camera_projections,range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)

            # extract bbox 
            veh_to_global = np.array(frame.pose.transform)
            ref_pose = np.reshape(np.array(frame.pose.transform), [4, 4])
            global_from_ref_rotation = ref_pose[:3, :3] 
            boxes = extract_objects(frame.laser_labels, global_from_ref_rotation)

            for eachbox in boxes:
                name = eachbox['name']
                speed = np.linalg.norm(eachbox['global_speed'])
                if(name in boundingboxes):
                    boundingboxes[name].append(speed)
                else:
                    boundingboxes[name]=[speed]

def vis_points_in_bbox(sequences):
    for c_sequence in sequences:
        dataset = tf.data.TFRecordDataset(c_sequence, compression_type='')

        for data in dataset:
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))

            (range_images, camera_projections,range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)

            #extract points
            points, _ = frame_utils.convert_range_image_to_point_cloud(frame,range_images,camera_projections,range_image_top_pose)
            points_all = np.concatenate(points, axis=0)
            pos_indice = []
            for idx, pts in enumerate(points):
                pos_indice.append(np.ones(pts.shape[0]) * idx)
            points_indice = np.concatenate(pos_indice, axis = 0).astype(np.int32)
            points_top = points[0]

            pcd_top = to_o3d_pcd(points[0])

            # extract bbox 
            veh_to_global = np.array(frame.pose.transform)
            ref_pose = np.reshape(np.array(frame.pose.transform), [4, 4])
            global_from_ref_rotation = ref_pose[:3, :3] 
            objects = extract_objects(frame.laser_labels, global_from_ref_rotation)

            # index points by bbox it belongs to 
            bboxes = []
            for eachbox in objects:
                bboxes.append(eachbox['box'])
            bboxes = np.array(bboxes)
            bboxes[:,-1] = - bboxes[:,-1]

            indices = points_in_rbbox(points_top, bboxes).astype(np.int)  
            indices = np.hstack([np.ones((indices.shape[0],1)) * 0.5,indices])
            ind_bbox = indices.argmax(1)  

            colors = color_dict[ind_bbox % 98]
            colors[ind_bbox==0] = get_blue()
            pcd_top.colors = to_o3d_vec(colors)

            # plot bbox and point cloud, here be careful with the yaw angle
            bbox = plot_boxes(objects)
            visuals = [pcd_top]
            visuals+=bbox
            vis_o3d(visuals,render=True)
            break



if __name__=='__main__':
    data_folder = '/scratch3/waymo/train/*.tfrecord'
    sequences = glob(data_folder)
    print(colored('Found %d sequences' % len(sequences),'green'))
    np.random.shuffle(sequences)
    # vis_lidar(sequences)
    vis_bbox_pcd(sequences)
    #analyse_bbox_velocity(sequences)
    # get_ground_height(sequences)
    # get_circle_radius(sequences)
    # vis_points_in_bbox(sequences)
            # analyse_bbox_velocity(frame)
            # break

            # # vis_bbox_pcd(frame, range_images, camera_projections, range_image_top_pose)
            # # break
            # #vis_lidar(frame, range_images, camera_projections, range_image_top_pose)

            #     # # camera projection corresponding to each point.
            #     # cp_points_all = np.concatenate(cp_points, axis=0)
            #     # cp_points_all_ri2 = np.concatenate(cp_points_ri2, axis=0)


            #     # images = sorted(frame.images, key=lambda i:i.name)
            #     # cp_points_all_concat = np.concatenate([cp_points_all, points_all], axis=-1)
            #     # cp_points_all_concat_tensor = tf.constant(cp_points_all_concat)

            #     # # The distance between lidar points and vehicle frame origin.
            #     # points_all_tensor = tf.norm(points_all, axis=-1, keepdims=True)
            #     # cp_points_all_tensor = tf.constant(cp_points_all, dtype=tf.int32)

            #     # mask = tf.equal(cp_points_all_tensor[..., 0], images[0].name)

            #     # cp_points_all_tensor = tf.cast(tf.gather_nd(
            #     #     cp_points_all_tensor, tf.where(mask)), dtype=tf.float32)
            #     # points_all_tensor = tf.gather_nd(points_all_tensor, tf.where(mask))

            #     # projected_points_all_from_raw_data = tf.concat(
            #     #     [cp_points_all_tensor[..., 1:3], points_all_tensor], axis=-1).numpy()

            #     # plot_points_on_image(projected_points_all_from_raw_data,
            #     #          images[0], rgba, point_size=5.0)