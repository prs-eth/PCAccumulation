"""
For Waymo,      we sample 80 scenes for training, 20 scenes for validation, 20 scenes for test     5h/epoch
For full dataset, we use 550 for training, 224 for validation, 202 for test
"""
import numpy as np
import random, os
from libs.utils import natural_key, setup_seed, save_pkl
setup_seed(42)
import glob

if __name__=='__main__':
    train_scene = np.loadtxt('../dataset_configs/waymo/full/train_scene.txt',dtype=str)
    train_scene = [ele.split('/')[-1].split('.')[0] for ele in train_scene]
    val_scene = np.loadtxt('../dataset_configs/waymo/full/validation_scene.txt',dtype=str)
    val_scene = [ele.split('/')[-1].split('.')[0] for ele in val_scene]

    train_sample = sorted(os.listdir('/scratch3/waymo/customise/training'), key=natural_key)
    val_sample = sorted(os.listdir('/scratch3/waymo/customise/validation'), key=natural_key)


    n_train = 550
    n_val = 224
    n_test = 202
    random_idx = np.random.permutation(len(train_sample))
    train_idx = random_idx[:n_train]
    val_idx = random_idx[n_train:n_train+n_val]
    random_idx = np.random.permutation(len(val_sample))
    test_idx = random_idx[:n_test]

    train_scenes = np.array(train_scene)[train_idx]
    val_scenes = np.array(train_scene)[val_idx]
    test_scenes = np.array(val_scene)[test_idx]
    np.savetxt('configs/datasets/waymo/full_split/train.txt', train_scenes, fmt = '%s')
    np.savetxt('configs/datasets/waymo/full_split/val.txt', val_scenes, fmt = '%s')
    np.savetxt('configs/datasets/waymo/full_split/test.txt', test_scenes, fmt = '%s')

    # create path to sample
    train_samples = np.array(train_sample)[train_idx].tolist()
    val_samples = np.array(train_sample)[val_idx].tolist()
    test_samples = np.array(val_sample)[test_idx].tolist()
    
    train_anchors = []
    for ele in train_samples:
        files = sorted(glob.glob(f'/scratch3/waymo/customise/training/{ele}/lidar/*.npy'), key=natural_key)
        sampled_files = files[5::5]
        train_anchors.extend(sampled_files)
    np.savetxt('configs/datasets/waymo/full_split/meta_train.txt', train_anchors, fmt = '%s')

    val_anchors = []
    for ele in val_samples:
        files = sorted(glob.glob(f'/scratch3/waymo/customise/training/{ele}/lidar/*.npy'), key=natural_key)
        sampled_files = files[5::5]
        val_anchors.extend(sampled_files)
    np.savetxt('configs/datasets/waymo/full_split/meta_val.txt', val_anchors, fmt = '%s')

    test_anchors = []
    for ele in test_samples:
        files = sorted(glob.glob(f'/scratch3/waymo/customise/validation/{ele}/lidar/*.npy'), key=natural_key)
        sampled_files = files[5::5]
        test_anchors.extend(sampled_files)
    np.savetxt('configs/datasets/waymo/full_split/meta_test.txt', test_anchors, fmt = '%s')

    # create mapping from index to scene name
    train_map, val_map, test_map = dict(), dict(), dict()
    for idx, ele in enumerate(train_samples):
        train_map[ele] = train_scenes[idx]
    for idx,ele in enumerate(val_samples):
        val_map[ele] = val_scenes[idx]
    for idx, ele in enumerate(test_samples):
        test_map[ele] = test_scenes[idx]    
    mapping = {
        'train': train_map,
        'val': val_map,
        'test': test_map
    }
    save_pkl(mapping, 'configs/datasets/waymo/full_split/mapping.pkl')
