import os, torch
from random import sample
from re import L
import numpy as np
from libs.dataset import DatasetSampler, NuSceneDataset, WaymoDataset
from collections import defaultdict


def collate_fn(batch):
    example_merged = defaultdict(list)
    for example in batch:
        for k,v in example.items():
            example_merged[k].append(v)
    results = dict()
    for key, elems in example_merged.items():
        if key in ['coordinates','time_indice']:  # pad batch indice
            samples = []
            for i, ele in enumerate(elems):
                n_sample = ele.shape[0]
                batch_idx = np.ones((n_sample, 1)) * i
                samples.append(np.concatenate((batch_idx, ele), axis = 1))
            results[key] = torch.tensor(np.concatenate(samples, axis=0))
        elif key in ['ego_motion_gt','shape',]:
            results[key] = torch.tensor(np.stack(elems, axis = 0))
        elif key in ['inst_motion_gt']:
            results[key] = [torch.tensor(ele) for ele in elems]
        elif key == 'data_path':
            results[key] = elems
        else:
            results[key] = torch.tensor(np.concatenate(elems, axis=0))
    
    num_points = results['num_points']
    num_voxels = results['num_voxels']
    batch_size = num_voxels.size(0)
    running_idx = 0
    n_accumulated = 0
    for batch_idx in range(batch_size):
        results['point_to_voxel_map'][running_idx:running_idx+num_points[batch_idx]] += n_accumulated
        running_idx += num_points[batch_idx]
        n_accumulated += num_voxels[batch_idx]
        
    return results            
   

DATASETS = {
    'nuscene': NuSceneDataset,
    'waymo': WaymoDataset
}

def get_dataloader(config):
    DATASET = DATASETS[config['data']['dataset']]

    train_set =DATASET(config, 'train', data_augmentation=True)
    val_set = DATASET(config,'val',  data_augmentation=False)
    test_set = DATASET(config,'test', data_augmentation=False)

    config['train']['train_loader'] = torch.utils.data.DataLoader(train_set, 
                                        batch_size=config['train']['batch_size'], 
                                        num_workers=config['train']['num_workers'],
                                        collate_fn=collate_fn,
                                        sampler = DatasetSampler(train_set),
                                        pin_memory=False,
                                        drop_last=True)
    config['train']['val_loader'] = torch.utils.data.DataLoader(val_set, 
                                    batch_size=config['val']['batch_size'], 
                                    num_workers=config['val']['num_workers'],
                                    collate_fn=collate_fn,
                                    sampler = DatasetSampler(val_set),
                                    pin_memory=False,
                                    drop_last=True)
    config['train']['test_loader'] = torch.utils.data.DataLoader(test_set, 
                                    batch_size=1, 
                                    num_workers=config['val']['num_workers'],
                                    collate_fn=collate_fn,
                                    sampler = DatasetSampler(test_set),
                                    pin_memory=False,
                                    drop_last=False)

    return config