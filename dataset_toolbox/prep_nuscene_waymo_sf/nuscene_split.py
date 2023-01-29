from libs.utils import load_pkl, save_pkl
from tqdm import tqdm
import random
from libs.utils import setup_seed
setup_seed(42)

"""
NuScenes provide 700 training scenes and 150 validation scenes
We use 550/150 scenes from training set as train/val split
   use 150 scenes from validation as hold-out test split
   For each scene, we take all samples that have 10 sweeps
"""

if __name__=='__main__':
    path = '/net/pf-pc04/scratch2/shengyu/datasets/nuscene/v1.0-trainval/infos_train.pkl'
    data_ours = load_pkl(path)['infos']

    path = '/net/pf-pc04/scratch2/shengyu/datasets/nuscene/v1.0-trainval/nuscenes_infos_train.pkl'
    data_used = load_pkl(path)['infos']

    n_samples = len(data_used)
    for idx in tqdm(range(n_samples)):
        assert data_ours[idx]['token'] == data_used[idx]['token']
        data_used[idx]['scene_token'] = data_ours[idx]['scene_token']
    
    train, val = [],[]
    scene_tokens = [ele['scene_token'] for ele in data_used]
    unique_scene_tokens = list(set(scene_tokens))
    random.shuffle(unique_scene_tokens)

    n_train = 550
    train_scenes = unique_scene_tokens[:n_train]
    val_scenes = unique_scene_tokens[n_train:]

    for idx in tqdm(range(n_samples)):
        scene_token = data_used[idx]['scene_token']
        n_sweeps = len(data_used[idx]['sweeps'])
        if n_sweeps==10:
            if scene_token in train_scenes:
                train.append(data_used[idx])
            elif scene_token in val_scenes:
                val.append(data_used[idx])
            else:
                print('Shit happens')

    print(len(train), len(val))
    save_pkl(train, 'configs/datasets/nuscene/train.pkl')
    save_pkl(val, 'configs/datasets/nuscene/val.pkl')



    path = '/net/pf-pc04/scratch2/shengyu/datasets/nuscene/v1.0-trainval/infos_val.pkl'
    data_ours = load_pkl(path)['infos']

    path = '/net/pf-pc04/scratch2/shengyu/datasets/nuscene/v1.0-trainval/nuscenes_infos_val.pkl'
    data_used = load_pkl(path)['infos']

    n_samples = len(data_used)
    for idx in tqdm(range(n_samples)):
        assert data_ours[idx]['token'] == data_used[idx]['token']
        data_used[idx]['scene_token'] = data_ours[idx]['scene_token']
    
    test = []
    for idx in tqdm(range(n_samples)):
        n_sweeps = len(data_used[idx]['sweeps'])
        if n_sweeps==10:
            test.append(data_used[idx])
    print(len(test))

    save_pkl(test, 'configs/datasets/nuscene/test.pkl')