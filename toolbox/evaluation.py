import os, sys, torch, nestargs
from glob import glob
import numpy as np
from toolbox.utils import to_tensor, makedirs, save_pkl, load_pkl
from toolbox.metrics import init_stats_meter, update_stats_meter
from toolbox.sf_eval_utils import compute_sf_metrics_torch, collect_scene_stats
from tqdm import tqdm


SAMPLE_FREQ = {
    'waymo': 4,
    'nuscene': 1
}

N_FRAMES = {
    'waymo': 5,
    'nuscene': 11
}

def collect_results(target_folder, save_dir, dataset):
    files = glob(target_folder + '/*/flow_error.npz')
    stats_meter = None
    
    scene_stats = dict()
    relative_error_list, epe_3d_error_list = [], []
    for eachfile in tqdm(files):
        data = np.load(eachfile)
        # convert to tensor
        data_torch = dict()
        for key, value in data.items():
            data_torch[key] = to_tensor(value).cuda()
            
        fb_label, sd_label = data_torch['fb_label'], data_torch['sd_label']
        epe_per_point = data_torch['epe_per_point'].float()
        relative_error = data_torch['relative_error'].float()
        
        # adjust time indice
        if 'length' in data_torch.keys():
            ts = data_torch['time_indice']
            length = data_torch['length']
            n_frames = length.size(0)
            running_idx = 0
            time_indice = torch.zeros_like(fb_label).long()
            for idx in range(n_frames):
                c_length, c_ts = length[idx], ts[idx]
                time_indice[running_idx:running_idx+c_length] = c_ts
                running_idx+=c_length
        else:
            time_indice = data_torch['time_indice']
        
        # store the dynamic part
        sel = sd_label == 1
        if sel.sum():
            relative_error_list.extend(relative_error[sel].half()[::SAMPLE_FREQ[dataset]].cpu())
            epe_3d_error_list.extend(epe_per_point[sel].half()[::SAMPLE_FREQ[dataset]].cpu())
        
        c_metrics = dict()
        # compute for the static part
        c_metrics['scene_overall'] =compute_sf_metrics_torch(epe_per_point, relative_error)
        sel = sd_label == 0
        c_metrics['static_overall'] = compute_sf_metrics_torch(epe_per_point[sel], relative_error[sel])

        sel = torch.logical_and(sd_label==0, fb_label ==0)
        c_metrics['static_BG'] = compute_sf_metrics_torch(epe_per_point[sel], relative_error[sel])
        sel = torch.logical_and(sd_label==0, fb_label ==1)
        if sel.sum():
            c_metrics['static_FG'] = compute_sf_metrics_torch(epe_per_point[sel], relative_error[sel])
        
        n_frames = time_indice.max().item() + 1
        # for t_idx in range(1, N_FRAMES[dataset]):
        for t_idx in range(1, n_frames):
            sel = torch.logical_and(sd_label==0, time_indice == t_idx)
            c_metrics[f'{t_idx}-th frame'] = compute_sf_metrics_torch(epe_per_point[sel], relative_error[sel])

        if(stats_meter is None):
            stats_meter = init_stats_meter(c_metrics)
        update_stats_meter(stats_meter, c_metrics)
        
        # collect scene stat
        scene_name = eachfile.split('/')[-2]
        scene_stats[scene_name] = collect_scene_stats(epe_per_point, relative_error,sd_label, fb_label)
        
    epe_per_point = torch.Tensor(epe_3d_error_list)
    relative_error =  torch.Tensor(relative_error_list)
    
    makedirs(save_dir)
    # save the intermediate for dynamic part
    dyanamic_dict = {
        'relative_error': relative_error,
        'epe_per_point': epe_per_point
    }
    torch.save(dyanamic_dict, f'{save_dir}/dynamic_dict.pth')
    
    # save the scene stats
    save_pkl(scene_stats, f'{save_dir}/scene_stats.pkl')
    
    # save the static results
    save_pkl(stats_meter, f'{save_dir}/static_stats.pkl')   

if __name__=='__main__':
    argparser = nestargs.NestedArgumentParser() 
    argparser.add_argument('path', type=str, help= 'Path to the results')
    argparser.add_argument('dataset', type=str, help= 'dataset')
    args, extra_args = argparser.parse_known_args()

    assert os.path.exists(args.path)
    save_dir = args.path.replace('results','metrics')

    collect_results(args.path, save_dir, args.dataset)

    static_stats = load_pkl(f'{save_dir}/static_stats.pkl')
    print('Results on the static BG part')
    print(round(static_stats['static_BG']['EPE3D'].avg, 3), round(static_stats['static_BG']['Acc3DS'].avg * 100, 1), round(static_stats['static_BG']['Acc3DR'].avg * 100, 1), round(static_stats['static_BG']['ROutlier'].avg * 100, 1))

    print('Results on the static FG part')
    print(round(static_stats['static_FG']['EPE3D'].avg, 3), round(static_stats['static_FG']['Acc3DS'].avg * 100, 1), round(static_stats['static_FG']['Acc3DR'].avg * 100, 1), round(static_stats['static_FG']['ROutlier'].avg * 100, 1))
    
    print('Results on the static part')
    print(round(static_stats['static_overall']['EPE3D'].avg, 3), round(static_stats['static_overall']['Acc3DS'].avg * 100, 1), round(static_stats['static_overall']['Acc3DR'].avg * 100, 1), round(static_stats['static_overall']['ROutlier'].avg * 100, 1))

    dynamic_results = torch.load(f'{save_dir}/dynamic_dict.pth')
    relative_error, epe_per_point = dynamic_results['relative_error'], dynamic_results['epe_per_point']
    sf_dynamics =compute_sf_metrics_torch(epe_per_point, relative_error)
    print('Results on the dynamic part')
    print(round(sf_dynamics['EPE3D'][0], 3), round(sf_dynamics['EPE3D_med'], 3), round(sf_dynamics['Acc3DS'][0] * 100, 1), round(sf_dynamics['Acc3DR'][0] * 100, 1), round(sf_dynamics['ROutlier'][0] * 100, 1))