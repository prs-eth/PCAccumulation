import torch
import torch.nn as nn
import numpy as np
from libs.utils import _EPS, save_pkl, load_pkl

from IPython.display import display
import pandas as pd


def display_results(path):
    
    def disp_df(index, data, category, message):
        keys = list(data['overall'][category].keys())
        data_dict = dict()
        for key in keys:
            data_dict[key] = [data[ele][category][key] for ele in index]
        df = pd.DataFrame(data_dict, index = index)
        df = df.round(decimals = 3)
        print(message)
        display(df)
        print('\n') 
        
    data = load_pkl(path)
    index = list(data.keys())
    disp_df(index, data, 'overall','Overall results')
    disp_df(index, data, 'BG', 'Detailed results on BG part')
    disp_df(index, data, 'FG', 'Detailed results on FG part')
    disp_df(index, data, 'Static', 'Detailed results on static part')
    disp_df(index, data, 'Dynamic', 'Detailed results on dynamic part')
    disp_df(index, data, 'percentile', 'Detailed results on dynamic part by percentile')
    

def display_from_stats_meter(stats_meter):
    def disp_df(index, data, category, message):
        keys = list(data['overall'][category].keys())
        data_dict = dict()
        for key in keys:
            data_dict[key] = [data[ele][category][key].avg for ele in index]
        df = pd.DataFrame(data_dict, index = index)
        df = df.round(decimals = 3)
        print(message)
        display(df)
        print('\n') 

    index = list(stats_meter.keys())
    disp_df(index, stats_meter, 'overall','Overall results')
    disp_df(index, stats_meter, 'BG', 'Detailed results on BG part')
    disp_df(index, stats_meter, 'FG', 'Detailed results on FG part')
    disp_df(index, stats_meter, 'Static', 'Detailed results on static part')
    disp_df(index, stats_meter, 'Dynamic', 'Detailed results on dynamic part')
    disp_df(index, stats_meter, 'percentile', 'Detailed results on dynamic part by percentile')
        

def compute_sf_metrics_torch(epe_per_point, relative_error):
    epe_3d_mean = epe_per_point.mean().item()
    epe_3d_median = torch.median(epe_per_point).item()
    acc3ds = torch.logical_or(epe_per_point < 0.05, relative_error < 0.05).float().mean().item()
    acc3dr = torch.logical_or(epe_per_point < 0.1, relative_error < 0.1).float().mean().item()
    outlier = torch.logical_or(epe_per_point > 0.3, relative_error > 0.1).float().mean().item()
    ROutlier = torch.logical_and(epe_per_point > 0.3, relative_error > 0.3).float().mean().item()
    size = epe_per_point.size(0)
    return {
        'EPE3D': [epe_3d_mean, size],
        'EPE3D_med': epe_3d_median,
        'Acc3DR': [acc3dr, size],
        'Acc3DS': [acc3ds, size],
        'Outlier': [outlier, size],
        'ROutlier': [ROutlier, size],
    }

def compute_flow_error_torch(gt_flow, est_flow, fb_label, sd_label, mask = None):
    """
    Compute 3d end-point-error

    Args:
        st_flow (torch.Tensor): estimated flow vectors [n,3]
        gt_flow  (torch.Tensor): ground truth flow vectors [n,3]
        flownet3d_eval (bool): compute the evaluation stats as defined in FlowNet3D
        mask (torch.Tensor): boolean mask used for filtering the epe [n]
        fb_label: foreground / background label
        sd_label: static / dynamic label

    Returns:
        epe (float): mean EPE for current batch
        epe_bckg (float): mean EPE for the background points
        epe_forg (float): mean EPE for the foreground points
        epe_static (float): mean EPE for the static points
        epe_dynamic (float): mean EPE for the dynamic points
        aee_5_5 (float): avg(epe_static, epe_dynamic)
        acc3d_strict (float): inlier ratio according to strict thresh (error smaller than 5cm or 5%)
        acc3d_relax (float): inlier ratio according to relaxed thresh (error smaller than 10cm or 10%)
        outlier (float): ratio of outliers (error larger than 30cm or 10%)
        ROutlier (float): ratio of outliers (error larger than 30cm and > 30%) follow SLIM setting
    """
    metrics = {}
    if mask is not None:
        est_flow, gt_flow = est_flow[mask], gt_flow[mask]
        fb_label, sd_label = fb_label[mask], sd_label[mask]

    error = est_flow - gt_flow
    epe_per_point = torch.norm(error, p=2, dim=1)
    gt_f_magnitude = torch.norm(gt_flow, p=2, dim=1)
    relative_err = epe_per_point / (gt_f_magnitude + _EPS)
    
    metrics['moving_ratio'] = sd_label.float().mean().item()
    metrics['FG_ratio'] = fb_label.float().mean().item()
    
    metrics['overall'] = compute_sf_metrics_torch(epe_per_point,relative_err)

    bckg_mask = (fb_label == 0)
    metrics['BG'] = compute_sf_metrics_torch(epe_per_point[bckg_mask],relative_err[bckg_mask])
    
    forg_mask = (fb_label == 1)
    if forg_mask.sum():
        metrics['FG'] =compute_sf_metrics_torch(epe_per_point[forg_mask],relative_err[forg_mask])
    
    static_mask = sd_label == 0
    metrics['Static'] = compute_sf_metrics_torch(epe_per_point[static_mask], relative_err[static_mask])
    
    dynamic_mask = sd_label == 1
    if dynamic_mask.sum():
        metrics['Dynamic'] = compute_sf_metrics_torch(epe_per_point[dynamic_mask], relative_err[dynamic_mask])
    
        # detailed results for the dynamic part
        percentiles = SF_Evaluator.get_percentile(epe_per_point[dynamic_mask].cpu().numpy()) 
        metrics['percentile'] = {
            '10%': percentiles[10],
            '25%': percentiles[25],
            '50%': percentiles[50],
            '75%': percentiles[75],
            '90%': percentiles[90],
        }
    return metrics

class SF_Evaluator():
    """
    scene flow evaluator
    """
    def __init__(self, n_frames, save_dir):
        self.n_frames = n_frames
        self.flow_error_list = []
        self.relative_errro_list = []
        self.fb_label_list = []
        self.sd_label_list = []
        self.time_indice_list = []
        self.save_dir = save_dir

    def update(self, gt_flow, est_flow, time_indice, fb_label, sd_label, mask = None):
        if mask is not None:
            est_flow, gt_flow = est_flow[mask], gt_flow[mask]
            fb_label, sd_label = fb_label[mask], sd_label[mask]

        error = est_flow - gt_flow
        epe_per_point = torch.norm(error, p=2, dim=1)
        gt_f_magnitude = torch.norm(gt_flow, p=2, dim=1)
        relative_err = epe_per_point / (gt_f_magnitude + _EPS)

        self.flow_error_list.extend(epe_per_point.cpu().numpy().astype(np.float16))
        self.relative_errro_list.extend(relative_err.cpu().numpy().astype(np.float16))
        self.fb_label_list.extend(fb_label.cpu().numpy().astype(np.bool))
        self.sd_label_list.extend(sd_label.cpu().numpy().astype(np.bool))
        self.time_indice_list.extend(time_indice.cpu().numpy().astype(np.int8))
        
    @staticmethod
    def get_percentile(data):
        percentile_tags = [5,10,25,50,75,90,95]
        percentiles = [round(np.percentile(data,[ele])[0],3) for ele in percentile_tags]

        results = dict()
        for idx, key in enumerate(percentile_tags):
            results[key] = percentiles[idx]

        return results
    
    @staticmethod
    def compute_sf_metrics(epe_per_point, relative_error):
        epe_3d_mean = epe_per_point.mean().astype(np.float64)
        epe_3d_median = np.median(epe_per_point).astype(np.float64)
        acc3ds = np.logical_or(epe_per_point < 0.05, relative_error < 0.05).mean()
        acc3dr = np.logical_or(epe_per_point < 0.1, relative_error < 0.1).mean()
        outlier = np.logical_or(epe_per_point > 0.3, relative_error > 0.1).mean()
        ROutlier = np.logical_and(epe_per_point > 0.3, relative_error > 0.3).mean()
        return {
            'EPE3D': epe_3d_mean,
            'EPE3D_med': epe_3d_median,
            'Acc3DR': acc3dr,
            'Acc3DS': acc3ds,
            'Outlier': outlier,
            'ROutlier': ROutlier
        }
        
    def sf_evaluate(self, fb_label, sd_label, epe_per_point, relative_err):
        metrics = {}
        assert fb_label.shape[0] == sd_label.shape[0] == epe_per_point.shape[0] == relative_err.shape[0]
        metrics['n_points'] = fb_label.shape[0]
        metrics['moving_ratio'] = sd_label.mean()
        metrics['FG_ratio'] = fb_label.mean()
        
        metrics['overall'] = self.compute_sf_metrics(epe_per_point,relative_err)

        bckg_mask = (fb_label == 0)
        metrics['BG'] = self.compute_sf_metrics(epe_per_point[bckg_mask],relative_err[bckg_mask])
        
        forg_mask = (fb_label == 1)
        metrics['FG'] =self.compute_sf_metrics(epe_per_point[forg_mask],relative_err[forg_mask])
        
        static_mask = sd_label == 0
        metrics['Static'] = self.compute_sf_metrics(epe_per_point[static_mask], relative_err[static_mask])
        
        dynamic_mask = sd_label == 1
        metrics['Dynamic'] = self.compute_sf_metrics(epe_per_point[dynamic_mask], relative_err[dynamic_mask])
        
        # detailed results for the dynamic part
        percentiles = self.get_percentile(epe_per_point[dynamic_mask]) 
        metrics['percentile'] = {
            '10%': percentiles[10],
            '25%': percentiles[25],
            '50%': percentiles[50],
            '75%': percentiles[75],
            '90%': percentiles[90],
        }
        return metrics

    
    def full_evaluation(self):
        results = {}
        fb_label = np.array(self.fb_label_list)
        sd_label = np.array(self.sd_label_list)
        epe_per_point = np.array(self.flow_error_list)
        relative_err = np.array(self.relative_errro_list)
        time_indice = np.array(self.time_indice_list)
    
        results['overall'] = self.sf_evaluate(fb_label, sd_label, epe_per_point, relative_err)

        for idx in range(1, self.n_frames):
            sel = time_indice == idx
            results[f'{idx}-th frame'] = self.sf_evaluate(fb_label[sel], sd_label[sel], epe_per_point[sel], relative_err[sel])

        save_pkl(results, f'{self.save_dir}/sf_results.pkl')
        display_results(f'{self.save_dir}/sf_results.pkl')