import os
import torch.nn as nn
import numpy as np
import torch

"""
Adopted from https://github.com/WXinlong/ASIS/blob/master/models/ASIS/eval_iou_accuracy.py
"""

def log_string(LOG_FOUT, out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

class ClusterEvaluation(nn.Module):
    def __init__(self, cfg):
        super(ClusterEvaluation, self).__init__()
        self.num_classes = 2
        self.iou_threshold = [0.5, 0.6, 0.7, 0.8, 0.9]
        
        self.all_mean_cov = [[] for _ in range(self.num_classes)]
        self.all_mean_weighted_cov = [[] for _ in range(self.num_classes)]
        self.total_gt_inst = np.zeros(self.num_classes)   
        
        self.tpsins, self.fpsins = dict(), dict()
        for threshold in self.iou_threshold:
            self.tpsins[f'@{threshold}'] = [[] for _ in range(self.num_classes)]
            self.fpsins[f'@{threshold}'] = [[] for _ in range(self.num_classes)]
    
        self.LOG_FOUT = open(os.path.join(cfg['save_dir'], 'cluster_eval.txt'), 'a')
        
        
    def final_eval(self):
        # compute final stats
        MUCov = np.zeros(self.num_classes)
        MWCov = np.zeros(self.num_classes)
        for sem_idx in range(self.num_classes):
            MUCov[sem_idx] = np.mean(self.all_mean_cov[sem_idx])
            MWCov[sem_idx] = np.mean(self.all_mean_weighted_cov[sem_idx])
        
        precision = np.zeros(self.num_classes)
        recall = np.zeros(self.num_classes)
        
        log_string(self.LOG_FOUT,'Instance Segmentation MUCov: {}'.format(MUCov))
        log_string(self.LOG_FOUT,'Instance Segmentation mMUCov: {}'.format(np.mean(MUCov)))
        log_string(self.LOG_FOUT,'Instance Segmentation MWCov: {}'.format(MWCov))
        log_string(self.LOG_FOUT,'Instance Segmentation mMWCov: {}'.format(np.mean(MWCov)))
        
        for threshold in self.iou_threshold:
            for sem_idx in range(self.num_classes):
                tp = np.asarray(self.tpsins[f'@{threshold}'][sem_idx]).astype(np.float)
                fp = np.asarray(self.fpsins[f'@{threshold}'][sem_idx]).astype(np.float)
                tp = np.sum(tp)
                fp = np.sum(fp)
                rec = tp / self.total_gt_inst[sem_idx]
                prec = tp / (tp + fp)

                precision[sem_idx] = prec
                recall[sem_idx] = rec

            # instance results
            log_string(self.LOG_FOUT, f'IoU threshold @{threshold}')
            log_string(self.LOG_FOUT,'Instance Segmentation Precision: {}'.format(precision))
            log_string(self.LOG_FOUT,'Instance Segmentation mPrecision: {}'.format(np.mean(precision)))
            log_string(self.LOG_FOUT,'Instance Segmentation Recall: {}'.format(recall))
            log_string(self.LOG_FOUT,'Instance Segmentation mRecall: {}'.format(np.mean(recall)))
        
        log_string(self.LOG_FOUT,'\n')
            
        
    def forward(self, inst_est, inst_gt, mos_label):
        """Evaluation of the clustering 
        Args:
            inst_est (array):   [N]  0: background
            inst_gt (array):    [N]  0: background
            mos_label (array):  [N]  0: static,   1: dynamic
        """
        # get instances from ground truth and prediction, respectively
        unique_est_inst = torch.unique(inst_est)
        pts_in_est_inst = [[] for _ in range(self.num_classes)]
        for idx, unique_id in enumerate(unique_est_inst): 
            if unique_id == 0:  #ignore background points
                continue 
            tmp = inst_est == unique_id
            sem_idx = round(mos_label[tmp].mean().item())
            pts_in_est_inst[sem_idx] +=[tmp]
        
        unique_gt_inst = torch.unique(inst_gt)
        pts_in_gt_inst = [[] for _ in range(self.num_classes)]
        for idx, unique_id in enumerate(unique_gt_inst):
            if unique_id == 0:
                continue
            tmp = inst_gt == unique_id
            sem_idx = round(mos_label[tmp].mean().item())
            pts_in_gt_inst[sem_idx] += [tmp]
            
        # compute instance mucov and mwcov
        for sem_idx in range(self.num_classes):
            sum_cov = 0
            mean_cov = 0
            mean_weighted_cov = 0
            num_gt_point = 0
            
            for idx, c_inst_gt in enumerate(pts_in_gt_inst[sem_idx]):
                ovmax = 0.
                num_inst_gt_point = c_inst_gt.sum().item()
                num_gt_point += num_inst_gt_point
                for idx_est, c_inst_est in enumerate(pts_in_est_inst[sem_idx]):
                    union = (c_inst_gt | c_inst_est)
                    intersect = (c_inst_gt & c_inst_est)
                    iou = float(intersect.sum() / union.sum())
                    
                    if iou > ovmax:  # get the max IoU
                        ovmax = iou
                
                sum_cov += ovmax
                mean_weighted_cov += ovmax * num_inst_gt_point
            
            n_inst = len(pts_in_gt_inst[sem_idx])
            if n_inst:
                mean_cov = sum_cov / n_inst
                self.all_mean_cov[sem_idx].append(mean_cov)
                mean_weighted_cov /= num_gt_point
                self.all_mean_weighted_cov[sem_idx].append(mean_weighted_cov)
                
        # instance precision & recall
        for sem_idx in range(self.num_classes):
            tp, fp = dict(), dict()
            for threshold in self.iou_threshold:
                tp[f'@{threshold}'] = [0.] * len(pts_in_est_inst[sem_idx])
                fp[f'@{threshold}'] = [0.] * len(pts_in_est_inst[sem_idx])
                
            self.total_gt_inst[sem_idx] += len(pts_in_gt_inst[sem_idx]) 
            for idx, c_inst_est in enumerate(pts_in_est_inst[sem_idx]):
                ovmax = -1.
                for idx_gt, c_inst_gt in enumerate(pts_in_gt_inst[sem_idx]):
                    union  = (c_inst_est | c_inst_gt)
                    intersect = (c_inst_est & c_inst_gt)   
                    iou = float(intersect.sum() / union.sum())                 
                    
                    if iou > ovmax:
                        ovmax = iou
                
                for threshold in self.iou_threshold:
                    if ovmax > threshold:
                        tp[f'@{threshold}'][idx] = 1
                    else:
                        fp[f'@{threshold}'][idx] = 1 
                        
            for threshold in self.iou_threshold:
                self.tpsins[f'@{threshold}'][sem_idx] += tp[f'@{threshold}']
                self.fpsins[f'@{threshold}'][sem_idx] += fp[f'@{threshold}']