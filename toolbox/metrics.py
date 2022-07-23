from toolbox.timer import AverageMeter, AverageMeterArray
import numpy as np
from toolbox.utils import _EPS

def update_stats_meter(stats_meter, stats):
    """
    Update stats meter from each batch
    """
    # make sure state_meter contain all keys of the stats
    for key, value in stats.items():
        if key not in stats_meter.keys():
            if(isinstance(value, dict)):
                stats_meter[key] = init_stats_meter(value)
            elif(isinstance(value, np.ndarray)):
                stats_meter[key] = AverageMeterArray(value)
            else:
                stats_meter[key]=AverageMeter()

    for key, value in stats.items():
        if(isinstance(value, dict)):
            update_stats_meter(stats_meter[key],value)
        else:
            if isinstance(value, list):
                stats_meter[key].update(value[0], value[1])
            else:
                stats_meter[key].update(value)

def init_stats_meter(stats):
    """
    Initialise the global stats meter
    """
    meters=dict()
    for key,value in stats.items():
        if(isinstance(value, dict)):
            meters[key] = init_stats_meter(value)
        elif(isinstance(value, np.ndarray)):
            meters[key] = AverageMeterArray(value)
        else:
            meters[key]=AverageMeter()
    return meters


def compute_mean_iou_recall_precision(stats, mapping):
    """
    Given a stats meter, we compute IoU, Recall and Precision
    """
    new_stats = dict()
    iou = stats['intersection'].sum / (stats['union'].sum + _EPS)
    recall =  stats['intersection'].sum / (stats['gt_positives'].sum + _EPS)
    precision = stats['intersection'].sum  / (stats['pred_positives'].sum + _EPS)

    message = ''
    for idx, ele in enumerate(mapping):
        c_iou, c_recall, c_precision = round(iou[idx],3), round(recall[idx], 3), round(precision[idx], 3)
        message += f'{ele}:  IoU: {c_iou},  Recall: {c_recall},  Precision: {c_precision} \n'

    assert len(mapping) == iou.shape[0]
    new_stats['iou'] = iou.mean()
    new_stats['recall'] = recall.mean()
    new_stats['precision'] = precision.mean()
    return new_stats, message