import os, torch, gc
from libs.utils import Logger,validate_gradient, _EPS, load_yaml, partial_load
from libs.metrics import init_stats_meter, update_stats_meter, compute_mean_iou_recall_precision

from tqdm import tqdm
from tensorboardX import SummaryWriter
import torch.nn as nn

class BaseTrainer(object):
    def __init__(self, args):
        self.config = args
        self.mode = args['misc']['mode']

        self.start_epoch = 1
        self.max_epoch = args['train']['max_epoch']
        self.save_dir = args['save_dir']
        self.device = args['instances']['device']

        self.model = args['instances']['model'].to(self.device)
        self.optimizer = args['instances']['optimizer']
        self.scheduler = args['instances']['scheduler']
        self.scheduler_freq = 1  # learning rate scheduler
        self.clip = args['train']['grad_clip']

        self.n_verbose = args['train']['n_verbose']
        self.iter_size = args['train']['iter_size']

        self.loss = args['train']['loss']
        self.mornitor_metric = args['train']['metric']

        self.best_loss = 1e5
        self.best_metric = -1e5

        self.loader =dict()
        self.loader['train']=args['train']['train_loader']
        self.loader['val']=args['train']['val_loader']
        self.loader['test'] = args['train']['test_loader']
        
        # save the logs
        self.logger = Logger(self.save_dir)
        self.logger.write(f'#parameters {sum([x.nelement() for x in self.model.parameters()])/1000000.} M\n')
        self.writer = SummaryWriter(log_dir=args['tboard_dir'])

        # load pretrain model
        if (args['misc']['pretrain'] !=''):
            self._load_pretrain(args['misc']['pretrain'])

        # save the model architecture
        with open(f'{self.save_dir}/model','w') as f:
            f.write(str(self.model))
        f.close()
        
        self._get_label_mapping(args)

    def _get_label_mapping(self, args):
        self.mos_mapping = load_yaml(args['data']['label_map'])['labels_mos']
        self.mos_mapping.pop(args['mos']['label_ignore'],None)
        self.mos_mapping = list(self.mos_mapping.values())
        
        self.fb_mapping = load_yaml(args['data']['label_map'])['labels_fb']
        self.fb_mapping.pop(args['fb']['label_ignore'],None)
        self.fb_mapping = list(self.fb_mapping.values())
        

    def _snapshot(self, epoch, name=None):
        """
        Save current model
        """
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'best_metric': self.best_metric
        }

        if name is None:
            filename = os.path.join(self.save_dir, f'model_{epoch}.pth')
        else:
            filename = os.path.join(self.save_dir, f'model_{name}.pth')
        self.logger.write(f"Save model to {filename}\n")
        torch.save(state, filename)

    def _load_pretrain(self, resume):
        """
        Load from pre-trained model
        """
        if os.path.isfile(resume):
            state = torch.load(resume)
            partial_load(state['state_dict'], self.model)
            
            #regressor_state = torch.load('snapshot/test_regression/model_best_loss.pth')
            #partial_load(regressor_state['state_dict'], self.model.reconstructor.alignment)
            
            # # self.model.load_state_dict(state['state_dict'])
            
            # self.start_epoch = state['epoch'] + 1 
            # self.scheduler.load_state_dict(state['scheduler'])
            # self.optimizer.load_state_dict(state['optimizer'])
            # self.best_loss = state['best_loss']
            # self.best_metric = state['best_metric']
             
            self.logger.write(f'Successfully load pretrained model from {resume} at epoch {self.start_epoch}!\n')
            self.logger.write(f'Current best loss {self.best_loss}\n')
            self.logger.write(f"Current best metric {self.best_metric}\n")
        else:
            raise ValueError(f"=> no checkpoint found at '{resume}'")

    def _get_lr(self, group=0):
        return self.optimizer.param_groups[group]['lr']

    def update_tensorboard(self, stats_meter, curr_iter, phase):
        """
        Update logs to tensorboard
        """
        stats, message = compute_mean_iou_recall_precision(stats_meter['mos_metric'], self.mos_mapping)
        for key, value in stats.items():
            self.writer.add_scalar(f'{phase}/mos_{key}', value, curr_iter)
            
        stats, message = compute_mean_iou_recall_precision(stats_meter['fb_metric'], self.fb_mapping)
        for key, value in stats.items():
            self.writer.add_scalar(f'{phase}/fb_{key}', value, curr_iter)

        for key, value in stats_meter.items():
            if(not isinstance(value, dict)):
                self.writer.add_scalar(f'{phase}/{key}', value.avg, curr_iter)

    def update_after_each_epoch(self, stats_meter, epoch, phase):
        """
        Update epoch-wise stats to both tensorboard and log file
        """
        message = f'{phase} Epoch: {epoch}\t'
        # 1. update to tensorboard
        stats, mos_iou_message = compute_mean_iou_recall_precision(stats_meter['mos_metric'], self.mos_mapping)
        for key, value in stats.items():
            self.writer.add_scalar(f'g_{phase}/mos_{key}', value, epoch)    

        for key, value in stats.items():
            message += f'mos_{key}: {value:.3f}\t'
            
        stats, fb_iou_message = compute_mean_iou_recall_precision(stats_meter['fb_metric'], self.fb_mapping)
        for key, value in stats.items():
            self.writer.add_scalar(f'g_{phase}/fb_{key}', value, epoch)    
            
        for key, value in stats.items():
            message += f'fb_{key}: {value:.3f}\t'

        for key, value in stats_meter.items():
            if(not isinstance(value, dict)):
                self.writer.add_scalar(f'g_{phase}/{key}', value.avg, epoch)

        for key, value in stats_meter.items():
            if(not isinstance(value, dict)):
                message += f"{key}: {value.avg:.3f}\t"
        self.logger.write(message+'\n')
        self.logger.write(mos_iou_message)
        self.logger.write(fb_iou_message+'\n')



    def inference_one_batch(self, input_dict, phase):
        assert phase in ['train','val']
        ##################################
        # put data to GPU
        for key, value in input_dict.items():
            if(not isinstance(value, list)):
                input_dict[key] = value.to(self.device)
        input_dict['phase'] = phase

        self.model.eval()
        with torch.no_grad():
            ###############################################
            # forward pass
            predictions = self.model(input_dict)
            stats = self.loss(predictions, input_dict)

        ##################################        
        # detach the gradients for loss terms
        for key, value in stats.items():
            if(key.find('loss')!=-1):
                stats[key] = float(value.detach())

        return stats


    def inference_one_epoch(self, epoch, phase):
        gc.collect()
        assert phase in ['train','val']

        # init stats meter
        stats_meter = None
        num_iter = int(len(self.loader[phase].dataset) // self.loader[phase].batch_size)        
        self.verbose_freq = num_iter // self.n_verbose
        
        c_loader_iter = self.loader[phase].__iter__()
        self.optimizer.zero_grad()
        
        for c_iter in tqdm(range(num_iter)): # loop through this epoch   
            ##################################
            try:
                inputs = c_loader_iter.next()
                ##################################
                # forward pass
                # with torch.autograd.detect_anomaly():
                stats = self.inference_one_batch(inputs, phase)
                ################################
                # update to stats_meter
                if(stats_meter is None):
                    stats_meter = init_stats_meter(stats)
                update_stats_meter(stats_meter, stats)
            except Exception as inst:
                print(inst)
            
            torch.cuda.empty_cache()

                
        ################################
        # epoch-wise stats
        self.update_after_each_epoch(stats_meter, epoch, phase)
        # print('cluster results from input')
        # self.loss.cluster_eval_input.final_eval()
        # print('cluster results from offseted points')
        # self.loss.cluster_eval_offset.final_eval()

        return stats_meter


    def train(self):
        print('start training...')
        
        for epoch in range(self.start_epoch, self.max_epoch):
            self.writer.add_scalar('lr', self._get_lr(), epoch)
            self.inference_one_epoch(epoch,'train')
            stats_meter = self.inference_one_epoch(epoch,'val')
            
            self.scheduler.step()
            if stats_meter['loss'].avg < self.best_loss:
                self.best_loss = stats_meter['loss'].avg
                self._snapshot(epoch,'best_loss')
            self._snapshot(epoch,'latest')

            # new_stats, message = compute_mean_iou_recall_precision(stats_meter['metric'], self.mapping)
            # if new_stats[self.mornitor_metric] > self.best_metric:
            #     self.best_metric = new_stats[self.mornitor_metric]
            #     self._snapshot(epoch,'best_metric')
            
        # finish all epoch
        print("Training finish!")


    def eval(self):
        print('Start to evaluate on validation datasets...')
        self.inference_one_epoch(0,'train')
        self.inference_one_epoch(0,'val')
