import os, torch, shutil, json, nestargs
from toolbox.utils import setup_seed, makedirs
from toolbox.config import get_config, get_scheduler, get_optimizer, parse_extra_args, update_recursive
from libs.dataloader import get_dataloader
from libs.loss import FuseLoss
from libs.tester import SegTrainer
from datetime import datetime
from models.motionnet import MotionNet

def update_config(args, config):
    config['config_file'] = args.config
    config['pillar_encoder']['voxel_size'] = config['voxel_generator']['voxel_size']
    config['pillar_encoder']['pc_range'] = config['voxel_generator']['range']
    config['pillar_encoder']['n_sweeps'] = config['voxel_generator']['n_sweeps']

# torch.autograd.set_detect_anomaly(True)
if __name__ == '__main__':
    ##############################
    # load configs
    argparser = nestargs.NestedArgumentParser() 
    argparser.add_argument('config', type=str, help= 'Path to the config file.')
    argparser.add_argument('batch_size', type=int, help = 'batch size of training and validation')
    argparser.add_argument('iter_size', type=int, help = 'gradient accumulation steps')

    args, extra_args = argparser.parse_known_args()
    config = get_config(args.config)
    update_config(args, config)

    # Parse extra arguments from a list to an overwrite dict
    if len(extra_args) != 0:
        override_config = parse_extra_args(extra_args)
        update_recursive(config, override_config)
    setup_seed(config['misc']['seed'])
       
    today = datetime.now().strftime("%d_%m_%H")
    config['save_dir'] = 'snapshot/%s' % (config['misc']['exp_name'])
    config['tboard_dir'] = 'snapshot/%s/tensorboard' % (config['misc']['exp_name'])
    config['loss']['save_dir'] = 'snapshot/%s' % (config['misc']['exp_name'])
    

    ##############################
    # save the configs and backup files
    makedirs(config['save_dir'])
    json.dump(config,open(os.path.join(config['save_dir'], 'config.json'), 'w'),indent=4)
    for folder in ['libs','models']:
        os.system(f'cp -r {folder} {config["save_dir"]}')
    shutil.copy2('main.py',config['save_dir'])
    

    ##############################
    # instantiate device
    config['instances'] = dict()
    if config['misc']['use_gpu']:
        config['instances']['device'] = torch.device('cuda')
    else:
        config['instances']['device']= torch.device('cpu')
    

    ##############################
    # instantiate modelc
    config['instances']['model'] = MotionNet(config)


    ##############################
    # instantiate optimizer and scheduler
    config['instances']['optimizer'] = get_optimizer(config,config['instances']['model'])
    config['instances']['scheduler'] = get_scheduler(config, config['instances']['optimizer'])


    ##############################
    # create dataset and dataloader
    config = get_dataloader(config)

    ##############################
    # create evaluation metrics and trainer
    config['train']['loss']= FuseLoss(config['loss'])
    trainer = SegTrainer(config)
    if(config['misc']['mode']=='train'):
        trainer.train()
    elif(config['misc']['mode'] =='val'):
        trainer.eval()
    elif(config['misc']['mode']=='test'):
        trainer.test()
    else:
        raise NotImplementedError