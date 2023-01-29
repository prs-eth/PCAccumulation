
import torch, yaml, json, copy, os
import torch.optim as optim

def get_optimizer(cfg, model):
    ''' 
    Returns an optimizer instance.
    Args:
        cfg (dict): config dictionary
        model (nn.Module): the model used for training

    Returns:
        optimizer (optimizer instance): optimizer used to train the network
    '''
    
    method = cfg['optimizer']['name']
    cfg['optimizer'] = cfg[method]

    if method == "SGD":
        optimizer = getattr(optim, method)(model.parameters(), lr=cfg['optimizer']['learning_rate'],momentum=cfg['optimizer']['momentum'],weight_decay=cfg['optimizer']['weight_decay'],nesterov = cfg['optimizer']['nesterov'])

    elif method == "Adam":
        optimizer = getattr(optim, method)(model.parameters(), lr=cfg['optimizer']['learning_rate'],weight_decay=cfg['optimizer']['weight_decay'])
    else: 
        print("{} optimizer is not implemented, must be one of the [SGD, Adam]".format(method))

    return optimizer


def get_scheduler(cfg, optimizer):
    ''' 
    Returns a learning rate scheduler
    Args:
        cfg (dict): config dictionary
        optimizer (torch.optim): optimizer used for training the network

    Returns:
        scheduler (optimizer instance): learning rate scheduler
    '''
    
    method = cfg['scheduler']['name']

    if method == "ExponentialLR":
        scheduler = getattr(optim.lr_scheduler, method)(optimizer, gamma=cfg['scheduler']['exp_gamma'])
    else: 
        print("{} scheduler is not implemented, must be one of the [ExponentialLR]".format(method))

    return scheduler

def parse_extra_args(extra_args):
    """
    Parses the extra args from a list to a dictionary
    Args:
        extra_args (list): extra command line arguments

    """

    override_config = {}
    # Parse the command line arguments to a dict
    for arg_id, arg in enumerate(extra_args):
        arg_name = arg.split('=')[0]
        if arg_name.startswith('--'):
            if "=" in arg:
                override_config[arg_name[2:]] = decode_type(arg.split('=')[1])
            else:
                override_config[arg_name[2:]] = decode_type(extra_args[arg_id + 1])

    return override_config



def decode_type(value):
    """
    Decodes the type of the command line argument.
    Currently handles, lists, int, strings and boolean values
    Args:
        value (str): value from the command line argument
    
    """

    # Check if the input value is a boolean
    if value.lower() == 'true':
        return True
    elif value.lower() == 'false':
        return False

    # If the input is a list convert it to a list and decode the type
    if value[0] == '[' and value[-1] == ']':
        list_value = value.strip('][').strip().split(',')
        # Check if the elements of the list are integers
        try:
            return [int(value) for value in list_value]
        except:
            pass
        # Check if the elements of the list are floats otherwise return the string
        try:
            return [float(value) for value in list_value]
        except:
            return [value for value in list_value]

    # If the input string represents a int convert it to int
    try:
        return int(value)
    except:
        pass

    # If the input string represents a float convert it to float
    try:
        return float(value)
    except:
        return value  # If non of the condition holds the values will be used a string
    

# General config
def get_config(path, default_path='./configs/default.yaml'):
    ''' 
    Loads config file.
    
    Args:  
        path (str): path to config file
        default_path (bool): whether to use default path
    '''
    # Load configuration from file itself
    with open(path, 'r') as f:
        cfg_special = yaml.safe_load(f)

    # load default setting
    with open(default_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Include main configuration
    update_recursive(cfg, cfg_special)

    return cfg


def update_recursive(dict1, dict2):
    ''' 
    Update two config dictionaries recursively.
    
    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used
    '''
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v


def update_configs(config):
    """
    If we are loading pre-trained models, then we also load its corresponding configs except for the misc part
    """
    pretrain_path = config['misc']['pretrain']
    if(pretrain_path!='' and os.path.exists(pretrain_path) and config['misc']['mode'] !='train'):
        old_misc = copy.deepcopy(config['misc'])
        config_path = os.path.join(os.path.dirname(pretrain_path), 'config.json')
        with open(config_path, 'r') as f:
            cfg = json.load(f)
        update_recursive(config, cfg)
        config['misc'] = old_misc

        model_name = os.path.dirname(pretrain_path)
        config['dump_dir'] = model_name.replace('snapshot','dump')
        os.makedirs(config['dump_dir'], exist_ok=True)