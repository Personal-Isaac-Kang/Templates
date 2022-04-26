import torch
import torch.nn as nn

def load_checkpoint(checkpoint_path, log, device, **kwargs):
    """Load saved checkpoint.

    Args:
        checkpoint_path : Path to saved checkpoint.
        log: Logger.
        device : Load device.
    """
    log.info(f'Loading pretrained model from : {checkpoint_path}.')
    ckpt = torch.load(checkpoint_path, map_location=device)
    for k, v in kwargs.items():
        if isinstance(v, nn.Module):
            v.load_state_dict(ckpt[k])
        elif k == 'optimizer':
            load_optimizer(v, ckpt['optimizer'])
        elif isinstance(v, (float, int)):
            v = ckpt['k']
            if k == 'epoch':
                loaded_epoch = v
        else:
            pass
    log.info(f'Loaded pretrained model from epoch : {loaded_epoch}')
        
            
def load_optimizer(optim, state_dict, device):
    optim.load_state_dict(state_dict)
    for state in optim.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()


model, optimizer, epoch = load_checkpoint(checkpoint_path, log, device, model=model, optimizer=optimizer)