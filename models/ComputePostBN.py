'''
Code credit for these operations to
https://github.com/taoyang1122/GradAug,
https://github.com/taoyang1122/MutualNet
'''

import torch
import torch.nn.functional as F
import torch.nn as nn
def adjust_bn_layers(module):
      
    if isinstance(module, nn.BatchNorm2d):
        
        # Removing the stats computed using exponential running average
        # and resetting count
        module.reset_running_stats()
        
        # Doing this so that we can restore it later
        module._old_momentum = module.momentum
        
        # Switching to cumulutive running average
        module.momentum = 0.1
        
        # This is necessary -- because otherwise the
        # newly observed batches will not be considered
        module._old_training = module.training
        module._old_track_running_stats = module.track_running_stats
        
        module.training = True
        # module.affine = True
        module.track_running_stats = True
            
def restore_original_settings_of_bn_layers(module):
      
    if isinstance(module, nn.BatchNorm2d):
        
        # Restoring old settings
        module.momentum = module._old_momentum
        module.training = module._old_training
        module.track_running_stats = module._old_track_running_stats

def adjust_momentum(module, t):
    if isinstance(module, nn.BatchNorm2d):
        module.momentum = 1 / (t+1)

def ComputeBN(net, postloader, resolution, device, num_batch=8):
    net.train()
    net.apply(adjust_bn_layers)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(postloader):
            img = inputs.to(device)
            net.apply(lambda m: adjust_momentum(m, batch_idx))
            _ = net(F.interpolate(img, (resolution, resolution), mode='bilinear', align_corners=True))
            if not batch_idx < num_batch:
                break
    net.apply(restore_original_settings_of_bn_layers)
    net.eval()
    return net