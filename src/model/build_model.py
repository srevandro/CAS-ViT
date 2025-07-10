#!/usr/bin/env python3
"""
model construction functions
"""

from tabnanny import verbose
import torch
import torch.nn as nn

#from .vit_models import ViT 
from src.model.rcvit import RCViT

# Supported model types
_MODEL_TYPES = {
    "vit": RCViT,
}

def build_model(cfg, args):
    """
    build model here
    """
    assert (
        cfg.MODEL.TYPE in _MODEL_TYPES.keys()
    ), "Model type '{}' not supported".format(cfg.MODEL.TYPE)
    assert (
        cfg.NUM_GPUS <= torch.cuda.device_count()
    ), "Cannot use more GPU devices than available"
       
    # Construct the model
    train_type = cfg.MODEL.TYPE
    #print('train_type: ', train_type)
    #model = _MODEL_TYPES[train_type](cfg)

    #EVANDRO: Adaptado para DAP/CAS-ViT rcvit_xs()
    model = _MODEL_TYPES[train_type](
        layers=[2, 2, 4, 2], embed_dims=[48, 56, 112, 220], mlp_ratios=4, downsamples=[True, True, True, True],
        norm_layer=nn.BatchNorm2d, attn_bias=False, act_layer=nn.GELU, num_classes=args.nb_classes, drop_rate=0.,
        drop_path_rate=args.drop_path, fork_feat=False, init_cfg=None, pretrained=False, distillation=False, cfg=cfg)  
    
    model, device = load_model_to_device(model, cfg) 
    return device, model


def get_current_device():
    if torch.cuda.is_available():
        # Determine the GPU used by the current process
        cur_device = torch.cuda.current_device()
    else:
        cur_device = torch.device('cpu')
    return cur_device


def load_model_to_device(model, cfg):
    cur_device = get_current_device()
    if torch.cuda.is_available():
        # Transfer the model to the current GPU device
        model = model.cuda(device=cur_device)
        # Use multi-process data parallel model in the multi-gpu setting
        if cfg.NUM_GPUS > 1:
            # Make model replica operate on the current device
            model = torch.nn.parallel.DistributedDataParallel(
                module=model, device_ids=[cur_device], output_device=cur_device,
                find_unused_parameters=True,  
            )
    else:
        model = model.to(cur_device)
    return model, cur_device
