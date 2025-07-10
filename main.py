import argparse
import datetime
import time
import torch.backends.cudnn as cudnn
import json
import torch
import numpy as np
import os
import shutil

from pathlib import Path

from timm.data.mixup import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma
from src.solver.optim_factory import create_optimizer

from src.data.datasets import build_dataset
from src.engine.engine import train_one_epoch, evaluate

from src.utils.utils import NativeScalerWithGradNormCount as NativeScaler
import src.utils.utils as utils

from src.model import *
from src.data.samplers import MultiScaleSamplerDDP 

#EVANDRO
from src.configs.config import get_cfg
from src.data import loader as data_loader
from src.engine.trainer import Trainer
from src.utils.file_io import PathManager
from src.model.build_model import build_model

from launch import default_argument_parser

import random

#=============================
# AUXILIARY FUNCTIONS
#=============================
def count_parameters(model):
    total_trainable_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        total_trainable_params += params
    return total_trainable_params

def log_wandb(args, global_rank):
    # if args.output_dir and args.log_dir is None:
    #     args.log_dir = args.output_dir

    log_writer = None
    wandb_logger = None
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None
    print("log writter dir: ", args.log_dir)

    if global_rank == 0 and args.enable_wandb:
        wandb_logger = utils.WandbLogger(args)
    else:
        wandb_logger = None

    return wandb_logger, log_writer


def validate_conditions(args):
    if args.eval and args.usi_eval:
        raise ValueError("Cannot use both --eval and --usi_eval at the same time.")
    if args.eval and args.finetune:
        raise ValueError("Cannot use --eval and --finetune at the same time.")
    if args.usi_eval and not args.finetune:
        raise ValueError("USI evaluation requires a finetuned model. Please provide a checkpoint with --finetune.")
    if args.finetune and not os.path.isfile(args.finetune):
        raise FileNotFoundError(f"Checkpoint file {args.finetune} does not exist.")
    
    if args.layer_decay < 1.0 or args.layer_decay > 1.0:
        # Layer decay not supported
        raise NotImplementedError
    else:
        assigner = None

    if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))

    return assigner 


#=============================
# Other model features
#=============================
def create_model_ema(args, model):
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        print("Using EMA with decay = %.8f" % args.model_ema_decay)
    return model_ema

def fine_tuning(args, model):    
    # Eval/USI_eval configurations
    if args.eval:
        if args.usi_eval:
            args.crop_pct = 0.95
            model_state_dict_name = 'state_dict'
        else:
            model_state_dict_name = 'model'
    else:
        model_state_dict_name = 'model'

    # Finetuning configurations
    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location="cpu")
        state_dict = checkpoint[model_state_dict_name]
        utils.load_state_dict(model, state_dict)
        print(f"Finetune resume checkpoint: {args.finetune}")
    return model_state_dict_name

def mixup(args):
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)
    return mixup_fn

def get_datasets(cfg):
    print("Loading training data...")
    train_dataset = data_loader._construct_dataset(cfg, split='train')
    print("Loading test data...")
    test_dataset = data_loader._construct_dataset(cfg, split='test')
    return train_dataset, test_dataset

#=============================
# TRAINING FUNCTIONS
#=============================
def train(args, cfg):
    #---------------------------------
    # Declarations of variables
    #---------------------------------
    print("Creating environments avariables and initializing systems") 
    mixup_fn = None
    model_ema = None
    max_accuracy = 0.0        

    #From DAP
    #------------------------------------------------
    # if torch.cuda.is_available():
    #     torch.cuda.empty_cache()

    # if cfg.SEED is not None:
    #     torch.manual_seed(cfg.SEED)
    #     np.random.seed(cfg.SEED)
    #     random.seed(0)
    #------------------------------------------------

    #FROM CAS-ViT
    #------------------------------------------------    
    utils.init_distributed_mode(args)
    print(args)    
    #device = torch.device(args.device)    
 
    #Fix the seed for reproducibility  
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True         
    #------------------------------------------------



    # init the logger name before other steps
    logger_name = time.strftime('%Y%m%d_%H%M%S', time.localtime())

    #---------------------------------
    # Building datasets
    #---------------------------------
    print("Constructing datasets...")
    #dataset_train, test_dataset = get_datasets(cfg)    
    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)

    if args.disable_eval:
        args.dist_eval = False
        dataset_val = None
    else:
        dataset_val, _ = build_dataset(is_train=False, args=args)

    #---------------------------------
    # Building tasks
    #---------------------------------
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    if args.multi_scale_sampler:
        sampler_train = MultiScaleSamplerDDP(base_im_w=args.input_size, base_im_h=args.input_size,
                                             base_batch_size=args.batch_size, n_data_samples=len(dataset_train),
                                             is_training=True, distributed=args.distributed,
                                             min_crop_size_w=args.min_crop_size_w, max_crop_size_w=args.max_crop_size_w,
                                             min_crop_size_h=args.min_crop_size_h, max_crop_size_h=args.max_crop_size_h)
    else:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True, seed=args.seed,
        )
        print("Sampler_train = %s" % str(sampler_train))

    if args.dist_eval:
        if len(dataset_val) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                  'This will slightly alter validation results as extra duplicate entries are added to achieve '
                  'equal num of samples per-process.')
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    # starting log metrics
    wandb_logger, log_writer = log_wandb(args, global_rank)

    if args.multi_scale_sampler:
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, batch_sampler=sampler_train,
            batch_size=1,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
        )
    else:
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )

    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
    else:
        data_loader_val = None

    #---------------------------------
    # Building features
    #---------------------------------
    print("Activating features")
    mixup_fn = mixup(args)   

    #---------------------------------
    # Building models
    #---------------------------------
    print("Constructing models...") 
    device, model = build_model(cfg, args)    #EVANDRO: Structure according to DAP 
    model.to(device)
    model_without_ddp = model
    model_ema = create_model_ema (args, model)     # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper 

    print("Activating fine-tuning")
    model_state_dict_name = fine_tuning(args, model)  #EVANDRO: Model is not used here, but it is required to load the model state dict

    #---------------------------------
    # Building distributed systems
    #---------------------------------
    print("Setting up trainer...")
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)
    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training examples = %d" % len(dataset_train))
    print("Number of training training per epoch = %d" % num_training_steps_per_epoch)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],
                                                          find_unused_parameters=args.find_unused_params)
        model_without_ddp = model.module


    assigner = validate_conditions(args)  #EVANDRO: Validate conditions and assign values if needed
    optimizer = create_optimizer(
        args, model_without_ddp, skip_list=None,
        get_num_layer=assigner.get_layer_id if assigner is not None else None,
        get_layer_scale=assigner.get_scale if assigner is not None else None
        )

    loss_scaler = NativeScaler()  # if args.use_amp is False, this won't be used

    #---------------------------------
    # Building activation functions
    #---------------------------------
    print("Use Cosine LR scheduler")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
        start_warmup_value=args.warmup_start_lr
    )

    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema, state_dict_name=model_state_dict_name)

    # if args.eval:
    #     print(f"Eval only mode")
    #     test_stats = evaluate(data_loader_val, model, device, use_amp=args.use_amp)
    #     print(f"Accuracy of the network on {len(dataset_val)} test images: {test_stats['acc1']:.5f}%")
    #     return 

    #---------------------------------
    # Building training procedures
    #---------------------------------
    ## EVANDRO: Starting training procedures    
    #ORIGINAL FROM CASVIT
    total_params = count_parameters(model)
    # fvcore to calculate MAdds
    # input_res = (3, args.input_size, args.input_size)
    # input = torch.ones(()).new_empty((1, *input_res), dtype=next(model.parameters()).dtype,
    #                                  device=next(model.parameters()).device)
    # flops = FlopCountAnalysis(model, input)
    # model_flops = flops.total()
    print(f"Total Trainable Params: {round(total_params * 1e-6, 2)} M")
    # print(f"MAdds: {round(model_flops * 1e-6, 2)} M")
    #     
    #EVANDRO: From DAP
    print("Setting up trainer...")
    print("Start training for %d epochs" % args.epochs)
    start_time = time.time()        
    #trainer = Trainer(cfg, model, device)
    #trainer.train_classifier(args, global_rank, dataset_train, test_dataset) #, num_training_steps_per_epoch


    #CAS-VIT TRAINING
    print("Start training for %d epochs" % args.epochs)
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.multi_scale_sampler:
            data_loader_train.batch_sampler.set_epoch(epoch)
        elif args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
        if wandb_logger:
            wandb_logger.set_steps()


        #Calling train_one_epoch function    
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer,
            device, epoch, loss_scaler, args.clip_grad, model_ema, mixup_fn,
            log_writer=log_writer, wandb_logger=wandb_logger, start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=args.update_freq,
            use_amp=args.use_amp
        )



        if args.output_dir and args.save_ckpt:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema)
        if data_loader_val is not None:
            test_stats = evaluate(data_loader_val, model, device, use_amp=args.use_amp)
            print(f"Accuracy of the model on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
            if max_accuracy < test_stats["acc1"]:
                max_accuracy = test_stats["acc1"]
                if args.output_dir and args.save_ckpt:
                    utils.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch=f"best_{args.input_size}", model_ema=model_ema)
            print(f'Max accuracy: {max_accuracy:.2f}%')

            if log_writer is not None:
                log_writer.update(test_acc1=test_stats['acc1'], head="perf", step=epoch)
                log_writer.update(test_acc5=test_stats['acc5'], head="perf", step=epoch)
                log_writer.update(test_loss=test_stats['loss'], head="perf", step=epoch)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

            # Repeat testing routines for EMA, if ema eval is turned on
            if args.model_ema and args.model_ema_eval:
                test_stats_ema = evaluate(data_loader_val, model_ema.ema, device, use_amp=args.use_amp)
                print(f"Accuracy of the model EMA on {len(dataset_val)} test images: {test_stats_ema['acc1']:.1f}%")
                if max_accuracy_ema < test_stats_ema["acc1"]:
                    max_accuracy_ema = test_stats_ema["acc1"]
                    if args.output_dir and args.save_ckpt:
                        utils.save_model(
                            args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                            loss_scaler=loss_scaler, epoch=f"best-ema_{args.input_size}", model_ema=model_ema)
                print(f'Max EMA accuracy: {max_accuracy_ema:.2f}%')
                if log_writer is not None:
                    log_writer.update(test_acc1_ema=test_stats_ema['acc1'], head="perf", step=epoch)
                log_stats.update({**{f'test_{k}_ema': v for k, v in test_stats_ema.items()}})
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, f"{logger_name}_{args.input_size}.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

        if wandb_logger:
            wandb_logger.log_epoch_metrics(log_stats)
    #ended training for

    #---------------------------------
    # Saving and reporting results 
    #---------------------------------
    if wandb_logger and args.wandb_ckpt and args.save_ckpt and args.output_dir:
        wandb_logger.log_checkpoints()

    if args.model_ema and args.model_ema_eval:
        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir, f"{logger_name}_{args.input_size}.txt"), mode="a",
                    encoding="utf-8") as f:
                info = f"Total Max accuracy: {max(max_accuracy, max_accuracy_ema):.2f}, with max accuracy: {max_accuracy:.2f} and max EMA accuracy: {max_accuracy_ema:.2f}"
                f.write(json.dumps(info) + "\n")

        # if max_accuracy > max_accuracy_ema:
        #     shutil.copyfile(os.path.join(args.output_dir, f"checkpoint-best_{args.input_size}.pth"),
        #                     os.path.join(args.output_dir, f"best_{args.input_size}_{max_accuracy:.2f}.pth"))
        # else:
        #     shutil.copyfile(os.path.join(args.output_dir, f"checkpoint-best-ema_{args.input_size}.pth"),
        #                     os.path.join(args.output_dir, f"best_{args.input_size}_{max_accuracy_ema:.2f}.pth"))
        print(f"Total max accuracy: {max(max_accuracy, max_accuracy_ema):.2f}%")


    #Loggin training time
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


#=============================
# MAIN
#=============================
def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    output_path = os.path.join(cfg.OUTPUT_DIR, cfg.DATA.NAME)

    if PathManager.exists(output_path):
        #raise ValueError(f"Already run for {output_path}")
        pass
    else:
        PathManager.mkdirs(output_path)
    cfg.OUTPUT_DIR = output_path
    return cfg 
 
def main(args):
    cfg = setup(args)
    train(args, cfg)

if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    #parser = argparse.ArgumentParser('CAS-ViT training and evaluation script', parents=[get_args_parser()])
    #args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args) 
