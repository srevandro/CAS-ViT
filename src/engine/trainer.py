#!/usr/bin/env python3
"""
a trainer class
"""

import datetime
import time
import torch
import torch.nn as nn
import os
import numpy as np
from fvcore.common.config import CfgNode

from cas.classification import model

from ..solver.lr_scheduler import make_scheduler
from ..solver.optimizer import make_optimizer
from ..solver.losses import build_loss

from src.utils import logger as logger_continual
from src.utils.train_utils import AverageMeter, gpu_mem_usage
from src.data.loader import _build_continual_dataset, _construct_continual_loader, _build_continual_dataset_pytorch

#CAS
#from src.engine import train_one_epoch, evaluate
import src.utils.utils as utils

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

class Trainer():
    def __init__(
        self,
        cfg: CfgNode,
        model: nn.Module,
        device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.device = device

        # solver related
        self.optimizer = make_optimizer([self.model], cfg.SOLVER)
        self.scheduler = make_scheduler(self.optimizer, cfg.SOLVER)
        self.cls_criterion = build_loss(self.cfg)

        self.cpu_device = torch.device("cpu")

        self.prev_task = -1
        self.task_changed = False

 
    def forward_one_batch(self, inputs, targets, is_train, task_id=None):
        """Train a single (full) epoch on the model using the given
        data loader.
        """
        inputs = inputs.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)

        # forward
        with torch.set_grad_enabled(is_train):
            #outputs, reduce_sim = self.model(inputs, is_train=is_train, cfg=self.cfg) # task_id=task_id,
            outputs, reduce_sim = self.model(inputs, is_train=is_train, cfg=self.cfg)    
            if torch.isnan(reduce_sim).any():
                print("[NaN DETECTED] forward_one_batch reduce_sim result is NaN!")

            if self.cls_criterion.is_local() and is_train:
                self.model.eval()
                loss = self.cls_criterion(
                    outputs, targets,
                    self.model, inputs
                )
            elif self.cls_criterion.is_local():
                return torch.tensor(1), outputs
            elif is_train:
                num_total_class = self.cfg.DATA.NUMBER_CLASSES
                class_mask = self.dataset_train._class_ids_mask
                not_mask = np.setdiff1d(np.arange(num_total_class), class_mask)
                outputs[:, not_mask] = -np.inf
                loss_cls = self.cfg.MODEL.DAP.CURRENT_LAMBDA * self.cls_criterion(
                        outputs, targets)
            else:
                loss_cls = self.cls_criterion(
                    outputs, targets)

            simloss = self.cfg.MODEL.DAP.SIM_LAMBDA * reduce_sim
            #loss -= simloss   
            loss = loss_cls - simloss
 
            print(
            f"CLS Loss: {loss_cls.item():.4f}, "
            f"ReduceSim: {reduce_sim.item():.4f}, "            
            f"Simloss: {simloss.item():.4f}, "
            f"Total Loss: {loss.item():.4f}, "
            f"Output min: {outputs.min().item():.4f}, max: {outputs.max().item():.4f}")
                           
        # =======backward and optim step only if in training phase... =========
        if is_train:
            self.optimizer.zero_grad()
            loss.backward()

            #EVANDRO: Gradient clipping prevents exploding gradients
            #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  

            if self.cfg.SOLVER.GRAD_CLIP_APPLY:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.SOLVER.GRAD_CLIP)
            self.optimizer.step()

        return loss, outputs

    def get_input(self, data):
        if not isinstance(data["image"], torch.Tensor):
            for k, v in data.items():
                data[k] = torch.from_numpy(v)

        inputs = data["image"].float()
        labels = data["label"]
        return inputs, labels
    
    def get_input_pytorch(self, data):
        # For (input, target) tuples (default in PyTorch DataLoader)
        if isinstance(data, (list, tuple)) and len(data) == 2:
            image, target = data
        elif isinstance(data, dict):  # Support older dict-style data
            image = data["image"]
            target = data["target"]
        else:
            raise ValueError("Unsupported data format")

        return image, target

    def forward_one_batch(self, inputs, targets, is_train, task_id=None):
        """Train a single (full) epoch on the model using the given
        data loader.
        """
        inputs = inputs.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)

        # forward
        with torch.set_grad_enabled(is_train): ## model(samples, task_id_emb=task_id_emb) 
            #outputs, reduce_sim = self.model(inputs, task_id=task_id, is_train=is_train, cfg=self.cfg) #dap original
            outputs, reduce_sim = self.model(inputs, task_id=task_id) #, is_train=is_train, cfg=self.cfg) #cas-vit
   
            if self.cls_criterion.is_local() and is_train:                
                #print('forward_one_batch: 1')
                self.model.eval()
                loss_cls = self.cls_criterion(
                    outputs, targets,
                    self.model, inputs
                )
            elif self.cls_criterion.is_local():
                #print('forward_one_batch: 2')                
                return torch.tensor(1), outputs
            elif is_train:
                #print('forward_one_batch: 3')              
                num_total_class = self.cfg.DATA.NUMBER_CLASSES
                class_mask = self.dataset_train._class_ids_mask #dap original
                #class_mask = self.dataset_train.class_ids_mask #cas-vit
                not_mask = np.setdiff1d(np.arange(num_total_class), class_mask)
                outputs[:, not_mask] = -np.inf
                loss_cls = self.cfg.MODEL.DAP.CURRENT_LAMBDA * self.cls_criterion(
                        outputs, targets)
            else:
                #print('forward_one_batch: 4')                
                loss_cls = self.cls_criterion(
                    outputs, targets)

            simloss = self.cfg.MODEL.DAP.SIM_LAMBDA * reduce_sim 
            #loss -= simloss   
            loss = loss_cls - simloss
     

           #print(f"Task_id {task_id}, CLS Loss: {loss_cls.item()}, ReduceSim Loss: {reduce_sim.item()}, Total Loss: {loss.item()}, is_train: {is_train}, islocal: {self.cls_criterion.is_local()}")
            #f"Epoch {epoch}, Step {data_iter_step}, "
            print(
            f"CLS Loss: {loss_cls.item():.4f}, "
            f"ReduceSim: {reduce_sim.item():.4f}, "            
            f"Simloss: {simloss.item():.4f}, "
            f"Total Loss: {loss.item():.4f}, "
            #f"Total Loss Value: {loss_value.item():.4f}, "
            f"Output min: {outputs.min().item():.4f}, max: {outputs.max().item():.4f}")
                           

        # =======backward and optim step only if in training phase... =========
        if is_train:
            self.optimizer.zero_grad()
            loss.backward()
            if self.cfg.SOLVER.GRAD_CLIP_APPLY:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.SOLVER.GRAD_CLIP)
            self.optimizer.step()

        return loss, outputs
 
 
    def train_classifier(self, args, global_rank, train_dataset, test_dataset):
        """
        Train a classifier using epoch
        """
        # starting log metrics
        #wandb_logger, log_writer = log_wandb(args, global_rank)


        # save the model prompt if required before training
        self.model.eval()

        # setup training epoch params
        total_epoch = self.cfg.SOLVER.TOTAL_EPOCH

        #DAP ORIGINAL
        self.scenario_train = _build_continual_dataset(self.cfg, train_dataset)
        self.scenario_test = _build_continual_dataset(self.cfg, test_dataset)

        #CAS-ViT dataset
        # self.scenario_train = _build_continual_dataset_pytorch(self.cfg, train
        # self.scenario_train = _build_continual_dataset_pytorch(self.cfg, train_dataset)
        # self.scenario_test = _build_continual_dataset_pytorch(self.cfg, test_dataset)

        self.LOG = logger_continual.logger_all('acc', n_tasks=self.cfg.CONTINUAL.N_TASKS)

        for task_id, dataset_train in enumerate(self.scenario_train):
            print(f"Starting task id {task_id}/{len(self.scenario_train) - 1}")
 
            if task_id == 1:
                #for k, p in self.model.enc.named_parameters(): #dap original
                for k, p in self.model.named_parameters():                
                    if "dap_downsample" in k:
                        p.requires_grad = False

            self.dataset_train = dataset_train

            loader_train = _construct_continual_loader(self.cfg, dataset_train, shuffle=True)

            total_data = len(loader_train)
            log_interval = self.cfg.SOLVER.LOG_EVERY_N

            losses = AverageMeter('Loss', ':.4e')
            batch_time = AverageMeter('Time', ':6.3f')
            data_time = AverageMeter('Data', ':6.3f')

            print(f"Start training for {total_epoch} epochs")
            for epoch in range(total_epoch):
                # reset averagemeters to measure per-epoch results
                losses.reset()
                batch_time.reset()
                data_time.reset()

                lr = self.scheduler.get_lr()[0]

                # Enable training mode
                self.model.train()

                end = time.time()
                for idx, input_data in enumerate(loader_train):
                    X, targets = self.get_input(input_data) #dap original 
                    #X, targets = self.get_input_pytorch(input_data)                     
                    data_time.update(time.time() - end)

                    train_loss, _ = self.forward_one_batch(X, targets, True, task_id) #dap original
                    #train_loss, _ =   self.forward_one_batch(X, targets, True, task_id)       

                    if train_loss == -1:
                        # continue
                        return None

                    losses.update(train_loss.item(), X.shape[0])

                    # measure elapsed time
                    batch_time.update(time.time() - end)
                    end = time.time()

                    # log during one batch
                    if (idx + 1) % log_interval == 0:
                        seconds_per_batch = batch_time.val
                        eta = datetime.timedelta(seconds=int(
                            seconds_per_batch * (total_data - idx - 1) + seconds_per_batch*total_data*(total_epoch-epoch-1)))
                        print(
                            "\tTraining {}/{}. train loss: {:.4f},".format(
                                idx + 1,
                                total_data,
                                train_loss
                            )
                            + "\t{:.4f} s / batch. (data: {:.2e}). ETA={}, ".format(
                                seconds_per_batch,
                                data_time.val,
                                str(eta),
                            )
                            + "max mem: {:.1f} GB ".format(gpu_mem_usage())
                        )
                print(
                    "Epoch {} / {}: ".format(epoch + 1, total_epoch)
                    + "learning rate: {:.2f}, avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
                        lr, data_time.avg, batch_time.avg)
                    + "average train loss: {:.4f}".format(losses.avg))
                self.scheduler.step()

                # Enable eval mode
                self.model.eval()

            self.eval_classifier_continual(task_id, self.scenario_test)

        task = self.cfg.CONTINUAL.N_TASKS - 1
        final_accs = self.LOG['acc'][:, task]
        logger_continual.per_task_summary(self.LOG, 'final_acc', value=np.round(np.mean(final_accs), 5))
        best_acc = np.max(self.LOG['acc'], 1)
        final_forgets = best_acc - self.LOG['acc'][:, task]
        logger_continual.per_task_summary(self.LOG, 'final_forget', value=np.round(np.mean(final_forgets[:-1]), 5))
        final_la = np.diag(self.LOG['acc'])
        logger_continual.per_task_summary(self.LOG, 'final_la', value=np.round(np.mean(final_la), 5))

        print('\n')
        print('final accuracy: {}'.format(final_accs))
        print('average: {}'.format(self.LOG['final_acc']))
        print('final forgetting: {}'.format(final_forgets))
        print('average: {}'.format(self.LOG['final_forget']))
        print('final LA: {}'.format(final_la))
        print('average: {}'.format(self.LOG['final_la']))

        with open(self.cfg.OUTPUT_DIR + '/final_results.txt', "w") as text_file:
            print(self.cfg, file=text_file)
            print("\n", file=text_file)
            print(self.LOG['acc'], file=text_file)
            print('\nFinal {} Accuracy: {:.5f}'.format('test', self.LOG['final_acc']), file=text_file)
            print('\nFinal {} Forget: {:.5f}'.format('test', self.LOG['final_forget']), file=text_file)
            print('\nFinal {} LA: {:.5f}'.format('test', self.LOG['final_la']), file=text_file)

    @torch.no_grad()
    def eval_classifier_continual(self, task_id, scenario_test):
        for task_t in range(task_id + 1):
            te_dataset = scenario_test[task_t]
            loader_te = _construct_continual_loader(self.cfg, te_dataset)

            LOG_eval = logger_continual.logger_eval('acc')

            for idx, input_data in enumerate(loader_te):
                X, targets = self.get_input(input_data) #dap original
                #X, targets = self.get_input_pytorch(input_data)
                loss, outputs = self.forward_one_batch(X, targets, False)
                if loss == -1:
                    return

                pred = outputs.argmax(dim=1, keepdim=True).cpu()
                LOG_eval['acc'] += [pred.eq(targets.view_as(pred)).sum().item() / pred.size(0)]

            logger_continual.per_task_summary(self.LOG, 'acc', task_id, task_t,
                                                  np.round(np.mean(LOG_eval['acc']), 5))

        print(self.LOG['acc'])