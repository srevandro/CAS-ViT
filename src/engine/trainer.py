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

from solver.lr_scheduler import make_scheduler
from solver.optimizer import make_optimizer
from solver.losses import build_loss
from classification.daputils import logger as logger_continual
from classification.daputils.train_utils import AverageMeter, gpu_mem_usage

#from data.loader import _build_continual_dataset, _construct_continual_loader

#CAS
from classification.engine import train_one_epoch, evaluate

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

    #, task_id=None
    def forward_one_batch(self, inputs, targets, is_train): 
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
                loss = self.cfg.MODEL.DAP.CURRENT_LAMBDA * self.cls_criterion(
                        outputs, targets)
            else:
                loss = self.cls_criterion(
                    outputs, targets)

            simloss = self.cfg.MODEL.DAP.SIM_LAMBDA * reduce_sim
            loss -= simloss

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

    def train_classifier(self, args, wandb_logger, log_writer, data_loader_train, data_loader_val, num_training_steps_per_epoch):
        """
        Train a classifier using epoch
        """
        # save the model prompt if required before training
        self.model.eval()

        # setup training epoch params
        total_epoch = self.cfg.SOLVER.TOTAL_EPOCH

        for epoch in range(args.start_epoch, args.epochs):
            if args.multi_scale_sampler:
                data_loader_train.batch_sampler.set_epoch(epoch)
            elif args.distributed:
                data_loader_train.sampler.set_epoch(epoch)
            if log_writer is not None:
                log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
            if wandb_logger:
                wandb_logger.set_steps()
            
            #treinamento
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

 
        #==============DAP´========================
        # self.scenario_train = _build_continual_dataset(self.cfg, train_dataset)
        # self.scenario_test = _build_continual_dataset(self.cfg, test_dataset)

        # self.LOG = logger_continual.logger_all('acc', n_tasks=self.cfg.CONTINUAL.N_TASKS)

        # for task_id, dataset_train in enumerate(self.scenario_train): #EVANDRO HERE TASK_ID
        #     print(f"Starting task id {task_id}/{len(self.scenario_train) - 1}")
        #     print('**************** task_id: ', task_id)

        #     if task_id == 1:
        #         for k, p in self.model.enc.named_parameters():
        #             if "dap_downsample" in k:
        #                 p.requires_grad = False

        #     self.dataset_train = dataset_train

        #     loader_train = _construct_continual_loader(self.cfg, dataset_train, shuffle=True)

        #     total_data = len(loader_train)
        #     log_interval = self.cfg.SOLVER.LOG_EVERY_N

        #     losses = AverageMeter('Loss', ':.4e')
        #     batch_time = AverageMeter('Time', ':6.3f')
        #     data_time = AverageMeter('Data', ':6.3f')

        #     print(f"Start training for {total_epoch} epochs")
        #     for epoch in range(total_epoch):
        #         # reset averagemeters to measure per-epoch results
        #         losses.reset()
        #         batch_time.reset()
        #         data_time.reset()

        #         lr = self.scheduler.get_lr()[0]

        #         # Enable training mode
        #         self.model.train()

        #         end = time.time()
        #         for idx, input_data in enumerate(loader_train):
        #             X, targets = self.get_input(input_data)
        #             data_time.update(time.time() - end)

        #             train_loss, _ = self.forward_one_batch(X, targets, True, task_id)

        #             if train_loss == -1:
        #                 # continue
        #                 return None

        #             losses.update(train_loss.item(), X.shape[0])

        #             # measure elapsed time
        #             batch_time.update(time.time() - end)
        #             end = time.time()

        #             # log during one batch
        #             if (idx + 1) % log_interval == 0:
        #                 seconds_per_batch = batch_time.val
        #                 eta = datetime.timedelta(seconds=int(
        #                     seconds_per_batch * (total_data - idx - 1) + seconds_per_batch*total_data*(total_epoch-epoch-1)))
        #                 print(
        #                     "\tTraining {}/{}. train loss: {:.4f},".format(
        #                         idx + 1,
        #                         total_data,
        #                         train_loss
        #                     )
        #                     + "\t{:.4f} s / batch. (data: {:.2e}). ETA={}, ".format(
        #                         seconds_per_batch,
        #                         data_time.val,
        #                         str(eta),
        #                     )
        #                     + "max mem: {:.1f} GB ".format(gpu_mem_usage())
        #                 )
        #         print(
        #             "Epoch {} / {}: ".format(epoch + 1, total_epoch)
        #             + "learning rate: {:.2f}, avg data time: {:.2e}, avg batch time: {:.4f}, ".format(
        #                 lr, data_time.avg, batch_time.avg)
        #             + "average train loss: {:.4f}".format(losses.avg))
        #         self.scheduler.step()

        #         # Enable eval mode
        #         self.model.eval()

        #     self.eval_classifier_continual(task_id, self.scenario_test)

        # task = self.cfg.CONTINUAL.N_TASKS - 1
        # final_accs = self.LOG['acc'][:, task]
        # logger_continual.per_task_summary(self.LOG, 'final_acc', value=np.round(np.mean(final_accs), 5))
        # best_acc = np.max(self.LOG['acc'], 1)
        # final_forgets = best_acc - self.LOG['acc'][:, task]
        # logger_continual.per_task_summary(self.LOG, 'final_forget', value=np.round(np.mean(final_forgets[:-1]), 5))
        # final_la = np.diag(self.LOG['acc'])
        # logger_continual.per_task_summary(self.LOG, 'final_la', value=np.round(np.mean(final_la), 5))

        # print('\n')
        # print('final accuracy: {}'.format(final_accs))
        # print('average: {}'.format(self.LOG['final_acc']))
        # print('final forgetting: {}'.format(final_forgets))
        # print('average: {}'.format(self.LOG['final_forget']))
        # print('final LA: {}'.format(final_la))
        # print('average: {}'.format(self.LOG['final_la']))

        # with open(self.cfg.OUTPUT_DIR + '/final_results.txt', "w") as text_file:
        #     print(self.cfg, file=text_file)
        #     print("\n", file=text_file)
        #     print(self.LOG['acc'], file=text_file)
        #     print('\nFinal {} Accuracy: {:.5f}'.format('test', self.LOG['final_acc']), file=text_file)
        #     print('\nFinal {} Forget: {:.5f}'.format('test', self.LOG['final_forget']), file=text_file)
        #     print('\nFinal {} LA: {:.5f}'.format('test', self.LOG['final_la']), file=text_file)

    # @torch.no_grad()
    # def eval_classifier_continual(self, task_id, scenario_test):
    #     for task_t in range(task_id + 1):
    #         te_dataset = scenario_test[task_t]
    #         loader_te = _construct_continual_loader(self.cfg, te_dataset)

    #         LOG_eval = logger_continual.logger_eval('acc')

    #         for idx, input_data in enumerate(loader_te):
    #             X, targets = self.get_input(input_data)
    #             loss, outputs = self.forward_one_batch(X, targets, False)
    #             if loss == -1:
    #                 return

    #             pred = outputs.argmax(dim=1, keepdim=True).cpu()
    #             LOG_eval['acc'] += [pred.eq(targets.view_as(pred)).sum().item() / pred.size(0)]

    #         logger_continual.per_task_summary(self.LOG, 'acc', task_id, task_t,
    #                                               np.round(np.mean(LOG_eval['acc']), 5))

    #     print(self.LOG['acc'])
