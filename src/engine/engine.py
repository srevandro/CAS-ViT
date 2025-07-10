import math
from typing import Iterable, Optional
import torch
from timm.data import Mixup
from timm.utils import accuracy, ModelEma

#import utils
from ..utils import utils

#EVANDRO: Tensorboard and logs
from torch.utils.tensorboard import SummaryWriter 
reduce_sim_log = []  # at the start of training

#2. Adjust lambda_dap dynamically during training
#a. Define scheduler function base=0.005, final=0.05
def get_lambda_dap(epoch, max_epoch, base=0.005, final=0.05):
    """ Linearly scale lambda_dap from base to final over max_epoch """
    return base + (final - base) * (epoch / max_epoch)


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
                    wandb_logger=None, start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, use_amp=False):
    model.train(True)

    writer = SummaryWriter(log_dir='./runs') #EVANDRO: Tensorboard

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    optimizer.zero_grad()
    
    best_sim = -1.0  # Initialize best_sim to a very low value
    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        global_step = epoch * len(data_loader) + data_iter_step #EVANDRO: Global step for Tensorboard

        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # Global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        #EVANDRO: DAP LOSS   
        #dap_lambda = 0.01 #antes era 0.1 # DAP uses something between 0.01 and 0.05 ?
        TASK_EMB = 256
        task_id_emb = torch.zeros((samples.size(0), TASK_EMB), device=samples.device) 


        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if use_amp:
            with torch.cuda.amp.autocast():
                output, reduce_sim_score = model(samples, task_id_emb=task_id_emb)                 
                #output = model(samples)
                cls_loss = criterion(output, targets)
        else:  # Full precision
            output, reduce_sim_score = model(samples, task_id_emb=task_id_emb)             
            #output = model(samples)
            cls_loss = criterion(output, targets)



        #auto lambda
        dap_lambda = get_lambda_dap(epoch, max_epoch=10)

        #EVANDRO: DAP LOSS
        simloss = dap_lambda * reduce_sim_score

        #loss_dap =  torch.sum(loss_cls) / targets.shape[0]
        #loss_dap -= simloss 
        
        #loss = loss_cls  
        #loss += simloss 


        # Apply ReduceSim the DAP way (as a reward)
        loss = cls_loss - dap_lambda * reduce_sim_score

        # Inside training loop
        reduce_sim_log.append(reduce_sim_score.item())

        writer.add_scalar('Loss/cls_loss', cls_loss.item(), global_step)
        writer.add_scalar('Loss/reduce_sim', reduce_sim_score.item(), global_step)
        writer.add_scalar('Loss/total_loss', loss.item(), global_step)
        writer.add_scalar('Hyper/dap_lambda', dap_lambda, global_step)

        # === Optional: Accuracy logging === #EVANDRO
        acc1 = accuracy(output, targets.cuda(), topk=(1,))[0]
        writer.add_scalar('Accuracy/top1', acc1.item(), global_step)        

        # === Save best ReduceSim model ===
        if reduce_sim_score.item() > best_sim:
            best_sim = reduce_sim_score.item()
            torch.save(model.state_dict(), "./output/best_sim_model.pth")  
            #torch.save(model.state_dict(), f"best_model_reducesim_{best_reducesim:.2f}.pth")            
 
        if step % 10 == 0:
            print(f"[Epoch {epoch}][{step}/{len(data_loader)}] "
                  f"Loss: {cls_loss.item():.4f}, ReduceSim: {reduce_sim_score.item():.4f}, "
                  f"λ: {dap_lambda:.4f}, Acc@1: {acc1.item():.2f}%")         
            
        if step % 50 == 0:
            avg_sim = sum(reduce_sim_log[-50:]) / 50
            print(f"Avg ReduceSim (last 50 steps): {avg_sim:.4f}")

        print(f"Epoch {epoch}, Step {data_iter_step}, "
            f"cls_loss: {cls_loss.item():.4f}, "
            #f"loss_dap: {loss_dap.item():.4f}, "
            f"reduce_sim_score: {reduce_sim_score.item():.4f}, " 
            f"simloss: {simloss.item():.4f}, " 
            f"loss: {loss.item():.4f}, "
            #f"loss_dap: {loss_dap.item():.4f}, "
            f"Output min: {output.min().item():.4f}, max: {output.max().item():.4f}")                    

        #EVANDRO: Adapted from DAP algorithm
        #if (loss == -1) or (math.isfinite(loss) == True):
            #return None             # continue
        #    loss = 0 

        #EVANDRO: DAP LOSS
        # sum_of_parameters = sum(p.sum() for p in model.parameters())
        # zero_sum = sum_of_parameters * 0.0
        # final_loss = loss + zero_sum
        # loss_value = final_loss.item()
        #OR
        loss= loss+ 0. * sum(p.sum() for p in model.parameters())     # MANTER PARA NÃO TER O ERRO THE RANK
        loss_value = loss.item()

        if not math.isfinite(loss_value):  # This could trigger if using AMP
            print("Loss is {}, stopping training".format(loss_value))
            assert math.isfinite(loss_value)        


        if use_amp:
            # This attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
        else:  # Full precision
            loss /= update_freq
            loss.backward()
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.step()
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)

        torch.cuda.synchronize()

        if mixup_fn is None:
            class_acc = (output.max(-1)[-1] == targets).float().mean()
        else:
            class_acc = None
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        if use_amp:
            metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            if use_amp:
                log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

        if wandb_logger:
            wandb_logger._wandb.log({
                'Rank-0 Batch Wise/train_loss': loss_value,
                'Rank-0 Batch Wise/train_max_lr': max_lr,
                'Rank-0 Batch Wise/train_min_lr': min_lr
            }, commit=False)
            if class_acc:
                wandb_logger._wandb.log({'Rank-0 Batch Wise/train_class_acc': class_acc}, commit=False)
            if use_amp:
                wandb_logger._wandb.log({'Rank-0 Batch Wise/train_grad_norm': grad_norm}, commit=False)
            wandb_logger._wandb.log({'Rank-0 Batch Wise/global_train_step': it})
    
    writer.close()
    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device, use_amp=False):
    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # Switch to evaluation mode
    model.eval()
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # Compute output
        # if use_amp: 
        #     with torch.cuda.amp.autocast(): 
        #         output = model(images)
        #         loss = criterion(output, target)
        # else:
        #     output = model(images)
        #     loss = criterion(output, target)

        #dap_lambda = 0.1   #EVANDRO
        if use_amp:
            with torch.cuda.amp.autocast():
                output, reduce_sim_score = model(images)
                if isinstance(output, tuple): #EVANDRO
                    output = output[0]  #EVANDRO
                loss = criterion(output, target)
        else:
            output, reduce_sim_score = model(images)
            if isinstance(output, tuple):  #EVANDRO
                output = output[0]  #EVANDRO
            loss = criterion(output, target)

        # if loss == -1:
        #     return
        #loss = loss_cls - dap_lambda * reduce_sim_score if reduce_sim_score is not None else loss_cls  #EVANDRO

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]      
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size) 

    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    # EVANDRO: Tensorboard
    # writer.add_scalar('Accuracy/top1', acc1.item(), step)
    # writer.add_scalar('Accuracy/top5', acc5.item(), step)
    # writer.add_scalar('Loss/total_loss', loss.item(), step)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
 