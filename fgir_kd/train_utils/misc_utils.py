import random

import numpy as np
import torch
import wandb
from statistics import mean, stdev

from torchprofile import profile_macs
# from ptflops import get_model_complexity_info


def count_flops(model, image_size=224, device='cuda', profiler='torchprofile', debugging=False):
    # https://github.com/Lyken17/pytorch-OpCounter
    # https://github.com/sovrasov/flops-counter.pytorch
    # https://github.com/zhijian-liu/torchprofile
    model.eval()
    with torch.no_grad():
        x = torch.rand(1, 3, image_size, image_size).to(device)
        if profiler == 'torchprofile':
            flops = profile_macs(model, x)
        # elif profiler == 'ptflops':
        #     flops, _ = get_model_complexity_info(
        #         model, tuple(x.shape[1:]), as_strings=False,
        #         print_per_layer_stat=debugging, verbose=debugging)
        else:
            raise NotImplemented

        flops = round(flops / 1e9, 4)
    print('FLOPs (G): ', flops)
    return flops


def count_params_module_list(module_list):
    return sum([count_params_single(model) for model in module_list])


def count_params_single(model):
    return sum([p.numel() for p in model.parameters()])


def count_params_trainable(model):
    return sum([p.numel() for p in model.parameters() if p.requires_grad])


def set_random_seed(seed=0, numpy=True):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if numpy:
        np.random.seed(seed)
    return 0


def summary_stats(epochs, time_total, best_acc, best_epoch, flops, max_memory,
                  no_params, no_params_trainable, class_deviation, offline=False):
    time_avg = round((time_total / epochs) / 60, 4)
    best_time = round((time_avg * best_epoch) / 60, 4)
    time_total = round(time_total / 60, 4)  # mins
    no_params = round(no_params / (1e6), 4)  # millions of parameters
    no_params_trainable = round(no_params_trainable / (1e6), 4)  # millions
    max_memory = round(max_memory, 4)

    print('''Total run time (minutes): {}
          Average time per epoch (minutes): {}
          Best accuracy (%): {} at epoch {}. Time to reach this (minutes): {}
          Class deviation: {}
          FLOPs (G): {}
          Max VRAM consumption (GB): {}
          Total number of parameters in all modules (M): {}
          Trainable number of parameters in all modules (M): {}
          '''.format(time_total, time_avg, best_acc, best_epoch, best_time,
                     class_deviation, flops, max_memory, no_params, no_params_trainable))

    if not offline:
        wandb.run.summary['time_total'] = time_total
        wandb.run.summary['time_avg'] = time_avg
        wandb.run.summary['best_acc'] = best_acc
        wandb.run.summary['best_epoch'] = best_epoch
        wandb.run.summary['best_time'] = best_time
        wandb.run.summary['flops'] = flops
        wandb.run.summary['max_memory'] = max_memory
        wandb.run.summary['no_params'] = no_params
        wandb.run.summary['no_params_trainable'] = no_params_trainable
        wandb.run.summary['class_deviation'] = class_deviation
    return 0



def stats_test(test_acc, class_deviation, flops, max_memory, no_params,
               no_params_trainable, time_total, num_images, offline=False):

    throughput = round(num_images / time_total, 4)
    no_params = round(no_params / (1e6), 4)  # millions of parameters
    no_params_trainable = round(no_params_trainable / (1e6), 4)  # millions
    max_memory = round(max_memory, 4)

    print('''Throughput (images / s): {}
          Test accuracy (%): {}
          Class deviation (%): {}
          FLOPs (G): {}
          Max VRAM consumption (GB): {}
          Total number of parameters in all modules (M): {}
          Trainable number of parameters in all modules (M): {}
          '''.format(throughput, test_acc, class_deviation, flops, max_memory, no_params, no_params_trainable))

    if not offline:
        wandb.run.summary['test_acc'] = test_acc
        wandb.run.summary['class_deviation'] = class_deviation
        wandb.run.summary['throughput'] = throughput
        wandb.run.summary['flops'] = flops
        wandb.run.summary['max_memory'] = max_memory
        wandb.run.summary['no_params'] = no_params
        wandb.run.summary['no_params_trainable'] = no_params_trainable
    return 0


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions
    for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_gradients(named_parameters):
    avg_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            avg_grads.append(p.grad.abs().mean().item())
            max_grads.append(p.grad.abs().max().item())

    max_grad = max(max_grads)
    mean_max_grad = mean(max_grads)
    std_max_grad = stdev(max_grads)
    
    mean_avg_grad = mean(avg_grads)
    std_avg_grad = stdev(avg_grads)

    gradients = {
        'max_grad_all': max_grad,
        'mean_max_grad_layer': mean_max_grad,
        'std_max_grad_layer': std_max_grad,
        'mean_avg_grad_layer': mean_avg_grad,
        'std_avg_grad_layer': std_avg_grad,
    }

    return gradients
