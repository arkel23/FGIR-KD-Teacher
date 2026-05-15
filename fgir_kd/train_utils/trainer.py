import sys
import os.path as osp
from contextlib import suppress

import wandb
import torch
from timm.models import model_parameters

from .misc_utils import AverageMeter, accuracy, count_params_single, count_params_trainable, get_gradients
from .dist_utils import reduce_tensor, distribute_bn
from .mix import get_mix
from .scaler import NativeScaler
from .save_vis_images import save_samples
from .per_class_acc import calc_per_class_acc, calc_class_deviation


class Trainer():
    def __init__(self, args, model, criterion, optimizer, lr_scheduler,
                 train_loader, val_loader, test_loader):
        self.args = args
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.model = model
        self.saved = False
        self.curr_iter = 0

        self.amp_autocast = torch.cuda.amp.autocast if args.fp16 else suppress
        self.loss_scaler = NativeScaler() if args.fp16 else None

    def train(self):
        val_acc = 0
        self.best_acc = 0
        self.best_epoch = 0
        self.max_memory = 0
        self.no_params = 0
        self.no_params_trainable = 0
        self.class_deviation = 0
        self.lr_scheduler.step(0)

        for epoch in range(self.args.epochs):
            self.epoch = epoch + 1

            if self.args.distributed or self.args.ra > 1:
                self.train_loader.sampler.set_epoch(epoch)

            train_acc, train_loss = self.train_epoch()

            if self.args.local_rank == 0 and \
                    ((self.epoch % self.args.eval_freq == 0) or (self.epoch == self.args.epochs)):
                val_acc, val_loss = self.validate_epoch(self.val_loader)

                if self.args.debugging:
                    return None, None, None, None, None, None

                self.epoch_end_routine(train_acc, train_loss, val_acc, val_loss)

        if self.args.local_rank == 0:
            self.train_end_routine(val_acc)

        return (self.best_acc, self.best_epoch, self.max_memory, self.no_params,
                self.no_params_trainable, self.class_deviation)

    def prepare_batch(self, batch):
        images, targets = batch
        if self.args.distributed:
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
        else:
            images = images.to(self.args.device, non_blocking=True)
            targets = targets.to(self.args.device, non_blocking=True)
        return images, targets

    def predict(self, images, targets, train=True):
        images, y_a, y_b, lam = get_mix(images, targets, train, self.args)

        with self.amp_autocast():
            output = self.model(images, targets)

            self.saved = save_samples(images, output, train, self.curr_iter, self.saved, self.args)

            output, loss = self.criterion(output, targets, y_a=y_a, y_b=y_b, lam=lam)

        return output, loss

    def train_epoch(self):
        """vanilla training"""
        self.model.train()

        losses = AverageMeter()
        top1 = AverageMeter()
        topk = AverageMeter()

        for idx, batch in enumerate(self.train_loader):
            images, targets = self.prepare_batch(batch)
            output, loss = self.predict(images, targets, train=True)

            acc1, acck = accuracy(output, targets, topk=(1, self.args.top_k))

            # ===================backward=====================
            if self.args.gradient_accumulation_steps > 1:
                with self.amp_autocast():
                    loss = loss / self.args.gradient_accumulation_steps
            if self.loss_scaler is not None:
                self.loss_scaler.scale_loss(loss)

            if (idx + 1) % self.args.gradient_accumulation_steps == 0:
                if self.loss_scaler is not None:
                    self.loss_scaler(self.optimizer, clip_grad=self.args.clip_grad,
                                     parameters=model_parameters(self.model))
                else:
                    loss.backward()
                    self.optimizer.step()

                if self.args.plot_gradients:
                    gradients = get_gradients(self.model.named_parameters())
                    if not self.args.debugging:
                        wandb.log(gradients)

                self.optimizer.zero_grad()
                self.lr_scheduler.step_update(num_updates=self.curr_iter)

            # ===================meters=====================
            torch.cuda.synchronize()

            if self.args.distributed:
                reduced_loss = reduce_tensor(loss.data, self.args.world_size)
                acc1 = reduce_tensor(acc1, self.args.world_size)
                acck = reduce_tensor(acck, self.args.world_size)
            else:
                reduced_loss = loss.data

            losses.update(reduced_loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            topk.update(acck.item(), images.size(0))

            self.curr_iter += 1

            # print info
            if idx % self.args.log_freq == 0 and self.args.local_rank == 0:
                lr_curr = self.optimizer.param_groups[0]['lr']
                print(
                    'Epoch: [{0}/{1}][{2}/{3}]\t'
                    'LR: {4}\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Acc@{k} {topk.val:.3f} ({topk.avg:.3f})'.format(
                        self.epoch, self.args.epochs, idx, len(self.train_loader), lr_curr,
                        k=self.args.top_k, loss=losses, top1=top1, topk=topk))
                sys.stdout.flush()

                if not self.args.debugging:
                    log_dict = {
                        'lr': lr_curr,
                        'acc_train1': top1.val,
                        'loss_train': losses.val
                    }
                    wandb.log(log_dict, step=self.curr_iter)

            if self.args.debugging:
                return None, None

        if self.args.local_rank == 0:
            print(' * Acc@1 {top1.avg:.3f} Acc@{k} {topk.avg:.3f}'.format(k=self.args.top_k, top1=top1, topk=topk))

        if self.args.distributed:
            distribute_bn(self.model, self.args.world_size, True)

        self.lr_scheduler.step(self.epoch)

        return round(top1.avg, 2), round(losses.avg, 3)

    def validate_epoch(self, val_loader):
        """validation"""
        # switch to evaluate mode
        self.model.eval()

        losses = AverageMeter()
        top1 = AverageMeter()
        topk = AverageMeter()

        if self.epoch == self.args.epochs:
            self.curr_img = 0
            class_correct = [0 for _ in range(self.args.num_classes)]
            class_total = [0 for _ in range(self.args.num_classes)]
            dic_preds = {}

        with torch.no_grad():
            for idx, batch in enumerate(val_loader):
                images, targets = self.prepare_batch(batch)
                output, loss = self.predict(images, targets, train=False)

                acc1, acck = accuracy(output, targets, topk=(1, self.args.top_k))

                torch.cuda.synchronize()

                if self.epoch == self.args.epochs:
                    self.curr_img = calc_per_class_acc(
                        self.args, self.curr_img, output, targets, class_correct,
                        class_total, images, dic_preds)

                reduced_loss = loss.data

                losses.update(reduced_loss.item(), images.size(0))
                top1.update(acc1.item(), images.size(0))
                topk.update(acck.item(), images.size(0))

                if idx % self.args.log_freq == 0 and self.args.local_rank == 0:
                    print('Val: [{0}/{1}]\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Acc@{k} {topk.val:.3f} ({topk.avg:.3f})'.format(
                              idx, len(val_loader), k=self.args.top_k,
                              loss=losses, top1=top1, topk=topk))

                if self.args.debugging:
                    return None, None

        if self.epoch == self.args.epochs:
            self.class_deviation = calc_class_deviation(self.args, class_correct, class_total, dic_preds)

        if self.args.local_rank == 0:
            print(' * Acc@1 {top1.avg:.3f} Acc@{k} {topk.avg:.3f}'.format(k=self.args.top_k, top1=top1, topk=topk))

        return round(top1.avg, 2), round(losses.avg, 3)

    def epoch_end_routine(self, train_acc, train_loss, val_acc, val_loss):
        lr_curr = self.optimizer.param_groups[0]['lr']
        print("Training...Epoch: {} | LR: {}".format(self.epoch, lr_curr))
        log_dic = {'epoch': self.epoch, 'lr': lr_curr,
                   'train_acc': train_acc, 'train_loss': train_loss,
                   'val_acc': val_acc, 'val_loss': val_loss}
        if not self.args.offline:
            wandb.log(log_dic)

        # save the best model
        if val_acc > self.best_acc:
            self.best_acc = val_acc
            self.best_epoch = self.epoch
            self.save_model(self.best_epoch, val_acc, mode='best')
        # regular saving
        if self.epoch % self.args.save_freq == 0:
            self.save_model(self.epoch, val_acc, mode='epoch')

    def train_end_routine(self, val_acc):
        # save last
        self.save_model(self.epoch, val_acc, mode='last')
        # VRAM and No. of params
        self.computation_stats()

    def computation_stats(self):
        # VRAM memory consumption
        self.max_memory = torch.cuda.max_memory_reserved() / (1024 ** 3)
        # summary stats
        self.no_params = count_params_single(self.model)
        self.no_params_trainable = count_params_trainable(self.model)

    def save_model(self, epoch, acc, mode):
        state = {
            'config': self.args,
            'epoch': epoch,
            'model': self.model.state_dict(),
            'accuracy': acc,
            'optimizer': self.optimizer.state_dict(),
        }

        if mode == 'best':
            save_file = osp.join(self.args.results_dir, f'{self.args.model_name}_best.pth')
            print('Saving the best model!')
            torch.save(state, save_file)
        elif mode == 'epoch':
            save_file = osp.join(self.args.results_dir, f'ckpt_epoch_{epoch}.pth')
            print('==> Saving each {} epochs...'.format(self.args.save_freq))
            torch.save(state, save_file)
        elif mode == 'last':
            save_file = osp.join(self.args.results_dir, f'{self.args.model_name}_last.pth')
            print('Saving last epoch')
            torch.save(state, save_file)

    def test(self):
        print(f'Evaluation on test dataloader: ')
        self.epoch = self.args.epochs
        test_acc, _ = self.validate_epoch(self.test_loader)
        self.computation_stats()

        if self.args.test_multiple:
            self.epoch = 0
            for i in range(self.args.test_multiple):
                print(f'Testing multiple times: {i}/{self.args.test_multiple}')
                test_acc, _ = self.validate_epoch(self.test_loader)

        if self.args.debugging:
            return None, None, None, None, None
        return (test_acc, self.max_memory, self.no_params,
                self.no_params_trainable, self.class_deviation)
