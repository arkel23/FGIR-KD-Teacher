import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    '''
    https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
    https://github.com/rwightman/pytorch-image-models/blob/main/timm/loss/cross_entropy.py
    https://github.com/jimitshah77/plant-pathology/blob/master/bilinear-efficientnet-focal-loss-label-smoothing.ipynb
    https://amaarora.github.io/2020/06/29/FocalLoss.html
    '''
    def __init__(self, gamma=2.0, alpha=None, smoothing=False, ls=0.1, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        elif isinstance(alpha,list):
            # weight for the classes (either do not use or use class distribution)
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average
        self.ls = ls if smoothing else 0

    def forward(self, logits, target):
        logprobs = F.log_softmax(logits, dim=-1)

        # this is equivalent to 
        # nll_loss = F.nll_loss(logprobs, target, reduction='none')
        nll_loss = logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)

        if self.alpha is not None:
            if self.alpha.type() != logits.type():
                self.alpha = self.alpha.type_as(logits.data)
            at = self.alpha.gather(0, target.view(-1))
            nll_loss = nll_loss * at

        if self.ls > 0:
            nll_loss = (1 - self.ls) * nll_loss + self.ls * (logprobs.mean(dim=-1))

        pt = nll_loss.data.exp()

        modulation_term = ((1 - pt) ** self.gamma)
        loss = - modulation_term * nll_loss
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class StudentTeacherDeltaFocalLoss(nn.Module):
    '''
    https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
    https://github.com/rwightman/pytorch-image-models/blob/main/timm/loss/cross_entropy.py
    https://github.com/jimitshah77/plant-pathology/blob/master/bilinear-efficientnet-focal-loss-label-smoothing.ipynb
    https://amaarora.github.io/2020/06/29/FocalLoss.html
    '''
    def __init__(self, focal_modulation=None, teacher_labels=0, gamma=2.0, alpha=None, smoothing=False, ls=0.1, size_average=True):
        super(StudentTeacherDeltaFocalLoss, self).__init__()
        self.focal_modulation = focal_modulation if focal_modulation else 'student_teacher'
        self.teacher_labels = teacher_labels
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        elif isinstance(alpha,list):
            # weight for the classes (either do not use or use class distribution)
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average
        self.ls = ls if smoothing else 0
        print(f'Using {self.focal_modulation} Modulated Focal Loss with {teacher_labels} teacher labels')

    def forward(self, logits, logits_t, target):
        logprobs = F.log_softmax(logits, dim=-1)
        logprobs_t = F.log_softmax(logits_t, dim=-1)

        if self.teacher_labels:
            nll_loss_t, teacher_labels = torch.topk(logprobs_t, k=self.teacher_labels, dim=-1, largest=True, sorted=True)

            nll_loss = logprobs.gather(dim=-1, index=teacher_labels)
            # nll_loss_t = logprobs_t.gather(dim=-1, index=teacher_labels)

            pt = torch.exp(nll_loss)
            pt_t = torch.exp(nll_loss_t)

            if self.focal_modulation == 'student_teacher':
                modulation = torch.abs(pt_t - pt)
            elif self.focal_modulation == 'student':
                modulation = (1 - pt)
            elif self.focal_modulation == 'teacher':
                modulation = (1 - pt_t)
            modulation_term = (modulation ** self.gamma)
            loss = - modulation_term * nll_loss
            # print(loss, modulation_term, teacher_labels, nll_loss, nll_loss_t, pt, pt_t)

        else:
            # this is equivalent to
            # nll_loss = F.nll_loss(logprobs, target, reduction='none')
            nll_loss = logprobs.gather(dim=-1, index=target.unsqueeze(1))
            nll_loss = nll_loss.squeeze(1)

            nll_loss_t = logprobs_t.gather(dim=-1, index=target.unsqueeze(1))
            nll_loss_t = nll_loss_t.squeeze(1)

            if self.alpha is not None:
                if self.alpha.type() != logits.type():
                    self.alpha = self.alpha.type_as(logits.data)
                at = self.alpha.gather(0, target.view(-1))
                nll_loss = nll_loss * at

            if self.ls > 0:
                nll_loss = (1 - self.ls) * nll_loss + self.ls * (logprobs.mean(dim=-1))

            pt = torch.exp(nll_loss)
            pt_t = torch.exp(nll_loss_t)

            if self.focal_modulation == 'student_teacher':
                modulation = torch.abs(pt_t - pt)
            elif self.focal_modulation == 'student':
                modulation = (1 - pt)
            elif self.focal_modulation == 'teacher':
                modulation = (1 - pt_t)
            modulation_term = (modulation ** self.gamma)
            loss = - modulation_term * nll_loss
            # print(loss, modulation_term, nll_loss, nll_loss_t, pt, pt_t)

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
