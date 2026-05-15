import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class TeacherRatioLoss(nn.Module):
    '''
    https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
    https://github.com/rwightman/pytorch-image-models/blob/main/timm/loss/cross_entropy.py
    https://github.com/jimitshah77/plant-pathology/blob/master/bilinear-efficientnet-focal-loss-label-smoothing.ipynb
    https://amaarora.github.io/2020/06/29/FocalLoss.html
    '''
    def __init__(self, gamma=2.0):
        super(TeacherRatioLoss, self).__init__()
        self.gamma = gamma
        print('Teacher ratio loss')

    def forward(self, logits, target, teacher_logits):
        logprobs = F.log_softmax(logits, dim=-1)

        nll_loss = logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)

        # implement the ratios metric and the entropy
        sorted_output,_ = torch.topk(teacher_logits, 2, largest=True, sorted=True) # sort the input predictions tensor
        top_pred = sorted_output[:, 0]
        second_pred = sorted_output[:, 1]

        # calculate the ratios
        ratios = top_pred / second_pred
        # print(top_pred, second_pred, ratios)

        # 1 (50% 1st, 50% 2nd), 10 (90% 1st, 10% 2nd)
        # modulation term depends on the statistics of the batch
        # modulation_term = (1 - (ratios - ratios.min()) / (ratios.max() - ratios.min()))
        modulation_term = (1 - F.softmax(ratios, dim=0))
        modulation_term = modulation_term ** self.gamma

        # original focal loss
        # pt = nll_loss.data.exp()
        # modulation_term = ((1 - pt) ** self.gamma)

        loss = - modulation_term * nll_loss
        return loss.mean()
    

class KLTeacherRatioLoss(nn.Module):
    '''
    https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
    https://github.com/jimitshah77/plant-pathology/blob/master/bilinear-efficientnet-focal-loss-label-smoothing.ipynb
    https://amaarora.github.io/2020/06/29/FocalLoss.html
    https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
    https://pytorch.org/docs/stable/generated/torch.nn.functional.kl_div.html
    '''
    def __init__(self, gamma=2.0):
        super(KLTeacherRatioLoss, self).__init__()
        self.gamma = gamma
        print('KL Teacher ratio loss')

    def forward(self, logits, target, teacher_logits):
        # from the equation of KL div = (p * (p/q).log()).sum()
        # logprobs = F.softmax(logits, dim=-1) # student probs
        # t_probs = F.softmax(teacher_logits, dim=-1) # teacher probs
        # kl_loss = (t_probs * (t_probs/logprobs).log())

        logprobs = F.log_softmax(logits, dim=-1)
        t_probs = F.softmax(teacher_logits, dim=-1)

        # input = log probs, target = softmax probs
        kl_loss = nn.functional.kl_div(logprobs, t_probs, reduction='none')

        # implement the ratios metric and the entropy
        sorted_output,_ = torch.topk(teacher_logits, 2, largest=True, sorted=True) # sort the input predictions tensor
        top_pred = sorted_output[:, 0]
        second_pred = sorted_output[:, 1]

        # calculate the ratios
        ratios = top_pred / second_pred
        # print(top_pred, second_pred, ratios)

        # 1 (50% 1st, 50% 2nd), 10 (90% 1st, 10% 2nd)
        # modulation term depends on the statistics of the batch
        # modulation_term = (1 - (ratios - ratios.min()) / (ratios.max() - ratios.min()))
        modulation_term = (1 - F.softmax(ratios, dim=0))
        modulation_term = modulation_term ** self.gamma

        # original focal loss
        # pt = nll_loss.data.exp()
        # modulation_term = ((1 - pt) ** self.g"amma)

        modulation_term = rearrange(modulation_term, 'b -> b 1' )
        loss = modulation_term * kl_loss
        return loss.mean()



class FocalLoss(nn.Module):
    '''
    https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
    https://github.com/rwightman/pytorch-image-models/blob/main/timm/loss/cross_entropy.py
    https://github.com/jimitshah77/plant-pathology/blob/master/bilinear-efficientnet-focal-loss-label-smoothing.ipynb
    https://amaarora.github.io/2020/06/29/FocalLoss.html
    '''
    def __init__(self, gamma=2.0, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        elif isinstance(alpha,list):
            # weight for the classes (either do not use or use class distribution)
            self.alpha = torch.Tensor(alpha)

    def forward(self, logits, target):
        logprobs = F.log_softmax(logits, dim=-1)

        nll_loss = logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)

        # imbalanced data: [10, 100, 1000, 10000]
        # alpha: [10000+1000+100+10/10, 10000+1000+100+10/100, 10000+1000+100+10/1000, 10000+1000+100+10/10000]
        if self.alpha is not None:
            if self.alpha.type() != logits.type():
                self.alpha = self.alpha.type_as(logits.data)
            at = self.alpha.gather(0, target.view(-1))
            nll_loss = nll_loss * at

        pt = nll_loss.data.exp()

        modulation_term = ((1 - pt) ** self.gamma)
        loss = - modulation_term * nll_loss
        return loss.mean()


class CELossBarebones(nn.Module):
    '''
    https://github.com/rwightman/pytorch-image-models/blob/main/timm/loss/cross_entropy.py
    '''
    def __init__(self):
        super(CELossBarebones, self).__init__()
        # this is basically what happens under the hood in nn.CrossEntropyLoss()

    def forward(self, logits, target):
        # logits are the outputs of a classifier (unnormalized)
        # 4 classes tensor: [15, 10, 3, 1]
        logprobs = F.log_softmax(logits, dim=-1)

        # after softmax: [0.8, 0.1, 0.08, 0.02]
        # after log: [-0.01, -1, -1.1, -1.69]
        # targets: a tensor with the index corresponding to the class label [3]

        # nll_loss = F.nll_loss(logprobs, target, reduction='none')
        # the nll loss basically does the four steps below

        nll_loss = logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)

        loss = - nll_loss
        return loss.mean()
