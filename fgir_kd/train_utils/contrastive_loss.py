import torch
import torch.nn as nn
from torch.nn import functional as F

from einops import repeat


class SupConLoss(nn.Module):
    # Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf
    # Contrastive Deep Supervision also used same SupConLoss
    # https://github.com/ArchipLab-LinfengZhang/contrastive-deep-supervision/blob/main/ImageNet/Contrastive_Deep_Supervision/loss.py
    def __init__(self, temperature=0.07, base_temperature=None, norm_ind=False):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = temperature if base_temperature is None else base_temperature
        self.norm_ind = norm_ind

    def forward(self, features, labels=None):
        # features shape: B, N_POSITIVES/VIEWS, D
        device = features.device

        # normalize each group of features individually as in CDS
        # https://github.com/ArchipLab-LinfengZhang/contrastive-deep-supervision/blob/main/ImageNet/Contrastive_Deep_Supervision/resnet.py
        if self.norm_ind:
            features = torch.split(features, 1, dim=1)
            features = [F.normalize(ft, dim=2) for ft in features]
            features = torch.cat(features, dim=1)
        else:
            features = F.normalize(features, dim=2)

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                            'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]

        if labels is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        else:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)

        # B, NP, D
        contrast_count = features.shape[1]
        # contrast_feature = rearrange(features, 'b np d -> (np b) d')
        # B, NP, D - > tuple of tensor B, D of length NP -> B*NP, D
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss


class FocallyModulatedSupConLoss(nn.Module):
    # Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf
    # Contrastive Deep Supervision also used same SupConLoss
    # https://github.com/ArchipLab-LinfengZhang/contrastive-deep-supervision/blob/main/ImageNet/Contrastive_Deep_Supervision/loss.py
    def __init__(self, supcon=False, detach=False, device='cpu',
                 gamma=2.0, alpha=None,
                 temperature=0.07, base_temperature=0.07, norm_ind=False):
        super(FocallyModulatedSupConLoss, self).__init__()

        self.supcon = supcon
        self.device = device
        self.temperature = temperature
        self.base_temperature = temperature if base_temperature is None else base_temperature
        self.norm_ind = norm_ind

        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        elif isinstance(alpha,list):
            # weight for the classes (either do not use or use class distribution)
            self.alpha = torch.Tensor(alpha)

        self.detach_modulation_factor = detach

        print('Using Focally Modulated SupCon Loss')

    def forward(self, features, preds, labels, preds2=None):
        # features need to have shape B, POSITIVES, D
        # labels if used need to have same length as first dimension B

        # normalize each group of features individually as in CDS
        # https://github.com/ArchipLab-LinfengZhang/contrastive-deep-supervision/blob/main/ImageNet/Contrastive_Deep_Supervision/resnet.py
        if self.norm_ind:
            features = torch.split(features, 1, dim=1)
            features = [F.normalize(ft, dim=2) for ft in features]
            features = torch.cat(features, dim=1)
        else:
            features = F.normalize(features, dim=2)

        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
        elif len(features.shape) == 2 and labels is not None and self.supcon:
            features = features.unsqueeze(1)
        elif len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                            'at least 3 dimensions are required')

        batch_size = features.shape[0]
        n_pos = features.shape[1]


        if preds2 is not None:
            # focal modulation
            preds_s = repeat(preds, 'b d -> (np b) d', np=n_pos)
            preds_t = repeat(preds2, 'b d -> (np b) d', np=n_pos)

            # focal loss cross-entropy: log-softmax + negative log likelihood
            logprobs_s = F.log_softmax(preds_s, dim=-1)
            logprobs_t = F.log_softmax(preds_t, dim=-1)

            # Get probabilities from logits
            labels_repeat = repeat(labels, 'b -> (np b) 1', np=n_pos)
            nll_loss_s = logprobs_s.gather(dim=-1, index=labels_repeat)
            nll_loss_t = logprobs_t.gather(dim=-1, index=labels_repeat)

            # Probability of the true class (shape: B*NP)
            nll_loss_s = nll_loss_s.squeeze(1)
            nll_loss_t = nll_loss_t.squeeze(1)

            # weights (in case of class imbalance)
            if self.alpha is not None:
                if self.alpha.type() != input.type():
                    self.alpha = self.alpha.type_as(input.data)
                at = self.alpha.gather(0, labels.view(-1))
                nll_loss_s = nll_loss_s * at

            pt_s = torch.exp(nll_loss_s)
            pt_t = torch.exp(nll_loss_t)

            # focal loss modulation factor
            modulation_term = (torch.abs(pt_t - pt_s) ** self.gamma)

        else:
            # focal modulation
            preds = repeat(preds, 'b d -> (np b) d', np=n_pos)

            # focal loss cross-entropy: log-softmax + negative log likelihood
            logprobs = F.log_softmax(preds, dim=-1)

            # Get probabilities from logits
            labels_repeat = repeat(labels, 'b -> (np b) 1', np=n_pos)
            nll_loss = logprobs.gather(dim=-1, index=labels_repeat)

            # Probability of the true class (shape: B*NP)
            nll_loss = nll_loss.squeeze(1)

            # weights (in case of class imbalance)
            if self.alpha is not None:
                if self.alpha.type() != input.type():
                    self.alpha = self.alpha.type_as(input.data)
                at = self.alpha.gather(0, labels.view(-1))
                nll_loss = nll_loss * at

            # focal loss modulation factor
            pt = nll_loss.data.exp()
            modulation_term = ((1 - pt) ** self.gamma)


        if labels is None or not self.supcon:
            mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
        else:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(self.device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(self.device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # multiply by modulation factor
        # detach the modulation_term so that the modulation_term only depends on the classification but
        # still influences the contrastive loss
        if self.detach_modulation_factor:
            log_prob = modulation_term.detach() * log_prob
        else:
            log_prob = modulation_term * log_prob

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
