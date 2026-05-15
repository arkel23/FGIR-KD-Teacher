import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from timm.loss import LabelSmoothingCrossEntropy

from .focal_loss import FocalLoss, StudentTeacherDeltaFocalLoss
from .mix import mixup_criterion
from .contrastive_loss import SupConLoss, FocallyModulatedSupConLoss
from .teacher_ratio_loss import TeacherRatioLoss, KLTeacherRatioLoss
from .kd_losses import DKDLoss, RKDLoss, PKTLoss, HintLoss


# Center Loss for Attention Regularization
class CenterLoss(nn.Module):
    def __init__(self):
        super(CenterLoss, self).__init__()
        self.l2_loss = nn.MSELoss(reduction='sum')

    def forward(self, output, targets):
        return self.l2_loss(output, targets) / output.size(0)


# Overall CAL Loss
class CALLoss(nn.Module):
    def __init__(self):
        super(CALLoss, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.center_loss = CenterLoss()

    def forward(self, output, y):
        if isinstance(output, tuple) and len(output) == 7:
            (_, y_pred_raw, y_pred_aux, feature_matrix, feature_center_batch,
             y_pred_aug, _) = output

            y_aug = torch.cat([y, y], dim=0)
            y_aux = torch.cat([y, y_aug], dim=0)
 
            batch_loss = (self.cross_entropy_loss(y_pred_raw, y) / 3. +
                          self.cross_entropy_loss(y_pred_aug, y_aug) * 2. / 3. +
                          self.cross_entropy_loss(y_pred_aux, y_aux) * 3. / 3. +
                          self.center_loss(feature_matrix, feature_center_batch))

        elif isinstance(output, tuple) and len(output) == 2:
            y_pred, _ = output
            batch_loss = self.cross_entropy_loss(y_pred, y)

        else:
            batch_loss = self.cross_entropy_loss(output, y)

        return batch_loss


class OverallLoss(nn.Module):
    def __init__(self, args, kd=False):
        super(OverallLoss, self).__init__()

        self.args = args
        self.t = args.temp

        if args.selector == 'cal' and not kd:
            self.criterion = CALLoss()
        elif args.focal_gamma:
            self.criterion = FocalLoss(args.focal_gamma, smoothing=args.ls)
        elif args.ls:
            self.criterion = LabelSmoothingCrossEntropy(args.smoothing)
        else:
            self.criterion = torch.nn.CrossEntropyLoss()

        if kd:
            self.criterion_kd = torch.nn.KLDivLoss(reduction="batchmean")

            if args.train_both:
                if args.teacher_loss == 'cal':
                    self.criterion_teacher_cal = CALLoss()
                    self.criterion_teacher = True
                elif args.teacher_loss == 'ce':
                    self.criterion_teacher = True

                if args.kd_aux_loss == 'cekd':
                    self.criterion_cekd = True

            if args.kd_aux_loss == 'dkd':
                self.criterion_kd_aux = DKDLoss()
            elif args.kd_aux_loss == 'rkd':
                self.criterion_kd_aux = RKDLoss()
            elif args.kd_aux_loss == 'pkt':
                self.criterion_kd_aux = PKTLoss()
            elif args.kd_aux_loss == 'hint':
                self.criterion_kd_aux = HintLoss()

            elif args.kd_aux_loss == 'trl':
                self.criterion_trl = TeacherRatioLoss(args.cont_focal_gamma)

            elif args.kd_aux_loss == 'kl_trl':
                self.criterion_kl_trl = KLTeacherRatioLoss(args.cont_focal_gamma)

            elif args.kd_aux_loss == 'crd':
                self.criterion_crd = True

            elif args.kd_aux_loss == 'std_focal':
                self.modulation_augs = args.modulation_augs
                self.criterion_std_focal = StudentTeacherDeltaFocalLoss(
                    args.focal_modulation, args.modulation_teacher_labels,
                    args.cont_focal_gamma, args.cont_focal_alpha, args.ls, args.smoothing
                )

            elif args.cont_loss and args.cont_focal_modulation:
                self.focal_modulation = args.cont_focal_modulation
                self.criterion_cont = FocallyModulatedSupConLoss(
                    args.supcon, args.cont_focal_detach, args.device,
                    args.cont_focal_gamma, args.cont_focal_alpha,
                    args.cont_temp, args.cont_base_temp, args.cont_norm_ind)
                print('Modulation for Contrastive Loss: ', self.focal_modulation)

            elif args.cont_loss:
                self.criterion_cont = SupConLoss(
                    args.cont_temp, args.cont_base_temp, args.cont_norm_ind)

    def forward(self, output, targets, output_t=None, y_a=None, y_b=None, lam=None):
        t2 = self.t ** 2

        if hasattr(self, 'criterion_kd') and output_t is not None:

            if self.args.tgda and hasattr(self, 'criterion_teacher_cal'):
                # train teacher with cal (bilinear attention pool/maps) at the same time as student
                loss_teacher = self.criterion_teacher_cal(output_t, targets)

                output, output_aug, _ = output
                _, output_t, _, _, _, output_t_aug, _ = output_t

                loss_kd = self.criterion_kd(
                    F.log_softmax(output / self.t, dim=1),
                    F.softmax(output_t / self.t, dim=1)
                    ) * t2 / 2 + self.criterion_kd(
                    F.log_softmax(output_aug / self.t, dim=1),
                    F.softmax(output_t_aug / self.t, dim=1)
                    ) * t2 * 2 / 2

                if hasattr(self, 'criterion_cekd'):
                    loss_cekd = self.criterion_kd(
                        F.log_softmax(output_t / self.t, dim=1),
                        F.softmax(output / self.t, dim=1)
                        ) * t2 / 2 + self.criterion_kd(
                        F.log_softmax(output_t_aug / self.t, dim=1),
                        F.softmax(output_aug / self.t, dim=1)
                        ) * t2 * 2 / 2


            elif self.args.tgda:
                if isinstance(output, tuple) and len(output) == 4:
                    output, output_aug, feats, _ = output
                    output_t, output_t_aug, _, _, _, _, _ = output_t
                elif isinstance(output, tuple) and len(output) == 3:
                    output, output_aug, _ = output
                    output_t, output_t_aug, _  = output_t


                loss_kd = self.criterion_kd(
                    F.log_softmax(output / self.t, dim=1),
                    F.softmax(output_t / self.t, dim=1)
                    ) * t2 / 2 + self.criterion_kd(
                    F.log_softmax(output_aug / self.t, dim=1),
                    F.softmax(output_t_aug / self.t, dim=1)
                    ) * t2 * 2 / 2

                if hasattr(self, 'criterion_teacher'):
                    loss_teacher = self.criterion(output_t, targets)

                    targets_aug = torch.cat([targets, targets], dim=0)
                    loss_teacher += self.criterion(output_t_aug, targets_aug)

                if hasattr(self, 'criterion_cekd'):
                    loss_cekd = self.criterion_kd(
                        F.log_softmax(output_t / self.t, dim=1),
                        F.softmax(output / self.t, dim=1)
                        ) * t2 / 2 + self.criterion_kd(
                        F.log_softmax(output_t_aug / self.t, dim=1),
                        F.softmax(output_aug / self.t, dim=1)
                        ) * t2 * 2 / 2

                if hasattr(self, 'criterion_kd_aux'):
                    loss_kd_aux = self.criterion_kd_aux(output, output_t, targets)

                if hasattr(self, 'criterion_trl'):
                    loss_trl = self.criterion_trl(output, targets, output_t)
                    target_aug = torch.cat([targets, targets], dim = 0)
                    loss_trl_augs = self.criterion_trl(output_aug, target_aug, output_t_aug)
                    loss_trl = loss_trl / 2 + loss_trl_augs * 2 / 2

                elif hasattr(self, 'criterion_kl_trl'):
                    loss_kl_trl = self.criterion_kl_trl(output, targets, output_t)
                    target_aug = torch.cat([targets, targets], dim = 0)
                    loss_kl_trl_augs = self.criterion_kl_trl(output_aug, target_aug, output_t_aug)
                    loss_kl_trl = loss_kl_trl / 2 + loss_kl_trl_augs * 2 / 2

                if hasattr(self, 'criterion_std_focal') and hasattr(self, 'modulation_augs'):
                    loss_std = ((self.criterion_std_focal(output, output_t, targets)) / 2 +
                        (self.criterion_std_focal(output_aug, output_t_aug, targets) * 2 / 2)
                    )
                elif hasattr(self, 'criterion_std_focal'):
                    loss_std = self.criterion_std_focal(output, output_t, targets)


                if hasattr(self, 'criterion_cont') and hasattr(self, 'focal_modulation') and self.focal_modulation == 'student_teacher':
                    loss_cont = self.criterion_cont(feats, output, targets, output_t)
                elif hasattr(self, 'criterion_cont') and hasattr(self, 'focal_modulation') and self.focal_modulation == 'teacher':
                    loss_cont = self.criterion_cont(feats, output_t, targets)
                elif hasattr(self, 'criterion_cont') and hasattr(self, 'focal_modulation') and self.focal_modulation == 'student':
                    loss_cont = self.criterion_cont(feats, output, targets)
                elif hasattr(self, 'criterion_cont') and self.args.supcon:
                    loss_cont = self.criterion_cont(feats, targets)
                elif hasattr(self, 'criterion_cont'):
                    loss_cont = self.criterion_cont(feats)


            else:
                if isinstance(output, tuple) and len(output) == 3:
                    output, feats, loss_crd = output
                    output_t, _, _ = output_t
                elif isinstance(output, tuple) and len(output) == 2 and hasattr(self, 'criterion_crd'):
                    output, loss_crd = output
                    output_t, _ = output_t
                elif isinstance(output, tuple) and len(output) == 2:
                    output, feats = output
                    output_t, _, _ = output_t


                # can add a modulation term to the kd loss 
                # so that examples with low/high ratio of probabilities 
                # contribute more or less to overall loss
                # https://www.google.com/url?sa=i&url=https%3A%2F%2Fmedium.com%2Fswlh%2Ffocal-loss-what-why-and-how-df6735f26616&psig=AOvVaw0s411w3rMfSQ4f8KF-eU35&ust=1742978608112000&source=images&cd=vfe&opi=89978449&ved=0CBcQjhxqFwoTCJCI5MTrpIwDFQAAAAAdAAAAABAE

                loss_kd = self.criterion_kd(
                    F.log_softmax(output / self.t, dim=1),
                    F.softmax(output_t / self.t, dim=1)
                ) * t2


                if hasattr(self, 'criterion_kd_aux'):
                    loss_kd_aux = self.criterion_kd_aux(output, output_t, targets)

                if hasattr(self, 'criterion_trl'):
                    loss_trl = self.criterion_trl(output, targets, output_t)

                if hasattr(self, 'criterion_kl_trl'):
                    loss_kl_trl = self.criterion_kl_trl(output, targets, output_t)

                if hasattr(self, 'criterion_std_focal'):
                    loss_std = self.criterion_std_focal(output, output_t, targets)
                

                if hasattr(self, 'criterion_cont') and hasattr(self, 'focal_modulation') and self.focal_modulation == 'student_teacher':
                    loss_cont = self.criterion_cont(feats, output, targets, output_t)
                elif hasattr(self, 'criterion_cont') and hasattr(self, 'focal_modulation') and self.focal_modulation == 'teacher':
                    loss_cont = self.criterion_cont(feats, output_t, targets)
                elif hasattr(self, 'criterion_cont') and hasattr(self, 'focal_modulation') and self.focal_modulation == 'student':
                    loss_cont = self.criterion_cont(feats, output, targets)
                elif hasattr(self, 'criterion_cont') and self.args.supcon:
                    loss_cont = self.criterion_cont(feats, targets)
                elif hasattr(self, 'criterion_cont'):
                    loss_cont = self.criterion_cont(feats)


        if y_a is not None:
            loss = mixup_criterion(self.criterion, output, y_a, y_b, lam)
        else:
            loss = self.criterion(output, targets)


        if hasattr(self, 'criterion_kd') and output_t is not None:
            loss = self.args.loss_orig_weight * loss + self.args.loss_kd_weight * loss_kd

        if hasattr(self, 'criterion_teacher') and output_t is not None:
            loss = loss + self.args.loss_teacher_weight * loss_teacher

        if hasattr(self, 'criterion_cekd') and output_t is not None:
            loss = loss + self.args.loss_kd_aux_weight * loss_cekd
        elif hasattr(self, 'criterion_crd') and output_t is not None:
            loss = loss + self.args.loss_kd_aux_weight * loss_crd
        elif hasattr(self, 'criterion_std_focal') and output_t is not None:
            loss = loss + self.args.loss_kd_aux_weight * loss_std
        elif hasattr(self, 'criterion_trl') and output_t is not None:
            loss = loss + self.args.loss_kd_aux_weight * loss_trl
        elif hasattr(self, 'criterion_kl_trl') and output_t is not None:
            loss = loss + self.args.loss_kd_aux_weight * loss_kl_trl
        elif hasattr(self, 'criterion_kd_aux') and output_t is not None:
            # print(loss_kd_aux)
            loss = loss + self.args.loss_kd_aux_weight * loss_kd_aux
            
        if hasattr(self, 'criterion_cont') and output_t is not None:
            loss = loss + self.args.loss_cont_weight * loss_cont

        if self.args.selector == 'cal' and isinstance(output, tuple) and len(output) == 7:
            output, _, _, _, _, _, _ = output
        elif self.args.selector == 'cal' and isinstance(output, tuple) and len(output) == 2:
            output, _ = output


        assert math.isfinite(loss), f'Loss is not finite: {loss}, stopping training'

        return output, loss
