import collections
import numpy as np
from abc import ABC
import torch
import torch.nn.functional as F
from torch import nn, autograd
class CM(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)


        for x, y in zip(inputs, targets):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None


def cm(inputs, indexes, features, momentum=0.5):
    return CM.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class CM_Hard(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, targets.tolist()):
            batch_centers[index].append(instance_feature)

        for index, features in batch_centers.items():
            distances = []
            for feature in features:
                distance = feature.unsqueeze(0).mm(ctx.features[index].unsqueeze(0).t())[0][0]
                distances.append(distance.cpu().numpy())

            median = np.argmin(np.array(distances))
            ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * features[median]
            ctx.features[index] /= ctx.features[index].norm()

        return grad_inputs, None, None, None


def cm_hard(inputs, indexes, features, momentum=0.5):
    return CM_Hard.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


def _weighted_mean(losses, weights=None, eps=1e-12):
    if weights is None:
        return losses.mean()
    weights = weights.to(losses.device).float()
    return (losses * weights).sum() / weights.sum().clamp_min(eps)

    
class ClusterMemory_teacher(nn.Module, ABC):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2, use_hard=False, lambda2=None):
        super(ClusterMemory_teacher, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples
        self.momentum = momentum
        self.temp = temp
        self.use_hard = use_hard
        self.lambda2 = lambda2


    def forward(self, inputs, inputs_up, inputs_down, inputs_mix, targets, epoch, sample_weights=None):

        inputs = F.normalize(inputs, dim=1).cuda()
        inputs_up = F.normalize(inputs_up, dim=1).cuda()
        inputs_down = F.normalize(inputs_down, dim=1).cuda()
        inputs_mix = F.normalize(inputs_mix, dim=1).cuda()

        if epoch == -1:
            outputs = cm(inputs, targets, self.features, 1.0)
            outputs_up = cm(inputs_up, targets, self.features_up, 1.0)
            outputs_down = cm(inputs_down, targets, self.features_down, 1.0)
        else:
            outputs = cm(inputs, targets, self.features, self.momentum)
            outputs_up = cm(inputs_up, targets, self.features_up, self.momentum)
            outputs_down = cm(inputs_down, targets, self.features_down, self.momentum)

        outputs /= self.temp
        outputs_up /= self.temp
        outputs_down /= self.temp

        loss_main = F.cross_entropy(outputs, targets, reduction='none')
        loss_up = F.cross_entropy(outputs_up, targets, reduction='none')
        loss_down = F.cross_entropy(outputs_down, targets, reduction='none')
        loss = (
            (1.0 - self.lambda2) * _weighted_mean(loss_main, sample_weights)
            + self.lambda2 * _weighted_mean(loss_up, sample_weights)
            + self.lambda2 * _weighted_mean(loss_down, sample_weights)
        )
        return loss

class ClusterMemory(nn.Module, ABC):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2, use_hard=False, lambda2=None, mu=None):
        super(ClusterMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples
        self.momentum = momentum
        self.temp = temp
        self.use_hard = use_hard
        self.lambda2 = lambda2
        self.mu = mu

    def forward(self, inputs, inputs_up, inputs_down, inputs_teacher, inputs_up_teacher, inputs_down_teacher, targets, epoch, sample_weights=None):

        inputs = F.normalize(inputs, dim=1).cuda()
        inputs_up = F.normalize(inputs_up, dim=1).cuda()
        inputs_down = F.normalize(inputs_down, dim=1).cuda()

        if epoch == -1:
            outputs = cm(inputs, targets, self.features, 1.0)
            outputs_up = cm(inputs_up, targets, self.features_up, 1.0)
            outputs_down = cm(inputs_down, targets, self.features_down, 1.0)
        else:
            outputs = cm(inputs, targets, self.features, self.momentum)
            outputs_up = cm(inputs_up, targets, self.features_up, self.momentum)
            outputs_down = cm(inputs_down, targets, self.features_down, self.momentum)

        outputs /= self.temp
        outputs_up /= self.temp
        outputs_down /= self.temp

        loss_distill = nn.MSELoss(reduction='none')(inputs, inputs_teacher).sum(dim=1)
        loss_distill_up = nn.MSELoss(reduction='none')(inputs_up, inputs_up_teacher).sum(dim=1)
        loss_distill_down = nn.MSELoss(reduction='none')(inputs_down, inputs_down_teacher).sum(dim=1)
        loss = (
            self.mu * _weighted_mean(loss_distill, sample_weights)
            + self.mu * _weighted_mean(loss_distill_up, sample_weights)
            + self.mu * _weighted_mean(loss_distill_down, sample_weights)
        )

        return loss
