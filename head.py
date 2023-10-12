from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
import random
import numpy as np


class Normalized_Softmax_Loss(nn.Module):
    def __init__(self, in_features, out_features, m=0.4, s=64):
        super(Normalized_Softmax_Loss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.s = s
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.kaiming_normal_(self.weight, a=1, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, input, label):
        cos_theta = F.linear(F.normalize(input, eps=1e-5), F.normalize(self.weight, eps=1e-5))

        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros((label.size(0), self.out_features), dtype=torch.bool)
        one_hot = one_hot.cuda() if cos_theta.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        d_theta = one_hot * self.m
        logits = self.s * (cos_theta - d_theta)
        return F.cross_entropy(logits, label)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', m=' + str(self.m) \
               + ', s=' + str(self.s) + ')'


class Normalized_BCE_Loss(nn.Module):
    def __init__(self, in_features, out_features, m=0.4, s=64):
        super(Normalized_BCE_Loss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.s = s
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.kaiming_normal_(self.weight, a=1, mode='fan_in', nonlinearity='leaky_relu')
        self.bias = Parameter(torch.FloatTensor(1, out_features))
        nn.init.constant_(self.bias, math.log(out_features*10))

    def forward(self, input, label):
        cos_theta = F.linear(F.normalize(input, eps=1e-5), F.normalize(self.weight, eps=1e-5))

        cos_m_theta_p = self.s * (cos_theta - self.m) - self.bias
        cos_m_theta_n = self.s * cos_theta - self.bias
        p_loss = torch.log(1 + torch.exp(-cos_m_theta_p.clamp(min=-self.s, max=self.s)))
        n_loss = torch.log(1 + torch.exp(cos_m_theta_n.clamp(min=-self.s, max=self.s)))

        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros((label.size(0), self.out_features), dtype=torch.bool)
        one_hot = one_hot.cuda() if cos_theta.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        loss = one_hot * p_loss + (~one_hot) * n_loss

        return loss.sum(dim=1).mean()

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', m=' + str(self.m) \
               + ', s=' + str(self.s) + ')'


class Unified_Cross_Entropy_Loss(nn.Module):
    def __init__(self, in_features, out_features, m=0.4, s=64):
        super(Unified_Cross_Entropy_Loss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.s = s
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.kaiming_normal_(self.weight, a=1, mode='fan_in', nonlinearity='leaky_relu')
        self.bias = Parameter(torch.FloatTensor(1))
        nn.init.constant_(self.bias, math.log(out_features*10))

    def forward(self, input, label):
        cos_theta = F.linear(F.normalize(input, eps=1e-5), F.normalize(self.weight, eps=1e-5))

        cos_m_theta_p = self.s * (cos_theta - self.m) - self.bias
        cos_m_theta_n = self.s * cos_theta - self.bias
        p_loss = torch.log(1 + torch.exp(-cos_m_theta_p.clamp(min=-self.s, max=self.s)))
        n_loss = torch.log(1 + torch.exp(cos_m_theta_n.clamp(min=-self.s, max=self.s)))

        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros((label.size(0), self.out_features), dtype=torch.bool)
        one_hot = one_hot.cuda() if cos_theta.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        loss = one_hot * p_loss + (~one_hot) * n_loss

        return loss.sum(dim=1).mean()

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', m=' + str(self.m) \
               + ', s=' + str(self.s) + ')'


class Sample_to_Sample_Based_Softmax_Loss(nn.Module):
    def __init__(self, in_features, out_features, m=0.1, s=64):
        super(Sample_to_Sample_Based_Softmax_Loss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.s = s
        self.register_buffer('feat_mem', torch.FloatTensor(out_features, in_features))

    def forward(self, input, label):
        cos_theta = F.linear(F.normalize(input, eps=1e-5), F.normalize(self.feat_mem, eps=1e-5))

        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros((label.size(0), self.out_features), dtype=torch.bool)
        one_hot = one_hot.cuda() if cos_theta.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        d_theta = one_hot * self.m
        logits = self.s * (cos_theta - d_theta)
        return F.cross_entropy(logits, label)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', m=' + str(self.m) \
               + ', s=' + str(self.s) + ')'


class Sample_to_Sample_Based_BCE_Loss(nn.Module):
    def __init__(self, in_features, out_features, m=0.1, s=64):
        super(Sample_to_Sample_Based_BCE_Loss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.s = s
        self.register_buffer('feat_mem', torch.FloatTensor(out_features, in_features))
        self.bias = Parameter(torch.zeros(1, out_features))

    def forward(self, input, label):
        cos_theta = F.linear(F.normalize(input, eps=1e-5), F.normalize(self.feat_mem, eps=1e-5))

        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros((label.size(0), self.out_features), dtype=torch.bool)
        one_hot = one_hot.cuda() if cos_theta.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        if self.bias.data.mean() <= 0:
            cos_n = cos_theta[~one_hot].view(label.size(0), -1)
            cos_txs = torch.logsumexp(self.s*cos_n, dim=-1)
            self.bias.data += cos_txs.mean().unsqueeze(-1).float().data

        cos_m_theta_p = self.s * (cos_theta - self.m) - self.bias
        cos_m_theta_n = self.s * cos_theta - self.bias
        p_loss = torch.log(1 + torch.exp(-cos_m_theta_p.clamp(min=-self.s, max=self.s)))
        n_loss = torch.log(1 + torch.exp(cos_m_theta_n.clamp(min=-self.s, max=self.s)))

        loss = one_hot * p_loss + (~one_hot) * n_loss

        return loss.sum(dim=1).mean()

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', m=' + str(self.m) \
               + ', s=' + str(self.s) + ')'


class Unified_Threshold_Integrated_Sample_to_Sample_Loss(nn.Module):
    def __init__(self, in_features, out_features, m=0.1, s=64):
        super(Unified_Threshold_Integrated_Sample_to_Sample_Loss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.s = s
        self.register_buffer('feat_mem', torch.FloatTensor(out_features, in_features))
        self.bias = Parameter(torch.zeros(1))

    def forward(self, input, label):
        cos_theta = F.linear(F.normalize(input, eps=1e-5), F.normalize(self.feat_mem, eps=1e-5))

        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros((label.size(0), self.out_features), dtype=torch.bool)
        one_hot = one_hot.cuda() if cos_theta.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        if self.bias.data <= 0:
            cos_n = cos_theta[~one_hot].view(label.size(0), -1)
            cos_txs = torch.logsumexp(self.s*cos_n, dim=-1)
            self.bias.data = cos_txs.mean().unsqueeze(-1).float().data

        cos_m_theta_p = self.s * (cos_theta - self.m) - self.bias
        cos_m_theta_n = self.s * cos_theta - self.bias
        p_loss = torch.log(1 + torch.exp(-cos_m_theta_p.clamp(min=-self.s, max=self.s)))
        n_loss = torch.log(1 + torch.exp(cos_m_theta_n.clamp(min=-self.s, max=self.s)))

        loss = one_hot * p_loss + (~one_hot) * n_loss

        return loss.sum(dim=1).mean()

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', m=' + str(self.m) \
               + ', s=' + str(self.s) + ')'
