import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#import tsensor

steps = 2
dt = 5
simwin = dt * steps
a = 0.25
aa = 0.5 # a /2
Vth = 0.2#0.3
tau = 0.2#0.3


class SpikeAct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        # if input = u > Vth then output = 1
        output = torch.gt(input, Vth) 
        return output.float()

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors # input = u - Vth
        grad_input = grad_output.clone()
        # hu is an approximate func of df/du
        hu = abs(input) < aa
        hu = hu.float() / (2 * aa)
        return grad_input * hu

spikeAct = SpikeAct.apply


def state_update(u_t_n1, o_t_n1, W_mul_o_t1_n):
    u_t1_n1 = tau * u_t_n1 * (1 - o_t_n1) + W_mul_o_t1_n
    o_t1_n1 = spikeAct(u_t1_n1 - Vth)
    return u_t1_n1, o_t1_n1



class SpikeLayer(nn.Module):
    def __init__(self, layer, bn=None):
        super(SpikeLayer, self).__init__()
        self.layer = layer
        self.bn = bn
        #self.timeorder = False   #TODO Running By Time
        #TODO 是否需要预置u/out 以及是否需要预置时序优先的训练

    def forward(self, x):
        """ input shape is [N*step, C, H, W] / [N*step, channel]"""
        
        #x = x.reshape((-1,) + x.shape[1:] +(steps,))    # reshape to [N, C, H, W, step] / [N, channel, step]
        x_ = torch.zeros(self.layer(x[..., 0]).shape + (steps,), device=x.device)
        for step in range(steps):
            x_[..., step] = self.layer(x[..., step])

        #x_ = torch.squeeze(x_)
        if self.bn is not None:
            x_ = self.bn(x_)
        #x_ = x_.reshape(x_.shape + (1,))
        #u   = torch.zeros(x.shape[:-1] , device=x.device)               # u shape: [N, C, H, W] / [N, channel]
        #out = torch.zeros(x.shape, device=x.device)                     # out shape: [N, C, H, W, step] / [N, channel, step]
        u   = torch.zeros(x_.shape[:-1] , device=x.device)
        out = torch.zeros(x_.shape, device=x.device)
        for step in range(steps):
            #x_ = self.layer(x[..., step])
            #print(x_.shape, u.shape, out.shape)
            u, out[..., step] = state_update(u, out[..., max(step-1, 0)], x_[..., step])
        
        #out = out.reshape((-1,) + out.shape[1:-1])
        return out
        '''
        #x = x.reshape((-1,) + x.shape[1:] +(steps,))    # reshape to [N, C, H, W, step] / [N, channel, step]
        x = x.reshape((-1,) + x.shape[1:-1])
        #print(x.shape)
        #print(self.layer)
        x_ = self.layer(x)
        #print(x_.shape)
        x_ = x_.reshape((-1,) + x_.shape[1:] + (steps,))
        #print(x_.shape)
        #u   = torch.zeros(x.shape[:-1] , device=x.device)               # u shape: [N, C, H, W] / [N, channel]
        #out = torch.zeros(x.shape, device=x.device)                     # out shape: [N, C, H, W, step] / [N, channel, step]
        u   = torch.zeros(x_.shape[:-1] , device=x.device)
        out = torch.zeros(x_.shape, device=x.device)
        for step in range(steps):
            #x_ = self.layer(x[..., step])
            #print(x_.shape, u.shape, out.shape)
            u, out[..., step] = state_update(u, out[..., max(step-1, 0)], x_[..., step])
        #print(out.shape)
        #print("--------")
        #out = out.reshape((-1,) + out.shape[1:-1])
        return out
        '''
        

class tdBatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, alpha=1, affine=True, track_running_stats=True):
        super(tdBatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.alpha = 1

    def forward(self, input):
        #self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mean = input.mean([0, 2, 3, 4])
            # use biased var in train
            var = input.var([0, 2, 3, 4], unbiased=False)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        input = self.alpha * Vth * (input - mean[None, :, None, None, None]) / (torch.sqrt(var[None, :, None, None, None] + self.eps))
        if self.affine:
            input = input * self.weight[None, :, None, None, None] + self.bias[None, :, None, None, None]

        return input