import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


args = {
    'steps':    8,
    'dt':       5,
    'a':        0.25,   # 梯度近似项 
    'aa':       0.5,
    'Vth':      1.5,    # 阈值电压 V_threshold
    'tau':      0.1     # 漏电常数 tau
}

def get_args():
    return args


class SpikeAct(torch.autograd.Function):
    """ 定义脉冲激活函数，并根据论文公式进行梯度的近似。
        Implementation of the spiking activation function with an approximation of gradient.
    """
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        # if input = u > Vth then output = 1
        output = torch.gt(input, 0) 
        return output.float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors 
        grad_input = grad_output.clone()
        # hu is an approximate func of df/du
        hu = abs(input) < args['aa']
        hu = hu.float() / (2 * args['aa'])
        return grad_input * hu



spikeAct = SpikeAct.apply


def state_update(u_t_n1, o_t_n1, W_mul_o_t1_n):
    u_t1_n1 = args['tau'] * u_t_n1 * (1 - o_t_n1) + W_mul_o_t1_n
    o_t1_n1 = spikeAct(u_t1_n1 - args['Vth'])
    return u_t1_n1, o_t1_n1


class tdLayer(nn.Module):
    """将普通的层转换到时间域上。输入张量需要额外带有时间维，此处时间维在数据的最后一维上。前传时，对该时间维中的每一个时间步的数据都执行一次普通层的前传。
        Converts a common layer to the time domain. The input tensor needs to have an additional time dimension, which in this case is on the last dimension of the data. When forwarding, a normal layer forward is performed for each time step of the data in that time dimension.

    Args:
        layer (nn.Module): 需要转换的层。
            The layer needs to be converted.
        bn (nn.Module): 如果需要加入BN，则将BN层一起当做参数传入。
            If batch-normalization is needed, the BN layer should be passed in together as a parameter.
    """
    def __init__(self, layer, bn=None):
        super(tdLayer, self).__init__()
        self.layer = layer
        self.bn = bn

    def forward(self, x):
        steps = x.shape[-1]
        x_ = torch.zeros(self.layer(x[..., 0]).shape + (steps,), device=x.device)
        for step in range(steps):
            x_[..., step] = self.layer(x[..., step])

        if self.bn is not None:
            x_ = self.bn(x_)
        return x_

        
class LIFSpike(nn.Module):
    """对带有时间维度的张量进行一次LIF神经元的发放模拟，可以视为一个激活函数，用法类似ReLU。
        Generates spikes based on LIF module. It can be considered as an activation function and is used similar to ReLU. The input tensor needs to have an additional time dimension, which in this case is on the last dimension of the data.
    """
    def __init__(self):
        super(LIFSpike, self).__init__()

    def forward(self, x):
        steps = x.shape[-1]
        u   = torch.zeros(x.shape[:-1] , device=x.device)
        out = torch.zeros(x.shape, device=x.device)
        for step in range(steps):
            u, out[..., step] = state_update(u, out[..., max(step-1, 0)], x[..., step])
        return out


class LIFVoltage(nn.Module):
    """末尾输出电压的LIF神经元。
        Generate float voltage based on LIF module.
    """
    def __init__(self):
        super(LIFVoltage, self).__init__()

    def forward(self, x):
        steps = x.shape[-1]
        u   = torch.zeros(x.shape[:-1] , device=x.device)
        out = torch.zeros(x.shape, device=x.device)
        for step in range(steps):
            u, out[..., step] = state_update(u, out[..., max(step-1, 0)], x[..., step])
        return u


class RateCoding(nn.Module):
    """对输出进行频率编码。
        Rate coding of output.
    """
    def __init__(self):
        super(RateCoding, self).__init__()

    def forward(self, x):
        return torch.sum(x, dim=2) / x.shape[-1]


class Transpose(nn.Module):
    """维度变换，permute的Module化实现。
        Transpose the tensor. An modulize implementation of 'tensor.permute()'.
    """
    def __init__(self, *args):
        super(Transpose, self).__init__()
        self.trans_args = args

    def forward(self, x):
        return x.permute(self.trans_args)


class BroadCast(nn.Module):
    """将传统数据广播为时空域数据。
        Broadcast spacial tensor to spacial-temporal tensor.
    """
    def __init__(self):
        super(BroadCast, self).__init__()

    def forward(self, x):
        x = x.unsqueeze(len(x.shape))
        x = torch.broadcast_to(x, x.shape[:-1] + (args['steps'],)) 
        return x


class SNNCell(nn.Module):
    """用于某个脚本的一个Wrapper。
    """
    def __init__(self, snn):
        super(SNNCell, self).__init__()
        self.snn = snn

    def forward(self, x, hidden):    # u[layer, batch, hidden]
        x = self.snn(x)
        return x, hidden



class tdBatchNorm(nn.BatchNorm2d):
    """tdBN的实现。相关论文链接：https://arxiv.org/pdf/2011.05280。具体是在BN时，也在时间域上作平均；并且在最后的系数中引入了alpha变量以及Vth。
        Implementation of tdBN. Link to related paper: https://arxiv.org/pdf/2011.05280. In short it is averaged over the time domain as well when doing BN.
    Args:
        num_features (int): same with nn.BatchNorm2d
        eps (float): same with nn.BatchNorm2d
        momentum (float): same with nn.BatchNorm2d
        alpha (float): an addtional parameter which may change in resblock.
        affine (bool): same with nn.BatchNorm2d
        track_running_stats (bool): same with nn.BatchNorm2d
    """
    def __init__(self, num_features, eps=1e-05, momentum=0.1, alpha=1, affine=True, track_running_stats=True):
        super(tdBatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.alpha = alpha

    def forward(self, input):
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

        input = self.alpha * args['Vth'] * (input - mean[None, :, None, None, None]) / (torch.sqrt(var[None, :, None, None, None] + self.eps))
        if self.affine:
            input = input * self.weight[None, :, None, None, None] + self.bias[None, :, None, None, None]

        return input