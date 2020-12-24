# STBP-simple

STBP-simple是一个关于STBP方法的基于Pytorch的简单实现。



## 使用

通过GitHub下载：

```
git clone git@github.com:thiswinex/STBP-simple.git
```

STBP方法的核心实现在 `layers.py` 中。`model.py` 给出了一些网络的实现示例。具体来说，如果有一个卷积层需要变为脉冲层：

```
conv = nn.Conv2d(...)
```

只需要使用`layers.py` 中提供的 `tdLayer()` 方法，在网络定义中额外添加一句：

```
conv_s = tdLayer(conv)
```

就可以把该层变为可以处理时空域数据（fc、pool层同理）。此外，再额外定义脉冲激活函数：

```
spike = LIFSpike()
```

并在`forward()`中将`relu()`等激活函数替换为`spike()`，把`conv()`等调用替换为`conv_s()`，便可以实现高度自定义化的脉冲层实现。

如果需要使用BN层：

```
bn = nn.BatchNorm2d(...)
```

那么只需要：

```
conv_s = tdLayer(conv, bn)
```

其余步骤同上。

如果使用的是普通数据集（MNIST、CIFAR10等），在数据输入网络前需要将输入往时间维度广播。`example.py`中给出了使用STBP方法在普通数据集上训练SNN的示例实现。脉冲数据集（N-MNIST等）不需要广播操作，但脉冲数据集往往体积庞大且需要预处理。`example_dvs.py`中给出了使用STBP方法在脉冲数据集上训练SNN的示例实现，`dataset.py`中则给出了预处理N-MNIST数据集的示例实现。



## 关于STBP

- [Zheng, H., Wu, Y., Deng, L., Hu, Y., & Li, G. (2020). Going Deeper With Directly-Trained Larger Spiking Neural Networks. *arXiv preprint arXiv:2011.05280*.](https://arxiv.org/pdf/2011.05280)
- [Wu, Y., Deng, L., Li, G., Zhu, J., Xie, Y., & Shi, L. (2019, July). Direct training for spiking neural networks: Faster, larger, better. In *Proceedings of the AAAI Conference on Artificial Intelligence* (Vol. 33, pp. 1311-1318).](https://www.aaai.org/ojs/index.php/AAAI/article/view/3929/3807)
- [Wu, Y., Deng, L., Li, G., Zhu, J., & Shi, L. (2018). Spatio-temporal backpropagation for training high-performance spiking neural networks. *Frontiers in neuroscience*, *12*, 331.](https://www.frontiersin.org/articles/10.3389/fnins.2018.00331/full)