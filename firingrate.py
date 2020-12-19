from __future__ import print_function
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from model import *
from dataset import NMNIST
from tensorboardX import SummaryWriter


steps = 8
dt = 5
simwin = dt * steps
a = 0.25
aa = 0.5 # a /2
Vth = 0.3#0.3
tau = 0.25#0.3


def train(args, model, device, train_loader, optimizer, epoch, writer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        #data = data.transpose(0, 4)
        #print(data.shape)
        data, _ = torch.broadcast_tensors(data, torch.zeros((steps,) + data.shape))
        #print(data.shape)
        data = data.permute(1, 2, 3, 4, 0)

        output = model(data)
        #print(output.shape)
        #print(target.shape)
        loss = F.cross_entropy(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data / steps), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
        

def test(args, model, device, test_loader, epoch, writer):
    model.eval()
    test_loss = 0
    correct = 0
    isEval = False
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            #data = data.transpose(0, 4)
            #print(data.shape)
            data, _ = torch.broadcast_tensors(data, torch.zeros((steps,) + data.shape))
            #print(data.shape)
            data = data.permute(1, 2, 3, 4, 0)

            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            #print(pred.shape)
            #print(target.shape)
            #target_label = target.argmax(dim=1, keepdim=True)

            correct += pred.eq(target.view_as(pred)).sum().item()

            if isEval == False:
                x = data
                for name, layer in model.module._modules.items():
                    x = x.view(x.shape[0], -1, x.shape[4]) if "fc1" in name else x
                    x = layer(x)
                    x = F.relu(x) if 'conv' in name else x
                    if 'conv' in name:
                        writer.add_histogram(f'{name}_feature_maps', x, global_step=epoch)
                isEval = True

    test_loss /= len(test_loader.dataset)

    for i, (name, param) in enumerate(model.module.named_parameters()):
        writer.add_histogram(name, param, 0)


    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda:1" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    writer = SummaryWriter('./summaries/cifar10')
    
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                         ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=False, 
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    '''
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
    
    train_loader = torch.utils.data.DataLoader(
        NMNIST('./data/NMNIST_npy/Train', train=True,  step=steps, dt=dt),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        NMNIST('./data/NMNIST_npy/Test',  train=False, step=steps, dt=dt),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
    '''
    
    model = CifarNet().to(device)
    for _, (data, _) in enumerate(train_loader):
        data = data.to(device)
        model.initSpikeParam(data[:, :, :, :])
        break
    #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    model=nn.DataParallel(model, device_ids=[1,2])


    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, writer)
        test(args, model, device, test_loader, epoch, writer)

    writer.close()

    if (args.save_model):
        torch.save(model.state_dict(), "./tmp/cifar10/cifar10_spike.pt")
        torch.save(model, "./tmp/cifar10/cifar10_spike.pth")


if __name__ == '__main__':
    main()
