from __future__ import print_function
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from torchvision import datasets, transforms
from example_model import *
from dataset import NMNIST
from tensorboardX import SummaryWriter




def train(args, model, device, train_loader, optimizer, epoch, writer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # necessary for general dataset: broadcast input
        data, _ = torch.broadcast_tensors(data, torch.zeros((steps,) + data.shape)) 
        data = data.permute(1, 2, 3, 4, 0)

        output = model(data)
        loss = F.cross_entropy(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data / steps), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            
            writer.add_scalar('Train Loss /batchidx', loss, batch_idx + len(train_loader) * epoch)
        

def test(args, model, device, test_loader, epoch, writer):
    model.eval()
    test_loss = 0
    correct = 0
    isEval = False
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            data, _ = torch.broadcast_tensors(data, torch.zeros((steps,) + data.shape))
            data = data.permute(1, 2, 3, 4, 0)

            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()


            
            

    test_loss /= len(test_loader.dataset)

    writer.add_scalar('Test Loss /epoch', test_loss, epoch)
    writer.add_scalar('Test Acc /epoch', 100. * correct / len(test_loader.dataset), epoch)
    for i, (name, param) in enumerate(model.named_parameters()):
        if '_s' in name:
            writer.add_histogram(name, param, epoch)


    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    
def adjust_learning_rate(args, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 20 epochs"""
    lr = args.lr * (0.1 ** (epoch // 35))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=200, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=800, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
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

    device = torch.device("cuda:0" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    writer = SummaryWriter('./summaries/resnet_2')
    
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=True, download=True,
                         transform=transforms.Compose([
                             transforms.RandomHorizontalFlip(),
                             transforms.RandomCrop(32, padding=4),
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

    #model = resnet19().to(device)
    model = seq_net.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    checkpoint_path = './tmp/cifar10/cifar10_spike.pt'
    if os.path.isdir(checkpoint_path) or True:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint)
        print('Model loaded.')
    

    for epoch in range(401, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, writer)
        test(args, model, device, test_loader, epoch, writer)

    writer.close()

    if (args.save_model):
        torch.save(model.state_dict(), "./tmp/cifar10/cifar10_spike.pt")
        torch.save(model, "./tmp/cifar10/cifar10_spike.pth")


if __name__ == '__main__':
    main()
