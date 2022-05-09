'''Train CIFAR100 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import argparse
import random

from models import *
from utils import progress_bar

from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
parser.add_argument('--lr', default=0.02, type=float, help='learning rate')
parser.add_argument('--epochs', default=100, type=int, help='number of epochs')
args = parser.parse_args()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
device = 'cuda' if torch.cuda.is_available() else 'cpu'

start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
val_percent = 0.2
trainset = torchvision.datasets.CIFAR100(
    root='./data', train=True, download=True, transform=transform_train)

n_val = int(len(trainset) * val_percent)
n_train = len(trainset) - n_val
trainset, valset = torch.utils.data.random_split(trainset, [n_train, n_val])

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=256, shuffle=True, num_workers=2)
valloader = torch.utils.data.DataLoader(
    valset, batch_size=256, shuffle=True, num_workers=2)


testset = torchvision.datasets.CIFAR100(
    root='./data', train=False, download=True, transform=transform_test)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=256, shuffle=False, num_workers=2)


# Training
writer = SummaryWriter()
def train(net, optimizer, scheduler, criterion, epoch):
    # print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def train_val(net, optimizer, scheduler, criterion, epoch, train_loss_list, train_acc, name):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            train_loss_list.append(loss.item())

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            train_acc.append(100. * correct / total)
            
            # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            writer.add_scalar('wd_'+str(name)+"/loss_train", loss.item(), epoch)
            writer.add_scalar('wd_'+str(name)+"/acc_train", 100. * correct / total, epoch)  

def val(net, optimizer, scheduler, criterion, epoch, val_loss_list, val_acc, name):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            val_loss_list.append(loss.item())
            

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            val_acc.append(100.*correct/total)

            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            writer.add_scalar('wd_'+str(name)+"/loss_val", loss.item(), epoch) 
            writer.add_scalar('wd_'+str(name)+"/acc_val", 100.*correct/total, epoch)  

def test(net, optimizer, scheduler, criterion, epoch, best_acc):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        # print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt1e-3.pth')
        best_acc = acc
    return best_acc
        


def main():
    setup_seed(123)
    if not os.path.isdir('./val_loss_list'):
        os.mkdir('val_loss_list')
        os.mkdir('val_acc_list')
        os.mkdir('train_loss_list')
        os.mkdir('train_acc_list')
    for wd in [0, 1e-4, 1e-3]: 
        print("wd",wd)
        best_acc = 0  # best test accuracy
        net = ResNet18().to(device)
        if device == 'cuda':
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = True
        criterion = nn.CrossEntropyLoss()
        val_loss_list = []
        val_acc = []
        train_loss_list = []
        train_acc = []
        optimizer = optim.SGD(net.parameters(), lr=args.lr,
                            momentum=0.9, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        for epoch in range(start_epoch, start_epoch+args.epochs):
            train(net, optimizer, scheduler, criterion, epoch)
            train_val(net, optimizer, scheduler, criterion, epoch, train_loss_list, train_acc, str(wd))
            val(net, optimizer, scheduler, criterion, epoch, val_loss_list, val_acc, str(wd))
            best_acc = test(net, optimizer, scheduler, criterion, epoch, best_acc)
            scheduler.step()
            if epoch % 10 == 0:
                print("train_loss", train_loss_list[-1], "val_loss", val_loss_list[-1],
                    "train_acc", train_acc[-1], "val_acc", val_acc[-1])
        print("best_acc", best_acc)
        torch.save(val_loss_list, os.path.join('val_loss_list',str(wd)))
        torch.save(val_acc, os.path.join('val_acc_list', str(wd)))
        torch.save(train_loss_list, os.path.join('train_loss_list', str(wd)))
        torch.save(train_acc, os.path.join('train_acc_list', str(wd)))


if __name__=='__main__':
    main()
