import torch
import torchvision
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import parameters_to_vector
from skopt import Optimizer
from skopt.space import Real
from skopt.plots import plot_convergence, plot_objective
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.utils.prune as prune
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import scale
from torch.autograd import Variable
import argparse
import fl_utils

# 得到全局train_dataset和test_dataset，以及随机得到各个节点的数据集序号字典
def get_dataset(args):
    if args.dataset == 'cifar':
        data_dir = './data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))])
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                    transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                    transform=apply_transform)
        user_groups = fl_utils.cifar_iid(train_dataset, args)

    elif args.dataset == 'mnist' or 'fmnist':
        if args.dataset == 'mnist':
            data_dir = './data/mnist/'
        else:
            data_dir = './data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)
        user_groups = fl_utils.mnist_iid(train_dataset, args)

    return train_dataset, test_dataset, user_groups

class Net_finger(torch.nn.Module):
    def __init__(self, n_feature, n_hidden1, n_hidden2,n_output):
        super(Net_finger, self).__init__()
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden1)   # hidden layer
        # self.drop1 = torch.nn.Dropout(0.7)
        self.hidden2 = torch.nn.Linear(n_hidden1, n_hidden2)
        # self.drop2 = torch.nn.Dropout(0.7)
        self.hidden3 = torch.nn.Linear(n_hidden2, n_hidden2)
        # self.drop3 = torch.nn.Dropout(0.1)
        self.hidden4 = torch.nn.Linear(n_hidden2, n_hidden2)

        # # self.drop4 = torch.nn.Dropout(0.1)
        self.hidden5 = torch.nn.Linear(n_hidden2, n_hidden2)
        self.drop5 = torch.nn.Dropout(0.1)
        self.hidden6 = torch.nn.Linear(n_hidden2, n_hidden2)
        # # self.drop6 = torch.nn.Dropout(0.1)
        self.hidden7 = torch.nn.Linear(n_hidden2, n_hidden2)
        # # self.drop7 = torch.nn.Dropout(0.1)
        self.hidden8 = torch.nn.Linear(n_hidden2, n_hidden2)
        self.drop8 = torch.nn.Dropout(0.1)
        # self.bn1 = torch.nn.BatchNorm1d(n_hidden)
        # self.hidden3 = torch.nn.Linear(n_hidden2, n_hidden2)
        # self.predict = torch.nn.Linear(n_hidden2, n_output)   # output layer

        self.num = torch.nn.Linear(n_hidden2, 3)
        self.layer = torch.nn.Linear(n_hidden2, 5)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        #x = self.drop1(x)
        x = F.relu(self.hidden2(x))
        #x = self.drop2(x)
        xx = F.relu(self.hidden3(x))
        #xx = self.drop3(xx)
        xx = F.relu(self.hidden4(xx))
        #xx = self.drop4(xx)
        xx = F.relu(self.hidden5(xx))
        xx = self.drop5(xx)

        x=xx+x

        xxx = F.relu(self.hidden6(x))
        #xxx = self.drop6(xxx)
        xxx = F.relu(self.hidden7(xxx))
        #xxx = self.drop7(xxx)
        xxx = F.relu(self.hidden8(xxx))
        xxx = self.drop8(xxx)

        x=xxx+x

        # x = self.bn1(x)
        #x = F.relu(self.hidden3(x))
        

        num = self.num(x)
        # num = torch.softmax(num,dim=1)

        layer = self.layer(x)
        # layer = torch.softmax(layer,dim=1)

        return num, layer

class CNNCifar(torch.nn.Module):
    def __init__(self):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3,16,3,padding=1)
        # self.drop1 = torch.nn.Dropout(0.7)
        self.pol1 = nn.MaxPool2d(2,stride=2)
        # self.drop2 = torch.nn.Dropout(0.7)
        self.conv2 = nn.Conv2d(16,32,3,padding=1)
        # self.drop3 = torch.nn.Dropout(0.1)
        self.pol2 = nn.MaxPool2d(2,stride=2)

        # # self.drop4 = torch.nn.Dropout(0.1)
        self.conv3 = nn.Conv2d(32,32,3,padding=1)
        self.drop = torch.nn.Dropout(0.1)
        self.pol3 = nn.MaxPool2d(2,stride=2)
        # # self.drop6 = torch.nn.Dropout(0.1)
        
        self.lin1 = nn.Linear(32*4*4,32*4*4)
        # # self.drop7 = torch.nn.Dropout(0.1)
        self.lin2 = nn.Linear(32*4*4,32*2*2)
        self.lin3 = nn.Linear(32*2*2,10)
        # self.bn1 = torch.nn.BatchNorm1d(n_hidden)
        # self.hidden3 = torch.nn.Linear(n_hidden2, n_hidden2)
        # self.predict = torch.nn.Linear(n_hidden2, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pol1(x)
        x = F.relu(self.conv2(x))
        x = self.drop(x)
        x = self.pol2(x)
        x = F.relu(self.conv3(x))
        x = self.pol3(x)
        x = x.view(x.size()[0], -1)
        x = self.lin1(x)
        x = self.lin2(x)
        x = self.lin3(x)
        
        return F.log_softmax(x, dim=1)


    # def __init__(self):
    #     super(Net, self).__init__()
    #     self.conv1 = nn.Conv2d(3, 6, 5)
    #     self.pool = nn.MaxPool2d(2, 2)
    #     self.conv2 = nn.Conv2d(6, 16, 5)
    #     self.fc1 = nn.Linear(16 * 5 * 5, 120)
    #     self.fc2 = nn.Linear(120, 84)
    #     self.fc3 = nn.Linear(84, 10)

    # def forward(self, x):
    #     x = self.pool(F.relu(self.conv1(x)))
    #     x = self.pool(F.relu(self.conv2(x)))
    #     x = x.view(-1, 16 * 5 * 5)
    #     x = F.relu(self.fc1(x))
    #     x = F.relu(self.fc2(x))
    #     x = self.fc3(x)
    #     return F.log_softmax(x, dim=1)

class CNNMnist(nn.Module):
    def __init__(self):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        # self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        # x = self.dropout(x)
        x = self.sigmoid(x)
        x = self.layer_hidden(x)
        return x

   

if __name__ == '__main__':
    num=10
    train_dataset, test_dataset, user_groups = get_dataset(num)
    print(len(train_dataset))
    print(len(test_dataset))
    print(len(user_groups))
