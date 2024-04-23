import torch
import torchvision
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import parameters_to_vector
import skopt
from skopt import Optimizer
from skopt.space import Real
from skopt.plots import plot_convergence, plot_objective
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.utils.prune as prune
import copy
import math
import random
import more_itertools
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import scale
from torch.autograd import Variable
import time
import argparse
from torchvision.models import mobilenet_v2
from torch.utils.data import DataLoader, Dataset
# from tensorboardX import SummaryWriter
import fl_utils
import model
from collections import OrderedDict

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# logger = SummaryWriter('../logs/SGD')

class Server:
    def __init__(self, model, test_dataset, optimizer,device,learning_rate,N_us):
        self.model = model
        self.test_dataset = test_dataset
        self.optimizer = optimizer
        self.device = device
        self.learning_rate = learning_rate
        self.N_us = N_us

    # 梯度聚合
    def aggregate(self, w, alpha):
        w_avg = copy.deepcopy(w[0])
        alpha_weight = [self.N_us[i]*alpha[i] for i in range(len(w))]
        alpha_weight = [i/sum(alpha_weight) for i in alpha_weight]
        if torch.all(alpha==0):
            print('alpha all zero')
            for key in w_avg.keys():
                w_avg[key] = torch.zeros_like(w_avg[key])
        else:
            for key in w_avg.keys():
                for i in range(1, len(w)):
                    w_avg[key] += w[i][key]*torch.tensor(alpha_weight)[i]
                # w_avg[key] = torch.div(w_avg[key], sum(alpha))
        return w_avg
    # 梯度下降更新模型参数
    def update(self, avg_weights):
        # with torch.no_grad():
        #     # checkg = [p.grad.data for p in self.model.parameters()]
        #     self.model.train()
        #     self.model.zero_grad()
        #     gradients = list(avg_gradients.values())
        #     for p, g in zip(self.model.parameters(), gradients):
        #         # p.grad.data = g.to(device)
        #         p.grad.data = g
        #     # checkg2 = [p.grad.data for p in self.model.parameters()]
        #     self.optimizer.step()
        self.model.train()
        
        self.model.load_state_dict(avg_weights)

    # 测试准确率
    def test(self):
        self.model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        criterion = nn.NLLLoss().to(device)
        testloader = DataLoader(self.test_dataset, batch_size=128,
                                shuffle=False)

        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = self.model(images)
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy
    
    # def test(self):
    #     test_data, test_target = self.x_test.to(self.device,dtype=torch.float32),self.y_test.to(self.device,dtype=torch.long)
    #     num_correct = 0
    #     num_samples = 0
    #     self.model.eval()   # set model to evaluation mode
    #     with torch.no_grad():
    #         scores = self.model(test_data)
    #         _,preds = scores.max(1)
    #         num_correct = (preds==test_target).sum()
    #         num_samples = preds.size(0)
    #         acc = float(num_correct) / num_samples
    #         print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 *acc ))
    #         return acc

class Client: 
    def __init__(self, model, logger, bitwidth, prune_rate, train_dataset, test_dataset, idxs, device, loss_func, local_bs, optimizer='sgd', learning_rate=0.01, local_ep=1):
        self.model = model
        self.logger = logger
        self.bitwidth = bitwidth
        self.prune_rate = prune_rate
        self.local_bs = local_bs
        self.trainloader = DataLoader(fl_utils.DatasetSplit(train_dataset, idxs),
                                 batch_size=self.local_bs, shuffle=True)
        self.testloader = DataLoader(test_dataset, batch_size=self.local_bs, shuffle=False)
        self.device = device
        self.optimizer = optimizer
        self.learning_rate = learning_rate 
        # if optimizer == 'sgd':
        #     self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.5)
        # elif optimizer == 'adam':
        #     self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.local_ep = local_ep
        if loss_func =='crossentropy':
            self.criterion = nn.CrossEntropyLoss()
        elif loss_func == 'nll':
            self.criterion = nn.NLLLoss()
    # 梯度量化

    def quantize(self, grad, g_max, g_min): 
        g_max = abs(g_max)
        g_min = abs(g_min)
        sign = torch.sign(grad)
        grad = abs(grad)

        # 计算最值，确定边界
        # max = grad.max()
        # min = grad.min()

        # 计算量化间隔大小
        scale = (g_max - g_min)/(2.**self.bitwidth - 1.) 

        if scale==0:
            print("scale=0")
            print(g_min,g_max)
            print(grad)
            raise ValueError("量化区间为0")
            return grad
        else:
            # 计算可能的量化值
            level = (grad - g_min)/scale
            level_down = level.floor_()
            level_up = (level+0.0001).ceil_()
            q_down = level_down*scale + g_min
            q_up = level_up*scale + g_min
            
            # 计算概率tensor
            pro = (grad-q_down)/(q_up-q_down)

            # 生成随机tensor
            rand_tensor = torch.randn(grad.shape).to(self.device)

            # 生成量化梯度
            q_grad = torch.where(rand_tensor>pro, q_up, q_down)

            if torch.isnan(q_grad).any():
                print(scale)
                print(g_min,g_max)
                print(grad)
                raise ValueError("梯度有nan")
            if torch.all(q_grad==0):
                print(scale)
                print(g_min,g_max)
                print(grad)
                raise ValueError("量化梯度全0")
            q_grad = q_grad*sign
            return q_grad
    
    # 剪枝
    def prune(self, model): 
        parameters_to_prune = ()
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                # parameters_to_prune += ((module, 'weight'),)
                prune.l1_unstructured(module, 'weight', amount=self.prune_rate)
                prune.l1_unstructured(module, 'bias', amount=self.prune_rate)
        
    def re_prune(self, model):
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                # parameters_to_prune += ((module, 'weight'),)
                prune.remove(module, 'weight')
                prune.remove(module, 'bias')
    
    # 训练
    def train(self,global_epoch, args):  
        self.model.train()
        epoch_loss = []
        epoch_gradients = OrderedDict()

        # Initialize epoch_gradients with zero tensors
        for name, param in self.model.named_parameters():
            epoch_gradients[name] = torch.zeros_like(param.data)

        # Set optimizer for the local updates
        if self.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.5)
        elif self.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        
        if args.if_batch:
            for iter in range(self.local_ep):
                batch_loss = []

                for batch_idx, (images, labels) in enumerate(self.trainloader):
                    if args.if_prune:
                        self.prune(self.model)                        # 剪枝
                    images, labels = images.to(self.device), labels.to(self.device)

                    self.model.zero_grad()
                    log_probs = self.model(images)
                    loss = self.criterion(log_probs, labels)
                    loss.backward()

                    # Accumulate gradients for each parameter
                    if args.if_prune:
                        for name, param in self.model.named_parameters():
                            epoch_gradients[name[:-len('_orig')]] += param.grad.data
                    else:
                        for name, param in self.model.named_parameters():
                            epoch_gradients[name] += param.grad.data

                    optimizer.step()

                    if (batch_idx % 1000 == 0):
                        print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            global_epoch, iter, batch_idx * len(images),
                            len(self.trainloader.dataset),
                                                100. * batch_idx / len(self.trainloader), loss.item()))
                    # self.logger.add_scalar('loss', loss.item())
                    batch_loss.append(loss.item())
                    if args.if_prune:
                        self.re_prune(self.model)                        # 去剪枝
                epoch_loss.append(sum(batch_loss) / len(batch_loss))

        else:
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                if args.if_prune:
                    self.prune(self.model)                        # 剪枝
                images, labels = images.to(self.device), labels.to(self.device)

                self.model.zero_grad()
                log_probs = self.model(images)

                loss = self.criterion(log_probs, labels)

                loss.backward()

                
                # Condition to initialize delta and previous gradients (gold) at the start
                if args.if_SCG:
                    if batch_idx == 0:
                        gold = [copy.deepcopy(parm.grad) for parm in self.model.parameters()]
                        delta = [torch.zeros_like(parm.grad) for parm in self.model.parameters()]


                    # Compute conjugate gradient direction
                    with torch.no_grad():
                        gnew = [parm.grad for parm in self.model.parameters()]
                        beta = [((gn**2).sum()/(go**2).sum()).item() for gn, go in zip(gnew, gold)]
                        delta = [-gn + bet * d for gn, bet, d in zip(gnew, beta, delta)]
                        
                        for param, dt in zip(self.model.parameters(), delta):
                            param.data.add_(dt)   # Update parameters

                    # Update gold
                    gold = gnew
                else:
                    optimizer.step()

                # optimizer.step()
                if args.if_prune:
                    self.re_prune(self.model)                        # 去剪枝
                epoch_loss.append(loss.item()) 

                break    # 只训练一次，一个batch就结束

        weight = self.model.state_dict()

        w_maxs = [weight[name].max().cpu() for name in weight.keys()]   # 此处的g_max为当前client各分量的上界列表
        w_mins = [weight[name].min().cpu() for name in weight.keys()]
        w_max = max(w_maxs)
        w_min = min(w_mins)
        # gradients = [self.quantize(p.grad.data, g_max, g_min) for p in self.model.parameters()]  # p为遍历g_u的每一个分量
        if args.if_quantize:
            for name in weight.keys():
                weight[name] = self.quantize(weight[name], w_max, w_min)
        #----------------------------------------------------------------------------------------------------------
        
        max_weight = float('-inf')
        for name,param in self.model.named_parameters():
            if 'weight' in name:
                max_weight = max(max_weight, param.max().item())
        
        q_max = w_max
        
        xi = 64

        # 计算准确率
        self.model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.trainloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = self.model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
            
        # print(accuracy)

        # self.re_prune(self.model)                        # 剪枝
        # return gradients
        # print(g_max, max(q_max))
        return weight, sum(epoch_loss) / len(epoch_loss), w_max, w_min, q_max, xi, accuracy
    

def FEDAVG(args, test_dataset, train_dataset, user_groups, N_us, file_path, transmit_power, bitwidths, prune_rates, computing_resources, I_us, h_us):
    start_time = time.time()
    L = 1/args.learning_rate

    print('现在进行实验:FEDAVG')
    # global_model = model.Net(n_feature=520, n_hidden1=64, n_hidden2=64, n_output=8)
    
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = model.CNNMnist()
        
        elif args.dataset == 'cifar':
            global_model = model.CNNCifar()

    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = model.MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=10)
    else:
        exit('Error: unrecognized model')
    # global_model = mobilenet_v2(num_classes=10)
    global_model = global_model.to(device)

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(global_model.parameters(), lr=args.learning_rate, momentum=0.5, nesterov=True)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(global_model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    global_model.train()

    server = Server(model=global_model, test_dataset=test_dataset, optimizer=optimizer, device=device, learning_rate=args.learning_rate, N_us=N_us)
    # 创建客户端
    
    alpha = torch.tensor([1 for i_ in range(args.num_clients)])
    
    print('试训练一次，获取梯度的最大值和最小值')
    client_gmaxs = []
    client_gmins = []
    for idx in range(args.num_clients):
        epo = 0
        print('client:',idx)
        client = Client(model=copy.deepcopy(global_model), logger=None, bitwidth=bitwidths[idx], prune_rate=prune_rates[idx], train_dataset=train_dataset, test_dataset=test_dataset, idxs=user_groups[idx], device=device, loss_func=args.loss_func, local_bs=args.local_bs, optimizer=args.optimizer, learning_rate=args.learning_rate, local_ep=args.local_ep)
        gradient, loss, g_max, g_min, q_max, xi, accuracy = client.train(global_epoch=epo, args=args)
        client_gmaxs.append(g_max)
        client_gmins.append(g_min)
    
    xis = [64+args.V for i in range(args.num_clients)]

    T_AVG = fl_utils.calculate_T(transmit_power,bitwidths,prune_rates,computing_resources,N_us,args.B_u,h_us,args.N0,I_us,args.V,xis,args.c0,args.s)
    E_AVG, T_1max, T_1min, T_2max, T_2min, E_1max, E_1min, E_2max, E_2min  = fl_utils.calculate_E(transmit_power,bitwidths,prune_rates,computing_resources,N_us,args.B_u,h_us,args.N0,I_us,args.V,xis,args.c0,args.k,args.sigma)
    dataamount_AVG = fl_utils.calculate_dataamount(bitwidths,prune_rates,args.V,xis)
    
    fl_utils.generate_alpha(alpha, transmit_power, args.num_clients, I_us, h_us, args.B_u, args.N0, args.waterfall_thre)
    a = sum(alpha)
    gama = fl_utils.Gamma(prune_rates,bitwidths,transmit_power, client_gmaxs, client_gmaxs, h_us, I_us, args.num_clients, N_us, args.B_u, args.N0, args.V, args.waterfall_thre, args.L, args.D)

    datatosave2 = {'T':[T_AVG],'E':[E_AVG],'dataamount':[dataamount_AVG],'a':[a], 'gamma':[gama],'T_calculate_max': [T_1max], 'T_calculate_min':[T_1min], 'T_transmit_max':[T_2max], 'T_transmit_min':[T_2min], 'E_calculate_max':[E_1max], 'E_calculate_min':[E_1min], 'E_transmit_max':[E_2max], 'E_transmit_min':[E_2min]}
    file_name2 = f'TE_AVG_T{args.Tmax}_E{args.Emax}_w{args.wer}_c{args.count_py}'
    fl_utils.Save_to_Csv(data = datatosave2, file_name = file_name2, Save_format = 'csv',Save_type = 'row', file_path=file_path)
    #--------------------------------------------------开始训练---------------------------------
    losses_train = []
    losses_test = []
    accuracies_train = []
    accuracies_test = []

    # epo用于记录训练轮数，但其实每个iteration都会更新一次，所以其实没啥用
    epo = 0

    # step1,step2,step3用于记录训练到0.6,0.7,0.8（可调节）的轮数
    step1 = 0
    step2 = 0
    step3 = 0
    step_count = 0

    for epo in range(args.num_epoch):
        local_weights, local_losses = [], []
        local_gradient = []
        list_acc = []
        xis = []
        print(f'\n | Global Training Round : {epo+1} |\n')
                                        
        # 获取训练结果
        for idx in range(args.num_clients):
            print('client:',idx)
            client = Client(model=copy.deepcopy(global_model), logger=None, bitwidth=bitwidths[idx], prune_rate=prune_rates[idx], train_dataset=train_dataset, test_dataset=test_dataset, idxs=user_groups[idx], device=device, loss_func='crossentropy', local_bs=10, optimizer='sgd', learning_rate=0.01, local_ep=1)
            weight, loss, g_max, g_min, q_max, xi, accuracy = client.train(global_epoch=epo, args=args)
            # local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            local_weights.append(copy.deepcopy(weight))   
            list_acc.append(accuracy) 
            xis.append(xi)
        
        acc_train = sum(list_acc)/len(list_acc)
        loss_avg = sum(local_losses) / len(local_losses)
        
        # client_loss = [client.train()[1] for client in clients]

        # 打印每个iteration的部分训练情况，可注释
        # print(f'epoch:{epo}, [{i + 1}] train_loss of client 0: {client_loss[0]}, accuracy of client 0: {client_acces[0]}, g_max[0]: {client_gmaxs[0]}, q_max[0]: {client_qmaxs[0][0]}, alpha: {alpha}')                                 # 打印训练时的loss
        if torch.all(alpha==0):
            print('alpha all zero')

        # 更新alpha，即各个client是否成功传输    
        fl_utils.generate_alpha(alpha, transmit_power, args.num_clients, I_us, h_us, args.B_u, args.N0, args.waterfall_thre)
            
        # 服务器操作（梯度聚合+更新）
        avg_weights = server.aggregate(w=local_weights, alpha=alpha.to(device))   # 梯度聚合   
        server.update(avg_weights=avg_weights)                                                           # 梯度更新
        
        #------------------测试-------------------------------
        acc_test = server.test()
        if acc_test >= 0.6 and step_count== 0:
            step1 = epo
            step_count = 1
        elif acc_test >= 0.7 and step_count== 1:
            step2 = epo
            step_count = 2
        elif acc_test >= 0.8 and step_count== 2:
            step3 = epo
            step_count = 3
        

        print(f'epoch:{epo+1}, [{epo + 1}] train_loss: {loss_avg},acccuracy_train:{acc_train}, accuracy_v: {acc_test}, g_max[0]: {client_gmaxs[0]}, alpha: {alpha}') 
        losses_train.append(loss_avg)
        accuracies_train.append(acc_train)
        accuracies_test.append(acc_test)

        end_time = time.time()
        train_time = end_time - start_time
        
        print(f'已训练时间：{train_time}')
    end_time = time.time()
    train_time = end_time - start_time
        
    print(f'训练时间：{train_time}')
    
    losses_train_AVG = torch.tensor(losses_train)
    accuracies_train_AVG = torch.tensor(accuracies_train)
    accuracies_test_AVG = torch.tensor(accuracies_test)

    datatosave = {'losses_train':losses_train_AVG, 'accuracies_train':accuracies_train_AVG, 'accuracies_test':accuracies_test_AVG}
    file_name1 = f'LA_AVG_T{args.Tmax}_E{args.Emax}_w{args.wer}_c{args.count_py}'
    fl_utils.Save_to_Csv(data = datatosave, file_name = file_name1, Save_format = 'csv',Save_type = 'row', file_path=file_path)

    T_AVG_step = [T_AVG*step1,T_AVG*step2,T_AVG*step3]
    E_AVG_step = [E_AVG*step1,E_AVG*step2,E_AVG*step3]
    datatosave = {'T_step':T_AVG_step, 'E_step':E_AVG_step, 'step':[step1,step2,step3]}
    file_name3 = f'TE_AVG_step_T{args.Tmax}_E{args.Emax}_w{args.wer}_c{args.count_py}'
    fl_utils.Save_to_Csv(data = datatosave, file_name = file_name3, Save_format = 'csv',Save_type = 'row', file_path=file_path)

