import torch
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

import fl_utils
import model

import MINISGD
import FEDAVG
import SIGNSGD
import PROPOSED

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Example script with global variable.')

parser.add_argument('--dataset', type=str, help='dataset, cifar或者mnist', default='mnist')
parser.add_argument('--model', type=str, help='model, cnn(cifar的cnn 343946个参数)或者mlp(50890个参数)', default='mlp')
parser.add_argument('--if_batch', type=int, help='是否使用minibatchgd', default=0)
parser.add_argument('--if_prune', type=int, help='是否prune', default=0)
parser.add_argument('--if_quantize', type=int, help='是否quantize', default=0)
parser.add_argument('--if_one_hot', type=int, help='是否独热编码', default=0)
parser.add_argument('--if_SCG', type=int, help='是否使用SCG, 注意不能与MINIbatchSGD一起用,还没写', default=1)
parser.add_argument('--pattern', type=str, help='pattern', default='FEDAVG')

parser.add_argument('--loss_func', type=str, help='loss_func, 可以为crossentropy或nll', default='crossentropy')
parser.add_argument('--local_bs', type=int, help='local_bs', default=290)
parser.add_argument('--local_ep', type=int, help='local_ep', default=1)
parser.add_argument('--optimizer', type=str, help='optimizer, 可以为sgd或adam', default='sgd')
parser.add_argument('--num_epoch', type=int, help='num_epoch', default=100)   # num_epoch是每轮全局迭代的最大轮次  
parser.add_argument('--mean_datanum', type=int, help='节点平均的数据量', default=300)

parser.add_argument('--wer', type=float, help='wer', default=0.0065)    # wer是信道条件Rayleigh fading factor
parser.add_argument('--Tmax', type=float, help='Tmax', default=40)     # Tmax是每轮全局迭代的最大时延（约束条件）
parser.add_argument('--Emax', type=float, help='Emax', default=0.08)     # Emax是每轮全局迭代的最大能耗（约束条件）
parser.add_argument('--num_clients', type=int, help='num_clients', default=10) # num_clients是参与训练的节点数量

parser.add_argument('--count_py', type=int, help='count_py', default=1)       # count_py是文件名序号，用于扫参数的时候区分随机性
parser.add_argument('--learning_rate', type=int, help='learning_rate', default=0.01)# learning_rate是学习率
parser.add_argument('--c0', type=int, help='c0', default=200000) # c0是通过反向传播算法训练一个样本数据所需的CPU周期数
parser.add_argument('--s', type=float, help='s', default=0.1)    # s是梯度聚合、模型更新并广播的时延。一个常数。
parser.add_argument('--waterfall_thre', type=int, help='waterfall_thre', default=1) # waterfall_thre是阈值
parser.add_argument('--D', type=float, help='D', default=0.3)
parser.add_argument('--sigma', type=int, help='sigma', default=3)
parser.add_argument('--V', type=int, help='V', default=50890)
parser.add_argument('--B_u', type=int, help='B_u', default=1000000*10)
parser.add_argument('--N0', type=float, help='N0', default=3.98e-21)
parser.add_argument('--k', type=float, help='k', default=1.25e-26)
parser.add_argument('--I_min', type=str, help='I_min', default=1e-8)
parser.add_argument('--I_max', type=str, help='I_max', default=2e-8)
parser.add_argument('--dis_min', type=str, help='dis_min', default=100)
parser.add_argument('--dis_max', type=str, help='dis_max', default=300)
'''
pattarn:
    FEDSGD:单独进行FEDSGD算法
    FEDAVG:单独进行FEDAVG算法
    SIGNSGD:单独进行SIGNSGD算法
    PROPOSED:单独进行PROPOSED算法
    exp1:四种算法的对比实验, 在相同特定信道条件、节点数量下, 不同的方案在不同通信轮次上的收敛性、能耗、时延
    exp2:四种算法的对比实验, 在相同特定信道条件下, 不同的方案在不同节点上的能耗、时延
'''
parser.add_argument('--bcd_epoch', type=int, help='bcd_epoch', default=5)                # 块坐标下降法的迭代次数
parser.add_argument('--BO_epoch', type=int, help='BO_epoch', default=100)                # 贝叶斯优化的迭代次数
parser.add_argument('--power_min', type=float, help='power_min', default=0.05)
parser.add_argument('--power_max', type=float, help='power_max', default=0.1)
parser.add_argument('--bitwidth_min', type=int, help='bitwidth_min', default=1)
parser.add_argument('--bitwidth_max', type=int, help='bitwidth_max', default=8)
parser.add_argument('--prune_rate_min', type=float, help='prune_rate_min', default=0.0)
parser.add_argument('--prune_rate_max', type=float, help='prune_rate_max', default=0.0)

parser.add_argument('--acq_func', type=str, help='acq_func', default='PI')

parser.add_argument('--markevery', type=int, help='画折线图时点的间隔', default=1)

parser.add_argument('--L', type=float, help='optimizer, 可以为sgd或adam', default=100)
args = parser.parse_args()

# wer = args.wer
# Tmax = args.Tmax               
# Emax = args.Emax               
# num_clients = args.num_clients          
# num_epoch = args.num_epoch
# count_py = args.count_py
# learning_rate = args.learning_rate
# c0 = args.c0
# s = args.s
# waterfall_thre = args.waterfall_thre
# D = args.D
# sigma = args.sigma
# V = args.V
# B_u = args.B_u
# N0 = args.N0
# k = args.k
# pattern = args.pattern
# I_min = args.I_min
# I_max = args.I_max
# dis_min = args.dis_min
# dis_max = args.dis_max


# bcd_epoch = args.bcd_epoch
# BO_epoch = args.BO_epoch
# power_min = args.power_min
# power_max = args.power_max
# bitwidth_min = args.bitwidth_min
# bitwidth_max = args.bitwidth_max
# prune_rate_min = args.prune_rate_min
# prune_rate_max = args.prune_rate_max
# acq_func = args.acq_func

ini_bitwidths = [8 for i in range(args.num_clients)]
ini_prune_rates = [random.uniform(args.prune_rate_min,args.prune_rate_max) for i in range(args.num_clients)]
ini_transmit_power = [0.1 for i in range(args.num_clients)]

# I_us=[random.uniform(I_min, I_max) for i in range(num_clients)]
# computing_resources=[random.uniform(30,80)*1e6 for i in range(num_clients)]
# distance = [random.uniform(dis_min,dis_max) for i in range(num_clients)]                     
# h_us = [wer/(i**(2)) for i in distance]        # [random.uniform(fading_min, fading_max)/(i**(2)) for i in distance]

I_us,computing_resources,distance = fl_utils.read_condition(file_name='./condition/condition.csv')
I_us = I_us[:args.num_clients]
computing_resources = computing_resources[:args.num_clients]
distance = distance[:args.num_clients]
h_us = [args.wer/(i**(2)) for i in distance] 

def main():
    train_dataset, test_dataset, user_groups = model.get_dataset(args=args)
    N_us = [len(user_groups[i]) for i in range(args.num_clients)]
    fl_utils.cal_ref(wer=args.wer,bitwidth_max=args.bitwidth_max,dis_max=args.dis_max,dis_min=args.dis_min,power_max=args.power_max,power_min=args.power_min,I_max=args.I_max,I_min=args.I_min,N_us=N_us,B_u=args.B_u,N0=args.N0,V=args.V,k=args.k,c0=args.c0,s=args.s)
    # server_test_x, server_test_y, clients_x, clients_y, N_us = model.get_data(num_clients)
    # fl_utils.cal_ref(wer=wer,bitwidth_max=bitwidth_max,dis_max=dis_max,dis_min=dis_min,power_max=power_max,power_min=power_min,I_max=I_max,I_min=I_min,N_us=N_us,B_u=B_u,N0=N0,V=V,k=k,c0=c0,s=s)
    if args.pattern=='FEDSGD':
        MINISGD.FEDSGD(args=args, train_dataset=train_dataset, test_dataset=test_dataset, user_groups=user_groups, N_us=N_us, file_path='./FEDSGD/', transmit_power=ini_transmit_power, bitwidths=ini_bitwidths, prune_rates=ini_prune_rates,computing_resources=computing_resources, I_us=I_us, h_us=h_us)
        # file_path='./FEDSGD/', train_dataset=None, test_dataset=None, user_groups=None, num_epoch=100, num_clients=10, learning_rate=0.01, bitwidths=[8 for i in range(10)], prune_rates=[0.5 for i in range(10)], transmit_power=[0.1 for i in range(10)], Tmax=0.1, Emax=0.1, wer=0.1, count_py=0.1, N_us=[100 for i in range(10)], I_us=[1.5e-08 for i in range(10)], h_us=[0.1 for i in range(10)], computing_resources=[6e7 for i in range(10)], c0=200000, s=0.1, waterfall_thre=1, D=0.3, sigma=3, V=62984, B_u=1000000*10, N0=3.98e-21, k=1.25e-26
        fl_utils.plot_single_converg(args=args, save_path='./FEDSGD/', file_name=f'./FEDSGD/LA_SGD_T{args.Tmax}_E{args.Emax}_w{args.wer}_c{args.count_py}.csv')
    
    elif args.pattern=='FEDAVG':
        FEDAVG.FEDAVG(args=args, train_dataset=train_dataset, test_dataset=test_dataset, user_groups=user_groups, N_us=N_us, file_path='./FEDAVG/', transmit_power=ini_transmit_power, bitwidths=ini_bitwidths, prune_rates=ini_prune_rates,computing_resources=computing_resources, I_us=I_us, h_us=h_us)
        # file_path='./FEDAVG/', train_dataset=None, test_dataset=None, user_groups=None, num_epoch=100, num_clients=10, learning_rate=0.01, bitwidths=[8 for i in range(10)], prune_rates=[0.5 for i in range(10)], transmit_power=[0.1 for i in range(10)], Tmax=0.1, Emax=0.1, wer=0.1, count_py=0.1, N_us=[100 for i in range(10)], I_us=[1.5e-08 for i in range(10)], h_us=[0.1 for i in range(10)], computing_resources=[6e7 for i in range(10)], c0=200000, s=0.1, waterfall_thre=1, D=0.3, sigma=3, V=62984, B_u=1000000*10, N0=3.98e-21, k=1.25e-26
        fl_utils.plot_single_converg(args=args, save_path='./FEDAVG/', file_name=f'./FEDAVG/LA_AVG_T{args.Tmax}_E{args.Emax}_w{args.wer}_c{args.count_py}.csv')
    
    elif args.pattern=='SIGNSGD':
        SIGNSGD.SIGNSGD(args=args, train_dataset=train_dataset, test_dataset=test_dataset, user_groups=user_groups, N_us=N_us, file_path='./SIGNSGD/', transmit_power=ini_transmit_power, bitwidths=ini_bitwidths, prune_rates=ini_prune_rates,computing_resources=computing_resources, I_us=I_us, h_us=h_us)
        # file_path='./SIGNSGD/', train_dataset=None, test_dataset=None, user_groups=None, num_epoch=100, num_clients=10, learning_rate=0.01, bitwidths=[8 for i in range(10)], prune_rates=[0.5 for i in range(10)], transmit_power=[0.1 for i in range(10)], Tmax=0.1, Emax=0.1, wer=0.1, count_py=0.1, N_us=[100 for i in range(10)], I_us=[1.5e-08 for i in range(10)], h_us=[0.1 for i in range(10)], computing_resources=[6e7 for i in range(10)], c0=200000, s=0.1, waterfall_thre=1, D=0.3, sigma=3, V=62984, B_u=1000000*10, N0=3.98e-21, k=1.25e-26
        fl_utils.plot_single_converg(args=args, save_path='./SIGNSGD/', file_name=f'./SIGNSGD/LA_AVG_T{args.Tmax}_E{args.Emax}_w{args.wer}_c{args.count_py}.csv')
    
    elif args.pattern=='PROPOSED':
        PROPOSED.PROPOSED(args=args, train_dataset=train_dataset, test_dataset=test_dataset, user_groups=user_groups, N_us=N_us, file_path='./PROPOSED/', transmit_power=ini_transmit_power, bitwidths=ini_bitwidths, prune_rates=ini_prune_rates,computing_resources=computing_resources, I_us=I_us, h_us=h_us)
        # file_path='./PROPOSED/', train_dataset=None, test_dataset=None, user_groups=None, num_epoch=100, num_clients=10, learning_rate=0.01, bitwidths=[8 for i in range(10)], prune_rates=[0.5 for i in range(10)], transmit_power=[0.1 for i in range(10)], Tmax=0.1, Emax=0.1, wer=0.1, count_py=0.1, N_us=[100 for i in range(10)], I_us=[1.5e-08 for i in range(10)], h_us=[0.1 for i in range(10)], computing_resources=[6e7 for i in range(10)], c0=200000, s=0.1, waterfall_thre=1, D=0.3, sigma=3, V=62984, B_u=1000000*10, N0=3.98e-21, k=1.25e-26
        fl_utils.plot_single_converg(args=args, save_path='./PROPOSED/', file_name=f'./PROPOSED/LA_AVG_T{args.Tmax}_E{args.Emax}_w{args.wer}_c{args.count_py}.csv')
    
    elif args.pattern=='exp1':
        SIGNSGD.SIGNSGD(args=args, train_dataset=train_dataset, test_dataset=test_dataset, user_groups=user_groups, N_us=N_us, file_path='./SIGNSGD/', transmit_power=ini_transmit_power, bitwidths=ini_bitwidths, prune_rates=ini_prune_rates,computing_resources=computing_resources, I_us=I_us, h_us=h_us)
        MINISGD.FEDSGD(args=args, train_dataset=train_dataset, test_dataset=test_dataset, user_groups=user_groups, N_us=N_us, file_path='./FEDSGD/', transmit_power=ini_transmit_power, bitwidths=ini_bitwidths, prune_rates=ini_prune_rates,computing_resources=computing_resources, I_us=I_us, h_us=h_us)
        FEDAVG.FEDAVG(args=args, train_dataset=train_dataset, test_dataset=test_dataset, user_groups=user_groups, N_us=N_us, file_path='./FEDAVG/', transmit_power=ini_transmit_power, bitwidths=ini_bitwidths, prune_rates=ini_prune_rates,computing_resources=computing_resources, I_us=I_us, h_us=h_us)
        PROPOSED.PROPOSED(args=args, train_dataset=train_dataset, test_dataset=test_dataset, user_groups=user_groups, N_us=N_us, file_path='./PROPOSED/', transmit_power=ini_transmit_power, bitwidths=ini_bitwidths, prune_rates=ini_prune_rates,computing_resources=computing_resources, I_us=I_us, h_us=h_us)
        fl_utils.plot_multi_converg(save_path='./exp1/', file_fedsgd=f'./exp1/LA_SGD_T{args.Tmax}_E{args.Emax}_w{args.wer}_c{args.count_py}.csv', file_signsgd=f'./exp1/LA_signSGD_T{args.Tmax}_E{args.Emax}_w{args.wer}_c{args.count_py}.csv', file_fedavg=f'./exp1/LA_avg_T{args.Tmax}_E{args.Emax}_w{args.wer}_c{args.count_py}.csv', file_proposed=f'./exp1/LA_PROPOSED_T{args.Tmax}_E{args.Emax}_w{args.wer}_c{args.count_py}.csv', file_name=f'_T{args.Tmax}_E{args.Emax}_w{args.wer}_c{args.count_py}')
        fl_utils.plot_bar(save_path='./exp1/', file_fedsgd=f'./exp1/TE_SGD_step_T{args.Tmax}_E{args.Emax}_w{args.wer}_c{args.count_py}.csv', file_signsgd=f'./exp1/TE_signSGD_step_T{args.Tmax}_E{args.Emax}_w{args.wer}_c{args.count_py}.csv', file_fedavg=f'./exp1/TE_avg_step_T{args.Tmax}_E{args.Emax}_w{args.wer}_c{args.count_py}.csv', file_proposed=f'./exp1/TE_PROPOSED_step_T{args.Tmax}_E{args.Emax}_w{args.wer}_c{args.count_py}.csv', file_name=f'_T{args.Tmax}_E{args.Emax}_w{args.wer}_c{args.count_py}')
        fl_utils.record_condition(save_path='./exp1/',vers=0,I_us=I_us,computing_resources=computing_resources,distance=distance,h_us=h_us)

    elif args.pattern=='exp2':
        num_1 = 3
        num_2 = 7
        num_3 = 9

        args.num_clients = num_1
        SIGNSGD.SIGNSGD(args=args, train_dataset=train_dataset, test_dataset=test_dataset, user_groups=user_groups, N_us=N_us, file_path='./SIGNSGD/', transmit_power=ini_transmit_power, bitwidths=ini_bitwidths, prune_rates=ini_prune_rates,computing_resources=computing_resources, I_us=I_us, h_us=h_us)
        MINISGD.FEDSGD(args=args, train_dataset=train_dataset, test_dataset=test_dataset, user_groups=user_groups, N_us=N_us, file_path='./FEDSGD/', transmit_power=ini_transmit_power, bitwidths=ini_bitwidths, prune_rates=ini_prune_rates,computing_resources=computing_resources, I_us=I_us, h_us=h_us)
        FEDAVG.FEDAVG(args=args, train_dataset=train_dataset, test_dataset=test_dataset, user_groups=user_groups, N_us=N_us, file_path='./FEDAVG/', transmit_power=ini_transmit_power, bitwidths=ini_bitwidths, prune_rates=ini_prune_rates,computing_resources=computing_resources, I_us=I_us, h_us=h_us)
        PROPOSED.PROPOSED(args=args, train_dataset=train_dataset, test_dataset=test_dataset, user_groups=user_groups, N_us=N_us, file_path='./PROPOSED/', transmit_power=ini_transmit_power, bitwidths=ini_bitwidths, prune_rates=ini_prune_rates,computing_resources=computing_resources, I_us=I_us, h_us=h_us)
        
        args.num_clients = num_2
        SIGNSGD.SIGNSGD(args=args, train_dataset=train_dataset, test_dataset=test_dataset, user_groups=user_groups, N_us=N_us, file_path='./SIGNSGD/', transmit_power=ini_transmit_power, bitwidths=ini_bitwidths, prune_rates=ini_prune_rates,computing_resources=computing_resources, I_us=I_us, h_us=h_us)
        MINISGD.FEDSGD(args=args, train_dataset=train_dataset, test_dataset=test_dataset, user_groups=user_groups, N_us=N_us, file_path='./FEDSGD/', transmit_power=ini_transmit_power, bitwidths=ini_bitwidths, prune_rates=ini_prune_rates,computing_resources=computing_resources, I_us=I_us, h_us=h_us)
        FEDAVG.FEDAVG(args=args, train_dataset=train_dataset, test_dataset=test_dataset, user_groups=user_groups, N_us=N_us, file_path='./FEDAVG/', transmit_power=ini_transmit_power, bitwidths=ini_bitwidths, prune_rates=ini_prune_rates,computing_resources=computing_resources, I_us=I_us, h_us=h_us)
        PROPOSED.PROPOSED(args=args, train_dataset=train_dataset, test_dataset=test_dataset, user_groups=user_groups, N_us=N_us, file_path='./PROPOSED/', transmit_power=ini_transmit_power, bitwidths=ini_bitwidths, prune_rates=ini_prune_rates,computing_resources=computing_resources, I_us=I_us, h_us=h_us)
        
        args.num_clients = num_3
        SIGNSGD.SIGNSGD(args=args, train_dataset=train_dataset, test_dataset=test_dataset, user_groups=user_groups, N_us=N_us, file_path='./SIGNSGD/', transmit_power=ini_transmit_power, bitwidths=ini_bitwidths, prune_rates=ini_prune_rates,computing_resources=computing_resources, I_us=I_us, h_us=h_us)
        MINISGD.FEDSGD(args=args, train_dataset=train_dataset, test_dataset=test_dataset, user_groups=user_groups, N_us=N_us, file_path='./FEDSGD/', transmit_power=ini_transmit_power, bitwidths=ini_bitwidths, prune_rates=ini_prune_rates,computing_resources=computing_resources, I_us=I_us, h_us=h_us)
        FEDAVG.FEDAVG(args=args, train_dataset=train_dataset, test_dataset=test_dataset, user_groups=user_groups, N_us=N_us, file_path='./FEDAVG/', transmit_power=ini_transmit_power, bitwidths=ini_bitwidths, prune_rates=ini_prune_rates,computing_resources=computing_resources, I_us=I_us, h_us=h_us)
        PROPOSED.PROPOSED(args=args, train_dataset=train_dataset, test_dataset=test_dataset, user_groups=user_groups, N_us=N_us, file_path='./PROPOSED/', transmit_power=ini_transmit_power, bitwidths=ini_bitwidths, prune_rates=ini_prune_rates,computing_resources=computing_resources, I_us=I_us, h_us=h_us)
        
        fl_utils.plot_exp2_bar(save_path='./exp2/', file_path='./exp2/',file_fedsgd=f'TE_SGD_step_T{args.Tmax}_E{args.Emax}_w{args.wer}_c{args.count_py}.csv', file_signsgd=f'TE_signSGD_step_T{args.Tmax}_E{args.Emax}_w{args.wer}_c{args.count_py}.csv', file_fedavg=f'TE_avg_step_T{args.Tmax}_E{args.Emax}_w{args.wer}_c{args.count_py}.csv', file_proposed=f'TE_PROPOSED_step_T{args.Tmax}_E{args.Emax}_w{args.wer}_c{args.count_py}.csv',file_name=f'_T{args.Tmax}_E{args.Emax}_w{args.wer}_c{args.count_py}')
      


    else:
        pass
    
    
      
if __name__ == "__main__":
    start_time = time.time()
    
    main()
    end_time = time.time()
    execution_time = end_time - start_time
    print("程序运行时间：", execution_time, "秒")