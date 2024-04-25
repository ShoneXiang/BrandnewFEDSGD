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
from itertools import combinations
import argparse
import os
from torch.utils.data import Dataset
from matplotlib import rcParams

def constraint_T_forBO(power,bitwidths,prune_rates,computing_resources,N_us,B_u,h_us,N0,I_us,V,xis,c0,s,Tmax):
    data_rates = np.array([data_rate(p_u=p_u, B_u=B_u, h_u=h_us[index], N0 = N0, I_u=I_us[index]) for index,p_u in enumerate(power)])
    bit_totals = np.array(bitwidths)*V + xis
    T_1 = np.array(N_us)*c0*(1-np.array(prune_rates))/np.array(computing_resources)
    T_2 = bit_totals*(1-np.array(prune_rates))/data_rates
    return T_1+T_2+s-Tmax

# np.maximum(np.array(N_us)*c0*(1-np.array(prune_rates))/np.array(computing_resources), np.array([data_rate(p_u=p_u, B_u=B_u, h_u=h_us[index], N0 = N0, I_u=I_us[index]) for index,p_u in enumerate(power)])*(1-np.array(prune_rates))/np.array(bitwidths)*V + xis)+s-Tmax

def constraint_E_forBO(power,bitwidths,prune_rates,computing_resources,N_us,B_u,h_us,N0,I_us,V,xis,c0,k,sigma,Emax):
    data_rates = np.array([data_rate(p_u=p_u, B_u=B_u, h_u=h_us[index], N0 = N0, I_u=I_us[index]) for index,p_u in enumerate(power)])
    bit_totals = np.array(bitwidths)*V + xis
    T_1 = np.array(N_us)*c0*(1-np.array(prune_rates))/np.array(computing_resources)
    T_2 = bit_totals*(1-np.array(prune_rates))/data_rates
    E_1 = k*np.array(computing_resources)**sigma*T_1
    E_2 = np.array(power)*T_2
    return E_1+E_2-Emax

def calculate_T(power,bitwidths,prune_rates,computing_resources,N_us,B_u,h_us,N0,I_us,V,xis,c0,s):
    data_rates = np.array([data_rate(p_u=p_u, B_u=B_u, h_u=h_us[index], N0 = N0, I_u=I_us[index]) for index,p_u in enumerate(power)])
    bit_totals = np.array(bitwidths)*V + xis
    T_1 = np.array(N_us)*c0*(1-np.array(prune_rates))/np.array(computing_resources)
    T_2 = bit_totals*(1-np.array(prune_rates))/data_rates
    return (T_1+T_2+s).max()

def calculate_E(power,bitwidths,prune_rates,computing_resources,N_us,B_u,h_us,N0,I_us,V,xis,c0,k,sigma):
    data_rates = np.array([data_rate(p_u=p_u, B_u=B_u, h_u=h_us[index], N0 = N0, I_u=I_us[index]) for index,p_u in enumerate(power)])
    bit_totals = np.array(bitwidths)*V + xis
    T_1 = np.array(N_us)*c0*(1-np.array(prune_rates))/np.array(computing_resources)
    T_2 = bit_totals*(1-np.array(prune_rates))/data_rates
    E_1 = k*np.array(computing_resources)**sigma*T_1
    E_2 = np.array(power)*T_2
    return sum(E_1+E_2), max(T_1), min(T_1), max(T_2), min(T_2), max(E_1), min(E_1), max(E_2), min(E_2)

def calculate_dataamount(bitwidths,prune_rates,V,xis):
    bit_totals = np.array(bitwidths)*V + xis
    dataamount = sum(bit_totals*(1-np.array(prune_rates)))
    return dataamount


def Gamma(prune_rates, bitwidths, transmit_power, g_maxs, g_mins, h_us, I_us, num_clients=10, N_us=[100 for i in range(10)], B_u=1000000*10, N0=3.98e-21, V=62984, waterfall_thre=1, L=100, D=0.3):
    index_set = {i for i in range(num_clients)}
    gamma = 0
    error_rates = [error_rate(p_u=p_u, B_u=B_u, h_u=h_us[index], N0 = N0, I_u=I_us[index], thre=waterfall_thre) for index, p_u in enumerate(transmit_power)]
    
    #-----------------------------------开始计算---------------------------------
    for t in range(num_clients):
        
        # 对通信节点为t个的每一种情况遍历
        for u1 in combinations(index_set, t+1):
            u1 = set(u1)
            u2 = index_set - u1

            # 计算概率
            p = 1
            for u in u1:
                p = p*(1-error_rates[u])
            for u in u2:
                p = p*error_rates[u]
            
            # 计算N_u的平方和
            n2 = 0
            for u in u1:
                n2 += N_us[u]**2 

            # 计算N_u的和平方
            n3 = 0
            for u in u1:
                n3 += N_us[u]
            n3 = n3**2

            # 计算小期望和
            g_list = [t*V for t in (np.array(g_maxs)-np.array(g_mins))**2]
            b_list = [1/(4 * (2**(i)-1)**2) for i in bitwidths] 
            temp = np.multiply(g_list, b_list)
            e = 0
            for u in u1:
                e += temp[u] + (L**2) * (D**2) * prune_rates[u]
            
            # 计算该情况的最终数值并加上去
            gamma += p*n2*e/n3

    return gamma

def Gamma_for_BO_v2(transmit_power, bitwidths, prune_rates, client_gmins, client_gmaxs, h_us, I_us, num_clients=10, N_us=[100 for i in range(10)], B_u=1, N0=1, V=1, waterfall_thre=1, L=100, D=0.3):
    # if num_clients==5:
    #     p0,p1,p2,p3,p4 = transmit_power
    #     # for i in range(num_clients):
    #     #     globals()['p'+str(i)] = transmit_power[i]
    #     error_rates = [error_rate_forBO(p_u=p0, B_u=B_u, h_u=h_us[0], N0 = N0, I_u=I_us[0], thre=waterfall_thre),
    #                 error_rate_forBO(p_u=p1, B_u=B_u, h_u=h_us[1], N0 = N0, I_u=I_us[1], thre=waterfall_thre),
    #                 error_rate_forBO(p_u=p2, B_u=B_u, h_u=h_us[2], N0 = N0, I_u=I_us[2], thre=waterfall_thre),
    #                 error_rate_forBO(p_u=p3, B_u=B_u, h_u=h_us[3], N0 = N0, I_u=I_us[3], thre=waterfall_thre),
    #                 error_rate_forBO(p_u=p4, B_u=B_u, h_u=h_us[4], N0 = N0, I_u=I_us[4], thre=waterfall_thre)
    #                 ]
    #     # error_rates = [error_rate_forBO(p_u=globals()['p'+str(i)], B_u=B_u, h_u=h_us[i], N0 = N0, I_u=I_us[i], thre=waterfall_thre) for i in range(num_clients)]
    # elif num_clients==3:
    #     p0,p1,p2 = transmit_power
    #     error_rates = [error_rate_forBO(p_u=p0, B_u=B_u, h_u=h_us[0], N0 = N0, I_u=I_us[0], thre=waterfall_thre),
    #                 error_rate_forBO(p_u=p1, B_u=B_u, h_u=h_us[1], N0 = N0, I_u=I_us[1], thre=waterfall_thre),
    #                 error_rate_forBO(p_u=p2, B_u=B_u, h_u=h_us[2], N0 = N0, I_u=I_us[2], thre=waterfall_thre)
    #                 ]
    # elif num_clients==4:
    #     p0,p1,p2,p3 = transmit_power
    #     error_rates = [error_rate_forBO(p_u=p0, B_u=B_u, h_u=h_us[0], N0 = N0, I_u=I_us[0], thre=waterfall_thre),
    #                 error_rate_forBO(p_u=p1, B_u=B_u, h_u=h_us[1], N0 = N0, I_u=I_us[1], thre=waterfall_thre),
    #                 error_rate_forBO(p_u=p2, B_u=B_u, h_u=h_us[2], N0 = N0, I_u=I_us[2], thre=waterfall_thre),
    #                 error_rate_forBO(p_u=p3, B_u=B_u, h_u=h_us[3], N0 = N0, I_u=I_us[3], thre=waterfall_thre)
    #                 ]
    # elif num_clients==7:
    #     p0,p1,p2,p3,p4,p5,p6 = transmit_power
    #     error_rates = [error_rate_forBO(p_u=p0, B_u=B_u, h_u=h_us[0], N0 = N0, I_u=I_us[0], thre=waterfall_thre),
    #                 error_rate_forBO(p_u=p1, B_u=B_u, h_u=h_us[1], N0 = N0, I_u=I_us[1], thre=waterfall_thre),
    #                 error_rate_forBO(p_u=p2, B_u=B_u, h_u=h_us[2], N0 = N0, I_u=I_us[2], thre=waterfall_thre),
    #                 error_rate_forBO(p_u=p3, B_u=B_u, h_u=h_us[3], N0 = N0, I_u=I_us[3], thre=waterfall_thre),
    #                 error_rate_forBO(p_u=p4, B_u=B_u, h_u=h_us[4], N0 = N0, I_u=I_us[4], thre=waterfall_thre),
    #                 error_rate_forBO(p_u=p5, B_u=B_u, h_u=h_us[5], N0 = N0, I_u=I_us[5], thre=waterfall_thre),
    #                 error_rate_forBO(p_u=p6, B_u=B_u, h_u=h_us[6], N0 = N0, I_u=I_us[6], thre=waterfall_thre)
    #                 ]
    # elif num_clients==10:
    #     p0,p1,p2,p3,p4,p5,p6,p7,p8,p9 = transmit_power
    #     error_rates = [error_rate_forBO(p_u=p0, B_u=B_u, h_u=h_us[0], N0 = N0, I_u=I_us[0], thre=waterfall_thre),
    #                 error_rate_forBO(p_u=p1, B_u=B_u, h_u=h_us[1], N0 = N0, I_u=I_us[1], thre=waterfall_thre),
    #                 error_rate_forBO(p_u=p2, B_u=B_u, h_u=h_us[2], N0 = N0, I_u=I_us[2], thre=waterfall_thre),
    #                 error_rate_forBO(p_u=p3, B_u=B_u, h_u=h_us[3], N0 = N0, I_u=I_us[3], thre=waterfall_thre),
    #                 error_rate_forBO(p_u=p4, B_u=B_u, h_u=h_us[4], N0 = N0, I_u=I_us[4], thre=waterfall_thre),
    #                 error_rate_forBO(p_u=p5, B_u=B_u, h_u=h_us[5], N0 = N0, I_u=I_us[5], thre=waterfall_thre),
    #                 error_rate_forBO(p_u=p6, B_u=B_u, h_u=h_us[6], N0 = N0, I_u=I_us[6], thre=waterfall_thre),
    #                 error_rate_forBO(p_u=p7, B_u=B_u, h_u=h_us[7], N0 = N0, I_u=I_us[7], thre=waterfall_thre),
    #                 error_rate_forBO(p_u=p8, B_u=B_u, h_u=h_us[8], N0 = N0, I_u=I_us[8], thre=waterfall_thre),
    #                 error_rate_forBO(p_u=p9, B_u=B_u, h_u=h_us[9], N0 = N0, I_u=I_us[9], thre=waterfall_thre)
    #                 ]
    # elif num_clients==15:
    #     p0,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14 = transmit_power
    #     error_rates = [error_rate_forBO(p_u=p0, B_u=B_u, h_u=h_us[0], N0 = N0, I_u=I_us[0], thre=waterfall_thre),
    #                 error_rate_forBO(p_u=p1, B_u=B_u, h_u=h_us[1], N0 = N0, I_u=I_us[1], thre=waterfall_thre),
    #                 error_rate_forBO(p_u=p2, B_u=B_u, h_u=h_us[2], N0 = N0, I_u=I_us[2], thre=waterfall_thre),
    #                 error_rate_forBO(p_u=p3, B_u=B_u, h_u=h_us[3], N0 = N0, I_u=I_us[3], thre=waterfall_thre),
    #                 error_rate_forBO(p_u=p4, B_u=B_u, h_u=h_us[4], N0 = N0, I_u=I_us[4], thre=waterfall_thre),
    #                 error_rate_forBO(p_u=p5, B_u=B_u, h_u=h_us[5], N0 = N0, I_u=I_us[5], thre=waterfall_thre),
    #                 error_rate_forBO(p_u=p6, B_u=B_u, h_u=h_us[6], N0 = N0, I_u=I_us[6], thre=waterfall_thre),
    #                 error_rate_forBO(p_u=p7, B_u=B_u, h_u=h_us[7], N0 = N0, I_u=I_us[7], thre=waterfall_thre),
    #                 error_rate_forBO(p_u=p8, B_u=B_u, h_u=h_us[8], N0 = N0, I_u=I_us[8], thre=waterfall_thre),
    #                 error_rate_forBO(p_u=p9, B_u=B_u, h_u=h_us[9], N0 = N0, I_u=I_us[9], thre=waterfall_thre),
    #                 error_rate_forBO(p_u=p10, B_u=B_u, h_u=h_us[10], N0 = N0, I_u=I_us[10], thre=waterfall_thre),
    #                 error_rate_forBO(p_u=p11, B_u=B_u, h_u=h_us[11], N0 = N0, I_u=I_us[11], thre=waterfall_thre),
    #                 error_rate_forBO(p_u=p12, B_u=B_u, h_u=h_us[12], N0 = N0, I_u=I_us[12], thre=waterfall_thre),
    #                 error_rate_forBO(p_u=p13, B_u=B_u, h_u=h_us[13], N0 = N0, I_u=I_us[13], thre=waterfall_thre),
    #                 error_rate_forBO(p_u=p14, B_u=B_u, h_u=h_us[14], N0 = N0, I_u=I_us[14], thre=waterfall_thre)
    #                 ]

    for i in range(num_clients):
            globals()['p'+str(i)] = transmit_power[i]
    error_rates = [error_rate_forBO(p_u=globals()['p'+str(i)], B_u=B_u, h_u=h_us[i], N0 = N0, I_u=I_us[i], thre=waterfall_thre) for i in range(num_clients)]
    
    index_set = {i for i in range(num_clients)}
    gamma = 0

    #-----------------------------------开始计算---------------------------------
    for t in range(num_clients):
        
        # 对通信节点为t个的每一种情况遍历
        for u1 in combinations(index_set, t+1):
            u1 = set(u1)
            u2 = index_set - u1

            # 计算概率
            p = 1
            for u in u1:
                p = p*(1-error_rates[u])
            for u in u2:
                p = p*error_rates[u]
            
            # 计算N_u的平方和
            n2 = 0
            for u in u1:
                n2 += N_us[u]**2 

            # 计算N_u的和平方
            n3 = 0
            for u in u1:
                n3 += N_us[u]
            n3 = n3**2

            # 计算小期望和
            g_list = [t*V for t in (np.array(client_gmaxs)-np.array(client_gmins))**2]
            b_list = [1/(4 * (2**(i)-1)**2) for i in bitwidths] 
            temp = np.multiply(g_list, b_list)
            e = 0
            for u in u1:
                e += temp[u] + (L**2) * (D**2) * prune_rates[u]
            
            # 计算该情况的最终数值并加上去
            gamma += p*n2*e/n3

    return gamma

def data_rate(p_u, B_u, h_u, N0, I_u): 
    # B_u is the allocated bandwidth for u; h_u is the channel gain; N0 is the power spectral density of noise; I_u is the interference
    temp = 1 + p_u*h_u/(I_u + B_u*N0)
    E = math.log(temp,2)
    R = B_u*E
    return R
 
def error_rate(p_u, B_u, h_u, N0, I_u, thre):
    # q^n_u; thre is the waterfall threshold
    temp = -thre*(I_u + B_u*N0)/(p_u*h_u)
    rate = 1 - math.exp(temp)
    return rate

def error_rate_forBO(p_u, B_u, h_u, N0, I_u, thre):
    # q^n_u; thre is the waterfall threshold
    temp = -thre*(I_u + B_u*N0)/(p_u*h_u)
    rate = 1 - np.exp(temp)
    return rate

def generate_alpha(alpha, transmit_power, num_clients, I_us, h_us, B_u=1, N0=1, waterfall_thre=1):
    error_rates = [error_rate(p_u=p_u, B_u=B_u, h_u=h_us[index], N0 = N0, I_u=I_us[index], thre=waterfall_thre) for index, p_u in enumerate(transmit_power)]
    np.random.seed(0)
    p = list(zip(np.array(error_rates), 1-np.array(error_rates)))
    for i in range(num_clients): 
        alpha[i] = np.random.choice([0,1], p=p[i])


G = []

def Save_to_Csv(data, file_name, Save_format = 'csv', Save_type = 'col', file_path = './'):
    # data
    # 输入为一个字典，格式： { '列名称': 数据,....} 
    # 列名即为CSV中数据对应的列名， 数据为一个列表
    
    # file_name 存储文件的名字
    # Save_format 为存储类型， 默认csv格式， 可改为 excel
    # Save_type 存储类型 默认按列存储， 否则按行存储
    
    # 默认存储在当前路径下
    
    import pandas as pd
    import numpy as np
    
    Name = []
    times = 0
 
    if Save_type == 'col':
        for name, List in data.items():
            Name.append(name)
            if times == 0:
                Data = np.array(List).reshape(-1,1)
            else:
                Data = np.hstack((Data, np.array(List).reshape(-1,1)))
                
            times += 1
            
        Pd_data = pd.DataFrame(columns=Name, data=Data) 
        
    else:
        for name, List in data.items():
            Name.append(name)
            if times == 0:
                Data = np.array(List)
            else:
                Data = np.vstack((Data, np.array(List)))
        
            times += 1
    
        Pd_data = pd.DataFrame(index=Name, data=Data)  

    if not os.path.exists(file_path):
        os.makedirs(file_path)

    if Save_format == 'csv':
        Pd_data.to_csv(file_path + file_name +'.csv',encoding='utf-8')
    else:
        Pd_data.to_excel(file_path + file_name +'.xls',encoding='utf-8')

# h_min = wer/(dis_max**2)
# h_max = wer/(dis_min**2)
# Tmax_ref = s + max(N_us)*c0/3e7 + (bitwidth_max*V+64+V)/data_rate(power_min,B_u,h_min,N0,I_max)
# Emax_ref = k*(8e7)**2*max(N_us)*c0+(bitwidth_max*V+64+V)*power_max/data_rate(power_min,B_u,h_min,N0,I_max)
# Tmin_ref = s + min(N_us)*c0/8e7 + (1*V+64+V)/data_rate(power_max,B_u,h_max,N0,I_min)
# Emin_ref = k*(3e7)**2*min(N_us)*c0+(1*V+64+V)*power_min/data_rate(power_max,B_u,h_max,N0,I_min)
# print(f'当前T范围:({Tmin_ref},{Tmax_ref}), 当前E范围:({Emin_ref},{Emax_ref})')

def cal_ref(wer,bitwidth_max,resource_max,resource_min,dis_max,dis_min,power_max,power_min,I_max,I_min,N_us,B_u,N0,V,k,c0,s):
    h_min = wer/(dis_max**2)
    h_max = wer/(dis_min**2)
    Tmax_ref = s + max(N_us)*c0/resource_min + (bitwidth_max*V+64+V)/data_rate(power_min,B_u,h_min,N0,I_max)
    Emax_ref = k*(resource_max)**2*max(N_us)*c0+(bitwidth_max*V+64+V)*power_max/data_rate(power_min,B_u,h_min,N0,I_max)
    Tmin_ref = s + min(N_us)*c0/resource_max + (1*V+64+V)/data_rate(power_max,B_u,h_max,N0,I_min)
    Emin_ref = k*(resource_min)**2*min(N_us)*c0+(1*V+64+V)*power_min/data_rate(power_max,B_u,h_max,N0,I_min)
    print(f'当前T范围:({Tmin_ref},{Tmax_ref}), 当前E范围:({Emin_ref},{Emax_ref})')

    with open(f'referce.txt', 'w') as f:
        # 写入数组字符串
        f.write(f'当前T范围:({Tmin_ref},{Tmax_ref}), 当前E范围:({Emin_ref},{Emax_ref})')


def adjust(bitwidths,prune_rates,power,I_us,h_us,g_maxs,g_mins,computing_resources,xis ,num_clients=10, power_min=0.001, power_max=0.1, bitwidth_min=1, bitwidth_max=8, prune_rate_min=0, prune_rate_max=0.5, bcd_epoch=5, BO_epoch=100, N0=3.98e-21, V=62984, c0=200000, s=0.1, k=1.25e-26, sigma=3, Emax=1, Tmax=1, N_us=[100 for i in range(10)], B_u=1000000*10, waterfall_thre = 1, L=100, D=0.3, client_gmaxs=[0.006 for i in range(10)], client_gmins=[-0.006 for i in range(10)], acq_func='PI'):
    ''' 
    bitwidths: [delta^n_1,...,delta^n_U],量化比特
    prune_rates: [rho^n_1,...,rho^n_U],剪枝率
    power: [p^n_1,...p^n_U],传输功率
    threshold: 判断块梯度下降是否收敛的阈值
    I_us: [I^n_1,...I^n_U],interference,用于计算误码率
    h_us: [h^n_1,...h^n_U],channel gain,用于计算误码率
    g_maxs: [[gm_1_1,...gm_1_V],...,[gm_U_1,...gm_U_V]],各个client的各个分量的上界列表
    g_mins: 各个client的各个分量的下界列表
    f: [f^n_1,...,f^n_u],computing_resources
    xis: [xi_1,...,xi_U]表达g_max和g_min和符号的bit数
    max_iter: 贝叶斯优化的最大优化次数
    '''
    # N_us = [3601,3601,3601,3601,3602]

    time_ = 1
    # 深拷贝，用于判断是否收敛
    # b = copy.deepcopy(bitwidths)
    # r = copy.deepcopy(prune_rates)
    # p = copy.deepcopy(power)
    # check = Gamma(r,b,p, g_maxs, g_mins, h_us, I_us)
    bitwidths = [8 for i in range(num_clients)]
    prune_rates = [random.uniform(prune_rate_min, prune_rate_max) for i in range(num_clients)]
    power = [random.uniform(power_min, power_max) for i in range(num_clients)]
    # power = [0.1 for i in range(num_clients)]

    best_power = copy.deepcopy(power)
    best_bitwidth = copy.deepcopy(bitwidths)
    best_prune_rate = copy.deepcopy(prune_rates)
    best_Gamma = 1e9
    # while abs(Gamma(r,b,p, g_max, g_min, h_us, I_us)-Gamma(prune_rates,bitwidths,power, g_max, g_min, h_us, I_us)) > threshold:
    for e in range(bcd_epoch):
        # b = copy.deepcopy(bitwidths)
        # r = copy.deepcopy(prune_rates)
        # p = copy.deepcopy(power)
        print('当前块坐标下降轮次：',time_)
        
        time_ += 1
        # update prune_rates

        data_rates = np.array([data_rate(p_u=p_u, B_u=B_u, h_u=h_us[index], N0 = N0, I_u=I_us[index]) for index,p_u in enumerate(power)])        # R_u arrray
        bit_totals = np.array(bitwidths)*V + xis                                                                                                 # sim_delta_u arrray
        
        f_us = np.array(computing_resources)    # f_u array                                                          !!!!undefined
        # N_us = np.array(data_amount)    # N_u array                                                          
        phi_1 = (Tmax - s)/(bit_totals/data_rates+c0*np.array(N_us)/f_us)
        phi_2 = Emax/(k*f_us**(sigma-1)*np.array(N_us)*c0 + np.array(power)*bit_totals/data_rates)
        min_1 = np.minimum(phi_1,phi_2)
        for i in range(num_clients):
            if (1-min_1[i])>prune_rate_max: 
                prune_rates[i] = prune_rate_max
            elif (1-min_1[i])<prune_rate_min:
                prune_rates[i] = prune_rate_min
            else:
                prune_rates[i] = 1-min_1[i]
        for i in range(num_clients):
            prune_rates[i] = 1-min_1[i]
            
            if prune_rates[i] >= prune_rate_max:
                prune_rates[i] = prune_rate_max
            elif prune_rates[i] <= prune_rate_min:
                prune_rates[i] = prune_rate_min
            

        # update bitwidths

        phi_3 = (Tmax - s -np.array(N_us)*c0*(1-np.array(prune_rates))/f_us)*data_rates/(1-np.array(prune_rates))
        phi_4 = (Emax - k*f_us**(sigma-1)*np.array(N_us)*c0*(1-np.array(prune_rates)))*data_rates/(np.array(power)*(1-np.array(prune_rates)))
        for bi in np.minimum((phi_3-xis)/V, (phi_4-xis)/V):
            if math.isnan(bi):
                raise ValueError("The list contains a NaN value.")
        bitwidths = [int(i) for i in np.minimum((phi_3-xis)/V, (phi_4-xis)/V)]
        for i in range(num_clients):
            if bitwidths[i] >= bitwidth_max:
                bitwidths[i] = bitwidth_max
            elif bitwidths[i] <= bitwidth_min:
                bitwidths[i] = bitwidth_min

        # update power by BO
        bit_totals = np.array(bitwidths)*V + xis   
        temp_array = Tmax - s -np.array(N_us)*c0*(1-np.array(prune_rates))/f_us
        temp_array = bit_totals*(1-np.array(prune_rates))/(B_u*temp_array)
        temp_array = 2**temp_array - 1
        p_min = temp_array*(np.array(I_us)+B_u*N0)/np.array(h_us)
        print('p_min:',p_min)
        if np.all(p_min<=power_max):
            print('至少有满足Tmax的解')
        else:
            print('在此时的约束条件、剪枝率和量化比特下, power无可行解!')
            for i in range(num_clients):
                p_min=power_max
        #   break
        for i in range(num_clients):
            if p_min[i] >= power_max:
                p_min[i] = power_max
            elif p_min[i] <= power_min:
                p_min[i] = power_min
        namespace = ['p'+str(i) for i in range(num_clients)]
        space = [Real(p_min[i],power_max,name=namespace[i]) for i in range(num_clients)]
        
        opt = Optimizer(space, base_estimator="GP", random_state=0, acq_func=acq_func)
        n_calls = BO_epoch
        BO_results = []
        count = 0
        for i in range(n_calls):
            suggested = opt.ask()
            if np.all(constraint_T_forBO(suggested,bitwidths,prune_rates,computing_resources,N_us,B_u,h_us,N0,I_us,V,xis,c0,s,Tmax)<=0) and np.all(constraint_E_forBO(suggested,bitwidths,prune_rates,computing_resources,N_us,B_u,h_us,N0,I_us,V,xis,c0,k,sigma,Emax)<=0):
                y = Gamma_for_BO_v2(suggested, bitwidths, prune_rates, client_gmins, client_gmaxs, h_us, I_us, num_clients, N_us, B_u, N0, V, waterfall_thre, L, D)
                count += 1
            else:
                # y = np.array([1e8 for i in range(num_clients)])
                y = 1e10
            BO_results.append(y)
            opt.tell(suggested,y)
        print('kexingjie:',count)    
        
        result = opt.get_result()
        power = result.x

        # plt.figure(figsize=(12,5))
        # plot_convergence(result)
        # plt.show()
        # plt.figure(figsize=(12,5))
        # plot_objective(result,size=3,dimensions=['p0','p1'])

        # if np.all(np.array(prune_rates)==0) or np.std(np.array(prune_rates))==0:
        #     power = [random.uniform(power_max/2, power_max) for i in range(num_clients)]
        #     print('random')
        #     temp = 0
        #     while np.any(constraint_E_forBO(power)>0) or np.any(constraint_T_forBO(power)>0):
        #         # print('not available')
        #         print(temp)
        #         temp +=1
        #         power = [random.uniform(power_min, power_max) for i in range(num_clients)]
            

        # print(Gamma(r,b,p, g_maxs, g_mins, h_us, I_us))
        if np.any(constraint_E_forBO(power,bitwidths,prune_rates,computing_resources,N_us,B_u,h_us,N0,I_us,V,xis,c0,k,sigma,Emax)>0) or np.any(constraint_T_forBO(power,bitwidths,prune_rates,computing_resources,N_us,B_u,h_us,N0,I_us,V,xis,c0,s,Tmax)>0):
            print('no it isnt!!!!')
            power = [random.uniform(p_min[c], power_max) for c in range(num_clients)]
        else:
            print('yes it is!!!!!')
            temp_g = Gamma(prune_rates,bitwidths,power, g_maxs, g_mins, h_us, I_us, num_clients, N_us, B_u, N0, V, waterfall_thre, L, D)
            # output_record.append([power,bitwidths,prune_rates])
            if temp_g <= best_Gamma:
                best_Gamma = temp_g
                best_power = copy.deepcopy(power)
                best_bitwidth = copy.deepcopy(bitwidths)
                best_prune_rate = copy.deepcopy(prune_rates)

            print(temp_g)
            G.append(temp_g)
        print('power',power,'bitwidth',bitwidths,'prune_rate',prune_rates)
        end = time.perf_counter()
        # print(f'当前用时：{end-start}')

        # if abs(Gamma(r,b,p, g_maxs, g_mins, h_us, I_us)-Gamma(prune_rates,bitwidths,power, g_maxs, g_mins, h_us, I_us)) < threshold:
        #     break
    return best_power,best_bitwidth,best_prune_rate


def read_converg(file_name):
    df = pd.read_csv(file_name, header=None)

    # 将第一列设置为变量名称
    variable_names = df.iloc[:, 0]
    df = df.iloc[:, 1:]

    # 转置DataFrame
    df_transposed = df.T

    # 给DataFrame设置列名
    df_transposed.columns = variable_names

    # 提取数据
    index_values = df_transposed.index  # 序号
    losses_train = df_transposed['losses_train']  # 对应的losses_train列
    accuracies_train = df_transposed['accuracies_train']  # 对应的acc_train列
    accuracies_test = df_transposed['accuracies_test']  # 对应的acc_train列

    return index_values, losses_train, accuracies_train, accuracies_test

def read_TE(file_name):
    df = pd.read_csv(file_name, header=None)

        # 将第一列设置为变量名称
    variable_names = df.iloc[:, 0]
    df = df.iloc[:, 1:]

        # 转置DataFrame
    df_transposed = df.T

        # 给DataFrame设置列名
    df_transposed.columns = variable_names

        # 提取数据
    # index_values = df_transposed.index  # 序号
    data_T = df_transposed['T_step']  # 对应的losses_train列
    data_E = df_transposed['E_step']  # 对应的acc_train列

    return data_T, data_E

def read_condition(file_name):
    df = pd.read_csv(file_name, header=None)

    # 将第一列设置为变量名称
    variable_names = df.iloc[:, 0]
    df = df.iloc[:, 1:]

    # 转置DataFrame
    df_transposed = df.T

    # 给DataFrame设置列名
    df_transposed.columns = variable_names

    # 提取数据
    I_us = df_transposed['I_us']  # 对应的losses_train列
    computing_resources = df_transposed['computing_resources']  # 对应的computing_resources列
    distances = df_transposed['distances']  # 对应的distances列

    return list(I_us), list(computing_resources), list(distances)

def plot_single_converg(args, save_path, file_name, if_loss=True):
    # 读取CSV文件
    index_values, losses_train, accuracies_train, accuracies_test = read_converg(file_name)
    font_size = 30
    rcParams.update({'font.size': font_size, 'font.family': 'Times New Roman'})
    
    # 示例输出
    if if_loss:
        plt.plot(index_values, losses_train, label='Losses Train', marker='o',markevery=args.markevery)
    plt.plot(index_values, accuracies_train, label='Train Accuracy', marker='o',markevery=args.markevery)
    plt.plot(index_values, accuracies_test, label='Test Accuracy', marker='o',markevery=args.markevery)

    plt.xlabel('Index Values')
    plt.ylabel('Values')
    plt.title('Training Metrics Over Time')
    plt.legend()
    plt.grid(True)

    plt.savefig(save_path +'convergence.png')

    plt.show()

def plot_multi_converg(args, save_path, file_fedsgd, file_signsgd, file_fedavg, file_proposed, file_name, if_loss=False):
    
    index_values_fedsgd, losses_train_fedsgd, accuracies_train_fedsgd, accuracies_test_fedsgd = read_converg(file_fedsgd)
    index_values_signsgd, losses_train_signsgd, accuracies_train_signsgd, accuracies_test_signsgd = read_converg(file_signsgd)
    index_values_fedavg, losses_train_fedavg, accuracies_train_fedavg, accuracies_test_fedavg = read_converg(file_fedavg)
    index_values_proposed, losses_train_proposed, accuracies_train_proposed, accuracies_test_proposed = read_converg(file_proposed)

    font_size = 20
    rcParams.update({'font.size': font_size, 'font.family': 'Times New Roman'})

    # 示例输出
    
    plt.figure()

    plt.plot(index_values_fedsgd, accuracies_test_fedsgd, label='Accuracy_fedsgd', marker='o',markevery=args.markevery)
    plt.plot(index_values_signsgd, accuracies_test_signsgd, label='Accuracy_signsgd', marker='s',markevery=args.markevery)
    plt.plot(index_values_fedavg, accuracies_test_fedavg, label='Accuracy_fedavg', marker='D',markevery=args.markevery)
    plt.plot(index_values_proposed, accuracies_test_proposed, label='Accuracy_proposed', marker='*',markevery=args.markevery)

    plt.xlabel('Training Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training Metrics Over Time')
    plt.legend()
    plt.grid(True)

    plt.savefig(save_path +'convergence'+file_name+'.png')
    plt.show()

    if if_loss:
        plt.figure()
        plt.plot(index_values_fedsgd, losses_train_fedsgd, label='Losses_fedsgd', marker='o',markevery=args.markevery)
        plt.plot(index_values_signsgd, losses_train_signsgd, label='Losses_signsgd', marker='s',markevery=args.markevery)
        plt.plot(index_values_fedavg, losses_train_fedavg, label='Losses_fedavg', marker='D',markevery=args.markevery)
        plt.plot(index_values_proposed, losses_train_proposed, label='Losses_proposed', marker='*',markevery=args.markevery)

        plt.xlabel('Training Epochs')
        plt.ylabel('Loss')
        plt.title('Training Metrics Over Time')
        plt.legend()
        plt.grid(True)

        plt.savefig(save_path +'convergence'+file_name+'.png')
        plt.show()

def plot_bar(save_path, file_fedsgd, file_signsgd, file_fedavg, file_proposed, file_name):
    data_T_fedsgd, data_E_fedsgd = read_TE(file_fedsgd)
    data_T_signsgd, data_E_signsgd = read_TE(file_signsgd)
    data_T_fedavg, data_E_fedavg = read_TE(file_fedavg)
    data_T_proposed, data_E_proposed = read_TE(file_proposed)

    proposed = [data_T_proposed[1], data_T_proposed[2], data_T_proposed[3]]                  # Proposed 2.0 1.5 0.006
    fedavg = [data_T_fedavg[1],data_T_fedavg[2],data_T_fedavg[3]]               # SGD 2. 1.5
    signsgd = [data_T_signsgd[1],data_T_signsgd[2],data_T_signsgd[3]]                # signSGD
    fedsgd = [data_T_fedsgd[1],data_T_fedsgd[2],data_T_fedsgd[3]]             # avg
    font_size = 20
    labels = ['0.6', '0.7', '0.8']
    bar_width = 0.15
    rcParams.update({'font.size': font_size, 'font.family': 'Times New Roman'})

    maxlim = max([max(proposed),max(signsgd),max(fedsgd),max(fedavg)])

    # 绘图
    plt.figure(figsize=(15, 12))
    plt.bar(np.arange(3), proposed, label='LTFL (Proposed)', color='royalblue', alpha=1, width=bar_width, edgecolor="k", hatch='/')
    plt.bar(np.arange(3) + 3 *bar_width, fedsgd, label=u'FedSGD', color='orange', alpha=1, edgecolor="k",
            width=bar_width, hatch="o")
    plt.bar(np.arange(3) + 1 * bar_width, signsgd, label=u'SignSGD', color='limegreen', alpha=1, edgecolor="k",
            width=bar_width, hatch="x")
    plt.bar(np.arange(3) + 2 * bar_width, fedavg, label=u'FedAVG', color='saddlebrown', alpha=1, edgecolor="k",
            width=bar_width, hatch="/")
    # 添加刻度标签
    plt.xticks(np.arange(3) + 2*bar_width, labels)
    # plt.tick_params(labelsize=20)
    # 设置Y轴的刻度范围
    plt.ylim([0, math.ceil(maxlim/10)*10])
    plt.grid(True)
    # 为每个条形图添加数值标签
    # for x2016, proposed in enumerate(proposed):
    #     plt.text(x2016, proposed + 2, '%s' % proposed, ha='center', fontsize=font)

    # for x2017, fedsgd in enumerate(fedsgd):
    #     plt.text(x2017 + bar_width, fedsgd + 2, '%s' % fedsgd, ha='center', fontsize=font)

    # for x2018, signsgd in enumerate(signsgd):
    #     plt.text(x2018 + 2 * bar_width, signsgd + 2, '%s' % signsgd, ha='center', fontsize=font)

    # for x2019, y2019 in enumerate(Y2019):
    #     plt.text(x2019 + 3*bar_width, y2019 + 2, '%s' % y2019, ha='center', fontsize=font)

    # for x2020, y2020 in enumerate(Y2020):
    #     plt.text(x2020 + 4 * bar_width, y2020 + 2, '%s' % y2020, ha='center', fontsize=font)

    # 显示图例
    # plt.legend(bbox_to_anchor=(0.5, 1), loc=5, borderaxespad=0, fontsize=font)
    plt.legend(loc='upper left')

    # plt.title("test", fontsize=font)
    plt.xlabel("Accuracy")
    plt.ylabel("Delay(s)")
    plt.savefig(save_path +'Delay'+file_name+'.png', dpi=300, format = 'png')

    #显示图形
    plt.show()

    proposed = [data_E_proposed[1], data_E_proposed[2], data_E_proposed[3]]                  # Proposed 2.0 1.5 0.006
    fedavg = [data_E_fedavg[1],data_E_fedavg[2],data_E_fedavg[3]]               # SGD 2. 1.5
    signsgd = [data_E_signsgd[1],data_E_signsgd[2],data_E_signsgd[3]]                # signSGD
    fedsgd = [data_E_fedsgd[1],data_E_fedsgd[2],data_E_fedsgd[3]]             # avg
    font_size = 20
    labels = ['0.6', '0.7', '0.8']
    bar_width = 0.15
    rcParams.update({'font.size': font_size, 'font.family': 'Times New Roman'})
    
    maxlim = max([max(proposed),max(signsgd),max(fedsgd),max(fedavg)])

    # 绘图
    plt.figure(figsize=(15, 12))
    plt.bar(np.arange(3), proposed, label='LTFL (Proposed)', color='royalblue', alpha=1, width=bar_width, edgecolor="k", hatch='/')
    plt.bar(np.arange(3) + 3 *bar_width, fedsgd, label=u'FedSGD', color='orange', alpha=1, edgecolor="k",
            width=bar_width, hatch="o")
    plt.bar(np.arange(3) + 1 * bar_width, signsgd, label=u'SignSGD', color='limegreen', alpha=1, edgecolor="k",
            width=bar_width, hatch="x")
    plt.bar(np.arange(3) + 2 * bar_width, fedavg, label=u'FedAVG', color='saddlebrown', alpha=1, edgecolor="k",
            width=bar_width, hatch="/")
    # 添加刻度标签
    plt.xticks(np.arange(3) + 2*bar_width, labels)
    # plt.tick_params(labelsize=20)
    # 设置Y轴的刻度范围
    plt.ylim([0, math.ceil(maxlim/10)*10])
    plt.grid(True)
    # 为每个条形图添加数值标签
    # for x2016, proposed in enumerate(proposed):
    #     plt.text(x2016, proposed + 2, '%s' % proposed, ha='center', fontsize=font)

    # for x2017, fedsgd in enumerate(fedsgd):
    #     plt.text(x2017 + bar_width, fedsgd + 2, '%s' % fedsgd, ha='center', fontsize=font)

    # for x2018, signsgd in enumerate(signsgd):
    #     plt.text(x2018 + 2 * bar_width, signsgd + 2, '%s' % signsgd, ha='center', fontsize=font)

    # for x2019, y2019 in enumerate(Y2019):
    #     plt.text(x2019 + 3*bar_width, y2019 + 2, '%s' % y2019, ha='center', fontsize=font)

    # for x2020, y2020 in enumerate(Y2020):
    #     plt.text(x2020 + 4 * bar_width, y2020 + 2, '%s' % y2020, ha='center', fontsize=font)

    # 显示图例
    # plt.legend(bbox_to_anchor=(0.5, 1), loc=5, borderaxespad=0, fontsize=font)
    plt.legend(loc='upper left')

    # plt.title("test", fontsize=font)
    plt.xlabel("Accuracy")
    plt.ylabel("Energy Consumption(J)")
    plt.savefig(save_path +'Energy Consumption'+file_name+'.png', dpi=300, format = 'png')

    #显示图形
    plt.show()

def record_condition(save_path,vers,I_us,computing_resources,distance,h_us):
    with open(save_path+f'array_proposed_step_v{vers}.txt', 'w') as f:
        # 写入数组字符串
        f.write('I_us:'+str(I_us)+'\n')
        f.write('computing_resources:'+str(computing_resources)+'\n')
        f.write('distances:'+str(distance)+'\n')
        f.write('h_us:'+str(h_us)+'\n')

def plot_exp2_bar(save_path, file_path, num_1, num_2, num_3, file_fedsgd, file_signsgd, file_fedavg, file_proposed,file_name):
    data_T_fedsgd_1, data_E_fedsgd_1 = read_TE(file_path+'num_1/FEDSGD/'+file_fedsgd)
    data_T_signsgd_1, data_E_signsgd_1 = read_TE(file_path+'num_1/SIGNSGD/'+file_signsgd)
    data_T_fedavg_1, data_E_fedavg_1 = read_TE(file_path+'num_1/FEDAVG/'+file_fedavg)
    data_T_proposed_1, data_E_proposed_1 = read_TE(file_path+'num_1/PROPOSED/'+file_proposed)

    data_T_fedsgd_2, data_E_fedsgd_2 = read_TE(file_path+'num_2/FEDSGD/'+file_fedsgd)
    data_T_signsgd_2, data_E_signsgd_2 = read_TE(file_path+'num_2/SIGNSGD/'+file_signsgd)
    data_T_fedavg_2, data_E_fedavg_2 = read_TE(file_path+'num_2/FEDAVG/'+file_fedavg)
    data_T_proposed_2, data_E_proposed_2 = read_TE(file_path+'num_2/PROPOSED/'+file_proposed)

    data_T_fedsgd_3, data_E_fedsgd_3 = read_TE(file_path+'num_3/FEDSGD/'+file_fedsgd)
    data_T_signsgd_3, data_E_signsgd_3 = read_TE(file_path+'num_3/SIGNSGD/'+file_signsgd)
    data_T_fedavg_3, data_E_fedavg_3 = read_TE(file_path+'num_3/FEDAVG/'+file_fedavg)
    data_T_proposed_3, data_E_proposed_3 = read_TE(file_path+'num_3/PROPOSED/'+file_proposed)

    proposed = [data_T_proposed_1[2], data_T_proposed_2[2], data_T_proposed_3[2]]                  # Proposed 2.0 1.5 0.006
    fedavg = [data_T_fedavg_1[2],data_T_fedavg_2[2],data_T_fedavg_3[2]]               # SGD 2. 1.5
    signsgd = [data_T_signsgd_1[2],data_T_signsgd_2[2],data_T_signsgd_3[2]]                # signSGD
    fedsgd = [data_T_fedsgd_1[2],data_T_fedsgd_2[2],data_T_fedsgd_3[2]]             # avg

    font_size = 20
    labels = [str(num_1), str(num_2), str(num_3)]
    bar_width = 0.15
    rcParams.update({'font.size': font_size, 'font.family': 'Times New Roman'})

    maxlim = max([max(proposed),max(signsgd),max(fedsgd),max(fedavg)])
    # 绘图
    plt.figure(figsize=(15, 12))
    plt.bar(np.arange(3), proposed, label='LTFL (Proposed)', color='royalblue', alpha=1, width=bar_width, edgecolor="k", hatch='/')
    plt.bar(np.arange(3) + 3 *bar_width, fedsgd, label=u'FedSGD', color='orange', alpha=1, edgecolor="k",
            width=bar_width, hatch="o")
    plt.bar(np.arange(3) + 1 * bar_width, signsgd, label=u'SignSGD', color='limegreen', alpha=1, edgecolor="k",
            width=bar_width, hatch="x")
    plt.bar(np.arange(3) + 2 * bar_width, fedavg, label=u'FedAVG', color='saddlebrown', alpha=1, edgecolor="k",
            width=bar_width, hatch="/")
    # 添加刻度标签
    plt.xticks(np.arange(3) + 2*bar_width, labels)
    # plt.tick_params(labelsize=20)
    # 设置Y轴的刻度范围
    plt.ylim([0, math.ceil(maxlim/10)*10])
    plt.grid(True)
    # 为每个条形图添加数值标签
    # for x2016, proposed in enumerate(proposed):
    #     plt.text(x2016, proposed + 2, '%s' % proposed, ha='center', fontsize=font)

    # for x2017, fedsgd in enumerate(fedsgd):
    #     plt.text(x2017 + bar_width, fedsgd + 2, '%s' % fedsgd, ha='center', fontsize=font)

    # for x2018, signsgd in enumerate(signsgd):
    #     plt.text(x2018 + 2 * bar_width, signsgd + 2, '%s' % signsgd, ha='center', fontsize=font)

    # for x2019, y2019 in enumerate(Y2019):
    #     plt.text(x2019 + 3*bar_width, y2019 + 2, '%s' % y2019, ha='center', fontsize=font)

    # for x2020, y2020 in enumerate(Y2020):
    #     plt.text(x2020 + 4 * bar_width, y2020 + 2, '%s' % y2020, ha='center', fontsize=font)

    # 显示图例
    # plt.legend(bbox_to_anchor=(0.5, 1), loc=5, borderaxespad=0, fontsize=font)
    plt.legend(loc='upper left')

    # plt.title("test", fontsize=font)
    plt.xlabel("num_clients")
    plt.ylabel("Energy Consumption(J)")
    plt.savefig(save_path +'Energy Consumption'+file_name+'.png', dpi=300, format = 'png')

    #显示图形
    plt.show()

    proposed = [data_E_proposed_1[2], data_E_proposed_2[2], data_E_proposed_3[2]]                  # Proposed 2.0 1.5 0.006
    fedavg = [data_E_fedavg_1[2],data_E_fedavg_2[2],data_E_fedavg_3[2]]               # SGD 2. 1.5
    signsgd = [data_E_signsgd_1[2],data_E_signsgd_2[2],data_E_signsgd_3[2]]                # signSGD
    fedsgd = [data_E_fedsgd_1[2],data_E_fedsgd_2[2],data_E_fedsgd_3[2]]
    font_size = 20
    labels = [str(num_1), str(num_2), str(num_3)]
    bar_width = 0.15
    rcParams.update({'font.size': font_size, 'font.family': 'Times New Roman'})
    
    maxlim = max([max(proposed),max(signsgd),max(fedsgd),max(fedavg)])

    # 绘图
    plt.figure(figsize=(15, 12))
    plt.bar(np.arange(3), proposed, label='LTFL (Proposed)', color='royalblue', alpha=1, width=bar_width, edgecolor="k", hatch='/')
    plt.bar(np.arange(3) + 3 *bar_width, fedsgd, label=u'FedSGD', color='orange', alpha=1, edgecolor="k",
            width=bar_width, hatch="o")
    plt.bar(np.arange(3) + 1 * bar_width, signsgd, label=u'SignSGD', color='limegreen', alpha=1, edgecolor="k",
            width=bar_width, hatch="x")
    plt.bar(np.arange(3) + 2 * bar_width, fedavg, label=u'FedAVG', color='saddlebrown', alpha=1, edgecolor="k",
            width=bar_width, hatch="/")
    # 添加刻度标签
    plt.xticks(np.arange(3) + 2*bar_width, labels)
    # plt.tick_params(labelsize=20)
    # 设置Y轴的刻度范围
    plt.ylim([0, math.ceil(maxlim/10)*10])
    plt.grid(True)
    # 为每个条形图添加数值标签
    # for x2016, proposed in enumerate(proposed):
    #     plt.text(x2016, proposed + 2, '%s' % proposed, ha='center', fontsize=font)

    # for x2017, fedsgd in enumerate(fedsgd):
    #     plt.text(x2017 + bar_width, fedsgd + 2, '%s' % fedsgd, ha='center', fontsize=font)

    # for x2018, signsgd in enumerate(signsgd):
    #     plt.text(x2018 + 2 * bar_width, signsgd + 2, '%s' % signsgd, ha='center', fontsize=font)

    # for x2019, y2019 in enumerate(Y2019):
    #     plt.text(x2019 + 3*bar_width, y2019 + 2, '%s' % y2019, ha='center', fontsize=font)

    # for x2020, y2020 in enumerate(Y2020):
    #     plt.text(x2020 + 4 * bar_width, y2020 + 2, '%s' % y2020, ha='center', fontsize=font)

    # 显示图例
    # plt.legend(bbox_to_anchor=(0.5, 1), loc=5, borderaxespad=0, fontsize=font)
    plt.legend(loc='upper left')

    # plt.title("test", fontsize=font)
    plt.xlabel("num_clients")
    plt.ylabel("Delay(s)")
    plt.savefig(save_path +'Delay'+file_name+'.png', dpi=300, format = 'png')

    #显示图形
    plt.show()

def cifar_iid(dataset, args):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/args.num_clients)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(args.num_clients):
        dict_users[i] = set(np.random.choice(all_idxs, num_items+5*(-1)**(i+1),
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def mnist_iid(dataset, args):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    # num_items = int(len(dataset)/args.num_clients)
    num_items = args.mean_datanum
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(args.num_clients):
        dict_users[i] = set(np.random.choice(all_idxs, num_items+5*(-1)**(i+1),
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

# 根据（节点本地）字典划分全局数据集，得到各个节点的数据集
class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)
    
if __name__ == '__main__':
    file_name='./FEDSGD/LA_SGD_T2.5_E0.015_w0.0065_c1.csv'
    
    plot_single_converg('./', file_name, if_loss=True)
