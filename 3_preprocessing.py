# -*- coding: utf-8 -*-


import numpy as np
import os

path = ''  # set the data directory
## flow data
path_flow_train = path+'trainingforplot.csv'
path_flow_train1 = path+'training2forplot.csv'
path_flow_test = path+'test2forplot.csv'
flow = np.genfromtxt(path_flow_train, dtype=str, delimiter=',')[:,2:]
flow1 = np.genfromtxt(path_flow_train1, dtype=str, delimiter=',')[:,2:]
flow2 = np.genfromtxt(path_flow_test, dtype=str, delimiter=',')[:,2:]

# 19~24 -> 25~30
# 46~51 -> 52~57
flow = np.c_[flow[:,18:18+12],flow[:,45:45+12]].astype(np.float32)
flow = np.c_[flow[:,0:6],flow[:,12:18],flow[:,6:12],flow[:,18:24]]
flow1 = np.c_[flow1[:,18:18+12],flow1[:,45:45+12]].astype(np.float32)
flow1 = np.c_[flow1[:,0:6],flow1[:,12:18],flow1[:,6:12],flow1[:,18:24]]
flow2 = np.c_[flow2[:,18:18+12],flow2[:,45:45+12]].astype(np.float32)
flow2 = np.c_[flow2[:,0:6],flow2[:,12:18],flow2[:,6:12],flow2[:,18:24]]

###########################
# c1 c2 c3 c4 c5
n_train = int(flow.shape[0] / 5)
n_train1 = int(flow1.shape[0] / 5)
n_test = int(flow2.shape[0] / 5)
print(n_train)
print(n_train1)
print(n_test)
for i in range(5):
    ff = np.r_[flow[i * n_train: (i + 1) * n_train, :], flow1[i * n_train1: (i + 1) * n_train1, :], flow2[i * n_test: (i + 1) * n_test, :]]
    ff = np.r_[ff[:12,:], ff[19:,:]] # delete Guoqingjie
    if not os.path.exists('data/C%d'%(i+1)):
        os.makedirs('data/C%d'%(i+1))
    np.savetxt(path+'data/C%d/tensor_new.csv'%(i+1), ff, fmt = '%.4f', delimiter=',')
    
    
    
# c1 c2 c3 c4 c5 c6 c7 c8 c9 c10
n_train = int(flow.shape[0] / 5)
n_train1 = int(flow1.shape[0] / 5)
n_test = int(flow2.shape[0] / 5)
task_name = ["1_0", "1_1", "2_0", "3_0", "3_1"]
for i in range(5):
    ff = np.r_[flow[i * n_train: (i + 1) * n_train, :], flow1[i * n_train1: (i + 1) * n_train1, :], flow2[i * n_test: (i + 1) * n_test, :]]
    print(ff.shape)
    ff = np.r_[ff[:12,:], ff[19:,:]] # delete Guoqingjie
    print(ff.shape)
    path_am = path+'data_10/%sam'%(task_name[i])
    if not os.path.exists(path_am):
        os.makedirs(path_am)
    np.savetxt(path_am + '/tensor_new.csv', np.c_[ff[:,0:6],ff[:,12:18]], fmt = '%.4f', delimiter=',')
    
    path_pm = path+'data_10/%spm'%(task_name[i])
    if(i==2):
        ff = np.r_[ff[:4,:], ff[6:,:]] # 2-0pm delete 0
    if not os.path.exists(path_pm):
        os.makedirs(path_pm)
    np.savetxt(path_pm + '/tensor_new.csv', np.c_[ff[:,0:12],ff[:,18:24]], fmt = '%.4f', delimiter=',')
if not os.path.exists(path+'data_10/tmp'):
    os.makedirs(path+'data_10/tmp')

    