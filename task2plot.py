'''
Created on 2017-4-6
@author: Administrator
'''
# -*- coding: utf-8 -*-
from datetime import date,datetime,timedelta
from sklearn import linear_model
from sklearn.linear_model import Ridge
# clf = Ridge(alpha=.5)
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
gbr = GradientBoostingRegressor()
import numpy as np
path = 'C:\\Users\\zhilonglu\\Desktop\\kdd\\KDDCUP\\'
model_dict={}
linear_dict={}
outputs=[]
his_dict={}
his_x_dict={}
for tol_dir in ["1_1","1_0","2_0","3_1","3_0"]:
    for wd in ["wday","wend"]:
        for tm in ["am","pm"]:
            model_dict[(tol_dir,wd,tm)]=[]
with open(path+"volume_history.csv") as f:
    all=f.read()
    lines=all.split('\n')
    for line in lines:
        ls=line.split(',')
        his_dict[(ls[0]+"_"+ls[3],ls[1],ls[2])]=ls[4]
# print his_dict
for tol_dir in ["1_1","1_0","2_0","3_1","3_0"]:
    for wd in map(str,range(1,8)):
        his_x_dict[(tol_dir,wd)]=[]
        for h in ["08:","09:","17:","18:"]:
            for m in ["00:00","20:00","40:00"]:
                his_x_dict[(tol_dir,wd)].append(his_dict[(tol_dir,wd,h+m)])
# print his_x_dict
with open(path+"forplot.csv") as f:
    all=f.read()
    lines=all.split('\n')
    for line in lines:
        ls=line.split(',')
        if(len(ls)!=74):
            continue
        datei=datetime.strptime(ls[1],"%Y-%m-%d")
        wd=datei.weekday()
        hisi=his_x_dict[(tol_dir,str(wd+1))]
        if(wd>=0 and wd<=4):
            model_dict[(ls[0],"wday","am")].append([map(float,ls[20:26:1])+map(float,hisi[0:6:1]),map(float,ls[26:32:1])])
            model_dict[(ls[0],"wday","pm")].append([map(float,ls[47:53:1])+map(float,hisi[6:12:1]),map(float,ls[53:59:1])])
        else:
            model_dict[(ls[0],"wend","am")].append([map(float,ls[20:26:1])+map(float,hisi[0:6:1]),map(float,ls[26:32:1])])
            model_dict[(ls[0],"wend","pm")].append([map(float,ls[47:53:1])+map(float,hisi[6:12:1]),map(float,ls[53:59:1])])
for i in model_dict:
    # Create linear regression object
    linear = linear_model.LinearRegression()
    x_train=[]
    y_train=[]
    for j in model_dict[i]:
        x_train.append(j[0])
        y_train.append(j[1])
    # Train the model using the training sets and check score
    npx=np.array(x_train)
    npy=np.array(y_train)
    # print npx
    # print npy
    linear.fit(npx, npy)
    # clf.fit(npx, npy)
    #clf.score(npx, npy)
    #print('Coefficient: n', linear.coef_)
    #print('Intercept: n', linear.intercept_)
    # linear_dict[i]=clf
    linear_dict[i]=linear
    # with open(path+"train_x_"+"_".join(map(str,i))+".csv","w") as f1:
    #     for j in x_train:
    #         f1.write(",".join(map(str,j))+"\n")
    # with open(path+"train_y_"+"_".join(map(str,i))+".csv","w") as f2:
    #     for j in y_train:
    #         f2.write(",".join(map(str,j))+"\n")
with open(path+"forplottest1.csv") as f:
    all=f.read()
    lines=all.split('\n')
    for line in lines:
        ls=line.split(',')
        if(len(ls)!=74):
            continue
        datei=datetime.strptime(ls[1],"%Y-%m-%d")
        wd=datei.weekday()
        hisi=his_x_dict[(tol_dir,str(wd+1))]
        x_test_am=np.array([map(float,ls[20:26:1])+map(float,hisi[0:6:1])])
        x_test_pm=np.array([map(float,ls[47:53:1])+map(float,hisi[6:12:1])])
        if(wd>=0 and wd<=4):
            outputam=linear_dict[(ls[0],"wday","am")].predict(x_test_am)
            outputpm=linear_dict[(ls[0],"wday","pm")].predict(x_test_pm)
        else:
            outputam=linear_dict[(ls[0],"wend","am")].predict(x_test_am)
            outputpm=linear_dict[(ls[0],"wend","pm")].predict(x_test_pm)
        # print outputam
        # print outputpm
        start_window=datei+timedelta(hours=8)
        for i in outputam.tolist()[0]:
            stop_window=start_window+timedelta(minutes=20)
            strstart=start_window.strftime("%Y-%m-%d %H:%M:%S")
            strstop=stop_window.strftime("%Y-%m-%d %H:%M:%S")
            outputs.append([ls[0].split("_")[0],"\"["+strstart+","+strstop+")\"",ls[0].split("_")[1],str(i)])
            start_window=stop_window
        start_window=datei+timedelta(hours=17)
        for i in outputpm.tolist()[0]:
            stop_window=start_window+timedelta(minutes=20)
            strstart=start_window.strftime("%Y-%m-%d %H:%M:%S")
            strstop=stop_window.strftime("%Y-%m-%d %H:%M:%S")
            outputs.append([ls[0].split("_")[0],"\"["+strstart+","+strstop+")\"",ls[0].split("_")[1],str(i)])
            start_window=stop_window
# print outputs
with open(path+"outdict1.csv","w") as f:
    f.write("tollgate_id,time_window,direction,volume\n")
    for i in outputs:
        f.write(",".join(i)+"\n")


