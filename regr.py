'''
Created on 2017-4-6

@author: Administrator
'''
from datetime import date,datetime,timedelta
from sklearn import linear_model
import numpy as np
path = 'E:\\kddcup\\dataSets\\dataSets\\training\\'
model_dict={}
linear_dict={}
outputs=[]
for tol_dir in ["1_1","1_0","2_0","3_1","3_0"]:
    for wd in ["wday","wend"]:
        for tm in ["am","pm"]:
            model_dict[(tol_dir,wd,tm)]=[]
with open(path+"forplot.csv") as f:
    all=f.read()
    lines=all.split('\n')
    for line in lines:
        ls=line.split(',')
        if(len(ls)!=74):
            continue
        datei=datetime.strptime(ls[1],"%Y-%m-%d")
        wd=datei.weekday()
        if(wd>=0 and wd<=4):
            model_dict[(ls[0],"wday","am")].append([map(int,ls[20:26:1]),map(int,ls[26:32:1])])
            model_dict[(ls[0],"wday","pm")].append([map(int,ls[47:53:1]),map(int,ls[53:59:1])])
        else:
            model_dict[(ls[0],"wend","am")].append([map(int,ls[20:26:1]),map(int,ls[26:32:1])])
            model_dict[(ls[0],"wend","pm")].append([map(int,ls[47:53:1]),map(int,ls[53:59:1])])
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
    print npx
    print npy
    linear.fit(npx, npy)
    linear.score(npx, npy)
    print('Coefficient: n', linear.coef_)
    print('Intercept: n', linear.intercept_)
    linear_dict[i]=linear
print linear_dict
with open(path+"forplottest1.csv") as f:
    all=f.read()
    lines=all.split('\n')
    for line in lines:
        ls=line.split(',')
        if(len(ls)!=74):
            continue
        datei=datetime.strptime(ls[1],"%Y-%m-%d")
        wd=datei.weekday()
        x_test_am=np.array([map(int,ls[20:26:1])])
        x_test_pm=np.array([map(int,ls[47:53:1])])
        if(wd>=0 and wd<=4):
            outputam=linear_dict[(ls[0],"wday","am")].predict(x_test_am)
            outputpm=linear_dict[(ls[0],"wday","pm")].predict(x_test_pm)
        else:
            outputam=linear_dict[(ls[0],"wend","am")].predict(x_test_am)
            outputpm=linear_dict[(ls[0],"wend","pm")].predict(x_test_pm)
        print outputam
        print outputpm
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
print outputs
with open(path+"outdict.csv","w") as f:
    for i in outputs:
        f.write(",".join(i)+"\n")


