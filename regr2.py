from datetime import date,datetime,timedelta
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
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
        for tm in map(str,range(26,32)+range(53,59)):
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
for i in his_x_dict:
    print i,his_x_dict[i]
train_x_am_all=[]
train_y_am_all=[]
with open(path+"forplot.csv") as f:
    all=f.read()
    lines=all.split('\n')
    for line in lines:
        ls=line.split(',')
        if(len(ls)!=74):
            continue
        datei=datetime.strptime(ls[1],"%Y-%m-%d")
        if(datei>=datetime(2016,10,1) and datei<datetime(2016,10,8)):
            continue
        wd=datei.weekday()
        hisi=his_x_dict[(tol_dir,str(wd+1))]
        wdi=""
        if(wd>=0 and wd<=4):
            wdi="wday"
        else:
            wdi="wend"
        for timei in range(26,32):
            model_dict[(ls[0],wdi,str(timei))].append([map(float,ls[20:26:1])+map(float,hisi[0:6:1]),float(ls[timei])])
        for timei in range(53,59):
            model_dict[(ls[0],wdi,str(timei))].append([map(float,ls[47:53:1])+map(float,hisi[6:12:1]),float(ls[timei])])
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
    clf = Ridge(alpha=.5)
    svr = SVR(kernel='sigmoid', gamma=0.1)
    gbr = GradientBoostingRegressor()
    svr.fit(npx, npy)
    #clf.score(npx, npy)
    #print('Coefficient: n', linear.coef_)
    #print('Intercept: n', linear.intercept_)
    linear_dict[i]=svr
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
        wdi=""
        if(wd>=0 and wd<=4):
            wdi="wday"
        else:
            wdi="wend"
        outputam=[]
        outputpm=[]
        for timei in range(26,32):
            outputam.append(linear_dict[(ls[0],wdi,str(timei))].predict(x_test_am))
        for timei in range(53,59):
            outputpm.append(linear_dict[(ls[0],wdi,str(timei))].predict(x_test_pm))
        start_window=datei+timedelta(hours=8)
        for i in outputam:
            stop_window=start_window+timedelta(minutes=20)
            strstart=start_window.strftime("%Y-%m-%d %H:%M:%S")
            strstop=stop_window.strftime("%Y-%m-%d %H:%M:%S")
            outputs.append([ls[0].split("_")[0],"\"["+strstart+","+strstop+")\"",ls[0].split("_")[1],str(i.tolist()[0])])
            start_window=stop_window
        start_window=datei+timedelta(hours=17)
        for i in outputpm:
            stop_window=start_window+timedelta(minutes=20)
            strstart=start_window.strftime("%Y-%m-%d %H:%M:%S")
            strstop=stop_window.strftime("%Y-%m-%d %H:%M:%S")
            outputs.append([ls[0].split("_")[0],"\"["+strstart+","+strstop+")\"",ls[0].split("_")[1],str(i.tolist()[0])])
            start_window=stop_window
with open(path+"outdict_svr.csv","w") as f:
    f.write("tollgate_id,time_window,direction,volume\n")
    for i in outputs:
        f.write(",".join(i)+"\n")

# only test one trainset one test

# wdi="wday"
# tol_dir="1_1"
# timei='26'
# train_x=[]
# train_y=[]
# for i in model_dict[(tol_dir,wdi,timei)]:
#     train_x.append(i[0])
#     train_y.append(i[1])
# print train_x
# print train_y
# npx=np.array(train_x)
# npy=np.array(train_y)
# svr = SVR(kernel='sigmoid', gamma=0.1)
# svr.fit(npx, npy)
# test_X=[37.0,47.0,72.0,68.0,94.0,105.0]
# hisi=map(float,his_x_dict[(tol_dir,'2')][0:6])
# test_X=test_X+hisi
# print test_X
# print svr.predict(np.array([test_X]))
#
# wdi="wday"
# tol_dir="1_1"
# timei='26'
# train_x=[]
# train_y=[]
# for i in model_dict[(tol_dir,wdi,timei)]:
#     train_x.append(i[0])
#     train_y.append(i[1])
# print train_x
# print train_y
# npx=np.array(train_x)
# npy=np.array(train_y)
# svr = SVR(kernel='sigmoid', gamma=0.1)
# svr.fit(npx, npy)
# test_X=[480,280,570,670,980,1030]
# hisi=map(float,his_x_dict[(tol_dir,'3')][0:6])
# test_X=test_X+hisi
# print test_X
# print svr.predict(np.array([test_X]))