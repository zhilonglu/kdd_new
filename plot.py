from statsmodels.graphics.api import qqplot
from scipy import  stats
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import statsmodels.api as sm
path = 'C:\\Users\\zhilonglu\\Desktop\kdd\\trainData_lzl\\'
file_dict ={}
sort_file_dict={}
with open(path+'training_20min_avg_volume.csv','r') as f1:
    f1.readline()  # skip the header
    data_1 = f1.readlines()
    f1.close()
for i in range(len(data_1)):
    ls = data_1[i].replace('\"','').split(',')
    if ls[1].split(" ")[0] not in ["[2016-10-01","[2016-10-02","[2016-10-03","[2016-10-04","[2016-10-05","[2016-10-06","[2016-10-07"]:
        file_dict[(ls[0]+'_'+ls[3],ls[1]+','+ls[2])] = float(ls[4])
# print file_dict
sort_file_dict = sorted(file_dict.iteritems(),key=lambda d:d[0])
# print sort_file_dict
tollgate_dict={}
for tol_dir in ["1_1","1_0","2_0","3_1","3_0"]:
    tollgate_dict[tol_dir]=[]
with open(path+'avg_volume_tollgate_dire.csv','w') as f:
    for i in sort_file_dict:
        # print i[0][0]
        tollgate_dict[i[0][0]].append(i[1])
        f.write(','.join(i[0])+','+str(i[1])+'\n')
# with open(path+'tollgate_dict.csv','w') as f:
#     for i in tollgate_dict:
#         # print i
#         # print len(tollgate_dict[i])
#         # f.write(i+','+','.join(map(str,tollgate_dict[i]))+'\n')
#         dta=pd.Series(tollgate_dict[i])
#         print dta
#         dta.index = pd.Index(sm.tsa.datetools.dates_from_range('1700',length=len(tollgate_dict[i])))
#         # dta.plot(figsize=(12,8))
#         # plt.show()
dateTime=[]
for h in ["06:","07:","08:","09:","15:","16:","17:","18:"]:
        for m in ["00:00","20:00","40:00"]:
            dateTime.append(h+m)
# print dateTime
dta=[]
for i in sort_file_dict:
    # print sort_file_dict[i]
    if i[0][0]=='1_0':
        if i[0][1].split(',')[0].split(' ')[1] =="08:40:00":
            dta.append(i[1])
        # if i[0][1] in
    # print i[0][0]
    # print i[0][1].split(',')[0].split(' ')[1]

# dta = tollgate_dict['1_1']
print dta
# print len(dta)

# pd.Timestamp.min ('1677-09-22 00:12:43.145225')
# pd.Timestamp.max ('2262-04-11 23:47:16.854775807')
#584 years
dta=pd.Series(dta)
# start = '1760'
# end = str(int(1760)+len(dta)-1)
# dta.index = pd.Index(sm.tsa.datetools.dates_from_range(start,end))
dta.index = pd.Index(sm.tsa.datetools.dates_from_range('1760','1781'))
print dta

# dta.plot(figsize=(12,8))
# plt.show()
# first difference
# fig = plt.figure(figsize=(12,8))
# ax1= fig.add_subplot(111)
# diff1 = dta.diff(1)
# diff1.plot(ax=ax1)
# plt.show()
# second difference
# fig = plt.figure(figsize=(12,8))
# ax2= fig.add_subplot(111)
# diff2 = dta.diff(2)
# diff2.plot(ax=ax2)
# plt.show()
# third difference
# fig = plt.figure(figsize=(12,8))
# ax3= fig.add_subplot(111)
# diff3 = dta.diff(3)
# diff2.plot(ax=ax3)
# plt.show()

# dta= dta.diff(1)
# fig = plt.figure(figsize=(12,8))
# ax1=fig.add_subplot(211)
# fig = sm.graphics.tsa.plot_acf(dta,lags=40,ax=ax1)
# ax2 = fig.add_subplot(212)
# fig = sm.graphics.tsa.plot_pacf(dta,lags=40,ax=ax2)
# plt.show()


arma_mod20 = sm.tsa.AR(dta).fit()
print arma_mod20.aic,arma_mod20.bic,arma_mod20.hqic

# arma_mod20 = sm.tsa.ARMA(dta,(2,0)).fit()
# print(arma_mod20.aic,arma_mod20.bic,arma_mod20.hqic)
# arma_mod30 = sm.tsa.ARMA(dta,(0,1)).fit()
# print(arma_mod30.aic,arma_mod30.bic,arma_mod30.hqic)
# arma_mod40 = sm.tsa.ARMA(dta,(2,1)).fit()
# print(arma_mod40.aic,arma_mod40.bic,arma_mod40.hqic)
# arma_mod50 = sm.tsa.ARMA(dta,(2,4)).fit()
# print(arma_mod50.aic,arma_mod50.bic,arma_mod50.hqic)

predict_sunspots = arma_mod20.predict('1782', '1788', dynamic=True)
print predict_sunspots
fig, ax = plt.subplots(figsize=(12, 8))
ax = dta.ix['1760':].plot(ax=ax)
predict_sunspots.plot(ax=ax)
plt.show()