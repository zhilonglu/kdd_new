from __future__ import print_function
from statsmodels.graphics.api import qqplot
from scipy import  stats
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import statsmodels.api as sm

#step 1 load data plot figure
# dta=[10930,10318,10595,10972,7706,6756,9092,10551,9722,10913,11151,8186,6422,
# 6337,11649,11652,10310,12043,7937,6476,9662,9570,9981,9331,9449,6773,6304,9355,
# 10477,10148,10395,11261,8713,7299,10424,10795,11069,11602,11427,9095,7707,10767,
# 12136,12812,12006,12528,10329,7818,11719,11683,12603,11495,13670,11337,10232,
# 13261,13230,15535,16837,19598,14823,11622,19391,18177,19994,14723,15694,13248,
# 9543,12872,13101,15053,12619,13749,10228,9725,14729,12518,14564,15085,14722,
# 11999,9390,13481,14795,15845,15271,14686,11054,10395]
# print(len(dta))
dta=[5.8,5.9,6.1,5,14.8,5.5,8.8,4.8,19,3,5.6,6.6,6.3,5.5,4.4,4.9,7.2,5,4.8,5,15.7,4.9,5.5,6.2,5.6,6,5.2,43,4.8,6.6,6.9,10.3,4.9,5.1,5.3,5.9,5.3,5.6,16.9,5.9,4,5,7.2,5.9,5.9,7.5,6.5,5.5,5.6,5.4,5.8,9.9,5.5,5.3,8.7,10.4,6,6,6,8.9,5.1,
       5,7.5,16.4,6.4,6.3,4.6,5.3,4.7,5.7,6,5.3,6.6,9.2,4.9,7.6,5.6,12.8,7.3,4.2,5.6,6.6,5.3,5.9,8.4,6.7,18.9,7.3,6.4,5,5.2,7.4]
# print(len(dta))
dta=pd.Series(dta)
dta.index = pd.Index(sm.tsa.datetools.dates_from_range('2001','2092'))
# print(dta)
# dta.plot(figsize=(12,8))
# plt.show()

# step 2 find d infer difference
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

#step 3 find autocorrelation and partial autocorrelation
# dta= dta.diff(1)
# print(dta)
fig = plt.figure(figsize=(12,8))
ax1=fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(dta,lags=40,ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(dta,lags=40,ax=ax2)
# plt.show()

#step 4 find the best model param aic,bic,hqic
for i in sm.tsa.datetools.dates_from_range('2001','2092'):
    if math.isnan(dta[i]):#set nan to zero
        dta[i]=0
arma_mod20 = sm.tsa.ARMA(dta,(7,0)).fit()
print(arma_mod20.aic,arma_mod20.bic,arma_mod20.hqic)
# arma_mod30 = sm.tsa.ARMA(dta,(0,1)).fit()
# print(arma_mod30.aic,arma_mod30.bic,arma_mod30.hqic)
# arma_mod40 = sm.tsa.ARMA(dta,(7,1)).fit()
# print(arma_mod40.aic,arma_mod40.bic,arma_mod40.hqic)
# arma_mod50 = sm.tsa.ARMA(dta,(8,0)).fit()
# print(arma_mod50.aic,arma_mod50.bic,arma_mod50.hqic)


#step 5 plot figure autocorrelation and partial autocorrelation
# check the model
resid = arma_mod20.resid
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(resid.values.squeeze(), lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(resid, lags=40, ax=ax2)
# plt.show()

#step 6 d-w check
# print(sm.stats.durbin_watson(arma_mod20.resid.values))

#step 7 find if suit some distribution
resid = arma_mod20.resid
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
fig = qqplot(resid, line='q', ax=ax, fit=True)
# plt.show()

#step 8 Ljung-Box check
r,q,p = sm.tsa.acf(resid.values.squeeze(), qstat=True)
data = np.c_[range(1,41), r[1:], q, p]
table = pd.DataFrame(data, columns=['lag', "AC", "Q", "Prob(>Q)"])
# print(table.set_index('lag'))

#step 9 predict the result
predict_sunspots = arma_mod20.predict('2092', '2122', dynamic=True)
# print(type(predict_sunspots))
#将pandas.core.series.Series转化成list类型
dataList = predict_sunspots.values.tolist()
print(dataList)
fig, ax = plt.subplots(figsize=(12, 8))
ax = dta.ix['2001':].plot(ax=ax)
predict_sunspots.plot(ax=ax)
# plt.show()