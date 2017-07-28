'''
Created on 2017-4-18

@author: Administrator
'''
from datetime import datetime
import os
import numpy as np
import tensorflow as tf
import gc
import sys
from sklearn.model_selection import KFold

def listmap(o,p):
    return list(map(o,p))

# taskname="1_0pm"
# cutime="20170504154250"
# inpath="E:\\kddcup\\tensordir\\"+taskname+"\\"
# modelpath="E:\\kddcup\\tensordir\\"+taskname+"\\"+cutime+"\\"

lossplots=[]
outputs=None
# 
# a = np.arange(0,12,0.5).reshape(4,-1)
# 
# np.savetxt(path+"a.csv",a,fmt="%.8f",delimiter=',')
# 
# b=np.loadtxt(path+"a.csv",delimiter=',')
# 
# print(b)

def decompose(tensor, n_output, n_valid, n_pred):
    print(tensor.shape)
    n_train=tensor.shape[0]-n_valid-n_pred
    n_input=tensor.shape[1]-n_output
    trainX = tensor[0: n_train, 0: n_input]
    trainY = tensor[0: n_train, n_input: n_input + n_output]
    validX = tensor[n_train: n_train + n_valid, 0: n_input]
    validY = tensor[n_train: n_train + n_valid, n_input: n_input + n_output]
    preX = tensor[n_train + n_valid: n_train + n_valid + n_pred, 0: n_input]
    return (trainX,trainY,validX,validY,preX)

def decompose2(tensor,outputnum,validnum,prenum):
    print(tensor.shape)
    trainnum=tensor.shape[0]-validnum-prenum
    inputnum=tensor.shape[1]-outputnum
    validX=tensor[0:validnum,0:inputnum]
    validY=tensor[0:validnum,inputnum:inputnum+outputnum]
    trainX=tensor[validnum:trainnum+validnum,0:inputnum]
    trainY=tensor[validnum:trainnum+validnum,inputnum:inputnum+outputnum]
    preX=tensor[trainnum+validnum:trainnum+validnum+prenum,0:inputnum]
    return (trainX,trainY,validX,validY,preX)

def splitData(tensor,n_output,n_pred):
    print(tensor.shape)
    n_known=tensor.shape[0]-n_pred
    n_input=tensor.shape[1]-n_output
    knownX = tensor[0: n_known, 0: n_input]
    knownY = tensor[0: n_known, n_input: n_input + n_output]
    preX = tensor[n_known: n_known+n_pred, 0: n_input]
    return (knownX,knownY,preX)

def fcn(trainX,trainY,hiddennum,times,keep,modelname,cutime,validX,validY,lr):
    global outputs
    inputnum=len(trainX[0])
    outputnum=len(trainY[0])
    losscol=[]
    npx=trainX
    npy=trainY
    npx_test=validX
    npy_test=validY
    nodenums=[inputnum]+hiddennum+[outputnum]
    sess=tf.InteractiveSession()
    x=tf.placeholder(tf.float32, [None,inputnum])
    y_=tf.placeholder(tf.float32,[None,outputnum])
    keep_prob=tf.placeholder(tf.float32)
    hiddens=[]
    drops=[x]
    for i in range(len(nodenums)-1):
        if(i==len(nodenums)-2):
            Wi=tf.Variable(tf.truncated_normal([nodenums[i],nodenums[i+1]],mean=10, stddev=0.1), name="W"+str(i)+cutime)
        else:
            Wi=tf.Variable(tf.truncated_normal([nodenums[i],nodenums[i+1]],mean=0, stddev=0.1), name="W"+str(i)+cutime)
        bi= tf.Variable(tf.ones(nodenums[i+1]), name="b"+str(i)+cutime)
        if i<len(nodenums)-2:
            hiddeni = tf.nn.relu(tf.add(tf.matmul(drops[i],Wi),bi))
            hiddens.append(hiddeni)
            dropi=tf.nn.dropout(hiddeni,keep_prob)
            drops.append(dropi)
        else:
            #y=tf.add(tf.add(tf.matmul(drops[i],Wi),bi),tf.slice(x, [0,6], [-1,-1]))
            y=tf.add(tf.matmul(drops[i],Wi),bi)
    #saver = tf.train.Saver()
    lossfun=tf.reduce_mean(tf.abs(tf.subtract(y/y_,1)))
    train_step=tf.train.AdamOptimizer(learning_rate=lr).minimize(lossfun)
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(times):
        noiseX=npx+2*np.random.random(npx.shape)-1
        noiseY=npy+2*np.random.random(npy.shape)-1
        sess.run(train_step,feed_dict={x:noiseX,y_:noiseY,keep_prob:keep})
        loss1=sess.run(lossfun,feed_dict={x:npx,y_:npy,keep_prob:1})
        loss2=sess.run(lossfun,feed_dict={x:npx_test,y_:npy_test,keep_prob:1})
        losscol.append([loss1,loss2,loss1+loss2])
    losscolnp=np.array(losscol)
    #save_path = saver.save(sess, path+modelname+".ckpt")
    lossplots.append("'"+cutime+","+",".join(listmap(str,losscol[-1])))
    print(losscol[-1])
    
    # valid
    preY_valid = sess.run(y, feed_dict={x: npx_test, keep_prob: 1})
    np.savetxt(path + "preY_valid.csv", preY_valid.reshape(-1, 1), fmt="%.8f", delimiter=',')

    # test
    preY_test = sess.run(y, feed_dict={x: preX, keep_prob: 1})
    np.savetxt(path + "preY_test.csv", preY_test.reshape(-1, 1), fmt="%.8f", delimiter=',')
    
    if outputs is None:
        outputs=preY_test.reshape(-1, 1)
    else:
        outputs=np.c_[outputs, preY_test.reshape(-1, 1)]
    
    np.savetxt(path+"losscol.csv",losscolnp,fmt="%.8f",delimiter=',')
    
    del sess
    del hiddens
    del drops
    del losscolnp
    del losscol
    gc.collect()

# onehide(trainX, trainY, 6, int(1e4), 0.94, taskname)
# pred([18,18,18],6,taskname, cutime, preX)

for taskname in ["1_0am","1_1am","2_0am","3_0am","3_1am","1_0pm","1_1pm","2_0pm","3_0pm","3_1pm"]:
    rootpath="data_10\\"

    inpath=rootpath+taskname+"\\"
    cutime=datetime.strftime(datetime.now(),"%Y%m%d%H%M%S")
    tempath=rootpath+"tmp\\"
    # tensor=np.loadtxt(inpath+"tensor_new.csv",delimiter=',')
    tensor=np.loadtxt(inpath+"tensor_new.csv",delimiter=',')
    knownX,knownY,preX=splitData(tensor,6,7)
    alltimes=0
    for i in range(8):
        kf = KFold(n_splits=int(knownX.shape[0]))
    #    kf = KFold(n_splits=int(knownX.shape[0]/3))
    #    kf = KFold(n_splits=3)
        for train_index, valid_index in kf.split(knownX):
            print("TRAIN:", train_index, "VALID:", valid_index)
            cutime=datetime.strftime(datetime.now(),"%Y%m%d%H%M%S")
            path=inpath+cutime+"\\"
            os.makedirs(path)
            print(path)
            trainX, validX = knownX[train_index], knownX[valid_index]
            trainY, validY = knownY[train_index], knownY[valid_index]
            print(trainX.shape,trainY.shape,validX.shape,validY.shape,preX.shape)
            np.savetxt(path+"validYtrue.csv",validY.reshape(-1, 1), fmt="%.8f",delimiter=',')
            lossi=fcn(trainX, trainY, [18]*10, int(5e4), 0.9, taskname, cutime, validX, validY, 3e-4)
            with open(path +"lossplots"+datetime.strftime(datetime.now(),"%Y%m%d%H%M%S")+".csv","w") as f:
                f.write("\n".join(lossplots))
            #     if(lossi<0.09):
            #         break
            if(alltimes>=50):
                break
            else:
                alltimes=alltimes+1
        if(alltimes>=50):
            break
    cutime=datetime.strftime(datetime.now(),"%Y%m%d%H%M%S")
    np.savetxt(tempath+ taskname + "outputs.csv", outputs, fmt="%.8f", delimiter=',')
    np.savetxt(tempath+ taskname + "outputsmedian.csv", np.median(outputs,1), fmt="%.8f", delimiter=',')
    np.savetxt(tempath+ taskname + "outputsmean.csv", np.mean(outputs,1), fmt="%.8f", delimiter=',')