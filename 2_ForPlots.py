'''
Created on 2017-4-1

@author: Administrator
'''
from datetime import datetime,date,timedelta
path = ''  # set the data directory

fileins=["training_20min_avg_volume","training2_20min_avg_volume","test2_20min_avg_volume"]
mydates=[date(2016,9,19),date(2016,10,18),date(2016,10,25)]
dayses=[29,7,7]
for myind in [0,1,2]:
    filein=fileins[myind]
    days=dayses[myind]
    v_dict={}
    res_dict={}
    with open(path+filein+".csv") as f:
        all=f.read()
        lines=all.split('\n')
        for line in lines:
            ls=line.split(',')
            if(len(ls)!=4):
                continue
            dt=ls[1].split(' ')
            datei=dt[0]
            time=dt[1]
            v_dict[(ls[0]+"_"+ls[2],datei,time)]=ls[3]
    
    with open(path+ filein.split('_')[0] +"forplot"+".csv", "w") as f:
        for tol_dir in ["1_0","1_1","2_0","3_0","3_1"]:
            mydate=mydates[myind]
            for i in range(days):
                strmydate=mydate.strftime('%Y-%m-%d')
                res_dict[(tol_dir,strmydate)]=[]
                for j in range(24):
                    for k in [0,20,40]:
                        if j<10:
                            strj="0"+str(j)
                        else:
                            strj=str(j)
                        if k<10:
                            strk="0"+str(k)
                        else:
                            strk=str(k)
                        if (tol_dir,strmydate,strj+":"+strk+":00") in v_dict:
                            res_dict[(tol_dir,strmydate)].append(str(v_dict[(tol_dir,strmydate,strj+":"+strk+":00")]))
                        else:
                            res_dict[(tol_dir,strmydate)].append("0")
                f.write(tol_dir+","+strmydate+","+",".join(res_dict[(tol_dir,strmydate)])+'\n')
                mydate=mydate + timedelta(days=1)