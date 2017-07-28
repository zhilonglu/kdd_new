'''
Created on 2017-6-7

@author: Administrator
'''


from datetime import date,datetime,timedelta

path=""


with open(path+ "final"+".csv", "w") as f:
    f.write(','.join(['"tollgate_id"', '"time_window"', '"direction"', '"volume"']) + '\n')
    for tol_dir in ["1_0","1_1","2_0","3_0","3_1"]:
        datei=datetime(2016,10,25)
        for i in range(7):
            start_window=datei+timedelta(hours=8)
            for i in range(6):
                stop_window=start_window+timedelta(minutes=20)
                strstart=start_window.strftime("%Y-%m-%d %H:%M:%S")
                strstop=stop_window.strftime("%Y-%m-%d %H:%M:%S")
                start_window=stop_window
                f.write(",".join([tol_dir.split("_")[0],"\"["+strstart+","+strstop+")\"",tol_dir.split("_")[1],"0"]))
                f.write("\n")
        datei=datei + timedelta(days=1)
    for tol_dir in ["1_0","1_1","2_0","3_0","3_1"]:
        datei=datetime(2016,10,25)
        for i in range(7):
            start_window=datei+timedelta(hours=17)
            for i in range(6):
                stop_window=start_window+timedelta(minutes=20)
                strstart=start_window.strftime("%Y-%m-%d %H:%M:%S")
                strstop=stop_window.strftime("%Y-%m-%d %H:%M:%S")
                start_window=stop_window
                f.write(",".join([tol_dir.split("_")[0],"\"["+strstart+","+strstop+")\"",tol_dir.split("_")[1],"0"]))
                f.write("\n")
        datei=datei + timedelta(days=1)
