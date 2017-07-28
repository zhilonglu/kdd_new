__author__ = 'zhilonglu'
from datetime import date,datetime,timedelta
file_suffix = '.csv'
path = 'C:\\Users\\zhilonglu\\Desktop\kdd\\KDDCUP\\'
def transform(in_file1,in_file2):
    in_file_name1 = in_file1 + file_suffix
    in_file_name2 = in_file2 + file_suffix
    # not differ tollgate
    volume_total_dict={}
    precipitation_dict={}
    volume_precipitation_dict={}
    # tollgate dire
    volume_dire_dict={}
    precipitation_dire_dict={}
    volume_precipitation_dire_dict={}
    with open(path+in_file_name1,'r') as f1:
        f1.readline()  # skip the header
        data_1 = f1.readlines()
        f1.close()
    with open(path+in_file_name2,'r') as f2:
        f2.readline()  # skip the header
        data_2 = f2.readlines()
        f2.close()
    #total
    for i in range(len(data_2)):
        ls=data_2[i].replace('"','').split(',')
        if ls[0] not in ["[2016-10-01","[2016-10-02","[2016-10-03","[2016-10-04","[2016-10-05","[2016-10-06","[2016-10-07"]:
            if '\"'+ls[1]+','+ls[2]+'\"' not in volume_total_dict:
                volume_total_dict['\"'+ls[1]+','+ls[2]+'\"'] = float(ls[4])
            else:
                volume_total_dict['\"'+ls[1]+','+ls[2]+'\"'] += float(ls[4])
            volume_dire_dict['\"'+ls[1]+','+ls[2]+'\",'+ls[0]+'_'+ls[3]] = float(ls[4])
    # print volume_total_dict
    # 06-08 forcast 08-10;15-17 forcast 17-19
    with open(path+'weather_train.csv','w') as f:
        for i in range(len(data_1)):
            ls=data_1[i].replace('"','').split(',')
            if ls[0] not in ["2016-10-01","2016-10-02","2016-10-03","2016-10-04","2016-10-05","2016-10-06","2016-10-07"]:
                # ls[0]=2016-07-01 ls[8]=0.0
                datei=datetime.strptime(ls[0],"%Y-%m-%d")
                start_window=datei+timedelta(hours=int(ls[1]))
                for j in range(9):
                    stop_window=start_window+timedelta(minutes=20)
                    strstart=start_window.strftime("%Y-%m-%d %H:%M:%S")
                    strstop=stop_window.strftime("%Y-%m-%d %H:%M:%S")
                    precipitation_dict["\"["+strstart+","+strstop+")\""] = float(ls[8])
                    f.write("\"["+strstart+","+strstop+")\","+ls[8])
                    start_window=stop_window
    # tollgate dire
    with open(path+'weather_train_dire.csv','w') as f:
        for i in range(len(data_1)):
            ls=data_1[i].replace('"','').split(',')
            if ls[0] not in ["2016-10-01","2016-10-02","2016-10-03","2016-10-04","2016-10-05","2016-10-06","2016-10-07"]:
                # ls[0]=2016-07-01 ls[8]=0.0
                datei=datetime.strptime(ls[0],"%Y-%m-%d")
                start_window=datei+timedelta(hours=int(ls[1]))
                for j in range(9):
                    stop_window=start_window+timedelta(minutes=20)
                    strstart=start_window.strftime("%Y-%m-%d %H:%M:%S")
                    strstop=stop_window.strftime("%Y-%m-%d %H:%M:%S")
                    for tol_dir in ["1_1","1_0","2_0","3_1","3_0"]:
                        precipitation_dire_dict["\"["+strstart+","+strstop+")\","+tol_dir] = float(ls[8])
                        f.write("\"["+strstart+","+strstop+")\","+ls[8])
                    start_window=stop_window
    # print precipitation_dict
    with open(path+'volume_precipitation.csv','w') as f:
        for i in precipitation_dict:
            if i in volume_total_dict:
                volume_precipitation_dict[i] = str(volume_total_dict[i])+','+str(precipitation_dict[i])
                f.write(i+','+volume_precipitation_dict[i]+'\n')
        # print volume_precipitation_dict
    with open(path+'volume_precipitation_dire.csv','w') as f:
        for i in precipitation_dire_dict:
            if i in volume_dire_dict:
                volume_precipitation_dire_dict[i] = str(volume_dire_dict[i])+','+str(precipitation_dire_dict[i])
                f.write(i+','+volume_precipitation_dire_dict[i]+'\n')
def main():
    in_file_1 = 'weather (table 7)_training_update'
    in_file_2 = 'training_20min_avg_volume'
    transform(in_file_1,in_file_2)
if __name__ == '__main__':
    main()