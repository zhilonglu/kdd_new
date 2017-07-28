import os
import datetime

path1 = "C:\\Users\\zhilonglu\\Desktop\\tensor_x\\"
path2 = "C:\\Users\\zhilonglu\\Desktop\\tensor_y\\"
path = "C:\\Users\\zhilonglu\\Desktop\\tensor\\"

def dateRange(start, end, step=1, format="%Y%m%d"):
    strptime, strftime = datetime.datetime.strptime, datetime.datetime.strftime
    days = (strptime(end, format) - strptime(start, format)).days
    return [strftime(strptime(start, format) + datetime.timedelta(i), format) for i in range(0, days, step)]

date_range_x = dateRange("20161015","20170101")
date_range_y = dateRange("20170103","20170118")
date_range = date_range_x + date_range_y
data_dir_x={}
data_dir_y={}
fileName =[]
data_x={}
data_y={}
files = os.listdir(path1)
for file in files:
    fileName.append(file)
    with open(path1+file+"\\tensor2.csv","w") as f2:
        for date in date_range:
            data_dir_x[date] = []
            for i in range(0, 120):
                data_dir_x[date].append(0)
        with open(path1+file+"\\tensor.csv") as f:
            all=f.read()
            lines=all.split('\n')
            for line in lines:
                ls=line.split(',')
                if (len(ls)!= 5):
                    continue
                idx = int(ls[3])-420
                data_dir_x[ls[0]][idx] = int(ls[4])
        title =[]
        for i in range(120):
            title.append(420+i)
        f2.write(" ,"+",".join(map(str,title))+"\n")
        data_x[file] = data_dir_x
        for i in sorted(data_dir_x.iterkeys()):
            f2.write(i+","+",".join(map(str,data_dir_x[i])) + "\n")
files = os.listdir(path2)
for file in files:
    with open(path2+file+"\\tensor2.csv","w") as f2:
        for date in date_range:
            data_dir_y[date] = []
            for i in range(0, 6):
                data_dir_y[date].append(0)
        with open(path2+file+"\\tensor.csv") as f:
            all=f.read()
            lines=all.split('\n')
            for line in lines:
                ls=line.split(',')
                if (len(ls)!= 5):
                    continue
                idx = int(ls[3])-27
                data_dir_y[ls[0]][idx] = int(ls[4])
        title =[]
        for i in range(6):
            title.append(27+i)
        f2.write(" ,"+",".join(map(str,title))+"\n")
        data_y[file] = data_dir_y
        for i in sorted(data_dir_y.iterkeys()):
            f2.write(i+","+",".join(map(str,data_dir_y[i])) + "\n")
tensor_data = {}
for i in fileName:
    tensor_data[i]={}
    for j in date_range:
        tensor_data[i][j]=[]
for i in fileName:
    for j in date_range:
        tensor_data[i][j] = data_x[i][j] + data_y[i][j]
        # print len(tensor_data[i][j])
for i in fileName:
    temp_dict = tensor_data[i]
    writepath = path+i+"\\tensor.csv"
    if not os.path.exists(path):
        os.mkdir(path)
    if not os.path.exists(path+i):
        os.mkdir(path+i)
    with open(writepath, "w") as f:
        for i in sorted(temp_dict.iterkeys()):
            f.write(i + "," + ",".join(map(str, temp_dict[i])) + "\n")


