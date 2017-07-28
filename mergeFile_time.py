__author__ = 'zhilonglu'
file_suffix = '.csv'
path = 'C:\\Users\\zhilonglu\\Desktop\kdd\\submission_lzl\\time\\'
# meege file with last week when the value is minus
# set the data directory
def megerLastWeek(in_file1,in_file2):
    in_file_name1 = in_file1 + file_suffix
    in_file_name2 = in_file2 + file_suffix
    file1_dict={}
    file2_dict={}
    with open(path+in_file_name1,'r') as f1:
        f1.readline()  # skip the header
        data_1 = f1.readlines()
        f1.close()
    for i in range(len(data_1)):
            ls=data_1[i].replace('"','').replace('\n','').split(',')
            file1_dict[(ls[0],ls[1]+","+ls[2],ls[3])]=float(ls[4])
    with open(path+in_file_name2,'r') as f2:
        f2.readline()  # skip the header
        data_2 = f2.readlines()
        f2.close()
    for i in range(len(data_2)):
            ls=data_2[i].replace('"','').replace('\n','').split(',')
            file2_dict[(ls[0],ls[1]+","+ls[2],ls[3])]=float(ls[4])
    for i in file2_dict:
        if i in file1_dict:
            if file1_dict[i] < 0 :
                file1_dict[i] = file2_dict[i]
        else:
            print i
            print 'there is a error'
    # print outputs
    with open(path+"outdict_AR.csv","w") as f:
        f.write("intersection_id,tollgate_id,time_window,avg_travel_time\n")
        for i in file1_dict:
            f.write(",".join(i).replace('[','"[').replace(')',')"')+','+str(file1_dict[i])+"\n")
def main():
    in_file1 = 'output_travel_time_final'
    in_file2 = 'travel_time_final_2'
    megerLastWeek(in_file1,in_file2)
if __name__ == '__main__':
    main()