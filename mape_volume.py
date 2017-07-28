__author__ = 'zhilonglu'
file_suffix = '.csv'
path = 'C:\\Users\\zhilonglu\\Desktop\kdd\\submission_lzl\\volume\\'
# set the data directory
def mape(in_file1,in_file2):
    in_file_name1 = in_file1 + file_suffix
    in_file_name2 = in_file2 + file_suffix
    error_dict={}
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
    for i in file1_dict:
        if i in file2_dict:
            if (i[0],i[2]) not in error_dict:
                error_dict[(i[0],i[2])] = [0,1]
                error_dict[(i[0],i[2])][0] += abs(file1_dict[i]-file2_dict[i])/file2_dict[i]
            else:
                error_dict[(i[0],i[2])][0] += abs(file1_dict[i]-file2_dict[i])/file2_dict[i]
                error_dict[(i[0],i[2])][1] += 1
        else:
            print i
            print 'there is a error'
    mape_value = 0
    for j in error_dict:
        print j,error_dict[j][0],error_dict[j][1]
        mape_value += error_dict[j][0]/error_dict[j][1]
    print mape_value/5
def main():
    # file2 as the baseline
    # judge file1
    in_file1 = 'outdict_AR'
    in_file2 = 'volume_final_2'
    mape(in_file1,in_file2)
if __name__ == '__main__':
    main()