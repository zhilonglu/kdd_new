'''
Created on 2017-5-28

@author: Administrator
'''

# path = "E:\\kddcup\\tensordir\\tem\\"
path = "data_10\\tmp\\"

for suf in ["outputs.csv","outputsmean.csv","outputsmedian.csv"]:
    with open(path+"all"+suf,"w") as fout:
        for apm in ["am","pm"]:
            for tol_dir in ["1_0","1_1","2_0","3_0","3_1"]:
                with open(path+tol_dir+apm+suf) as fin:
                    fall=fin.read()
                    fout.write(fall)

