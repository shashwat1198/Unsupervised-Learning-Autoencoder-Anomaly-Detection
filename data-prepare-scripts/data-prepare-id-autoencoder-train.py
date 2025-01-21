import csv
import numpy as np

file = open("DoS_dataset.csv")
csvreader = csv.reader(file)
rows = []
for row in csvreader:
    rows.append(row)
file.close()
a = np.array(rows)
file_size = len(a)

# For generating the X variable of the training data uncomment the below for loop.
for i in range(file_size):#file_size
    # print(a[i][1])
    inter_id_int = int(a[i][1],16) #Hexadecimal to integer conversion step.
    #print(inter_id_int)
    end_length = 12
    res = bin(inter_id_int)#Integer to binary conversion step.
    id_bin = res[2:].zfill(end_length)
    print(id_bin[0]+','+id_bin[1]+','+id_bin[2]+','+id_bin[3]+','+id_bin[4]+','+id_bin[5]+','+id_bin[6]+','+id_bin[7]+','+id_bin[8]+','+id_bin[9]+','+id_bin[10]+','+id_bin[11])
