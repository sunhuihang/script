import os
import csv
import pandas as pd
import glob
import time
path = '/data/Atmos/SHARE/atmosData/obs/station/CHN/2022/sorted_by_province/山东/'
file_list=glob.glob(path+"*202207*.csv")
datas = []
# 读取需要的csv文件
for file in sorted(file_list):
    print(file)
    with open(file, "r") as csvfile:
        reader = csv.reader(csvfile)
        # 去掉表头（第一行）
        reader = list(reader)[1:]
        for line in reader:
            datas.append(line)

#读最后一个文件的表头
csv_head = list(pd.read_csv(file))

excel_name = './山东20220707-20220731.csv'
with open(excel_name, 'w') as csvfile2:
    writer = csv.writer(csvfile2)
    # 写表头
    writer.writerow(csv_head)
    writer.writerows(datas)

print("finish")
