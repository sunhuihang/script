#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import datetime
import numpy as np

###########################################
### 对从中国气象数据网上下载的数据进行处理 ###
###########################################



time='2022' #下订单的年 例‘2022’
time_fmt='%Y%m%d%H'
file_station  = os.path.join(r'/data/Atmos/wenjk/autorun_station_data_process','SURF_CHN_LIST.csv')
file_raw_data = os.path.join(r'/data/Atmos/SHARE/atmosData/obs/station/rawdata_CHN/need_processed')
file_reprocess_data = os.path.join(r'/data/Atmos/SHARE/atmosData/obs/station/CHN/2022/sorted_by_province')

station_info = pd.read_csv(file_station,encoding='gbk')
station_info = station_info[['Station_Name','Province','Station_Id_C','Lat','Lon','Alti']]
station_info['Province'] = station_info['Province'].str.slice(0,2)

############### 获取file_raw_data目录下的所有站点气象txt文件 ###########
def get_files():
    list_file = []
    for filepath, dirnames, filenames in os.walk(file_raw_data):
        for filename in filenames:
            prefix,suffix = filename.split('.')
            #print(prefix[0:7] == ('S'+time) and suffix == 'txt' )
            if prefix[0:5] == ('S'+time) and suffix == 'txt':
                 list_file.append(os.path.join(filepath, filename))
    return list_file

def remove_files(list_files):
    for file_name in list_files:
        target_file = file_name.replace("need_processed", "have_processed")
        os.replace(file_name, target_file)


############## 根据站点的station ID 获取该站点位于哪个省份 ##########
def get_province(station_id,station_info):
    station_info_data = station_info[station_info['Station_Id_C'].isin([station_id])]
    station_province  = station_info_data['Province'].iloc[0]
    return station_province
    
#list_raw_data_file= [os.path.join(r'D:\数据存放\中国气象数据网数据\原始文件\文金科', 'SURF_CHN_MUL_HOR_20220707-20220810.txt')]
list_raw_data_file = get_files()
print(list_raw_data_file)
file_name_list = []
################### 对你给定的文件夹下面的每一个站点文件进行循环读取 ##############
for filename in list_raw_data_file:
    raw_data = pd.read_csv(filename,sep='\\s+')
    province_current_file = get_province(raw_data['Station_Id_C'].iloc[1],station_info)
    #### 获取文件的起始时间 #####
    #print(raw_data.info())
    time_start_data = raw_data.iloc[0]
    time_end_data = raw_data.iloc[raw_data.shape[0]-1]
    datetime_start = datetime.datetime(int(time_start_data['Year']),int(time_start_data['Mon']),int(time_start_data['Day']),int(time_start_data['Hour']))
    datetime_end = datetime.datetime(int(time_end_data['Year']),int(time_end_data['Mon']),int(time_end_data['Day']),int(time_end_data['Hour']))
    
    
    ####### 读取站点文件中的数据有几天 ############
    delta = datetime.timedelta(hours=1)
    interval = int((datetime_end - datetime_start).days*24+(datetime_end-datetime_start).seconds/60/60)+1
    time_list = [(datetime_start+delta*i).strftime(time_fmt) for i in range(0,interval,1)]
    time_name = time_list[0][:8]
    time_day_list = list(set([time[:8] for time  in time_list]))
    time_day_list.sort(key=[time[:8] for time  in time_list].index) 
    #print(type(time_day_list),time_day_list)
    
    ######## 对day进行循环并读取数据写入csv文件 #####
    for time_day in time_day_list:
        folder_file = os.path.join(file_reprocess_data,province_current_file)
        file_name_write = os.path.join(folder_file,province_current_file+'SURF_CHN_MUL_HOR_'+time_day+'.csv')
        data_write_today  = raw_data[raw_data['Day'].isin([np.int64(time_day[-2:])])]
        
        if not os.path.exists(folder_file):
            os.mkdir(folder_file)
        with open(file_name_write,"w") as file:  
            data_write_today.to_csv(file_name_write,index=False, sep=',')
        
remove_files( list_raw_data_file ) 
        
        

