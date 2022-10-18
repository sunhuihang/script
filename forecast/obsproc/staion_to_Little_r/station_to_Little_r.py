import os
import little_r as lr 
import pandas as pd
import datetime
import numpy as np

start_date   = "2022073021"
end_date   = "2022073103"
province = 'CHN'
#province = '山东'
dir_data_weather = '/data/Atmos/SHARE/atmosData/obs/station/CHN/2022'
folder_data_little_R = "/data/Atmos/sunhh/Little_R/2022073100"
file_data_little_R = os.path.join(folder_data_little_R,province+start_date+'_'+end_date+'.little_r')
print(file_data_little_R)
station_info = pd.read_csv("/data/Atmos/SHARE/atmosData/obs/station/CHN/SURF_CHN_LIST.csv")
#station_info = pd.read_excel("/mnt/d/work/sunhh/data/station/SURF_CHN_LIST.xlsx")
station_info = station_info[['Station_Id_C','Lat','Lon','Alti']]

def find_lat_lon(station_id,station_info):
    station_lat_lon = station_info[station_info['Station_Id_C'].isin([station_id])]
    return station_lat_lon.iloc[0,1],station_lat_lon.iloc[0,2], station_lat_lon.iloc[0,3]   #返回三个参数，分别时观测站点的纬度、经度以及海拔高度

# def hour_range(start_date,end_date):
#     fmt='%Y%m%d%H'
#     begin = datetime.datetime.strptime(start_date,fmt)
#     end   = datetime.datetime.strptime(end_date,fmt)
#     delta = datetime.timedelta(hours=24)
#     #interval = int((end - begin).days*24+(end-begin).seconds/60/60)+1
#     interval = int((end - begin).days+(end-begin).seconds/60/60/24)+1
#     return [(begin+delta*i).strftime(fmt) for i in range(0,interval,1)]

def hour_range(bgn,end):
    fmt = '%Y%m%d%H'
    begin=datetime.datetime.strptime(bgn,fmt)
    end=datetime.datetime.strptime(end,fmt)
    delta=datetime.timedelta(hours=1)
    interval=int((end-begin).days*24+(end-begin).seconds/60/60) +1  
    return [(begin+delta*i).strftime(fmt) for i in range(0,interval,1)]


def str_I2(int0):
    space = '0'*(2-len(str(int0)))
    return space+str(int0)

#删除存在的同名little文件
if os.path.exists(file_data_little_R):
       os.remove(file_data_little_R)

with open(file_data_little_R,"w") as file:  
    #对每个小时进行循环
    print(hour_range(start_date, end_date))
    for time in hour_range(start_date, end_date):
        print(time)
        # file_in_time_loop = os.path.join(dir_data_weather,province+'SURF_CHN_MUL_HOR_'+time[:-2]+'.csv')
        file_in_time_loop = os.path.join(dir_data_weather,time[:8]+'/SURF_CHN_MUL_HOR_'+time+'.csv')
        data_today = pd.read_csv(file_in_time_loop)
        # data_today = data_today[['Station_Id_C','PRS','PRS_Sea','WIN_D_Avg_2mi','WIN_S_Avg_2mi','TEM','RHU','PRE_1h','Hour']]
        data_today = data_today[['Station_Id_C','PRS','PRS_Sea','WIN_D_Avg_2mi','WIN_S_Avg_2mi','TEM','RHU','PRE_1h']]
        
        
        #将原始数据中的缺测值进行替换
        data_today = data_today.replace(to_replace=[999017,999999,999998],value=np.nan)     
        
        #print(data_today)
        #进行单位换算
        data_today[['PRS']] = data_today[['PRS']] * 100
        data_today[['PRS_Sea']] = data_today[['PRS_Sea']] * 100
        data_today[['TEM']] = data_today[['TEM']] + 273.15
        
        #使用little_R格式中的缺测值替换na
        data_today = data_today.fillna(value=-888888.00000)   
        
        #对每个小时的中每一个站点进行循环
        for i in range(len(data_today)):
            station_id_in_loop = data_today.iloc[i,0]

            #for header
            lat,lon,elevation = find_lat_lon(station_id_in_loop,station_info)
            name  = 'WESTERMARKELSDORF / GERMANY'
            FM    = 'FM-12 SYNOP'  #SYNOP 表示Surface station “synoptic” report
            bogus = False #是否虚假文件
            # hour  = str_I2(data_today.iloc[i,8]) 
            hour  = str_I2(time[-2:]) 
            date  = time[:-2]+hour+'0000'
            slp = data_today.iloc[i,2]
            header_str = lr.header_record(lat, lon, station_id_in_loop, FM, elevation, bogus, date,slp,name)
            file.write(header_str+"\n")
            
            
            # for data
            pres  = data_today.iloc[i,1] # Pa
            h     = elevation # m
            t     = data_today.iloc[i,5]  # K
            td    = -888888.00000 # K
            wspd  = data_today.iloc[i,4] # m/s
            wdir  = data_today.iloc[i,3] # deg
            u     = -888888.00000 # m/s
            v     = -888888.00000 # m/s
            rh    = data_today.iloc[i,6] # %
            tk    = -888888.00000 # m
            
            #if time == '2022051218':
                #print(pres,t,wspd,wdir,rh)
            
            data_str   = lr.data_record(pres, h, t, td, wspd, wdir, u, v, rh, tk)
            file.write(data_str+"\n")
            ending_str = lr.ending_record()
            file.write(ending_str+"\n")

            
            # for tail
            num_fields = 1

            tail_str   = lr.tail_record(num_fields)
            file.write(tail_str+"\n")

            
file.close()
            






