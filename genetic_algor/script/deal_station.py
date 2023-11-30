import meteva.base as meb      # 该模块用于IO和基础计算
import meteva.method as mem    # 该模块基础了检验的基础算法
import meteva.product as mpd   # 该模块包含了检验的工具
import numpy as np
np.set_printoptions(suppress=True)
import datetime
import copy
import pandas as pd
import sys 
import xarray as xr
# coding=utf-8

input_dir = '/data/Atmos/sunhh/cycling/genetic_algor/data/station'
input_time = str(sys.argv[1])
#设置关注的起始时段
YYYY = int(input_time[0:4])
MM = int(input_time[4:6])
DD = int(input_time[6:8])
time_begin = datetime.datetime(YYYY,MM,DD,0,0)
input_time2 = (datetime.datetime.strptime(input_time, "%Y%m%d%H") + datetime.timedelta(days=1)).strftime("%Y%m%d%H")
YYYY = int(input_time2[0:4])
MM = int(input_time2[4:6])
DD = int(input_time2[6:8])
time_end = datetime.datetime(YYYY,MM,DD,0,0)
std_time = datetime.datetime.strftime(time_begin,"%Y%m%d%H")
std_time = std_time[0:8]

#处理站点数据
Initime = datetime.datetime.strftime(time_begin,"%Y%m%d%H")
Endtime = datetime.datetime.strftime(time_end,"%Y%m%d%H")

while Initime <= Endtime:
    data1 = input_dir+'/SURF_CHN_LIST.csv'
    info = pd.read_csv(data1, sep=',', index_col=False)
    df_info = pd.DataFrame(info)
    data2 = f'{input_dir}/{std_time}/origin/SURF_CHN_MUL_HOR_{Initime}.csv'
    obs = pd.read_csv(data2, sep=',', index_col=False)
    df_obs = pd.DataFrame(obs)
    df_total = df_obs.merge(df_info, how='inner', on='Station_Id_C')
    #df_shandong = df_total[(df_total['Province'] == '山东省')]
    df_shandong = df_total[(df_total.Lat>=34)&(df_total.Lat<=38.5)&(df_total.Lon>=114.5)&(df_total.Lon<=123)]
    df_shandong.to_csv(f'{input_dir}/{std_time}/SURF_CHN_MUL_HOR_{Initime}.csv', sep=',', index=None)
    Initime = (datetime.datetime.strptime(Initime, "%Y%m%d%H") + datetime.timedelta(hours=1)).strftime("%Y%m%d%H")
print(f'{input_dir}/{std_time}/SURF_CHN_MUL_HOR_{Initime}.csv')
