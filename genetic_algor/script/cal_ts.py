import meteva.base as meb
import meteva.method as mem
import numpy as np
import sys
import datetime
import pandas as pd
import os
np.set_printoptions(suppress=True)
import netCDF4 as nc
from wrf import getvar
from scipy.interpolate import griddata

# input_dir = '/root/Atmos/kezx/wrf_forecast/satadata/qixiang/input'
# wrfout_dir = '/satadata/qixiang/Atmos/kezx'
# plot_dir = '/root/Atmos/kezx/wrf_forecast/scripts/postprocess/output'
input_dir = '/data/Atmos/sunhh/cycling/genetic_algor/data/station'
wrfout_dir = '/data/Atmos/sunhh/cycling/genetic_algor'
plot_dir = '/data/Atmos/sunhh/cycling/genetic_algor/figure'
script_dir = '/data/Atmos/sunhh/cycling/genetic_algor/script'
input_time = str(sys.argv[1])
mp_para = str(sys.argv[2])
cu_para = str(sys.argv[3])
pbl_para = str(sys.argv[4])
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

Initime = datetime.datetime.strftime(time_begin,"%Y%m%d%H")

####以下开始为数据收集部分的程序
#读取站点列表，并将站点内容为缺省值，当其作为读取站点数据的参数时，如果站点文件中某个站号不存在时,返回结果中该站点保持为缺省值
#station = meb.read_stadata_from_csv(f"{input_dir}/OBS/station/{std_time}/SURF_CHN_MUL_HOR_{std_time}00.csv", columns = ["id","time","PRS","PRS_Sea","PRS_Max","PRS_Min","WIN_S_Max","WIN_S_Inst_Max","WIN_D_INST_Max","WIN_D_Avg_2mi","WIN_S_Avg_2mi","WIN_D_S_Max","TEM","TEM_Max","TEM_Min","RHU","RHU_Min","OBS","windpower","tigan","Station_Name","Country","Province","City","Cnty","Station_Id_d","lat","lon","Alti"], member_list = ["OBS"], sep=',', skiprows=1)
station = meb.read_stadata_from_csv(f"{input_dir}/{std_time}/SURF_CHN_MUL_HOR_{std_time}00.csv", columns = ["id","time","PRS","PRS_Sea","PRS_Max","PRS_Min","WIN_S_Max","WIN_S_Inst_Max","WIN_D_INST_Max","WIN_D_Avg_2mi","WIN_S_Avg_2mi","WIN_D_S_Max","TEM","TEM_Max","TEM_Min","RHU","RHU_Min","OBS","windpower","tigan","Station_Name","Country","Province","City","Cnty","Station_Id_d","lat","lon","Alti"], member_list = ["OBS"], sep=',', skiprows=1)
station.iloc[:,-1] = meb.IV

##读取收集观测数据
sta_list = []

time0 = time_begin + datetime.timedelta(hours = 0)
time1 = datetime.datetime.strftime(time0,"%Y%m%d%H")
#path_ob = f"{input_dir}/OBS/station/{std_time}/SURF_CHN_MUL_HOR_{time1}.csv"
path_ob = f"{input_dir}/{std_time}/SURF_CHN_MUL_HOR_{time1}.csv"
sta = meb.read_stadata_from_csv(path_ob, columns=["id","time","PRS","PRS_Sea","PRS_Max","PRS_Min","WIN_S_Max","WIN_S_Inst_Max","WIN_D_INST_Max","WIN_D_Avg_2mi","WIN_S_Avg_2mi","WIN_D_S_Max","TEM","TEM_Max","TEM_Min","RHU","RHU_Min","OBS","windpower","tigan","Station_Name","Country","Province","City","Cnty","Station_Id_d","lat","lon","Alti"], member_list=["OBS"], sep=',', skiprows=1)
sta_list.append(sta)
ob_sta = meb.concat(sta_list).reset_index(drop=True)

time0 = time_begin + datetime.timedelta(hours = 1)
time_end_ob = time_end + datetime.timedelta(hours = 0)

while time0 < time_end_ob:
    sta_list = []
    time1 = datetime.datetime.strftime(time0,"%Y%m%d%H")
    path_ob = f"{input_dir}/{std_time}/SURF_CHN_MUL_HOR_{time1}.csv"
    sta = meb.read_stadata_from_csv(path_ob, columns=["id","time","PRS","PRS_Sea","PRS_Max","PRS_Min","WIN_S_Max","WIN_S_Inst_Max","WIN_D_INST_Max","WIN_D_Avg_2mi","WIN_S_Avg_2mi","WIN_D_S_Max","TEM","TEM_Max","TEM_Min","RHU","RHU_Min","OBS","windpower","tigan","Station_Name","Country","Province","City","Cnty","Station_Id_d","lat","lon","Alti"], member_list=["OBS"], sep=',', skiprows=1)
    sta_list.append(sta)  # 将数据加入到列表当中
    ob_sta_oth = meb.concat(sta_list).reset_index(drop=True)
    ob_sta['OBS'] = ob_sta['OBS'] + ob_sta_oth['OBS']
    time0 += datetime.timedelta(hours = 1)

ob_sta_all = ob_sta

##读取预报数据
domain = "d01"
time1 = time_begin + datetime.timedelta(hours = 24)
time2 = datetime.datetime.strftime(time1, "%Y%m%d%H")
YY4 = time2[0:4]
MM4 = time2[4:6]
DD4 = time2[6:8]
HH4 = time2[8:10]
path_wrfout = f"{wrfout_dir}/WRFOUT/{Initime}/{mp_para}_{cu_para}_{pbl_para}/wrfout_{domain}_{YY4}-{MM4}-{DD4}_{HH4}:00:00"
wrfnc   = nc.Dataset(path_wrfout)
#rainc   = getvar(wrfnc, "RAINC", timeidx=-1)
#rainnc  = getvar(wrfnc, "RAINNC", timeidx=-1)
rainc   = getvar(wrfnc, "RAINC")
rainnc  = getvar(wrfnc, "RAINNC")
rain    = rainc + rainnc 

lat     = getvar(wrfnc, "lat")
lon     = getvar(wrfnc, "lon")

grid_x = station['lon'].tolist()
grid_y = station['lat'].tolist()
points = ( np.array(lon).flatten(),  np.array(lat).flatten() )
fo = griddata(points, np.array(rain).flatten(), (grid_x, grid_y), method='nearest')

###################以上为数据收集部分的程序
grade_list = list(range(10,260,10))
ob = np.array(ob_sta_all['OBS'])
ts_score = mem.ts(ob,fo,grade_list)
ts_score[ts_score==999999.] = 0.
ts_sum = round(np.sum(ts_score),4)

ts_file = os.path.join(script_dir,"ts_score.txt")
f= open(ts_file, "a")
f.write(f"mp={mp_para},cu={cu_para},pbl={pbl_para},ts_sum={ts_sum}\n")
f.close()

# # 根据ts_sum进行排序
# with open(ts_file, 'r') as f:
#     lines = f.readlines()
# lines_with_ts_sum = [(float(line.split('ts_sum=')[1]), line) for line in lines]
# lines_with_ts_sum.sort(key=lambda x: x[0], reverse=True)
# with open(ts_file, 'w') as f:
#     for line in lines_with_ts_sum:
#         f.write(line[1])

print(ts_sum)
