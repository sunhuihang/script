import os,sys
from datetime import datetime,timedelta
import bisect
import pandas as pd 

def str2time(timestr, format = "%Y%m%d%H"):
    return datetime.strptime(timestr, format)

def time2str(time, format = "%Y%m%d%H"):
    return datetime.strftime(time, format)

def download_cfs():
    '''
    下载实时预报数据
    '''
    dt_init  = str2time(init)
    
    HH       = time2str(dt_init, "%H")
    YYYY     = time2str(dt_init, "%Y")
    YYYYMM   = time2str(dt_init, "%Y%m")
    YYYYMMDD = time2str(dt_init, "%Y%m%d")
    
    url_root = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/cfs/prod/cfs"
    url      = f"{url_root}/cfs.{YYYYMMDD}/{HH}/6hrly_grib_01"
    
    valid = time2str(dt_valid)
    # 等压面数据
    os.system(f"axel {url}/pgbf{valid}.01.{init}.grb2")
    # 地表数据
    os.system(f"axel {url}/flxf{valid}.01.{init}.grb2")
    
def download_cfs_his():
    '''
    下载历史数据
    '''
    dt_init  = str2time(init)
    YYYY     = dt_init.strftime("%Y")
    YYYYMM   = dt_init.strftime("%Y%m")
    YYYYMMDD = dt_init.strftime("%Y%m%d")
    
    url_root = "https://www.ncei.noaa.gov/data/climate-forecast-system/access/operational-9-month-forecast"
    url_pres = f"{url_root}/6-hourly-by-pressure/{YYYY}/{YYYYMM}/{YYYYMMDD}/{init}"
    url_sfc  = f"{url_root}/6-hourly-flux/{YYYY}/{YYYYMM}/{YYYYMMDD}/{init}"
    
    valid = time2str(dt_valid)
    # 等压面数据
    os.system(f"axel {url_pres}/pgbf{valid}.01.{init}.grb2")
    # 地表数据
    os.system(f"axel {url_sfc}/flxf{valid}.01.{init}.grb2")

if __name__ == "__main__":
    
    init  = "2022090100"  # 模式起报时间
    
    # 预报时段很长，仅下载其中的一部分
    start = "2022090100"  # 下载数据开始时间
    end   = "2022090200"  # 下载数据结束时间
    yyyy = init[:4]
    mm = init[4:6]
    dd = init[6:8]
    hh = init[8:10]
    # 下载目录设置
    wrk_dir = f"/home/qixiang/SHARE/cfs/{yyyy}.{mm}/{init}"
    if not os.path.exists(wrk_dir): os.makedirs(wrk_dir)
    os.chdir(wrk_dir)

    idx = pd.date_range(str2time(start), str2time(end), freq='6H')
    for dt_valid in idx:
        print(dt_valid)
        download_cfs_his()
