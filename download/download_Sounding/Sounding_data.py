from urllib import request
import pandas as pd
from fake_useragent import UserAgent
import datetime as dt
ua = UserAgent()
import re
from metpy.units import units
from siphon.simplewebservice.wyoming import WyomingUpperAir
import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor

def fnt_sounding_data_spider(url):
    req = request.Request(url=url, headers={'User-Agent': ua.chrome})
    res = request.urlopen(req)
    html = res.read().decode("utf-8", "ignore")

    # 获取探空数据
    pattern = re.compile(r'</I>\n<PRE>\n-*\n(.*)\n</PRE>\n\n', re.S)
    r = pattern.findall(html)
    if len(r) ==0:
        return 0,0

    # 处理探空数据
    pattern = re.compile(r'(.*)\n-*\n(.*)', re.S)
    r = pattern.findall(r[0])

    table = r[0][0]
    sounding_data  = r[0][1]

    # 数据逐行写入csv
    pattern=re.compile('(.*)\n')
    table_list = pattern.findall(table+'\n' )
    return table_list ,sounding_data

def update_csv(id):
    url = f'http://weather.uwyo.edu/cgi-bin/bufrraob.py?src=bufr&datetime={pd_time[0]}&id={id}&type=TEXT:LIST'
    req = request.Request(url=url, headers={'User-Agent': ua.chrome})
    res = request.urlopen(req)
    html = res.read().decode("utf-8", "ignore")
    pattern = re.compile(r'<I>Latitude: (.*) Longitude: (.*)</I>')
    dd = pattern.findall(html)
    if len(dd)==0:
        url = f'http://weather.uwyo.edu/cgi-bin/bufrraob.py?src=bufr&datetime={pd_time[1]}&id={id}&type=TEXT:LIST'
        req = request.Request(url=url, headers={'User-Agent': ua.chrome})
        res = request.urlopen(req)
        html = res.read().decode("utf-8", "ignore")
        pattern = re.compile(r'<I>Latitude: (.*) Longitude: (.*)</I>')
        dd = pattern.findall(html)
        if len(dd)==0:
            print(f'{id}无站点数据')
            return 0
    lat = float(dd[0][0])
    lon = float(dd[0][1])
    station = pd.DataFrame([[id,lat,lon]],columns=['id','lat','lon'])
    station[['id']] = station[['id']].astype(str)
    station_csv = pd.read_csv('../sounding_data/station.csv',dtype={"id":str,"lat": float,"lon":float})
    station_csv = pd.concat((station_csv,station))
    station_csv = station_csv.drop_duplicates()
    station_csv.to_csv('../sounding_data/station.csv',index=False)
    print(f'已更新{id}站点信息')

# 新建文件夹函数，便于分站点存储数据
def mkdir(path):
    folder = os.path.exists(path)

    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
    else:
        pass
def fnt_child_spider(t,id):
    dd = re.sub('%20', ' ', t)
    dd = dt.datetime.strptime(dd,'%Y-%m-%d %H:%M:%S')
    str_time = dt.datetime.strftime(dd,'%Y%m%d_%H')
    file_path = f'../sounding_data/{id}/{str_time}.txt'
    if os.path.exists(file_path):
        print(id+t+'数据已存在')
        return 0
    url = f'http://weather.uwyo.edu/cgi-bin/bufrraob.py?src=bufr&datetime={t}&id={id}&type=TEXT:LIST'
    # 找到当前探空数据的时间
    # pattern = re.compile(r'.*datetime=(.*?)&')
    # dd = pattern.findall(url)
    # dd = re.sub('%20', ' ', dd[0])
    # dd = dt.datetime.strptime(dd,'%Y-%m-%d %H:%M:%S')
    table_list ,sounding_data = fnt_sounding_data_spider(url)
    if table_list==0:
        print(id+t.strftime('%Y%m%d_%H')+'缺失数据')
        return 0

    with open(file_path , "w", encoding='utf-8') as f:
        f.write(table_list[0]+'\n')
        f.write(sounding_data)
        f.close()
    print(id+t+'下载成功')
    

if __name__ == '__main__':
    # 爬取时间段
    pd_time = pd.date_range(start='2022-01-01 0:00', end='2022-10-13 0:00', freq='12H') # 下载时间
    pd_time_list = [dt.datetime.strftime(t, '%Y-%m-%d %H:%M:%S') for t in pd_time]
    pd_time = [re.sub(r' ', "%20", r_) for r_ in pd_time_list]
    # 雷达站点文件信息
    id_list = pd.read_csv('../sounding_data/station.csv',dtype={"id": str})
    pool = ThreadPoolExecutor(max_workers=5)
    for id in id_list['id']:
        mkdir('../sounding_data/'+id)
        for t in pd_time:
            pool.submit(fnt_child_spider,t,id)
    pool.shutdown(wait=True)


