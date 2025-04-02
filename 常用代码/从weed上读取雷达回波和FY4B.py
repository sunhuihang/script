#filer.py
import io
import os
from urllib import parse
import requests
from loguru import logger
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

class WeedFiler(object):
    """ weed filer service.
    """

    def __init__(self, url_base='http://localhost:27100'):
        """ construct WeedFiler

        Arguments:
        - `host`: defaults to '127.0.0.1'
        - `port`: defaults to 27100
        :param url_base:
        """
        self.url_base = url_base

    def _create_retry_session(self, max_retries=5, backoff_factor=0.3, status_forcelist=(500, 502, 504,404)):
        """ Create a retry-enabled session for requests """
        session = requests.Session()
        retry = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=status_forcelist
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session

    def get(self, remote_path) -> None or {}:
        """ put a file @fp to @remote_path on seaweedfs

        returns @remote_path if succeeds else None
        Arguments:
        - `self`:
        - `remote_path`:
        - `echo`: if True, print response
        """
        url = parse.urljoin(self.url_base, remote_path)
        try:
            session = self._create_retry_session()
            rsp = session.get(url)
            if rsp.ok:
                result = {'content_length': rsp.headers.get('content-length'),
                          'content_type': rsp.headers.get('content-type'),
                          'content': rsp.content}
                return result
            else:
                logger.error('%d GET %s' % (rsp.status_code, url))
                return None
        except Exception as e:
            logger.error('Error GETting %s. e:%s' % (url, e))
            return None

    def put(self, fp, remote_path) -> None or str:
        """ put a file @fp to @remote_path on seaweedfs

        returns @remote_path if succeeds else None
        :arg
        - `fp`: either a file-handler by method open(with binary mode) or a str to the file-path.
        - `remote_path`:
        :returns
        None or str-of-remote-path

        """
        url = parse.urljoin(self.url_base, remote_path)
        is_our_responsibility_to_close_file = False
        if isinstance(fp, str):
            _fp = open(fp, 'rb')
            is_our_responsibility_to_close_file = True
        else:
            _fp = fp
        result = None
        try:
            session = self._create_retry_session()
            rsp = session.post(url, files={'file': _fp})
            if rsp.ok:
                result = remote_path
            else:
                logger.error('%d POST %s' % (rsp.status_code, url))
        except Exception as e:
            logger.error('Error POSTing %s. e:%s' % (url, e))

        # close fp if parameter fp is a str
        if is_our_responsibility_to_close_file:
            try:
                _fp.close()
            except Exception as e:
                logger.warning('Could not close fp: %s. e: %s' % (_fp, e))

        return result

    def delete(self, remote_path) -> bool:
        """ remove a @remote_path by http DELETE """
        url = parse.urljoin(self.url_base, remote_path)
        try:
            session = self._create_retry_session()
            rsp = session.delete(url)
            if not rsp.ok:
                logger.error('Error deleting file: %s. ' % remote_path)
            return rsp.ok
        except Exception as e:
            logger.error('Error deleting file: %s. e: %s' % (remote_path, e))
            return False

    def list(self, directory) -> None or {}:
        """ list sub folders and files of @dir. show a better look if you turn on @pretty

        returns a dict of "sub-folders and files'
        """
        d = directory if directory.endswith('/') else (directory + '/')
        url = parse.urljoin(self.url_base, d+'?limit=1000')
        headers = {'Accept': 'application/json'}
        try:
            session = self._create_retry_session()
            rsp = session.get(url, headers=headers)
            if not rsp.ok:
                logger.error('Error listing "%s". [HTTP %d]' % (url, rsp.status_code))
                return None
            return rsp.json()
        except Exception as e:
            logger.error('Error listing "%s". e: %s' % (url, e))
        return None


    def mkdir(self, directory) -> None or str:
        """ make dir on filer.

        eg:
           mkdir('/image/avatar').
           mkdir('/image/avatar/helloworld')

        We will post a file named '.info' to @_dir.
        """
        self.put(io.StringIO('.info'), os.path.join(directory, '.info'))
        # try:
        #     self.delete(os.path.join(directory, '.info'))
        # except:
        #     pass
            
        return True




##############################################################
### 从weed 上获取雷达 和 卫星数据
from filer import WeedFiler
import numpy as np
import pandas as pd
import datetime as dt
from loguru import logger
from PIL import Image
import xarray as xr
import io
from glob import glob

def get_dbz(t,weed_url='http://192.168.0.162:8888/'):
    '''
    输入例如 '20240506 0030' 或datetime格式时间
    '''
    try:
        t = pd.to_datetime(t)
    except:
        print('无法解析时间，尝试当前时刻进行预报')
        t = dt.datetime.now()
    # 通过wf 获取起报时间前文件列表
    WF = WeedFiler(url_base=weed_url)
    fl = WF.list(f'/radar/china_radar_map/{t.year}/{t.month}/{t.day}')
    fl = [f['FullPath'] for f in fl['Entries'] if f['FullPath'].endswith('png')]
    t_1 = t - pd.Timedelta(hours=1)
    if t_1.day != t.day:
        file_list = WF.list(f'/radar/china_radar_map/{t_1.year}/{t_1.month}/{t_1.day}')
        fl.extend([f['FullPath'] for f in file_list['Entries'] if f['FullPath'].endswith('png')])
    tl = pd.to_datetime(pd.Series(['_'.join(f.split('/')[-4:])[:-4] for f in fl]), format='%Y_%m_%d_%H%M')
    radar_data = pd.DataFrame({'file': fl, 'time': tl})
    radar_data = radar_data.sort_values('time')
    # 更新起报时间
    t = radar_data['time'].values[np.argmin(abs(radar_data['time'] - t))]
    t = pd.to_datetime(t)
    logger.info(f'起报时刻为{t}')
    # radar_data = get_files_list(work_dir, t)
    time_list = pd.date_range(t - pd.Timedelta(hours=1), t, freq='6min')[-10:]
    dbz_list = radar_data.file[radar_data.time.isin(time_list)].values
    return dbz_list

def get_dbz_values(t,weed_url='http://192.168.0.162:8888/'):
    WF = WeedFiler(url_base=weed_url)
    dbz_list = get_dbz(t,weed_url='http://192.168.0.162:8888/')
    if len(dbz_list) == 10:
        logger.info(f"数据匹配完成,数据列表为{dbz_list}")
        dbz = np.stack([Image.open(io.BytesIO(WF.get(f)['content'])) for f in dbz_list], axis=0).astype(np.float32) / 2
        lon_start,lon_step = 70.0,0.02
        lat_start,lat_step = 54.0,-0.02
        _,h,w = dbz.shape
        lon = lon_start + np.arange(w) * lon_step
        lat = lat_start + np.arange(h) * lat_step
        return dbz,np.round(lon,2),np.round(lat,2)
    else:
        logger.info(f"数据数量错误,数据数量为{len(dbz_list)}")
        return

import numpy as np
import xarray as xr
from loguru import logger

def get_sat_values(t,dbz_lon,dbz_lat,weed_url='http://192.168.0.162:8888/'):
    try:
        t = pd.to_datetime(t)
    except:
        print('无法解析时间')

    # 通过wf 获取起报时间前文件列表
    WF = WeedFiler(url_base=weed_url)
    fl = WF.list(f'/radar/FY4B_L1/{t.year}/{t.year}{str(t.month).zfill(2)}{str(t.day).zfill(2)}')
    fl = [f['FullPath'] for f in fl['Entries'] if f['FullPath'].endswith('nc')]
    t_1 = t - pd.Timedelta(hours=1)
    if t_1.day != t.day:
        file_list = WF.list(f'/radar/FY4B_L1/{t_1.year}/{t_1.year}{str(t_1.month).zfill(2)}{str(t_1.day).zfill(2)}')
        fl.extend([f['FullPath'] for f in file_list['Entries'] if f['FullPath'].endswith('nc')])
    tl = pd.to_datetime(pd.Series([os.path.basename(f).split('_')[-4] for f in fl]), format='%Y%m%d%H%M%S') + dt.timedelta(minutes=15)
    sat_data = pd.DataFrame({'file': fl, 'time': tl})
    sat_data = sat_data.sort_values('time')
    # 更新起报时间
    t = sat_data['time'].values[np.argmin(abs(sat_data['time'] - t))]
    t = pd.to_datetime(t)
    logger.info(f'起报时刻为{t}')
    # sat_data = get_files_list(work_dir, t)
    time_list = pd.date_range(t - pd.Timedelta(hours=1), t, freq='15min')[-4:]
    sat_list = sat_data.file[sat_data.time.isin(time_list)].values

    if len(sat_list) == 4:
        logger.info(f"数据匹配完成,数据列表为{sat_list}")
        sat_list = [io.BytesIO(WF.get(f)['content']) for f in sat_list]
        sat_all = []
        for f in sat_list:
            f_ds = xr.open_dataset(f)
            data_list = []
            for v in range(1,16):
                f_ds['lat'] = f_ds.lat.data.round(2)
                f_ds['lon'] = f_ds.lon.data.round(2)
                data_tmp = f_ds.sel(lat=dbz_lat[::2], lon=dbz_lon[::2])[f'Channel{str(v).zfill(2)}']
                data_list.append(data_tmp.data/10.)
            sat_data = np.stack(data_list,axis=0)
            sat_all.append(sat_data)
        sat = np.stack(sat_all,axis=0) 
        return sat
    else:
        logger.info(f"数据数量错误,数据数量为{len(sat_list)}")
        return

def get_sat_mean_std(use_channel_list=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14],file_statistics = '/data/qixiang/sunhh/FY_project/data/statistics_channel.npy'):
    data_normal = np.load(file_statistics, allow_pickle=True).item()
    feat_mean, feat_std = [], []
    for _, num_channel in enumerate(use_channel_list):
        if num_channel < 6:
            feat_mean.append(0)
            feat_std.append(100)
        else:
            feat_mean.append(data_normal['channel_{}_mean'.format(str(num_channel))])
            feat_std.append(data_normal['channel_{}_std'.format(str(num_channel))])
    feat_mean, feat_std = np.array(feat_mean).reshape((1,len(use_channel_list),1,1)), np.array(feat_std).reshape((1,len(use_channel_list),1,1))
    return feat_mean,feat_std

def get_dem(dbz_lon,dbz_lat):
    dem = xr.open_dataset('/data/qixiang/sunhh/FY_project/data/China_dem_2km.nc')
    dem = dem.rename({'Band1':'dem'})
    lsm = dem['dem']
    dem['dem']= dem['dem'].fillna(0)
    vmax,vmin = dem.dem.max(),dem.dem.min()
    dem['dem'] = (dem['dem'] - vmin) / (vmax - vmin)  


    dem = dem['dem'].sel(lat= dbz_lat, lon= dbz_lon,method='nearest').data
    # lon,lat = np.meshgrid(dbz_lon,dbz_lat)
    # geo = np.stack([lon,lat],axis=0)
    # geo = np.cos(np.radians(geo))
    # geo = np.concatenate([geo,dem[None]])
    return dem
