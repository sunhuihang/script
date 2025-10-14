'''
此脚本下载当日00时的这一天的国家站点数据，与该目录下的其他处理脚本并不是匹配关系，自行更改匹配
'''
import arrow
from tqdm import tqdm
import pandas as pd
import os
import time
import requests
from datetime import datetime, timedelta, timezone
from collections import Counter
import numpy as np

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
STATION_INFO_FP = os.path.join(STATIC_DIR, "station_info_new.csv")
TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp")
SHARE_DIR = "/mnt/glusterfs/qixiang/SHARE/atmos/obs/download"
OBS_DATA_URL_PATTERN = "http://www.nmc.cn/rest/weather?stationid={code}"


def get_station_info():
    df = pd.read_csv(STATION_INFO_FP)
    return df

def parse_obs_data(data, sid, ts):

    try:
        for passedchart in data["passedchart"]:
            timestr = passedchart["time"]
            dt = (
                datetime.fromisoformat(timestr)
                .replace(tzinfo=timezone(timedelta(hours=8)))
                .astimezone(timezone.utc)
            )

            if int(dt.timestamp()) != ts:
                continue
            else:
                wind_speed = passedchart["windSpeed"]
                wind_direction = passedchart["windDirection"]
                temperature = passedchart["temperature"]
                humidity = passedchart["humidity"]
                break
        else:
            return False
    except (KeyError, TypeError):
        return False
    else:
        if wind_speed > 9000 or temperature > 9000:
            return False
        return {
            "sid": sid,
            "datetime": dt,
            "wind_speed": wind_speed,
            "wind_direction": wind_direction,
            "temperature": temperature,
            "humidity": humidity,
        }


def prepare_observation():
    station_df = get_station_info()

    url_error_list = []
    data_error_list = []
    codes = station_df["code"].tolist()
    sids = station_df["sid"].tolist()
    lons = station_df["lon"].tolist()
    lats = station_df['lat'].tolist()
    records = []
    # dts = []
    print("Downloading observation data...")
    now_dt = arrow.now(tz="utc").floor("hour")
    round3dt = now_dt.replace(hour=now_dt.hour // 3 * 3)
    want_dt = round3dt.shift(hours=-3)
    # want_ts = int(want_dt.timestamp())

    now_time = datetime.now()
    Initime = datetime.strftime(now_time,"%Y%m%d%H")[:8] + "00"

    for cycle_num, code in enumerate(tqdm(codes)):
        sid = sids[cycle_num]
        lon = np.round(lons[cycle_num], 3)
        lat = np.round(lats[cycle_num], 3)
        URL = OBS_DATA_URL_PATTERN.format(code=code) + f"&_={int(time.time()*1000)}"
        try:
            resp = requests.get(URL, timeout=5)
        except Exception:
            url_error_list.append(code)
            continue
        if resp.ok:
            data = resp.json()["data"]
            if data:
                for passedchart in data["passedchart"]:
                    wind_speed = passedchart["windSpeed"]
                    wind_direction = passedchart["windDirection"]
                    temperature = passedchart["temperature"]
                    humidity = passedchart["humidity"]
                    pressure = passedchart["pressure"]
                    rain1h = passedchart["rain1h"]
                    timestr = passedchart["time"]
                    dt = (
                        datetime.fromisoformat(timestr)
                        .replace(tzinfo=timezone(timedelta(hours=8)))
                        .astimezone(timezone.utc)
                    )
                    parsed_data = {
                                "sid": sid,
                                "lon": lon,
                                "lat": lat,
                                "datetime": dt,
                                "wind_speed": wind_speed,
                                "wind_direction": wind_direction,
                                "temperature": temperature,
                                "humidity": humidity,
                                "pressure": pressure,
                                "rain1h": rain1h,
                            }
                    if parsed_data:
                        records.append(parsed_data)
                    else:
                        data_error_list.append(code)
            else:
                continue

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(TMP_DIR, f"{Initime}.csv"), index=False)
    #df.to_csv(os.path.join(SHARE_DIR, f"{Initime}.csv"), index=False)
    print(
        "Observation data download is completed, "
        f"a total of {len(df)} observation stations' data downloaded, "
    )


def prepare_all():
    os.makedirs(TMP_DIR, exist_ok=True)
    prepare_observation()


if __name__ == "__main__":
    prepare_all()
