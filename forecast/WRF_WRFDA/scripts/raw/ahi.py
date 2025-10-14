from datetime import datetime,timedelta
import sys,os

start_date = datetime(2021, 1, 5, 0, 0)
end_date = datetime(2021, 12, 31, 23, 59)

time_interval = timedelta(minutes=10)

current_date = start_date
while current_date <= end_date:
    print('开始下载',current_date)
    os.system(f'./download_ahi.sh {current_date.strftime("%Y%m%d%H%M")}')
    current_date += time_interval
