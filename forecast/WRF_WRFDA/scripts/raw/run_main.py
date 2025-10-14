import os,sys
import datetime
import env
if (len(sys.argv) != 2 ):
    print("输入参数有误")
    print(f"例python {sys.argv[0]} 2022080800")
    sys.exit(1)

Initime = sys.argv[1]
date_format = "%Y%m%d%H"
current_date = datetime.datetime.strptime(Initime, date_format)
current_date_string = current_date.strftime(date_format)
print("now running ",current_date_string)
WORK_DIR = f'../working/{current_date_string}'
###运行wps
#wps_work = f'{WORK_DIR}/wps'
#wps_work = os.path.abspath(wps_work)
#if not os.path.exists(wps_work):
#    os.makedirs(wps_work)
#os.system(f'python run_wps.py {current_date_string} {wps_work}')
#
#####运行real
#i = env.sf_surface_physics
#real_work = f'{WORK_DIR}/real'
#real_work = os.path.abspath(real_work)
#if not os.path.exists(real_work):
#    os.makedirs(real_work)
#os.system(f'python run_real.py {current_date_string} {real_work}')
#os.system(f'rm {wps_work}/met*nc')
#####运行wrfda
#da_work = f'{WORK_DIR}/da'
#da_work = os.path.abspath(da_work)
#if not os.path.exists(da_work):
#    os.makedirs(da_work)
#os.system(f'python run_da.py {current_date_string} {da_work}')
######运行wrf
wrf_work = f'{WORK_DIR}/wrf'
os.system(f'python run_wrf.py {current_date_string} {wrf_work}')


