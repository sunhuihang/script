import os 

root = '/home/qixiang/sunhh/LIBRARIES'
WPS_DIR = os.path.join(root, 'WPS')
WRF_DIR = os.path.join(root, 'WRF')
WRFDA_DIR = os.path.join(root, 'WRFDA')
WRFOUT_DIR = '/home/qixiang/sunhh/realtime/WRFOUT'
geog_data_path='/home/qixiang/SHARE/dem/WPS_GEOG'
GFS_DIR = '/home/qixiang/SHARE/gfs'
gfsres=25

STATION_DIR = '/home/qixiang/sunhh/realtime/ob'
GDAS_DIR = '/home/qixiang/SHARE/gdas'
AHI_DIR = '/home/qixiang/SHARE/ahi'
BE_DIR = '/home/qixiang/sunhh/realtime/be'
VARBC_DIR = '/home/qixiang/sunhh/realtime/VARBC'
#Length of predictable time
d01_len=24
d02_len=24
d03_len=24

#Time resolution of boundary field
interval_seconds=10800

#For domain 
max_dom=1
e_we1=139
e_sn1=121
e_we2=199
e_sn2=175
e_we3=235
e_sn3=205
i_parent_start2=64
j_parent_start2=55
i_parent_start3=93
j_parent_start3=100
geog_res = 'default'
dx=12000
dy=12000
dx2=9000
dy2=9000
dx3=3000
dy3=3000
map_proj = 'lambert'
ref_lat=36.0
ref_lon=118.0
stand_lon=112.0
truelat1=30.0
truelat2=60.0


#For namelist.input
time_step=60
nproc_x=-1
nproc_y=-1
nio_tasks_per_group=0
nio_groups=1
e_vert=43
ptop=5000
num_metgrid_levels=34
num_metgrid_soil_levels=4
mp_physics=3
cu_physics1=5
cu_physics2=11
cu_physics3=11
ra_lw_physics=4
ra_sw_physics=4
sf_sfclay_physics=4
sf_surface_physics=1
bl_pbl_physics=4
num_land_cat=21

radt=12
hybrid_opt=2
etac=0.2
use_theta_m=1
#For WRFDA
bufr_type='gdas' #gdas or gfs
be_type = 5
ob_format=1
ob_format_gpsro=1
max_ext_its=2
ntmax=100
eps=0.0001
cv_options=5
rtm_option=1
#set the cores needed for each step
realcores=24
wrfcores=64
