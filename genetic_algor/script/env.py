import os 

root = '/data/Atmos/sunhh/'
top = os.path.join(root,'cycling','genetic_algor')
script_dir = os.path.join(top,'scripts')
input_dir = os.path.join(top, 'data', 'input')
output_dir = os.path.join(top, 'data','output','fctprocess')
work_dir = os.path.join(top, 'workdir')

#for source 
wps_dir = os.path.join(root, 'WPS')
wrf_dir = os.path.join(root, 'WRF')
wrfda_dir = os.path.join(root, 'WRFDA')

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
geog_data_path=os.path.join(root, 'WPS_GEOG')
gfsdir='/satadata/qixiang/Atmos/SHARE/atmosData/gfs'
gfsres=25

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
mp_physics=8
cu_physics1=1
cu_physics2=1
cu_physics3=0
ra_lw_physics=4
ra_sw_physics=4
sf_sfclay_physics=1
sf_surface_physics=1
bl_pbl_physics=1
num_land_cat=21

#For WRFDA

#set the cores needed for each step
realcores=24
wrfcores=64
