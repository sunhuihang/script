import sys,os
import env
import datetime
import re
from env import ( WRFDA_DIR, STATION_DIR, GDAS_DIR,
                 AHI_DIR, BE_DIR, VARBC_DIR )

Initime=sys.argv[1]
da_work=sys.argv[2]
bufr_type=env.bufr_type
sf_surface_physics = env.sf_surface_physics
be_type = env.be_type
ccyy = Initime[0:4]
mm = Initime[4:6]
dd = Initime[6:8]
hh = Initime[8:10]
os.chdir(da_work)

Begin_time = datetime.datetime.strptime(Initime, "%Y%m%d%H")
Start_time = Begin_time.strftime("%Y%m%d%H")
Finish_time = (Begin_time + datetime.timedelta(hours=env.d01_len)).strftime("%Y%m%d%H")

ccyy_e = Finish_time[0:4]
mm_e = Finish_time[4:6]
dd_e = Finish_time[6:8]
hh_e = Finish_time[8:10]

analysis_date = datetime.datetime.strftime(Begin_time,"%Y-%m-%d_%H:%M:%S") 
time_window_min = datetime.datetime.strftime(Begin_time - datetime.timedelta(hours=3),"%Y-%m-%d_%H:%M:%S") 
time_window_max = datetime.datetime.strftime(Begin_time + datetime.timedelta(hours=3),"%Y-%m-%d_%H:%M:%S") 
#########  生成3dvar的namelist #########
namelist_da = """&wrfvar1
print_detail_grad=false
/
&wrfvar2
/
&wrfvar3
ob_format={ob_format}
/
&wrfvar4
thin_mesh_conv = 20
use_amsuaobs = T,
use_hirs4obs = F,
use_mhsobs = T,
use_iasiobs = F,
use_ahiobs = T,
use_mwhs2obs = F,
use_satwnd_bufr = T,
use_ssmisobs = F,
/
&wrfvar5
MAX_ERROR_T =    5.0     ,
MAX_ERROR_UV =    5.0     ,
MAX_ERROR_PW =    5.0     ,
MAX_ERROR_REF =    5.0     ,
MAX_ERROR_Q =    5.0     ,
MAX_ERROR_P =    5.0   ,
/
&wrfvar6
max_ext_its=1,
ntmax=100,
eps=0.0001,
orthonorm_gradient=.true.,
/
&wrfvar7
cv_options=5
cloud_cv_options=1
var_scaling1 = 1.0, !psi
var_scaling2 = 1.0, !chi
var_scaling3 = 1.0, !un t
var_scaling4 = 1.5, !rh
var_scaling5 = 1.0, !un psfc
len_scaling1 = 1.0,
len_scaling2 = 1.0,
len_scaling3 = 1.0,
len_scaling4 = 1.0,
len_scaling5 = 1.0,
/
&wrfvar8
/
&wrfvar9
/
&wrfvar10
test_transforms=false,
test_gradient=false,
/
&wrfvar11
check_rh=2
calculate_cg_cost_fn=.true.
write_detail_grad_fn=.true.
/
&wrfvar12
/
&wrfvar13
/
&wrfvar14
rtminit_nsensor=7,
rtminit_platform=1,1,1,10,10,10,31
rtminit_satid=15,18,19,1,3,1,8
rtminit_sensor=3,3,3,3,3,15,56
thinning_mesh=60,60,60,60,60,60,60
thinning=true,
qc_rad=true,
write_iv_rad_ascii=false,
write_oa_rad_ascii=true,
rtm_option=1,
only_sea_rad=false,
use_varbc=.true.
crtm_coef_path='./rttov_coeffs'
crtm_irland_coef='IGBP.IRland.EmisCoeff.bin'
use_blacklist_rad = true
/
&wrfvar15
/
&wrfvar16
/
&wrfvar17
analysis_type="QC-OBS"
/
&wrfvar18
analysis_date="{analysis_date}",
/
&wrfvar19
/
&wrfvar20
/
&wrfvar21
time_window_min="{time_window_min}",
/
&wrfvar22
time_window_max="{time_window_max}",
/
&wrfvar23
/
&time_control
start_year                          = {start_year}
start_month                         = {start_month}
start_day                           = {start_day}
start_hour                          = {start_hour}
start_minute                        = 00
start_second                        = 00
end_year                            = {end_year}
end_month                           = {end_month}
end_day                             = {end_day}
end_hour                            = {end_hour}
end_minute                          = 00
end_second                          = 00
/
&dfi_control
/
&domains
e_we                                = 139
e_sn                                = 121
e_vert                              = 43
dx                                  = 12000
dy                                  = 12000
/
&physics
mp_physics                          = 2,
cu_physics                          = 3,
ra_lw_physics                       = 4,
ra_sw_physics                       = 4,
radt                                = 12
sf_sfclay_physics                   = 2,
sf_surface_physics                  = 1,
bl_pbl_physics                      = 2,
bldt                                = 0
cudt                                = 0
kfeta_trigger                       = 1
icloud                              = 1,
num_land_cat                        = 21,
sf_urban_physics                    = 0,0,0,
mp_zero_out                         = 2,
mp_zero_out_thresh                  = 1.e-12
cu_rad_feedback                     = .false., 
/
&fdda
/
&dynamics
hybrid_opt                          = 2,
etac                                = 0.2
rk_ord                              = 3,
w_damping                           = 1,
diff_opt                            = 1,
km_opt                              = 4,
diff_6th_opt                        = 0,
diff_6th_factor                     = 0.12,
base_temp                           = 290.
damp_opt                            = 0,
zdamp                               = 5000.,
dampcoef                            = 0.01, 
khdif                               = 0,
kvdif                               = 0,
epssm                               = 0.4,
smdiv                               = 0.1,
emdiv                               = 0.1,
non_hydrostatic                     = .true.,
moist_adv_opt                       = 2,
scalar_adv_opt                      = 2,
tke_adv_opt                         = 2,
time_step_sound                     = 0,
h_sca_adv_order                     = 5,
use_theta_m                         = 1
/
&bdy_control
spec_bdy_width                      = 5,
specified                           = .true.,
nested                              = .false.,
spec_zone                           = 1,
relax_zone                          = 4,  
/
&grib2
/
&namelist_quilt
nio_tasks_per_group = 0,
nio_groups = 1,
/ 
""".format(ob_format=env.ob_format,analysis_date=analysis_date,
           time_window_min=time_window_min,time_window_max=time_window_max,
           start_year=ccyy,start_month=mm,start_day=dd,start_hour=hh,
           end_year=ccyy_e,end_month=mm_e,end_day=dd_e,end_hour=hh_e)

namelist_file = open("namelist.da", "w")
namelist_file.write(namelist_da)
namelist_file.close()

last_day = Begin_time - datetime.timedelta(days=1)
last_day = datetime.datetime.strftime(last_day,"%Y%m%d%H")
#链接VARBC偏差订正文件
if os.path.exists(f'{VARBC_DIR}/{hh}/VARBC.out'):
    os.system(f'cp {VARBC_DIR}/{hh}/VARBC.out VARBC.in')
else:
    os.system(f'ln -sf {WRFDA_DIR}/var/run/VARBC.in .')
#链接运行同化需要的文件
os.system('ln -sf ../real/wrfinput_d01 fg')
os.system('cp ../real/wrfbdy_d01 .')

os.system(f'ln -sf {BE_DIR}/be.dat_cv{be_type} be.dat')
os.system(f'ln -sf {WRFDA_DIR}/run/LANDUSE.TBL .')
os.system(f'ln -sf {WRFDA_DIR}/var/run/radiance_info .')
os.system(f'ln -sf {WRFDA_DIR}/var/run/rttov_coeffs .')
os.system(f'ln -sf {WRFDA_DIR}/var/run/crtm_coeffs .')
os.system(f'ln -sf {WRFDA_DIR}/var/build/da_wrfvar.exe .')

#指定格式的辐射数据，并链接为固定名称
os.system(f'ln -sf {GDAS_DIR}/{ccyy}.{mm}/{ccyy}{mm}{dd}{hh}/{bufr_type}.t{hh}z.1bamua.tm00.bufr_d amsua.bufr')
os.system(f'ln -sf {GDAS_DIR}/{ccyy}.{mm}/{ccyy}{mm}{dd}{hh}/{bufr_type}.t{hh}z.1bhrs4.tm00.bufr_d hirs4.bufr')
os.system(f'ln -sf {GDAS_DIR}/{ccyy}.{mm}/{ccyy}{mm}{dd}{hh}/{bufr_type}.t{hh}z.1bmhs.tm00.bufr_d mhs.bufr')
os.system(f'ln -sf {GDAS_DIR}/{ccyy}.{mm}/{ccyy}{mm}{dd}{hh}/{bufr_type}.t{hh}z.prepbufr.nr ob.bufr')
os.system(f'ln -sf {AHI_DIR}/{ccyy}.{mm}/{ccyy}{mm}{dd}{hh}/NC_H09_{ccyy}{mm}{dd}_{hh}00_R21_FLDK.02401_02401.nc  L1AHITBR')

#ahi.info 
ahi_info = """data source:1.cma hdf5;2.geocat netcdf4;3.jaxa netcdf4;4.ncep bufr
3 
nscan
3000
area information for geocat netcdf4 data: lonstart latstart nlongitude nlatitude
1,1,2401,2401
date infomation for cma hdf5 data
{ccyy},{mm},{dd},{hh},00,00
""".format(ccyy=ccyy,mm=mm,dd=dd,hh=hh)

ahi_file = open("ahi_info", "w")
ahi_file.write(ahi_info)
ahi_file.close()

os.system(f'ln -sf {WRFDA_DIR}/var/run/radiance/himawari-8-ahi.info_rttov {WRFDA_DIR}/var/run/radiance/himawari-8-ahi.info')
os.system('ln -sf namelist.da namelist.input')
os.system(f'mpirun -np {env.wrfcores} ./da_wrfvar.exe')
if not os.path.exists('wrfvar_output'):
    print(f'mpirun -np {env.wrfcores} ./da_wrfvar.exe failed')
    os.system(f'mpirun -np 32 ./da_wrfvar.exe')
    if not os.path.exists('wrfvar_output'):
        print(f'{da_work} crtm failed')
        if not os.path.exists('wrfvar_output'):
            with open('namelist.input', 'r') as f:
                lines = f.readlines()
            for i in range(len(lines)):
                if "rtm_option=1" in lines[i]:
                    lines[i] = "rtm_option=2\n"
                if "crtm_coef_path='./rttov_coeffs'" in lines[i]:
                    lines[i] = "crtm_coef_path='./crtm_coeffs'\n"
                    os.system(f'ln -sf {WRFDA_DIR}/var/run/radiance/himawari-8-ahi.info_crtm {WRFDA_DIR}/var/run/radiance/himawari-8-ahi.info')
            with open('namelist.input', 'w') as f:
                f.writelines(lines)
            os.system(f'mpirun -np {env.wrfcores} ./da_wrfvar.exe')
if not os.path.exists(f'wrfvar_output'):
    print('da failed')
    sys.exit(-1)

### 把VARBC.out 中的 **********改为0.0
with open('VARBC.out', 'r') as f:
    text = f.read()
fixed_text = re.sub(r'\*{10}','       0.0', text)
with open('VARBC.out', 'w') as f:
    f.write(fixed_text)
os.system(f'cp VARBC.out {VARBC_DIR}/{hh}')
