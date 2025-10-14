import os,glob,sys
import datetime
import random
import env

if (len(sys.argv) != 3 ):
    print("输入参数有误")
    print(f"例python {sys.argv[0]} 2022080800 $work_path")
    sys.exit(1)

Initime = sys.argv[1]
work_dir = sys.argv[2]
WRF_DIR = env.WRF_DIR
WRFOUT_DIR =env.WRFOUT_DIR
#wrfout提取的变量
varlist="XLONG,XLAT,XLONG_U,XLAT_U,XLONG_V,XLAT_V,XTIME,Times,T,T2,P,PB,PH,PHB,PSFC,Q2,QVAPOR,U10,U,V10,V,HGT,RAINC,RAINNC,RAINSH,PBLH"
fc_hour = env.d01_len
#设置关注的起始时段
ccyy_s = Initime[0:4]
mm_s = Initime[4:6]
dd_s = Initime[6:8]
hh_s = Initime[8:10]
endtime = (datetime.datetime.strptime(Initime, "%Y%m%d%H") + datetime.timedelta(hours=fc_hour)).strftime("%Y%m%d%H")
ccyy_e = endtime[0:4]
mm_e = endtime[4:6]
dd_e = endtime[6:8]
hh_e = endtime[8:10]

TOP = os.path.abspath(".")

mp = env.mp_physics
cu = env.cu_physics1
lw = env.ra_lw_physics
sw = env.ra_sw_physics
sf_surf = env.sf_surface_physics
pbl = env.bl_pbl_physics

if pbl == 2:
    sf_sfclay = 2
elif pbl == 4:
    sf_sfclay = 4
elif pbl == 6:
    sf_sfclay = 5
else:
    sf_sfclay = 1
mp_radii = [3, 4, 6, 7, 8, 10, 14, 16, 17, 24, 26, 28, 50, 51, 52, 53, 55]
if mp in mp_radii:
    use_mp_re=1
else:
    use_mp_re=0
wrf_work = work_dir 
wrf_work = os.path.abspath(wrf_work)
if not os.path.exists(wrf_work):os.makedirs(wrf_work)
os.chdir(wrf_work)

namelist_string="""&time_control
 start_year                          = {ccyy_s},{ccyy_s}
 start_month                         = {mm_s},{mm_s}
 start_day                           = {dd_s},{dd_s}
 start_hour                          = {hh_s},{hh_s}
 start_minute                        = 00,00,
 start_second                        = 00,00,
 end_year                            = {ccyy_e},{ccyy_e}
 end_month                           = {mm_e},{mm_e}
 end_day                             = {dd_e},{dd_e}
 end_hour                            = {hh_e},{hh_e}
 end_minute                          = 00,00,00,
 end_second                          = 00,00,00,
 interval_seconds                    = 10800,
 input_from_file                     = .true.,.true., .true.,
 history_interval                    = 60,360,360 
 frames_per_outfile                  = 1000,1000,1000,
 restart                             = .false.,
 !restart_interval                    = 1440
 !iofields_filename                   = "my_file_d01.txt", "my_file_d01.txt"
 io_form_history                     = 2,
 io_form_input                       = 2,
 io_form_boundary                    = 2,
 debug_level                         = 0,
 auxinput1_inname                    = "met_em.d<domain>.<date>",
 adjust_output_times                 = .true.
/

 &dfi_control
/
 &domains
 time_step                           = 60,
 time_step_fract_num                 = 0,
 time_step_fract_den                 = 1,
 max_dom                             = 1,
 e_we                                = 139,142,235,
 e_sn                                = 121,116,205,
 e_vert                              = 43,43,43,
 p_top_requested                     = 5000,
 num_metgrid_levels                  = 34,
 num_metgrid_soil_levels             = 4,
 dx                                  = 12000,12000,3000,
 dy                                  = 12000,12000,3000,
 grid_id                             = 1,2,3,
 parent_id                           = 0,1,2,
 i_parent_start                      = 1,64,93,
 j_parent_start                      = 1,55,100,
 parent_grid_ratio                   = 1,3,3,
 parent_time_step_ratio              = 1,3,3,
 nproc_x                             = -1
 nproc_y                             = -1
 feedback                            = 0,
 smooth_option                       = 0
 interp_type                         = 0
 force_sfc_in_vinterp                = 1
 lagrange_order                      = 2
 smooth_cg_topo                      = .true.
 sfcp_to_sfcp                        = .false.
 use_adaptive_time_step              = .false.
 step_to_output_time                 = .true.
 target_cfl                          = 1.2,1.2,1.2,
 target_hcfl                         = .84,.84,.84 
 max_step_increase_pct               = 5,51,51,
 max_time_step                       = 27,9,3
 starting_time_step                  = 9,3,1
 min_time_step                       = 9,3,1
 adaptation_domain                   = 1
 /

 &physics
 mp_physics                          = {mp},{mp},{mp},
 cu_physics                          = {cu},{cu},{cu},
 ra_lw_physics                       = {lw},{lw},{lw},
 ra_sw_physics                       = {sw},{sw},{sw},
 radt                                = 12
 sf_sfclay_physics                   = {sf_sfclay},{sf_sfclay},{sf_sfclay},
 !iz0tlnd                             = 0
 sf_surface_physics                  = {sf_surf},{sf_surf},{sf_surf},
 bl_pbl_physics                      = {pbl},{pbl},{pbl},
 bldt                                = 0
 cudt                                = 0
 kfeta_trigger                       = 1
 icloud                              = 1,
 num_land_cat                        = 21,
 sf_urban_physics                    = 0,0,0,
 mp_zero_out                         = 2,
 mp_zero_out_thresh                  = 1.e-12
 cu_rad_feedback                     = .false.,
 prec_acc_dt                         = 60
 /

 &fdda
 /
 
&dynamics
 hybrid_opt                          = 2,
 etac                                = 0.2,
 rk_ord                              = 3,
 w_damping                           = 1,
 diff_opt                            = 1,      1,      1,  
 km_opt                              = 4,      4,      4,  
 diff_6th_opt                        = 0,      0,      0,  
 diff_6th_factor                     = 0.12,0.12,0.12
 base_temp                           = 290.
 damp_opt                            = 0,
 zdamp                               = 5000.,  5000.,  5000.,
 dampcoef                            = 0.01,   0.01,   0.01
 khdif                               = 0,      0,      0,  
 kvdif                               = 0,      0,      0,  
 epssm                               = 0.4,   0.4,    0.4   
 smdiv                               = 0.1,   0.1,    0.1
 emdiv                               = 0.1,   0.1,    0.1
 non_hydrostatic                     = .true., .true., .true.,
 moist_adv_opt                       = 2,      2,      2,  
 scalar_adv_opt                      = 2,      2,      2,  
 tke_adv_opt                         = 2,      0,      0,  
 time_step_sound                     = 0,      0,      0,  
 h_sca_adv_order                     = 5,      5,      5, 
 use_theta_m = 1, 
/

 &bdy_control
 spec_bdy_width                      = 5,
 specified                           = .true.,.false.,.false.,
 real_data_init_type=3,
 nested                              = .false.,.true.,.true.,
 spec_zone                           = 1,
 relax_zone                          = 4, 
/
 &grib2
 /

 &namelist_quilt
 nio_tasks_per_group = 0
 nio_groups = 1
 / 

""".format(ccyy_s=ccyy_s,mm_s=mm_s,dd_s=dd_s,hh_s=hh_s,
ccyy_e=ccyy_e,mm_e=mm_e,dd_e=dd_e,hh_e=hh_e,
mp=mp,cu=cu,lw=lw,sw=sw,pbl=pbl,sf_sfclay=sf_sfclay,sf_surf=sf_surf)

input_file = open("namelist.input","w")
input_file.write(namelist_string)
input_file.close()
os.system(f"ln -sf ../da/wrfvar_output wrfinput_d01")
os.system(f"cp ../da/wrfbdy_d01 wrfbdy_d01")
parame_string="""&control_param
 da_file            = 'wrfinput_d01'
 wrf_bdy_file       = 'wrfbdy_d01'
 wrf_input          = '../real/wrfinput_d01'
 domain_id          = 1
 debug              = .false.
 update_lateral_bdy = .true.
 update_low_bdy     = .false
 update_lsm         = .false.
 iswater            = 17
/
"""
input_file = open("parame.in","w")
input_file.write(parame_string)
input_file.close()

os.system("ln -sf $WRFDA_DIR/var/build/da_update_bc.exe .")
os.system("./da_update_bc.exe")

os.system(f'ln -sf {WRF_DIR}/run/*_DATA .')
os.system(f'ln -sf {WRF_DIR}/run/*.txt .')
os.system(f'ln -sf {WRF_DIR}/run/*.TBL .')
os.system(f'ln -sf {WRF_DIR}/run/*.tbl .')
os.system(f'ln -sf {WRF_DIR}/run/tr* .')
os.system(f'ln -sf {WRF_DIR}/run/ozone* .')
os.system(f'ln -sf {WRF_DIR}/run/CAMtr_volume_mixing_ratio .')
os.system(f'ln -sf {WRF_DIR}/main/wrf.exe .')
os.system("mpirun -np 64 ./wrf.exe")
run_num = 1
wrfout_path=glob.glob('wrfout_d01*')
if wrfout_path == []:
    if run_num == 1:
        print("wrf 运行失败,降低步长重新运行一次")
        with open('namelist.input','r+') as f:
            lines = f.readlines()
        for i in range(len(lines)):
            if "time_step                           = 60," in lines[i]:
                lines[i]="time_step                           = 30,\n"
        f.close()
        os.system("mpirun -np 64 ./wrf.exe")
        run_num += 1
    if run_num ==2 :
        print("wrf 运行失败,且重试失败")
        sys.exit(1)
#os.system(f'ncks -O -v {varlist} wrfout_d01* wrfout.nc')
os.system('mv wrfout_d01* wrfout.nc')
os.system('ncl_filedump -v Times wrfout.nc &>log.wrfout')
wrfout_time_num = os.popen('cat log.wrfout |grep :00:00 |wc -l').read()
os.remove('log.wrfout')
if not os.path.exists(f'{WRFOUT_DIR}/{Initime}'):
    os.mkdir(f'{WRFOUT_DIR}/{Initime}')
os.system(f'mv wrfout.nc {WRFOUT_DIR}/{Initime}')
if int(wrfout_time_num) < fc_hour+1:
    print("wrf 运行失败")
    os.remove('wrfout.nc')
    sys.exit(1)

