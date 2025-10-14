import os,sys,datetime,subprocess
import env
Initime=sys.argv[1]
real_work=sys.argv[2]
sf_surface_physics = env.sf_surface_physics
WRF_DIR = env.WRF_DIR
os.chdir(real_work)
Begin_time = datetime.datetime.strptime(Initime, "%Y%m%d%H")
Start_time = Begin_time.strftime("%Y%m%d%H")
Finish_time = (Begin_time + datetime.timedelta(hours=env.d01_len)).strftime("%Y%m%d%H")



ccyy_s = Initime[0:4]
mm_s = Initime[4:6]
dd_s = Initime[6:8]
hh_s = Initime[8:10]

ccyy_e = Finish_time[0:4]
mm_e = Finish_time[4:6]
dd_e = Finish_time[6:8]
hh_e = Finish_time[8:10]
#定义一个检查文本中是否有某句话的函数，用以检查程序的执行情况
def check_sentence(filename, sentence):
    with open(filename, 'r') as f:
        for line in f:
            if sentence in line:
                print("Sentence found")
                return
        print("Sentence not found")
#生成namelist.input
namelist_string="""&time_control
start_year                          = {ccyy_s},{ccyy_s},{ccyy_s},
start_month                         = {mm_s},   {mm_s},   {mm_s},
start_day                           = {dd_s},   {dd_s},   {dd_s},
start_hour                          = {hh_s},   {hh_s},   {hh_s},
start_minute                        = 00,00,00,
start_second                        = 00,00,00,
end_year                            = {ccyy_e},{ccyy_e},{ccyy_e},
end_month                           = {mm_e},   {mm_e},   {mm_e},
end_day                             = {dd_e},   {dd_e},   {dd_e},
end_hour                            = {hh_e},   {hh_e},   {hh_e},
end_minute                          = 00,00,00,
end_second                          = 00,00,00,
interval_seconds                    = {interval_seconds}
input_from_file                     = .true.,.true.,.true.,
history_interval                    = 60,60,60,
frames_per_outfile                  = 1000,1000,1000,
restart                             = .false.,
!restart_interval                    = 1440,
io_form_history                     = 2
io_form_restart                     = 2
io_form_input                       = 2
io_form_boundary                    = 2
debug_level                         = 0
auxinput1_inname                    = "met_em.d<domain>.<date>",
adjust_output_times                 = .true.
/
&dfi_control
/
&domains
time_step                           = {time_step},
time_step_fract_num                 = 0,
time_step_fract_den                 = 1,
max_dom                             = {max_dom},
e_we                                = {e_we1},{e_we2},{e_we3},
e_sn                                = {e_sn1},{e_sn2},{e_sn3},
e_vert                              = {e_vert},{e_vert},{e_vert},
p_top_requested                     = {ptop},
num_metgrid_levels                  = {num_metgrid_levels},
num_metgrid_soil_levels             = {num_metgrid_soil_levels},
dx                                  = {dx},{dx2},{dx3},
dy                                  = {dy},{dy2},{dy3},
grid_id                             = 1,2,3,
parent_id                           = 0,1,2,
i_parent_start                      = 1,{i_parent_start2},{i_parent_start3},
j_parent_start                      = 1,{j_parent_start2},{j_parent_start3},
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
mp_physics                          = {mp_physics},{mp_physics},{mp_physics},
cu_physics                          = {cu_physics1},{cu_physics2},{cu_physics3},
ra_lw_physics                       = {ra_lw_physics},{ra_lw_physics},{ra_lw_physics},
ra_sw_physics                       = {ra_sw_physics},{ra_sw_physics},{ra_sw_physics},
radt                                = 12
sf_sfclay_physics                   = {sf_sfclay_physics},{sf_sfclay_physics},{sf_sfclay_physics},
sf_surface_physics                  = {sf_surface_physics},{sf_surface_physics},{sf_surface_physics},
bl_pbl_physics                      = {bl_pbl_physics},{bl_pbl_physics},{bl_pbl_physics},
bldt                                = 0
cudt                                = 0
kfeta_trigger                       = 1
icloud                              = 1,
num_land_cat                        = {num_land_cat},
sf_urban_physics                    = 0,0,0,
mp_zero_out                         = 0,
mp_zero_out_thresh                  = 1.e-12
cu_rad_feedback                     = .false., 
/
&fdda
/
&dynamics
hybrid_opt                          = 2,
etac                                = {etac}
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
use_theta_m = 1
/
&bdy_control
spec_bdy_width                      = 5,
specified                           = .true.,.false.,.false.,
nested                              = .false.,.true.,.true.,
spec_zone                           = 1,
relax_zone                          = 4,  
/
&grib2
/
&namelist_quilt
nio_tasks_per_group = 0,
nio_groups = 1,
/ 
""".format(ccyy_s=ccyy_s,mm_s=mm_s,dd_s=dd_s,hh_s=hh_s,ccyy_e=ccyy_e,mm_e=mm_e,dd_e=dd_e,hh_e=hh_e,
           interval_seconds=env.interval_seconds,time_step=env.time_step,max_dom=env.max_dom,
           e_we1=env.e_we1,e_we2=env.e_we2,e_we3=env.e_we3,e_sn1=env.e_sn1,e_sn2=env.e_sn2,
           e_sn3=env.e_sn3,e_vert=env.e_vert,ptop=env.ptop,num_metgrid_levels=env.num_metgrid_levels,
           num_metgrid_soil_levels=env.num_metgrid_soil_levels,dx=env.dx,dx2=env.dx2,dx3=env.dx3,
           dy=env.dy,dy2=env.dy2,dy3=env.dy3,i_parent_start2=env.i_parent_start2,
           i_parent_start3=env.i_parent_start3,j_parent_start2=env.j_parent_start2,
           j_parent_start3=env.j_parent_start3,mp_physics=env.mp_physics,cu_physics1=env.cu_physics1,
           cu_physics2=env.cu_physics2,cu_physics3=env.cu_physics3,ra_lw_physics=env.ra_lw_physics,
           ra_sw_physics=env.ra_sw_physics,sf_sfclay_physics=env.sf_sfclay_physics,
           bl_pbl_physics=env.bl_pbl_physics,num_land_cat=env.num_land_cat,etac=env.etac,
           sf_surface_physics=sf_surface_physics)

input_file = open("namelist.input","w")
input_file.write(namelist_string)
input_file.close()

os.system('ln -sf ../wps/met_em*.nc .')
os.system(f'ln -sf {WRF_DIR}/run/real.exe .')

print("Run real.exe...")
subprocess.check_call('mpirun -np 32 ./real.exe',shell=True)
state_real = check_sentence('rsl.error.0000',"SUCCESS COMPLETE REAL_EM INIT")
if state_real == "Sentence not found":
    shutil.move('rsl.error.0000',os.path.join(output_dir,'rsl.error.0000'))
    print("real.exe failed")
    sys.exit(1)
