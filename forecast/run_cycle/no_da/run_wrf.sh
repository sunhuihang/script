cyc_start=2022070100
cyc_end=2022073100
BIN_DIR=/data/Atmos/sunhh/WRFDA/var/build
RC_DATDIR=/data/Atmos/sunhh/cycling/rc
WRF_SRCDIR=/data/Atmos/sunhh/WRF
RUN_BASEDIR=/data/Atmos/sunhh/WRFDA/nmc/run
MET_DIR=/data/Atmos/sunhh/WRFDA/nmc/met
REAL_RUNDIR=/data/Atmos/sunhh/WRFDA/nmc/real_run
REAL_DIR=/data/Atmos/sunhh/WRFDA/nmc/real
###预报时效 单位h
FCST_RANGE=48
###预报间隔 单位h
INTERVAL=24

LBC_FREQ=3
MAX_DOM=1
NPROC=48
WPS_DIR=/data/Atmos/sunhh/WPS
GFS_DIR=/data/Atmos/SHARE/atmosData/gfs/2022.07

ccyymmdd_s=`echo $cyc_start|cut -c1-8`
ccyymmdd_e=`echo $cyc_end|cut -c1-8`
hh_s=`echo $cyc_start|cut -c9-10`

while [ $cyc_start -le $cyc_end ];do
DATE=${cyc_start}
##### date -d 命令只读到年月日，不读小时，默认为00时； 在此加上截断的启动时间和预报时效
START_DATE=${DATE}
END_DATE=`date -d "${ccyymmdd_s} +${hh_s}hours +${FCST_RANGE}hours" "+%Y%m%d%H"`

############################ wps
cd ${WPS_DIR}
START_WPS=`${BIN_DIR}/da_advance_time.exe $START_DATE 0 -w`
END_WPS=`${BIN_DIR}/da_advance_time.exe $END_DATE 0 -w`
cat >namelist.wps << EOF
&share
 wrf_core = 'ARW',
 max_dom = 1,
 start_date = ${START_WPS},${START_WPS}
 end_date   = ${END_WPS},${END_WPS}
 interval_seconds = 10800
/

&geogrid
 parent_id         =   0,   1,
 parent_grid_ratio =   1,   3, 
 i_parent_start    =   1,  25, 
 j_parent_start    =   1,  21, 
 e_we              = 142,  106,
 e_sn              = 116,  94,
 geog_data_res = 'default','default'
 dx = 12000, 
 dy = 12000,
 map_proj = 'lambert',
 ref_lat   =  37.00,
 ref_lon   = 120.00,
 truelat1  =  30.0,
 truelat2  =  60.0,
 stand_lon = 112.0,
 geog_data_path = '/data/Atmos/sunhh/WPS_GEOG'
/

&ungrib
 out_format = 'WPS',
 prefix = 'FILE',
/

&metgrid
 fg_name = 'FILE'
 io_form_metgrid = 2,
/

EOF

rm geo_em.d* GRIBFILE* PFILE* FILE* met_em*
./geogrid.exe
./link_grib.csh ${GFS_DIR}/gfs_4_${ccyymmdd_s}_0000*.grb2
./ungrib.exe
./metgrid.exe

if [ ! -d ${MET_DIR}/${DATE} ];then mkdir -p ${MET_DIR}/${DATE};fi
mv met_em* ${MET_DIR}/${DATE}

echo "Done ${DATE} wps "

############################ real 
if [ ! -d ${REAL_RUNDIR} ];then mkdir -p ${REAL_RUNDIR};fi
cd  ${REAL_RUNDIR}
rm -rf *
ccyy_s=`echo $START_DATE | cut -c1-4`
mm_s=`echo $START_DATE | cut -c5-6`
dd_s=`echo $START_DATE | cut -c7-8`
hh_s=`echo $START_DATE | cut -c9-10`
ccyy_e=`echo $END_DATE | cut -c1-4`
mm_e=`echo $END_DATE | cut -c5-6`
dd_e=`echo $END_DATE | cut -c7-8`
hh_e=`echo $END_DATE | cut -c9-10`

cat >namelist.input << EOF
 &time_control
 start_year                          = ${ccyy_s}, ${ccyy_s},
 start_month                         = ${mm_s},   ${mm_s},
 start_day                           = ${dd_s},   ${dd_s},
 start_hour                          = ${hh_s},   ${hh_s},
 start_minute                        = 00,   00,
 start_second                        = 00,   00,
 end_year                            = ${ccyy_e}, ${ccyy_e},
 end_month                           = ${mm_e},   ${mm_e},
 end_day                             = ${dd_e},   ${dd_e},
 end_hour                            = ${hh_e},   ${hh_e},
 end_minute                          = 00,   00,
 end_second                          = 00,   00,
 interval_seconds                    = 10800
 input_from_file                     = .true.,.true.,
 history_interval                    = 60,  60,
 frames_per_outfile                  = 1, 1,
 restart                             = .false.,
 restart_interval                    = 999999,
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
 time_step                           = 60,
 time_step_fract_num                 = 0,
 time_step_fract_den                 = 1,
 max_dom                             = ${MAX_DOM},
 e_we                                = 142,199,235,
 e_sn                                = 116,175,205,
 e_vert                              = 43,43,43,
 p_top_requested                     = 5000,
 num_metgrid_levels                  = 34,
 num_metgrid_soil_levels             = 4,
 dx                                  = 12000,9000,3000,
 dy                                  = 12000,9000,3000,
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
 mp_physics                          = 51,51,51,
 cu_physics                          = 93,93,93,
 ra_lw_physics                       = 4,4,4,
 ra_sw_physics                       = 4,4,4,
 radt                                = 27
 sf_sfclay_physics                   = 1,1,1,
 sf_surface_physics                  = 1,1,1,
 bl_pbl_physics                      = 8,8,8,
 bldt                                = 0
 cudt                                = 0
 kfeta_trigger                       = 1
 icloud                              = 1,
 num_land_cat                        = 21,
 sf_urban_physics                    = 0,0,0,
 mp_zero_out                         = 0,
 mp_zero_out_thresh                  = 1.e-12
 cu_rad_feedback                     = .false., 
 /
 &fdda
 /
 &dynamics
 hybrid_opt                          = 0,
 etac                                = 0.15
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
EOF

ln -sf ${WRF_SRCDIR}/run/real.exe .
ln -sf ${MET_DIR}/${DATE}/met_em* .
./real.exe
if [ ! -d ${REAL_DIR}/${DATE} ];then mkdir -p ${REAL_DIR}/${DATE};fi
mv namelist.input wrfbdy_* wrfinput_* ${REAL_DIR}/${DATE}
######################## WRF

WRF_RUNDIR=${RUN_BASEDIR}/${DATE}
if [ ! -d ${WRF_RUNDIR} ];then mkdir -p ${WRF_RUNDIR};fi
cd ${WRF_RUNDIR}
rm *
cp ${REAL_DIR}/${DATE}/namelist.input .
#
# link constant files
#
ln -fs ${WRF_SRCDIR}/run/*_DATA .
#ln -fs ${WRF_SRCDIR}/run/*_DATA_DBL .
#ln -fs ${WRF_SRCDIR}/run/*_TBL .
ln -fs ${WRF_SRCDIR}/run/*.TBL .
ln -sf ${WRF_SRCDIR}/run/p3_lookupTable_1.dat-5.3-2momI .
#ln -fs ${WRF_SRCDIR}/run/*_tbl .
#ln -fs ${WRF_SRCDIR}/run/*_txt .
ln -fs ${WRF_SRCDIR}/run/ozone* .
#ln -fs ${WRF_SRCDIR}/run/tr* .
ln -sf ${WRF_SRCDIR}/run/CAMtr_volume_mixing_ratio .


ln -fs ${WRF_SRCDIR}/main/wrf.exe .

ln -fs ${REAL_DIR}/${DATE}/wrfbdy_d01 .
ln -fs ${REAL_DIR}/${DATE}/wrfinput_d01 .
if [ ${MAX_DOM} == 2 ]; then
   ln -fs ${REAL_DIR}/${DATE}/wrfinput_d02 .
fi
if [ $NPROC > 1 ]; then
   mpirun -np ${NPROC} ./wrf.exe
else
   time ./wrf.exe >&! wrf_${DATE}.log
fi


###每天运行一次
cyc_start=$(date -d "${ccyymmdd_s} +${hh_s}hours +${INTERVAL}hours"  "+%Y%m%d%H" )
ccyymmdd_s=`echo $cyc_start|cut -c1-8`
done



