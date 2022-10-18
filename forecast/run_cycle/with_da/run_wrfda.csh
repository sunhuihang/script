#!/bin/csh
#
set echo
set DATE=2022070700
set RUN_TYPE     = cold
set DOMAIN_ID    = 01
set NPROC        = 12
set DA_SRCDIR    = /data/Atmos/sunhh/WRFDA
set BIN_DIR      = ${DA_SRCDIR}/var/build
set OB_DATDIR    = /data/Atmos/sunhh/cycling/ob
set BE_DATDIR    = /data/Atmos/sunhh/cycling/be
set RC_DATDIR    = /data/Atmos/sunhh/cycling/rc
set RUN_BASEDIR  = /data/Atmos/sunhh/cycling/run
set DA_RUNDIR    = ${RUN_BASEDIR}/${DATE}/da/d${DOMAIN_ID}
set TIMEWINDOW1  =  -1h00m
set TIMEWINDOW2  =  1h00m
set CYCLE_PERIOD =  06

set ccyy = `echo $DATE | cut -c1-4`
set   mm = `echo $DATE | cut -c5-6`
set   dd = `echo $DATE | cut -c7-8`
set   hh = `echo $DATE | cut -c9-10`
set DATE_MIN = `${BIN_DIR}/da_advance_time.exe ${DATE} ${TIMEWINDOW1} -w`
set DATE_MAX = `${BIN_DIR}/da_advance_time.exe ${DATE} ${TIMEWINDOW2} -w`

set PREV_DATE = `${BIN_DIR}/da_advance_time.exe ${DATE} -${CYCLE_PERIOD}`
set VARBC_PREV_DATE = `${BIN_DIR}/da_advance_time.exe ${DATE} -24`

if ( ! -d ${DA_RUNDIR} ) mkdir -p ${DA_RUNDIR}
cd ${DA_RUNDIR}

if ( ${DOMAIN_ID} == '01' ) then
   set WEST_EAST_GRID_NUMBER   = 142
   set SOUTH_NORTH_GRID_NUMBER = 116
   set VERTICAL_GRID_NUMBER    = 43
   set GRID_DISTANCE           = 12000
   set BE_FILE = be.dat
else if ( ${DOMAIN_ID} == '02' ) then
   set WEST_EAST_GRID_NUMBER   = 181
   set SOUTH_NORTH_GRID_NUMBER = 121
   set VERTICAL_GRID_NUMBER    = 41
   set GRID_DISTANCE           = 30000
   set BE_FILE = be_d02.dat
endif

rm -f ${DA_RUNDIR}/rsl.*
#
# link some constant files
#
ln -fs ${DA_SRCDIR}/run/LANDUSE.TBL  .
#
# link first-guess, observations, background-error
#
# link either ob.bufr (for ob_format=1) or ob.ascii (for ob_format=2)
ln -fs ${OB_DATDIR}/${DATE}/prepbufr.gdas.${ccyy}${mm}${dd}.t${hh}z.nr    ./ob.bufr
ln -fs ${OB_DATDIR}/${DATE}/ob.ascii                           ./ob.ascii
# BE file for cv_options=5 and 7
ln -fs ${BE_DATDIR}/${BE_FILE}                                 ./be.dat
# BE file for cv_options=3
#ln -fs ${DA_SRCDIR}/var/run/be.dat.cv3                      ./be.dat
#
# link radiance-related files
#
ln -fs ${OB_DATDIR}/${DATE}/gdas.t${hh}z.1bamua.tm00.bufr_d ./amsua.bufr
ln -sf ${OB_DATDIR}/${DATE}/gdas.t${hh}z.1bmhs.tm00.bufr_d ./mhs.bufr_d
#ln -fs ${OB_DATDIR}/${DATE}/gdas.1bamub.t${hh}z.${ccyy}${mm}${dd}.bufr   ./amsub.bufr
ln -fs ${DA_SRCDIR}/var/run/radiance_info .
# link crtm_coeffs if using CRTM
ln -fs ${DA_SRCDIR}/var/run/crtm_coeffs   .
# link rttov_coeffs if using RTTOV
#ln -fs /kumquat/wrfhelp/external/rttov11/rtcoef_rttov11/rttov7pred54L ./rttov_coeffs
#
set VARBC_DIR = ${RUN_BASEDIR}/${VARBC_PREV_DATE}/da/d${DOMAIN_ID}
if ( ! -e ${VARBC_DIR}/VARBC.out ) then
   ln -fs ${DA_SRCDIR}/var/run/VARBC.in ./VARBC.in
else
   ln -fs ${VARBC_DIR}/VARBC.out  ./VARBC.in
endif

if ( ${RUN_TYPE} == 'cold' ) then
   if ( ! -e ${RC_DATDIR}/${DATE}/wrfinput_d${DOMAIN_ID} ) then
      echo "ERROR in run_wrfda.csh : first guess ${RC_DATDIR}/${DATE}/wrfinput_d${DOMAIN_ID} not found..."
      exit 1
   endif
   ln -fs ${RC_DATDIR}/${DATE}/wrfinput_d${DOMAIN_ID} fg
else  # cycling
   if ( ! -e ${RUN_BASEDIR}/${PREV_DATE}/wrf/wrfout_d${DOMAIN_ID}_${ccyy}-${mm}-${dd}_${hh}:00:00 ) then
      echo "ERROR in run_wrfda.csh : first guess ${RUN_BASEDIR}/${PREV_DATE}/wrf/wrfout_d${DOMAIN_ID}_${ccyy}-${mm}-${dd}_${hh}:00:00 not found..."
      exit 1
   else
      ln -fs ${RUN_BASEDIR}/${PREV_DATE}/wrf/wrfout_d${DOMAIN_ID}_${ccyy}-${mm}-${dd}_${hh}:00:00 fg_orig
      cp -p ${RUN_BASEDIR}/${PREV_DATE}/wrf/wrfout_d${DOMAIN_ID}_${ccyy}-${mm}-${dd}_${hh}:00:00 fg
   endif
endif

if ( ${RUN_TYPE} == 'cycle' ) then
   # when not cold-starting, update lower boundary first, before running wrfvar
   # fields to be updated are TSK over water, TMN, SST, VEGFRA, ALBBCK, SEAICE, LANDMASK, IVGTYP, ISLTYP
   cd ${DA_RUNDIR}
   cat >! ${DA_RUNDIR}/parame.in << EOF
&control_param
 da_file            = '${DA_RUNDIR}/fg'
 wrf_input          = '${RC_DATDIR}/${DATE}/wrfinput_d${DOMAIN_ID}'
 domain_id          = ${DOMAIN_ID}
 debug              = .false.
 update_lateral_bdy = .false.
 update_low_bdy     = .true.
 update_lsm         = .false.
 iswater            = 17 /
EOF
   ln -fs ${DA_SRCDIR}/var/build/da_update_bc.exe .
   time ./da_update_bc.exe >&! update_low_bc_${DATE}.log
   mv parame.in parame.in.lowbdy
   # check status
   grep "Update_bc completed successfully" update_low_bc_${DATE}.log
   if ( $status != 0 ) then
      echo "ERROR in run_wrfda.csh : update low bdy failed..."
      exit 1
   endif
endif
#
# create namelist for WRFDA run
#
cat >! namelist.input << EOF
 &wrfvar1
 print_detail_grad=false
 /
 &wrfvar2
 /
 &wrfvar3
 ob_format=2
 /
 &wrfvar4
 thin_mesh_conv = 20
 USE_SYNOPOBS =  T,
 USE_SHIPSOBS =  T,
 USE_METAROBS =  T,
 USE_SOUNDOBS =  T,
 USE_MTGIRSOBS =  F,
 USE_TAMDAROBS =  F,
 USE_PILOTOBS =  F,
 USE_AIREPOBS =  T,
 USE_GEOAMVOBS =  T,
 USE_POLARAMVOBS =  T,
 USE_BOGUSOBS =  F,
 USE_BUOYOBS =  T,
 USE_PROFILEROBS =  T,
 USE_SATEMOBS =  F,
 USE_GPSZTDOBS =  F,
 USE_GPSPWOBS =  F,
 USE_GPSREFOBS =  F,
 USE_QSCATOBS =  T,
 USE_AIRSRETOBS =  F,
 use_ssmiretrievalobs=false
 use_amsuaobs = F,
 use_amsubobs = T,
 use_mhsobs   = F,
 use_airsobs  = F,
 use_eos_amsuaobs = F,
 USE_OBS_ERRFAC=F
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
 eps=0.01,
 orthonorm_gradient=.true.,
 /
 &wrfvar7
 !cv_options=3
cv_options=5
 !var_scaling1 = 1.,
 !var_scaling2 = 1.,
 !var_scaling3 = 1.,
 !var_scaling4 = 1.,
 !var_scaling5 = 1.,
 !len_scaling1 = 1.,
 !len_scaling2 = 1.,
 !len_scaling3 = 1.,
 !len_scaling4 = 1.,
 !len_scaling5 = 1.,
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
 !check_rh=1
 sfc_assi_options =2
 max_stheight_diff=100
 obs_err_inflate = false
 stn_ht_diff_scale = 200.0
 calculate_cg_cost_fn=.true.
 write_detail_grad_fn=.true.
 /
 &wrfvar12
 /
 &wrfvar13
 /
 &wrfvar14
 rtminit_nsensor=2,
 rtminit_platform=1,1,
 rtminit_satid=18,18,
 rtminit_sensor=3,15,
 thinning_mesh=30*200.0
 thinning=true,
 qc_rad=true,
 write_iv_rad_ascii=false,
 write_oa_rad_ascii=true,
 rtm_option=2,
 only_sea_rad=false,
 use_varbc=.true.
 use_crtm_kmatrix=.true.
 crtm_coef_path='./crtm_coeffs'
 /
 &wrfvar15
 /
 &wrfvar16
 /
 &wrfvar17
 analysis_type="QC-OBS"
 /
 &wrfvar18
 analysis_date="${ccyy}-${mm}-${dd}_${hh}:00:00",
 /
 &wrfvar19
 /
 &wrfvar20
 /
 &wrfvar21
 time_window_min="${DATE_MIN}",
 /
 &wrfvar22
 time_window_max="${DATE_MAX}",
 /
 &wrfvar23
 /
 &time_control
 start_year                          = ${ccyy}
 start_month                         = ${mm}
 start_day                           = ${dd}
 start_hour                          = ${hh}
 start_minute                        = 00
 start_second                        = 00
 end_year                            = ${ccyy}
 end_month                           = ${mm}
 end_day                             = ${dd}
 end_hour                            = ${hh}
 end_minute                          = 00
 end_second                          = 00
 interval_seconds                    = 10800
 input_from_file                     = .true.
 history_interval                    = 60,
 frames_per_outfile                  = 1,
 restart                             = .false.,
 restart_interval                    = 1440,
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
 s_we                                = 1
 e_we                                = ${WEST_EAST_GRID_NUMBER}
 s_sn                                = 1
 e_sn                                = ${SOUTH_NORTH_GRID_NUMBER}
 s_vert                              = 1
 e_vert                              = ${VERTICAL_GRID_NUMBER}
 dx                                  = ${GRID_DISTANCE}
 dy                                  = ${GRID_DISTANCE}
 p_top_requested                     = 5000,
 num_metgrid_levels                  = 34,
 num_metgrid_soil_levels             = 4,
 grid_id                             = 1,
 parent_id                           = 0,
 i_parent_start                      = 1,
 j_parent_start                      = 1,
 parent_grid_ratio                   = 1,
 parent_time_step_ratio              = 1,
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
 target_cfl                          = 1.2,
 target_hcfl                         = .84,
 max_step_increase_pct               = 5,
 max_time_step                       = 27,
 starting_time_step                  = 9,
 min_time_step                       = 9,
 adaptation_domain                   = 1
 /
 &physics
 mp_physics                          = 51,
 cu_physics                          = 93,
 ra_lw_physics                       = 4,
 ra_sw_physics                       = 4,
 radt                                = 27
 sf_sfclay_physics                   = 1,
 sf_surface_physics                  = 1,
 bl_pbl_physics                      = 8,
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

ln -sf ${DA_SRCDIR}/var/build/da_wrfvar.exe .
if ( $NPROC > 1 ) then
   mpirun -np ${NPROC} ./da_wrfvar.exe  >&! wrfda_${DATE}_d${DOMAIN_ID}.log
else
   time ./da_wrfvar.exe  >&! wrfda_${DATE}_d${DOMAIN_ID}.log
endif

# check status
if ( $NPROC > 1 ) then
   grep "WRF-Var completed successfully" rsl.out.0000
else
   grep "WRF-Var completed successfully" wrfda_${DATE}_d${DOMAIN_ID}.log
endif
if ( $status != 0 ) then
   echo "ERROR in run_wrfda.csh : da_wrfvar.exe failed..."
   exit 1
endif

rm gts_omb_oma_01.* unpert_obs.*
foreach file_rej ( `ls rej_obs_conv_01.*` )
   cat ${file_rej} >> rej_obs_conv_01
end
rm rej_obs_conv_01.* filtered_obs.0*

# update lateral bdy for coarse domain
if ( ${DOMAIN_ID} == '01' ) then
   cd ${DA_RUNDIR}
   cp -p ${RC_DATDIR}/${DATE}/wrfbdy_d${DOMAIN_ID} ${DA_RUNDIR}/wrfbdy_d${DOMAIN_ID}
   cat >! ${DA_RUNDIR}/parame.in << EOF
&control_param
 da_file            = '${DA_RUNDIR}/wrfvar_output'
 wrf_bdy_file       = '${DA_RUNDIR}/wrfbdy_d${DOMAIN_ID}'
 wrf_input          = '${RC_DATDIR}/${DATE}/wrfinput_d${DOMAIN_ID}'
 domain_id          = ${DOMAIN_ID}
 debug              = .false.
 update_lateral_bdy = .true.
 update_low_bdy     = .false
 update_lsm         = .false.
 iswater            = 17 /
EOF
   ln -sf ${DA_SRCDIR}/var/build/da_update_bc.exe .
   time ./da_update_bc.exe >&! update_lat_bc_${DATE}.log
   mv parame.in parame.in.latbdy
   # check status
   grep "Update_bc completed successfully" update_lat_bc_${DATE}.log
   if ( $status != 0 ) then
      echo "ERROR in run_wrfda.csh : update lateral bdy failed..."
      exit 1
   endif
endif

echo "Done run_wrfda.csh for domain ${DOMAIN_ID} `date`"
