import sys,os,datetime
import env 
run_geogrid  = 1 
run_ungrib   = 1  
run_metgrid  = 1  
run_real     = 1  
run_wrf      = 1 
run_plot     = 1 

Initime=sys.argv[1]
wps_work=sys.argv[2]
yyyy = Initime[0:4]
mm = Initime[4:6]
dd = Initime[6:8]
hh = Initime[8:10]
os.chdir(wps_work)
WPS_DIR = env.WPS_DIR
opt_geogrid_tbl_path = os.path.join(WPS_DIR,'geogrid')
opt_metgrid_tbl_path = os.path.join(WPS_DIR,'metgrid')
GFS_DIR = env.GFS_DIR
# --1-- Time information
Begin_time = datetime.datetime.strptime(Initime, "%Y%m%d%H")
Finish_time = datetime.datetime.strptime(Initime, "%Y%m%d%H") + datetime.timedelta(hours=env.d01_len)
Start_time = Begin_time.strftime("%Y-%m-%d_%H:00:00")
End_time = Finish_time.strftime("%Y-%m-%d_%H:00:00")


# --2-- Creat namelist.wps
namelist_string = """&share
    wrf_core           = 'ARW',
    max_dom            = {max_dom}
    start_date         = '{Start_time}','{Start_time}','{Start_time}'
    end_date           = '{End_time}','{End_time}','{End_time}',
    interval_seconds   = {interval_seconds},
    io_form_geogrid    = 2,
    debug_level        = 0,
/

&geogrid
    parent_id          = 0, 1, 2
    parent_grid_ratio  = 1, 3, 3
    e_we               = {e_we1},{e_we2},{e_we3},
    e_sn               = {e_sn1},{e_sn2},{e_sn3},
    i_parent_start     = 1,{i_parent_start2},{i_parent_start3},
    j_parent_start     = 1,{j_parent_start2},{j_parent_start3},
    geog_data_res      = '{geog_res}','{geog_res}','{geog_res}'
    dx                 = {dx},
    dy                 = {dy},
    map_proj           = '{map_proj}'
    ref_lat            = {ref_lat},
    ref_lon            = {ref_lon},
    truelat1           = {truelat1},
    truelat2           = {truelat2},
    stand_lon          = {stand_lon},
    geog_data_path     = '{geog_data_path}',
    opt_geogrid_tbl_path = '{opt_geogrid_tbl_path}'
/

&ungrib
    out_format         = 'WPS',
    prefix             = 'FILE',
/

&metgrid
    fg_name            = 'FILE',
    io_form_metgrid    = 2,
    opt_metgrid_tbl_path = '{opt_metgrid_tbl_path}'
/
""".format(max_dom=env.max_dom, Start_time=Start_time, End_time=End_time, interval_seconds=env.interval_seconds,
            e_we1=env.e_we1, e_sn1=env.e_sn1, e_we2=env.e_we2, e_sn2=env.e_sn2, e_we3=env.e_we3, e_sn3=env.e_sn3,
            i_parent_start2=env.i_parent_start2, j_parent_start2=env.j_parent_start2,
            i_parent_start3=env.i_parent_start3, j_parent_start3=env.j_parent_start3, geog_res=env.geog_res,
            dx=env.dx, dy=env.dy, map_proj=env.map_proj, ref_lat=env.ref_lat, ref_lon=env.ref_lon,
            stand_lon=env.stand_lon, truelat1=env.truelat1, truelat2=env.truelat2, geog_data_path=env.geog_data_path,
            opt_geogrid_tbl_path=opt_geogrid_tbl_path,opt_metgrid_tbl_path=opt_metgrid_tbl_path)

wps_file = open("namelist.wps", "w")
wps_file.write(namelist_string)
wps_file.close()

############ geogrid ##########
if run_geogrid:
    print(" ")
    print('Begin to run geogrid')
    os.system(f'ln -sf {WPS_DIR}/geogrid.exe .')
    os.system('./geogrid.exe >& log.geogrid')

########## ungrib ##########
if run_ungrib :
    print(" ") 
    print('Begin to run ungrib(gfs)')
    print(f'{GFS_DIR}/{yyyy}.{mm}/{Initime}/gfs.t{hh}z.pgrb2.0p25.f*')
    os.system('rm -f GRIBFILE* FILE*')
    os.system(f'ln -sf {WPS_DIR}/ungrib.exe .')
    os.system(f'ln -sf {WPS_DIR}/link_grib.csh .')
    os.system(f'ln -sf {WPS_DIR}/ungrib/Variable_Tables/Vtable.GFS Vtable')
    os.system(f'./link_grib.csh {GFS_DIR}/{yyyy}.{mm}/{Initime}/gfs.t{hh}z.pgrb2.0p25.f*')
    os.system('./ungrib.exe >& log.ungrib')


########## metgrid ##########
if run_metgrid :
    print(" ")
    print('Begin to run metgrid')
    os.system(f'ln -sf {WPS_DIR}/metgrid.exe .')
    os.system('./metgrid.exe >& log.metgrid')
    os.system('rm FILE:* PFILE*')

















