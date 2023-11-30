import sys 
import re
import os
import datetime
import glob
import shutil
import env 

# --0-- User input
if (len(sys.argv) != 7 ):
    print("The user input is not correct")
    print("Usage: python " + str(sys.argv[0] + " Initime type mp_para cu_para pbl_para sf_sfclay_para"))
    print("For example: python " + str(sys.argv[0] + " 2020081800 wrfda 1 1 1 1"))
    sys.exit(1)
else:
    Initime = str(sys.argv[1])
    da = str(sys.argv[2])
    mp_para = str(sys.argv[3])
    cu_para = str(sys.argv[4])
    pbl_para = str(sys.argv[5])
    sf_sfclay_para = str(sys.argv[6])
    output_dir = os.path.join(env.output_dir, Initime)
    #data_dir = '/data/Atmos/kezx'
    data_dir = '/data/Atmos/sunhh/cycling/genetic_algor'
    wrfout_dir = os.path.join(data_dir, 'WRFOUT',Initime,f'{mp_para}_{cu_para}_{pbl_para}')
    log_dir = os.path.join(output_dir, 'LOG')
    if not os.path.isdir(log_dir): os.makedirs(log_dir)
    if not os.path.exists(wrfout_dir): os.makedirs(wrfout_dir)
    workdir = os.path.join(env.work_dir, Initime, 'tmp.wrf',f'{mp_para}_{cu_para}_{pbl_para}')
    if not os.path.exists(workdir): os.makedirs(workdir) 
    os.chdir(workdir)
    ls = os.listdir(workdir)
    for i in ls: 
        c_path = os.path.join(workdir, i)
        try:    
            os.remove(c_path)
        except:
            os.removedirs(c_path)

# --1-- Run wrf.exe
# for runfile in glob.glob(os.path.join(env.wrf_dir, 'run/*')):
#     runfile_name = os.path.basename(runfile)
#     shutil.copy(runfile, runfile_name)
link_list = ["*_DATA","*.TBL","p3_lookupTable_1.dat-5.3-2momI","ozone*","CAMtr_volume_mixing_ratio","my_file_d01.txt"]
for link in link_list:
    for runfile in glob.glob(os.path.join(env.wrf_dir,'run', link)):
        runfile_name = os.path.basename(runfile)
        os.symlink(runfile, runfile_name)
os.symlink(os.path.join(output_dir, f'REAL/wrfbdy_d01_{da}'), 'wrfbdy_d01')
for domain in range(1,env.max_dom+1):
    area = str(domain)
    if not os.path.isfile(os.path.join(output_dir, f'REAL/wrfinput_d0{area}_{da}')):
        print(f"ERROR: wrfinput_d0{area}_{da} does not exist, exit now!")
        sys.exit(1)
    os.symlink(os.path.join(output_dir, f'REAL/wrfinput_d0{area}_{da}'), f'wrfinput_d0{area}')

for exefile in glob.glob('./*exe'):
    exefile_name = os.path.basename(exefile)
    os.remove(exefile_name)
shutil.copy(os.path.join(env.wrf_dir, 'main','wrf.exe'),'wrf.exe')
shutil.copy(os.path.join(log_dir, f'namelist.{Initime}.input'), f'namelist.{Initime}.input')

r = open(f'namelist.{Initime}.input','r')
w = open('namelist.input','w')
lines = r.readlines()
for line in lines:
    if ("nproc_x                             = -1" in line):
        line = " nproc_x                             = " + str(env.nproc_x) + "\n"
    elif ("nproc_y                             = -1" in line):
        line = " nproc_y                             = " + str(env.nproc_y) + "\n"
    elif ("nio_tasks_per_group = 0," in line):
        line = " nio_tasks_per_group = " + str(env.nio_tasks_per_group) + "\n"
    elif ("nio_groups = 1," in line):
        line = " nio_groups = " + str(env.nio_groups) + "\n"
    elif ("mp_physics                          = 11,51,51," in line):
        line = f" mp_physics                          = {mp_para},{mp_para},{mp_para}," + "\n"
    elif ("cu_physics                          = 16,93,93," in line):
        line = f" cu_physics                          = {cu_para},{cu_para},{cu_para}," + "\n"
    elif ("bl_pbl_physics                      = 5,8,8," in line):
        line = f" bl_pbl_physics                      = {pbl_para},{pbl_para},{pbl_para}," + "\n"
    elif ("sf_sfclay_physics                   = 1,1,1," in line):
        line = f" sf_sfclay_physics                   = {sf_sfclay_para},{sf_sfclay_para},{sf_sfclay_para}," + "\n"
    w.write(line)
w.close()

#bdy_num = os.popen('ncl_filedump -v Times wrfbdy_d01.nc|grep :00:00  |wc -l').read()
os.system('ncl_filedump -v Times wrfbdy_d01.nc &>log.bdy')
bdy_num = os.popen('cat log.bdy |grep :00:00 |wc -l').read()
os.remove('log.bdy')
bdy_time = env.d01_len/(env.interval_seconds/3600) + 1 
if int(bdy_num) == int(bdy_time):
    print("Run wrf.exe...")
    os.system('ulimit -s unlimited')
    os.system('mpirun -np ' + str(env.wrfcores) + ' ./wrf.exe &')
    os.system('sleep 10')
else:
    print(f"ERROR: There is not enough time period for wrfbdy_d01_{da} or wrfbdy_d01_{da} does not exist")
    sys.exit(1)

# --2-- Fault-tolerant
path = os.path.join(workdir,'rsl.error.0000')
right_count = 0
same_count = 0
last_line = ' '
wait_time = 10 
while True:
    f = open(path,'r')
    lines = f.readlines()
    for line in lines:
        if ("SUCCESS COMPLETE WRF" in line):
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("!  Successful completion of wrf program  !")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            for wrfout in glob.glob('./wrfout*'):
                wrfout_name = os.path.basename(wrfout)
                shutil.move(wrfout_name, os.path.join(wrfout_dir,wrfout_name))
            shutil.copy('rsl.error.0000', os.path.join(wrfout_dir, 'log.wrf.rsl.error'))
            shutil.copy('rsl.out.0000', os.path.join(wrfout_dir, 'log.wrf.rsl.out'))
            shutil.copy('namelist.input', os.path.join(wrfout_dir, 'namelist.input'))
            sys.exit(0)
    if line == last_line:
        if same_count == wait_time:
            try:
                #pid = os.popen("ps -ef|grep 'mpirun -np 24'|awk 'NR==1 {printf $2}'").read().strip('\n')
                pid = os.popen("ps -ef|grep 'mpirun -np "+ str(env.wrfcores) + "'|awk 'NR==1 {printf $2}'").read().strip('\n')
                os.system('kill ' + pid)
            finally:
                sys.exit(1)
        else:
            same_count += 1 
            os.system('sleep 30')
            continue
    else:
        same_count = 0
        last_line = line
        os.system('sleep 5')

