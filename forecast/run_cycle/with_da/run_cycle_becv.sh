cv_list="5 7"
#be_list=`seq 0 0`
be_dir=/data/Atmos/sunhh/WRFDA/nmc
work_dir=`pwd`
for cv in ${cv_list};do
    if [ $cv == 5 ];then
        be_list=`seq 1 6`
    else
        be_list="0 2 4"
    fi
    
	for be in ${be_list};do
        cd $work_dir
		ln -sf $be_dir/gen_be${be}_cv${cv}/be.dat be/be.dat

######## 设置
cyc_start=2022070700
cyc_end=2022073100
###预报时效 单位h
FCST_RANGE=48
###预报间隔 单位h
INTERVAL=24
LBC_FREQ=3
MAX_DOM=1
NPROC=48
cycle_opt=False # False=cold ; Trye=cyc
cycle_period=06


		ccyymmdd_s=`echo $cyc_start|cut -c1-8`
		ccyymmdd_e=`echo $cyc_end|cut -c1-8`
		hh_s=`echo $cyc_start|cut -c9-10`

		while [ $cyc_start -le $cyc_end ];do
			DATE=${cyc_start}
			##### date -d 命令只读到年月日，不读小时，默认为00时； 在此加上截断的启动时间和预报时效
			START_DATE=${DATE}
			END_DATE=`date -d "${ccyymmdd_s} +${hh_s}hours +${FCST_RANGE}hours" "+%Y%m%d%H"`


			############### WRFDA
			cd ${work_dir}
			if [ ${cycle_opt} = "True" ];then
				sed -i "5c set RUN_TYPE     = cycle" run_wrfda.csh
				sed -i "17c set CYCLE_PERIOD =  ${cycle_period}" run_wrfda.csh
			else
				sed -i "5c set RUN_TYPE     = cold" run_wrfda.csh
			fi
			sed -i "4c set DATE=${cyc_start}" run_wrfda.csh
			sed -i "178c cv_options=${cv}" run_wrfda.csh
			./run_wrfda.csh

			############### WRF
			cd ${work_dir}
			sed -i "3c set DATE=${cyc_start}" run_wrf_from_wrfda.csh
			sed -i "9c set FCST_RANGE  = ${FCST_RANGE}" run_wrf_from_wrfda.csh
			sed -i "11c set MAX_DOM  = ${MAX_DOM}" run_wrf_from_wrfda.csh
			sed -i "12c set NPROC  = ${NPROC}" run_wrf_from_wrfda.csh
			./run_wrf_from_wrfda.csh



			###每天运行一次
			cyc_start=$(date -d "${ccyymmdd_s} +${hh_s}hours +${INTERVAL}hours"  "+%Y%m%d%H" )
			ccyymmdd_s=`echo $cyc_start|cut -c1-8`
		done

	################# 把各组试验分开
	if [ ! -d /data/Atmos/sunhh/cycling/run/be${be}_cv${cv} ];then
        mkdir -p /data/Atmos/sunhh/cycling/run/be${be}_cv${cv}
	fi
	mv /data/Atmos/sunhh/cycling/run/202207* /data/Atmos/sunhh/cycling/run/be${be}_cv${cv}
	done
done




