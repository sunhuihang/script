cyc_start=2022070100
cyc_end=2022073100
###预报时效 单位h
time_window=3
###预报间隔 单位h
INTERVAL=24
work=`pwd`
little_r_dir=/data/Atmos/sunhh/Little_R
link_dir=/data/Atmos/sunhh/cycling/ob
ccyymmdd_s=`echo $cyc_start|cut -c1-8`
ccyymmdd_e=`echo $cyc_end|cut -c1-8`
hh_s=`echo $cyc_start|cut -c9-10`

while [ $cyc_start -le $cyc_end ];do
	DATE=${cyc_start}
	##### date -d 命令只读到年月日，不读小时，默认为00时； 在此加上截断的启动时间和预报时效
	START_DATE=`date -d "${ccyymmdd_s} +${hh_s}hours -${time_window}hours" "+%Y-%m-%d_%H:00:00"`
	ANALYSIS_DATE=`date -d "${ccyymmdd_s} +${hh_s}hours" "+%Y-%m-%d_%H:00:00"`
	END_DATE=`date -d "${ccyymmdd_s} +${hh_s}hours +${time_window}hours" "+%Y-%m-%d_%H:00:00"`
	echo START_DATE=${START_DATE}
	echo ANALYSIS_DATE=${ANALYSIS_DATE}
	echo END_DATE=${END_DATE}

	if [ ! -d ${DATE} ];then mkdir ${DATE};fi
	cd ${work}/${DATE}
	cp ${work}/files/* .
	ln -sf ${little_r_dir}/${DATE}/*.little_r obs.little_r
	sed -i "8c  time_window_min  = \'${START_DATE}\'," namelist.obsproc
	sed -i "9c  time_analysis  = \'${ANALYSIS_DATE}\'," namelist.obsproc
	sed -i "10c  time_window_max  = \'${END_DATE}\'," namelist.obsproc
	./obsproc.exe
	sed -i "13c date = \"${cyc_start}\"" plot_ob_ascii_loc.ncl
	sed -i "14c obfile = \"obs_gts_${ANALYSIS_DATE}.3DVAR\"" plot_ob_ascii_loc.ncl
	sed -i "15c fgfile = \"/data/Atmos/sunhh/cycling/rc/${cyc_start}/wrfinput_d01\"   ; for retrieving mapping info" plot_ob_ascii_loc.ncl
	ncl plot_ob_ascii_loc.ncl

	if [ ! -d ${link_dir}/${DATE} ];then mkdir ${link_dir}/${DATE};fi
	ln -sf `pwd`/*.3DVAR ${link_dir}/${DATE}/ob.ascii

	cd ${work}
	### ${INTERVAL}小时运行一次
	cyc_start=$(date -d "${ccyymmdd_s} +${hh_s}hours +${INTERVAL}hours"  "+%Y%m%d%H" )
	ccyymmdd_s=`echo $cyc_start|cut -c1-8`
done