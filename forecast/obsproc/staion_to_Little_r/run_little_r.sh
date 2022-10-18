cyc_start=2022070100
cyc_end=2022073100
###预报时效 单位h
time_window=3
###预报间隔 单位h
INTERVAL=24

ccyymmdd_s=`echo $cyc_start|cut -c1-8`
ccyymmdd_e=`echo $cyc_end|cut -c1-8`
hh_s=`echo $cyc_start|cut -c9-10`

while [ $cyc_start -le $cyc_end ];do
	DATE=${cyc_start}
	##### date -d 命令只读到年月日，不读小时，默认为00时； 在此加上截断的启动时间和预报时效
	START_DATE=`date -d "${ccyymmdd_s} +${hh_s}hours -${time_window}hours" "+%Y%m%d%H"`
	END_DATE=`date -d "${ccyymmdd_s} +${hh_s}hours +${time_window}hours" "+%Y%m%d%H"`
	echo START_DATE=${START_DATE}
	echo END_DATE=${END_DATE}
	if [ ! -d ${DATE} ];then mkdir ${DATE};fi
	sed -i "7c start_date   = \"${START_DATE}\"" station_to_Little_r.py
	sed -i "8c end_date   = \"${END_DATE}\"" station_to_Little_r.py
	sed -i "12c folder_data_little_R = \"/data/Atmos/sunhh/Little_R/${DATE}\"" station_to_Little_r.py
	python3 station_to_Little_r.py


	### ${INTERVAL}小时运行一次
	cyc_start=$(date -d "${ccyymmdd_s} +${hh_s}hours +${INTERVAL}hours"  "+%Y%m%d%H" )
	ccyymmdd_s=`echo $cyc_start|cut -c1-8`
done