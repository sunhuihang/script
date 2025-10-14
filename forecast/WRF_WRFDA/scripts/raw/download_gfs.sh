#!/bin/sh 
initime=$1
interval_gfs=3  # 间隔
res_gfs=0.25      # 分辨率 0.5 or 0.25 
fstart_gfs=0    # 开始时间
fend_gfs=48     # 结束时间
thread=8        # 进程数
if [ -z "$initime" ];then
    echo "please input initime"
    exit
fi
ccyymmdd=`echo ${initime} | cut -c 1-8`
ccyymm=`echo ${initime} |cut -c 1-6`
ccyy=`echo ${initime} | cut -c 1-4`
mm=`echo ${initime} |cut -c 5-6`
hr=`echo ${initime} | cut -c 9-10`
home=`pwd`
data_dir=/home/qixiang/SHARE/gfs/${ccyy}.${mm}/${initime}   # 数据存放目录
##############################################################
test -d ${data_dir} || mkdir -p ${data_dir}
cd ${data_dir}

#URL="https://www.ncei.noaa.gov/data/global-forecast-system/access/grid-004-0.5-degree/forecast/${ccyymm}/${ccyymmdd}"
URL="https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/gfs.${ccyymmdd}/${hr}/atmos"
it=${fstart_gfs}
while [ ${it} -le ${fend_gfs} ]
do
  fhr=`expr ${it}  | awk '{printf "%02d\n",$1}'`
  fhr3=`expr ${it} | awk '{printf "%03d\n",$1}'`
  #filename="gfs_4_${ccyymmdd}_${hr}00_${fhr3}.grb2"
  filename="gfs.t${hr}z.pgrb2.0p25.f${fhr3}"
  echo $URL/$filename
  num=0
  while [ True ]
  do
    [ -f ${filename} ] || axel -n ${thread} ${URL}/${filename}  > ${home}/log.gfs_download
    if [ $? -eq 0 ]; then
      echo download ${filename} successfully
      break
    elif [ $num -lt 3 ]; then
      ((num++))
      sleep 3m
      echo sleep 3 min, waiting for ${filename}
    else
      echo WARNING: ${filename} is not available now, please check
      exit 1
    fi
  done
  it=`expr ${it} + ${interval_gfs}`
done
