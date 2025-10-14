#!/bin/sh 
initime=$1
type_list="gdas gfs"    #下载数据的类型，0为gfs类型，1为gdas类型
          #gfs延迟3小时，gdas延迟6小时，gdas数据量更多
filelist="1bamua 1bhrs4 1bmhs prepbufr"
thread=8  # 进程数

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
data_dir=/home/qixiang/SHARE/gdas/${ccyy}.${mm}/${initime}   # 数据存放目录
##############################################################
test -d ${data_dir} || mkdir -p ${data_dir}
cd ${data_dir}

for type in $type_list;do
URL="https://nomads.ncep.noaa.gov/pub/data/nccf/com/obsproc/prod/${type}.${ccyymmdd}"
for file in $filelist;do
    if [ "${file}" == "prepbufr" ];then
        filename="${type}.t${hr}z.${file}.nr"
    else
        filename="${type}.t${hr}z.${file}.tm00.bufr_d"
    fi
    echo $URL/$filename
    num=0
    while [ True ];do
        [ -f ${filename} ] || axel -n ${thread} ${URL}/${filename}  > ${home}/log.gdas_download
        if [ $? -eq 0 ]; then
            echo download ${filename} successfully
            break
        elif [ $num -lt 2 ]; then
            ((num++))
            sleep 10s
            echo sleep 10s, waiting for ${filename}
        else
            echo WARNING: ${filename} is not available now, please check
            break
            #exit 1
        fi
    done
done
done
