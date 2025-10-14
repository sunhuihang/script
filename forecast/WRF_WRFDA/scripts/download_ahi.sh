#! /usr/bin/bash
if [ -f "/home/qixiang/SHARE/anaconda3/etc/profile.d/conda.sh" ]; then
    . "/home/qixiang/SHARE/anaconda3/etc/profile.d/conda.sh"
else
    export PATH="/home/qixiang/SHARE/anaconda3/bin:$PATH"
fi
source /home/qixiang/SHARE/anaconda3/bin/activate sunhh_usual
cd /home/qixiang/sunhh/realtime/scripts

user=978982001_qq.com   #从20221214日起，由葵花8变为葵花9
password=SP+wari8       #ahi数据延迟30min
thread=8
initime=$(date -u +"%Y%m%d00") 

ccyymmdd=`echo $initime|cut -c 1-8`
ccyy=`echo $initime|cut -c 1-4`
mm=`echo $initime|cut -c 5-6`
dd=`echo $initime|cut -c 7-8`
hh=`echo $initime|cut -c 9-10`
min="00"

data_dir=/home/qixiang/SHARE/ahi/${ccyy}.${mm}/${initime}   # 数据存放目录
test -d ${data_dir} || mkdir -p ${data_dir}
cd $data_dir
if [ $ccyymmdd -ge 20221214 ];then
    filename="NC_H09_${ccyy}${mm}${dd}_${hh}${min}_R21_FLDK.02401_02401.nc"
    URL="ftp://ftp.ptree.jaxa.jp/jma/netcdf/${ccyy}${mm}/${dd}/${filename}"
    wget -c --ftp-user=${user} --ftp-password=${password} ${URL} -O ${data_dir}/${filename}
else
    filename="NC_H08_${ccyy}${mm}${dd}_${hh}${min}_R21_FLDK.02401_02401.nc"
    URL="ftp://ftp.ptree.jaxa.jp/jma/netcdf/${ccyy}${mm}/${dd}/${filename}"
    wget -c --ftp-user=${user} --ftp-password=${password} ${URL} -O ${data_dir}/${filename}
fi
#wget -c --ftp-user=${user} --ftp-password=${password} ${URL} -O ${data_dir}/
