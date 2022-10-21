cyc_start=2022101100 #开始下载的时间
cyc_end=2022102000

INTERVAL=6  # 间隔
FCST_RANGE=48    #最大预报时效
resolution=0.5      # 分辨率
thread=6        # 进程数
leftlon=70
rightlon=145
toplat=60   #南纬用负的
bottomlat=0
work=`pwd`


while [ $cyc_start -le $cyc_end ];do
    ccyymmdd_s=`echo $cyc_start|cut -c1-8`
    ccyymmdd_e=`echo $cyc_end|cut -c1-8`
    hh_s=`echo $cyc_start|cut -c9-10`
    ccyymm=`echo $cyc_start|cut -c1-6`
    data_dir=$work/${ccyymm}   # 数据存放目录
    test -d ${data_dir} || mkdir -p ${data_dir}
    # cd ${data_dir}
    source="https://nomads.ncep.noaa.gov/cgi-bin/"

    for fcst_start in `seq -w 00 $INTERVAL 18`;do
        for fcst_range in `seq -w 000 3 $FCST_RANGE`;do
            # echo $cyc_start $fcst_start $fcst_range
            if [ $resolution == 0.25 ];then
                filename="filter_gfs_0p25.pl?file=gfs.t${fcst_start}z.pgrb2.0p25.f${fcst_range}\&all_lev=on\&all_var=on\&subregion=\&leftlon=${leftlon}\&rightlon=${rightlon}\&toplat=${toplat}\&bottomlat=${bottomlat}\&dir=%2Fgfs.${ccyymmdd_s}%2F${fcst_start}%2Fatmos"
            elif [ $resolution == 0.5 ];then
                filename="filter_gfs_0p50.pl?file=gfs.t${fcst_start}z.pgrb2full.0p50.f${fcst_range}\&all_lev=on\&all_var=on\&subregion=\&leftlon=${leftlon}\&rightlon=${rightlon}\&toplat=${toplat}\&bottomlat=${bottomlat}\&dir=%2Fgfs.${ccyymmdd_s}%2F${fcst_start}%2Fatmos"
            fi
            data=${source}${filename}
            echo axel -n $thread $data -o ${data_dir}/china_gfs_${resolution}_${ccyymmdd_s}_${fcst_start}00_${fcst_range}.grb2 >> download_sub.sh
        done
    done
	cyc_start=$(date -d "${ccyymmdd_s} +${hh_s}hours +${INTERVAL}hours"  "+%Y%m%d%H" )
	ccyymmdd_s=`echo $cyc_start|cut -c1-8`
done
###在这个脚本里面直接执行axel会报错 不知道为啥
chmod 740 download_sub.sh
##然后手动 ./download_sub.sh
