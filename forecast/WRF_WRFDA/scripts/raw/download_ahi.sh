user=978982001_qq.com   #从20221214日起，由葵花8变为葵花9
password=SP+wari8       #ahi数据延迟30min
thread=8
initime=$1
if [ -z "$initime" ];then
    echo "please input initime"
    exit
fi

ccyymmdd=`echo $initime|cut -c 1-8`
ccyy=`echo $initime|cut -c 1-4`
mm=`echo $initime|cut -c 5-6`
dd=`echo $initime|cut -c 7-8`
hh=`echo $initime|cut -c 9-10`
#min="00"
min=`echo $initime|cut -c 11-12`
tmp_dir=/home/qixiang/SHARE/tmp
data_dir=/home/qixiang/SHARE/ahi/${ccyy}.${mm}/${ccyymmdd}${hh}   # 数据存放目录
test -d ${data_dir} || mkdir -p ${data_dir}
cd $data_dir
if [ $ccyymmdd -ge 20221214 ];then
    #filename="NC_H09_${ccyy}${mm}${dd}_${hh}${min}_R21_FLDK.02401_02401.nc"
    filename="NC_H09_${ccyy}${mm}${dd}_${hh}${min}_R21_FLDK.06001_06001.nc"
    URL="ftp://ftp.ptree.jaxa.jp/jma/netcdf/${ccyy}${mm}/${dd}/${filename}"
    wget -c --ftp-user=${user} --ftp-password=${password} ${URL} -O ${tmp_dir}/${filename}
    #URL="ftp.ptree.jaxa.jp/jma/netcdf/${ccyy}${mm}/${dd}/${filename}"
    #axel -n 6 -o ${data_dir}/${filename} "ftp://${user}:${password}@${URL}"
else
    #filename="NC_H08_${ccyy}${mm}${dd}_${hh}${min}_R21_FLDK.02401_02401.nc"
    filename="NC_H08_${ccyy}${mm}${dd}_${hh}${min}_R21_FLDK.06001_06001.nc"
    URL="ftp://ftp.ptree.jaxa.jp/jma/netcdf/${ccyy}${mm}/${dd}/${filename}"
    wget -c --ftp-user=${user} --ftp-password=${password} ${URL} -O ${tmp_dir}/${filename}
    #URL="ftp.ptree.jaxa.jp/jma/netcdf/${ccyy}${mm}/${dd}/${filename}"
    #axel -n 6 -o ${data_dir}/${filename} "ftp://${user}:${password}@${URL}"

fi
#tmp_file=${tmp_dir}/${filename}
#targ_file=${data_dir}/${filename}
#varlist="tbb_07,tbb_08,tbb_09,tbb_10,tbb_11,tbb_12,tbb_13,tbb_14,tbb_15,tbb_16,Hour"
#lonlatbox="104,125,31,45"
#cdo -selname,${varlist} -sellonlatbox,${lonlatbox} ${tmp_file} ${targ_file}
#rm ${tmp_file}
#wget -c --ftp-user=${user} --ftp-password=${password} ${URL} -O ${data_dir}/
