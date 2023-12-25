###
cdo 不好使的，就试试nco, cdo处理不了的 nco很多情况下可以处理
nco 作为cdo的补充和后备使用
###
#把其他格式的数据转为 nc格式
cdo -f nc copy infile outfile.nc

#文件合并
cdo cat infile1 infile2 outfile
#出现NetCDF: Numeric conversion not representable错误时，使用以下命令
cdo -b F64 mergetime infile1 infile2 outfile


#提取文件中的某个变量,可以提取多个变量及层次(selxxx,xxx系列命令，与select,xxx=xxx系列代码效果一样)
cdo select,name=t,u,v infile outfile

#将数据按年月分开(cdo版本要高于1.9.0) ，#splityear splitmon等等都有
cdo splityearmon infile

# 求月平均,自动判断时间
cdo monmean in out

# 集合平均(#多命令同时使用时发现会有问题，例如cdo -f nc copy -ensemean -selmon,5,6,7 infile*.grib out.nc,输出的不是ensmean，而是多成员cat结果，不止为何)
cdo ensmean infile1 infile2 outfile

# 选取层次 变量 ,提取25-900的变量T  （在25-900hpa之间的都提取）
cdo -sellevel,25/900 -selname,T infile outfile

# 将数据var.nc插值到var2.nc的网格上生成数据var3.nc
cdo griddes var2.nc > var2_grid.txt
cdo remapbil,var2_grid.txt var.nc var3.nc

#选变量 插值到360*180网格
cdo remapbil,r360x180 -selvar,sst HadISST_sst.nc sst.nc


#求气候态
#先看文件的时间格式是1985-01还是1985-01-01
cdo showdate infile
#以1985-01-01为例
cdo ymonmean -seldate,1985-01-01,2014-12-31 infile outfile

#算anomaly，不会输出气候态文件
cdo -ymonsub NOAA_sst.nc -ymonmean -seldate,1985-01-01,2014-12-31 NOAA_sst.nc anomaly.nc
#先-ymonmean -seldate,1985-01-01,2014-12-31 NOAA_sst.nc 为计算气候态，然后用 -ymonsub NOAA_sst.nc减去刚算的气候态，得到anomaly文件

#算逐月数据 减去 季节平均
cdo -yseassub 161_tsurf.nc 1d_198204_01.nc 198201_01_anomaly.nc

#例6：对于NCEP的风场数据，筛选1980-2010年中国地区(10-55N, 70-135E)的夏季850hPa的风场。
#分析：这个问题涉及到多步的操作，可以一次性的在cdo里面执行，但是注意两点：①设计到多步操作时，每个函数名称前面加上“-”；②不同函数名称中间用空格间隔。
cdo -selmon,6,7,8 -seldate,1981-01-01,2010-12-01 -sellonlatbox,70,135,10,55 uwnd.mon.mean.nc uwnd.


#求日最大值
cdo daymax infile outfile
#用逐小时资料（时间分辨率小于等于天），求月均日最高气温 数据的气候态
cdo -ymonmean -monmean -daymax infile(mergetime起来的逐小时数据) outfile 

#将所有输入场加入一个常数-273.15：
cdo -addc,-273.15 infile outfile 

#计算所有输入场的纬向平均：
cdo zonmean infile outfile

#两个数据相减 
cdo sub file1 file2 outfile 
#加减乘除 分别为 add sub mul div

#多个文件相加，不能用add ，用enssum
cdo enssum file1 file2 file3 outfile 

#求两个文件 每个时间每个点的最大最小，即比较两个文件数值的大小
cdo min(max) file1 file2 outfile 

#求两个文件的正切arc tangent
cdo atan2 file1 file2 outfile 

#裁取指定范围 82.5,154.3,18.75,54.5 -> lon.min(),lon.max(),lat.min(),lat.max()
cdo sellonlatbox,82.5,154.3,18.75,54.5 global.nc china.nc

#强制覆盖
cdo -O

###
cdo seldate/selmon/selyear,date input.nc output.nc
#选取特定时间范围的数据
e.g. cdo seldate,1959-01-01,2023-03-01 in.nc cut.nc
e.g. cdo -selmon,1,2,12 -selyear,2010/2012 in.nc cut.nc 

cdo selname/sellevel,name  input.nc output.nc
#选取特定的变量和高度场的信息
e.g.cdo -selname,sf in.nc cut.nc

cdo sellonlatbox,lonmin,lonmax,latmin,latmax input.nc output.nc
#选取特定经纬度的信息
e.g. cdo -sellonlatbox,70,135,10,55 in.nc cut.nc


#获取mask 后的数据 
1. 先获取一个mask文件（把不需要的网格数据设置为nodat）,
	用GDAL，其思路为：
	栅格化shp文件得到栅格文件；
	把上面的文件作为掩膜与全球数据做条件运算。
	栅格化shp文件
	我们可以在命令行很方便的用gdal_rasterize命令对shp文件进行栅格化：

gdal_rasterize -of netCDF -burn 1 -tr 0.01 0.01 china.shp china_mask.nc

2. cdo ifthen china_mask.nc global.nc china.nc


#选取指定月份数据做平均（其他同理）
cdo -yearmean -selmon,5,6,7 infile.nc outfile.nc
#季节平均(可添加选项 -select,season=DJF 指定季节月份)
cdo -seasmean infile.nc outfile.nc

#cdo时间步长插值,start_date格式为2021-11-11, start_time格式为15:00:00,step (second, minute, hour, day, month, year)
cdo inttime,start_date,start_time,step infile outfile
cdo inttime,2021-11-11,15:00:00,5minute test_cat.grib2 test_inttime2.grib2
#cdo把文件插值成n个时次（intntime 与上面的inttime不同）
cdo intntime,n infile outfile

#压缩存储 -z ：szip 按grib1的压缩，jpeg按照grib2的压缩（压缩最狠）
cdo -z jpeg -inttime,2021-11-11,15:00:00,5minute test_cat.grib2 test_inttime2.grib2
