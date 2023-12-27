#修改文件时间，时间 自加 1800  （变量名看文件，也可能叫time，与文件一致，这里是秒也可能是小时等）
ncap2 -s 'time_counter+=1800'  198205time.nc
#生成行文件198205time2，修改新文件时间，时间 自加 1800  （单位与文件一致，这里是秒）
ncap2 -s 'time_counter+=1800'  198205time.nc 198205time2.nc 

#修改变量属性,把lat的units改为 "degrees_north"
ncatted -O -a units,lat,o,c,"degrees_north" sst_ORCA1-360x180_bilin.nc

#求多个文件的集合平均
nces file1.nc file2.nc file3.nc fileout.nc
#把多个文件合并起来，跟cdo的cat一样（沿时间合并）
ncrcat file1.nc file2.nc file3.nc fileout.nc

#连接多个文件并生成新的维度
ncecat *nc -O merged.nc
#重命名新的维度名 为time
ncrename -d record,time merged.nc
#删除维度
ncwa -a dim_name input.nc output.nc

#提取in.nc文件中 depth的 0层，输出到out.nc
ncks -d deptht,0 in.nc out.nc

#提取in.nc文件中 时次624-635的数据，输出到out.nc
ncks -d time,624,635 in.nc out.nc

#把文件中的temper小于-30大于50的设为缺测(温度盐度范围都在-30到50之间)，缺测设为-1e10，输出到out文件里
ncap2 -s 'temper=temper;temper.change_miss(-1e10);where(temper<-30 || temper>50) temper=temper.get_miss()' temper_ORCA2-360x180_bilin.nc 198203new.nc

#两个文件相减(加减乘除 对应 +-*/) , 1.nc - 2.nc
ncbo --op_typ=- 1.nc 2.nc diff.nc

#去掉一个文件里的某个变量
ncks -x -v nav_lat in.nc out.nc

#从文件中提取某个变量
ncks -v nav_lat in.nc out.nc

#从WRFOUT中提取常用要素
varlist="XLONG,XLAT,XLONG_U,XLAT_U,XLONG_V,XLAT_V,XTIME,Times,T,T2,P,PB,PH,PHB,PSFC,Q2,QVAPOR,U10,U,V10,V,HGT,RAINC,RAINNC,RAINSH,PBLH"
ncks -O -v  ${varlist}  wrfout_d01*  wrfout_new.nc

#用in文件里的变量U  overwrite out文件中的同名变量（维度一致）
ncks -A -v U in.nc out.nc

#提取经纬度范围数据,范围一定要加小数点，只写整型 相当于是idx,注意 写lat,lon还是latitude,longitude根据数据里的命名来
ncea -O -d longitude,104.0,125.0 -d latitude,31.0,45.0 input.nc output.nc


#提取区域范围内的某些变量
varlist="U,V,Hour"
lonbox="104.0,125.0" #要有小数点
latbox="31.0,45.0" #要有小数点
ncea -O -v ${varlist} -d longitude,${lonbox} -d latitude,${latbox} input.nc output.nc
