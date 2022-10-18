cv_list="3 5 7"
home=/data/Atmos/sunhh/cycling
save_dir=${home}/process
# leadtime=`seq -w 06 12 48`
leadtime="03 06 12 24 48"
######## 设置
cyc_start=2022070700
cyc_end=2022073100
###预报间隔 单位h
INTERVAL=24


ccyymmdd_s=`echo $cyc_start|cut -c1-8`
ccyymmdd_e=`echo $cyc_end|cut -c1-8`
hh_s=`echo $cyc_start|cut -c9-10`

while [ $cyc_start -le $cyc_end ];do
	
	for lead in ${leadtime};do
		lead_dir=${home}/process/lead${lead}
		if [ ! -d ${lead_dir} ];then
			mkdir -p ${lead_dir}
		fi
		##### date -d 命令只读到年月日，不读小时，默认为00时； 在此加上截断的启动时间和预报时效
		DATE=`date -d "${ccyymmdd_s} +${hh_s}hours +${lead}hours" "+%Y-%m-%d_%H:00:00"`
		

		##########  根据leadtime进行分类 然后链接
		for cv in ${cv_list};do
		    if [ $cv == 3 ];then
		    	file_dir=$home/run/cv${cv}
		    	link_dir=${lead_dir}/cv${cv}
		    	# echo ${file_dir}/${cyc_start}/wrfout_d01_${DATE},${link_dir}
		    	if [ ! -d ${link_dir} ];then mkdir -p ${link_dir};fi
		    	ln -sf ${file_dir}/${cyc_start}/wrfout_d01_${DATE} ${link_dir}


		    elif [ $cv == 5 ];then
		        be_list="3 5"
		    	for be in $be_list;do
		    		file_dir=$home/run/be${be}_cv${cv}
		    		link_dir=${lead_dir}/be${be}_cv${cv}
		    		if [ ! -d ${link_dir} ];then mkdir -p ${link_dir};fi
		    		ln -sf ${file_dir}/${cyc_start}/wrf/wrfout_d01_${DATE} ${link_dir} 

		    		cd ${link_dir}
		    		# ncrcat wrfout_d01* wrfout_be${be}_cv${cv}_lead${lead}.nc
		    	done


		    elif [ $cv == 7 ];then
		    	be_list="2"
		    	for be in $be_list;do
		    		file_dir=$home/run/be${be}_cv${cv}
		    		link_dir=${lead_dir}/be${be}_cv${cv}
		    		if [ ! -d ${link_dir} ];then mkdir -p ${link_dir};fi
		    		ln -sf ${file_dir}/${cyc_start}/wrf/wrfout_d01_${DATE} ${link_dir} 

		    		cd ${link_dir}
		    		# ncrcat wrfout_d01* wrfout_be${be}_cv${cv}_lead${lead}.nc
		    	done
		    fi
		done
	done


	cyc_start=$(date -d "${ccyymmdd_s} +${hh_s}hours +${INTERVAL}hours"  "+%Y%m%d%H" )
	ccyymmdd_s=`echo $cyc_start|cut -c1-8`
done


##########  利用ncrcat对不同leadtime的链接进行合并
for lead in ${leadtime};do
	lead_dir=${home}/process/lead${lead}
	for cv in ${cv_list};do
	    if [ $cv == 3 ];then
	    	link_dir=${lead_dir}/cv${cv}
			cd ${link_dir}
			echo ${link_dir}
			ls wrfout_d01*
	    	ncrcat wrfout_d01* ${save_dir}/wrfout_cv3_lead${lead}.nc


	    elif [ $cv == 5 ];then
	        be_list="3 5"
	    	for be in $be_list;do
	    		link_dir=${lead_dir}/be${be}_cv${cv}
	    		cd ${link_dir}
	    		ncrcat wrfout_d01* ${save_dir}/wrfout_cv${cv}_be${be}_lead${lead}.nc
	    	done


	    elif [ $cv == 7 ];then
	    	be_list="2"
	    	for be in $be_list;do
	    		link_dir=${lead_dir}/be${be}_cv${cv}
	    		cd ${link_dir}
	    		ncrcat wrfout_d01* ${save_dir}/wrfout_cv${cv}_be${be}_lead${lead}.nc
	    	done
	    fi
	done
done
