run2 是早期使用的脚本，备份一下

forecast中为利用GFS预报数据驱动wrf进行循环预报的相关脚本
    #obsproc 为观测数据的处理，用于同化或者分析
        #station_data_process 将下载的国家站观测数据 txt文件，进行整理，处理为规整的csv
        #station_to_Little_r 将处理好的观测csv 处理成Little_r格式，用于WRFDA的obsproc
        #Little_r_to_ascii 运行WRFDA的obsproc程序并画图（需要WRFDA及ncl），用run_cycle.sh调用files中的文件完成
        #obs_concat 从处理好的观测csv文件中 挑选自己需要的并合并成一个csv

    #gen_be 制作新的背景误差 be.dat文件，run_gen.sh 调用gen_be_wrapper.ksh，批量生成多个be.dat，
    gen_be_plot_wrapper.ksh用于画图分析 （需要WRFDA和ncl）

    #run_cycle 循环预报
        #noda 进行无同化的循环预报
        #withda 循环同化预报
            run_cycle_becv.sh 利用不同的背景误差文件，调用run_wrfda.sh 和 run_wrf_from_wrfda.csh进行循环同化预报
            run_wrfda.sh 单时次同化 并订正边界场
            run_wrf_from_wrfda.csh 单时次利用同化后的初始场运行WRF
    
    #postproc 后处理预报数据，方便分析
        lead_concat.sh 分别将指定leadtime的wrfout 合并
        wrfout转csv.ipynb 把wrfout插值到站点，并输出成csv

download 下载数据的脚本
    #download_realtime_gfs 只能下载最近一周的gfs，脚本设置了经纬度范围 只下载中国区域
    网址 https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p50.pl
    #download_Sounding 下载探空数据 
    网址 http://weather.uwyo.edu/upperair/bufrraob.shtml
