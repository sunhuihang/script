cv_list="7"
bin_list=`seq 1 6`
dir=`pwd`
for cv in $cv_list;do
    for bin in $bin_list;do
        cd $dir 
        sed -i "22c export NL_CV_OPTIONS=${cv}" gen_be_wrapper.ksh
        sed -i "27c export BIN_TYPE=${bin}" gen_be_wrapper.ksh
        ./gen_be_wrapper.ksh
        cd gen_be${bin}_cv${cv}
        mkdir diag 
        cd diag
        cp ../../gen_be_plot_wrapper.ksh .
        chmod 740 gen_be_plot_wrapper.ksh
        sed -i "26c export BE_DIR=\${WRFVAR_DIR}/nmc/gen_be${bin}_cv${cv}/working" gen_be_plot_wrapper.ksh
        ./gen_be_plot_wrapper.ksh
    done
done
