#! /bin/ksh
#-----------------------------------------------------------------------
# Script gen_be_plot_wrapper.ksh
#
# Purpose: Plot the background error statistics (BES) for WRF-Var
# 
#   After finished the BES, this script gives the BES plots
#   by using ncl shell under var/graphics/ncl/gen_be.
#
#                             Y.-R. Guo  06/16/2008
#-----------------------------------------------------------------------
#export NCARG_ROOT=/contrib

#[1] Define job by overriding default environment variables:

export WRFVAR_DIR=/data/Atmos/sunhh/WRFDA
export GEN_BE_PLOT=${WRFVAR_DIR}/var/graphics/ncl/gen_be

export GRAPHIC_WORKS=png

export NUM_WE=141 # 1 point less than stagger points  90
export NUM_SN=115 # 1 point less than stagger points  60
export NUM_LEVELS=42 # 1 point less than stagger points 41
export RESOLUTION_KM=12.0  # km
export REGION=tutorial     # only for naming output purpose
export BE_DIR=${WRFVAR_DIR}/nmc/gen_be5_cv5/working
export BE_NROW=$NUM_LEVELS

#[2] Plot gen_be
#
# 1, First five eigenvectors:
#----------------------------
ncl ${GEN_BE_PLOT}/gen_be_global_evecs.ncl 
#
# 2, First five eigenvaluess:
#----------------------------
ncl ${GEN_BE_PLOT}/gen_be_global_evals.ncl 
#
# 3, Lengthscales:
#----------------------------
ncl ${GEN_BE_PLOT}/gen_be_lengthscales.ncl 
#
# 4, Correlation: chi_b.chi ans t_b.t
#-----------------------------------
ncl ${GEN_BE_PLOT}/gen_be_corr_z.ncl  
#
# 5, Correlation yz coross-section: <chi_b.chi> ans <t_b.t>
#---------------------------------------------------------
ncl ${GEN_BE_PLOT}/gen_be_corr_yz.ncl 
#
# 6, Correlation:  <ps_b.ps>
#--------------------------
ncl ${GEN_BE_PLOT}/gen_be_corr_ps.ncl 
#
