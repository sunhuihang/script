#yum添加额外的源
yum install -y epel-release
#安装系统用的动态库，要不然报错 libtinfo.so.5: cannot open shared object file
yum install libtiff

#yum换源
cd /etc/yum.repos.d/
mv CentOS-Base.repo CentOS-Base.repo.backup
wget http://mirrors.aliyun.com/repo/Centos-7.repo #阿里源
mv Centos-7.repo CentOS-Base.repo 
或者 wget http://mirrors.163.com/.help/CentOS7-Base-163.repo #网易源
    mv CentOS7-Base-163.repo CentOS-Base.repo
yum makecache
yum -y update


#先更新conda ，后面安装才顺利
conda update -n base -c defaults conda

需要安装的库
conda install -c conda-forge matplotlib cartopy gdal netcdf4 eccodes xarray pandas cfgrib h5py hdf4 hdf5 jupyter onnx pillow(PIL) scipy xesmf scikit-learn scikit-image pytorch-lightning tensorboard tqdm torchinfo einops timm pathlib pathlib2
pynvml tensorbord
pip install -U 'jsonargparse[signatures]>=4.26.1 #只能pip安装
pip install apache-airflow[kubernetes]
pickle是python自带 不用装 ，PIL的库名叫pillow
