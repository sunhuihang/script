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
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia #nvidia 11.4的cuda驱动版本可以装11.8的cuda-pytorch
conda install -c conda-forge matplotlib cartopy gdal netcdf4 eccodes xarray rioxarray pandas cfgrib h5py hdf4 hdf5 jupyter onnx pillow scipy xesmf scikit-learn scikit-image pytorch-lightning tqdm torchinfo einops timm pathlib pathlib2 pynvml imageio
conda install -c conda-forge tensorboard #这个要单独装
pip install -U 'jsonargparse[signatures]>=4.26.1' #只能pip安装
pip install apache-airflow[kubernetes]
pip install opencv-python
#pickle是python自带 不用装 ，PIL的安装库名叫pillow, cv2的安装库名为opencv-python
pip install frykit==0.4.2post1 #用来画中国地图
pip install py3nvml  #用py3smi 替代 nvidia-smi，即使在docker中都可以显示pid
pip install -U mplfonts #中文
mplfonts init
#安装自己下载的字体
https://blog.csdn.net/HLBoy_happy/article/details/131667829
#克隆一个环境，创建一个B环境，是A的克隆
conda create -n B --clone A

#使用清华源 下载
conda install numpy -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main


#根据environment.yml 创建环境
conda env create -f environment.yml -n newenv
