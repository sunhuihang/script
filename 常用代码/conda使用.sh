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
或 pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118   #12.2版本的cuda可以装新版本pytorch
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



1.安装 conda-pack
2.conda pack -j 8 -n wrf -o /mnt/d/work/wrf.tar.gz #用8线程 把名为'wrf'的环境打包到 /mnt/d/work/wrf.tar.gz
3.把wrf.tar.gz 拷贝解压到新环境的anaconda3/envs/wrf 下,即可


#conda 环境克隆
conda create -n B --clone A #生成一个B环境，为A的克隆
#移除环境
conda env remove -n nowcast_test
######## 在离线环境中，多安装一个库
1.在有网络的环境中，新建一个文件夹new，并进入
2.pip download scikit-image -d ./  #把库及需求的依赖库下载到当前目录
3.sshpass -p sh123456 scp -P 10030 ./* Atmos@124.128.14.90:/data/Atmos/sunhh/cycling/run/download #把刚才下载的文件 全部拷贝到new中
4.在离线环境中，进入new文件夹 pip install scikit-image --no-index -f ./  安装完毕

######## 安装xesmf和esmpy（esmpy是xesmf的库，且二者必须从同一个源安装） 
1.~/.condarc中不能同时存在 default和 conda-forge，删除default后，用conda install xesmf=0.7.1 -c conda-forge 安装

######## 移植带有esmpy的库，要修改esmf.mk 中的路径
1.vim anaconda3/envs/环境名/lib/esmf.mk 

######## 解决jupyter 无环境问题
方法1
conda install ipykernel
python -m ipykernel install --user --name sunhh  #把sunhh环境加入可选项

方法2
vscode中 按 ctl+shift+p 搜索 setting.jspn,就是"打开工作区设置"
打开setting.json
改这个 "python.defaultInterpreterPath": "/home/qixiang/anaconda3/envs/sunhh" 
###########################################################################
