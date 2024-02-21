############################## 导入自定义库 ##################################
import sys 
sys.path.append('./model')	#把model写入系统路径，后面导入model内的py文件才没问题
from test import * 	#test.py是model下的文件

############################### os ####################################
import os
os.getcwd() #显示当前路径
os.listdir('./model') #显示目录中的文件
os.path.exists('./model') #检查./model是否存在，返回False、True
dir='test/dir1'
file='abc.sh'
os.path.join(dir,file) #返回字符串'test/dir1/abc.sh'，且可以同时输入多个字符串，会自动按顺序生成路径字符串
os.path.basename(path) $返回最后一个'/'后的文件名字符串
os.mkdir('test', mode=0o777) #生成单层目录 test，可选权限设置为777 ，没有exist_ok选项，建议都用os.makedirs
os.makedirs('test/a/b', mode=0o777,exist_ok=True) #可以生成多级目录，可选exist_ok=True 即使目录存在也不会报错，一直用这个就可以了

os.system('cp a.sh b.sh') #调用linux的系统命令

############################### glob ####################################
#python中使用通配符较为少见，在shell中很常见，通过glob可以使用通配符匹配文件文，并返回列表
from glob import glob
file_list=glob('../csv/2020*.csv')

############################### pathlib.path ####################################



############################### list 列表 ####################################
#list去重
a = [11,1,3,5,3,8,9,1,9]
#方法一：可以保持原始顺序
	b = []
	for id in a:
	    if id not in b:
	        b.append(id)

#方法二：不保持原始顺序
	b = list(set(a))
	从大到小排序
	b = sorted(list(set(a)))

	b=np.unique(a) 返回的不是列表 是个数组 #去除其中重复的元素，并按元素由大到小返回一个新的数组

#删除列表中的某个元素：
#删除列表最后一个元素
name = names.pop(） 
#删除倒数第2个元素
name = names.pop(-2）

#根据元素值进行删除
names = ['张三', '李四', '王五', '赵六', '麦叔', '小麦']
#删掉赵六
n = names.remove('赵六')

#使用in关键词判断王八是否在列表中
if '王五' in names:
    n = names.remove('王五')
    print(n)
###########################################################################


############################## tuple 元组 #################################


############################## json ##############################
json.load()从json文件中读取数据
json.loads()将str类型的数据转换为dict类型
json.dumps()将dict类型的数据转成str
json.dump()将数据以json的数据类型写入文件中

写json
with open('text.json','w',encoding='utf-8') as f :
    json.dump('test',f)

读取json
with open('text.json','r',encoding='utf-8') as f :
    json.load(f)

###########################################################################

############################### pandas ####################################
#跳过前几行读取csv
f = pd.read_csv(file_path,skiprows=2) #跳过前两行
 
#list、array、dict转pd.Dataframe
a = ['20230506','20230607']
df = pd.DataFrame(a,columns=['time']) #必须要有[]
a=np.arange(3)
df = pd.DataFrame(a,columns=['time'])
a={'time':['20230506','20230607']}
df = pd.DataFrame(a)
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

##根据文件名中的时间相关字符串，转成df
test_radar_dir = '/home/qixiang/liucs/HuaBei/dataset/HuaBei_map/'
radar_files = glob(os.path.join(test_radar_dir,'*.png'))
radar_time = pd.to_datetime(pd.Series([f.split('/')[-1] for f in radar_files ]),format='%Y%m%d%H%M%S.png') #生成一个时间构成的df
radar_files = np.array(radar_files)[np.argsort(radar_time)] #把文件名按时间排序
radar_time = radar_time.sort_values().reset_index(drop=True)    #去重
radar_files = pd.DataFrame(dict(file=radar_files,time= radar_time)) #生成一个包含 radar_files和time的df


# 快速修改df中数据的类型
df.time = df.time.astype('str')
#给df添加一列或者一行
df['C'] = [11,12,13]	#添加名为'C'的列，如果已经存在就覆盖（注意写的长度要跟其他列一样）
df.loc[10] = [11,12,13] #添加index为10的行，如果已经存在就覆盖（注意写的长度要跟其他行一样）
#合并df
pd.merge(df_left,df_right,how='inner',on='Station_Id_C') #how:inner/outer/left/right,应用多个列 on=['a','b']
pd.concat([left,right])  #常用参数,axis=0,join='inner',当axis=1时与pd.merge类似,pd.concat 用join而不用how 
#根据其中一列去重
# 保留每个'time'的第一个行，删除后续重复的行
df.drop_duplicates(subset='time', keep='first', inplace=True)

#去除重复的列
duplicated_columns = df.columns[df.columns.duplicated()]
	# 打印重复的列名
	print("重复的列名:", duplicated_columns,len(duplicated_columns))
df = df.loc[:, ~df.columns.duplicated()]

#重命名df中的列名
student_df.rename(
    columns={"ID": "Student_ID", "name": "First_Name"},
    inplace=True）
	
#合并一个有df构成的列表
from functools import reduce
from glob import glob
import pandas as pd
files = sorted(glob('*csv'))
df_list = [pd.read_csv(f) for f in files]
df_all = reduce(lambda left,right: pd.concat([left,right]),df_list) #上下合并
df_all = reduce(lambda left,right: pd.merge(left,right,how='inner'),df_list) #左右合并


#时间处理
#np.datetime64计算, np.timedelat64 和 dt.timedelta不能处理年和月，要转成datetime 用 relativedelta 
np.timedelta64(1,'D') # 1天， 'D'为天 'H'为小时 'm'为分钟 's'为分钟
from dateutil.relativedelta import relativedelta
from datetime import datetime
datetime(2016, 2, 29) + relativedelta(months=1)  -> datetime.datetime(2016, 3, 29, 0, 0)   #relativedelta 可以使用 years、months、days、hours等，加's'说明是加减运算
datetime(2016, 2, 29) + relativedelta(month=1)  -> datetime.datetime(2016, 1, 29, 0, 0)    # 不加's' ，就是直接把对应的时间修改成设置的值

#np.datetime64 与 datetime.datetime互相转换
dt = datetime.datetime(year=2020, month=6, day=1, hour=20, minute=5, second=30) 
dt64 = np.datetime64(dt, 's') 
print(dt64, dt64.dtype) # 2020-06-01T20:05:30 datetime64[s] 
dt2 = dt64.astype(datetime.datetime) 
print(dt2, type(dt2)) # 2020-06-01 20:05:30 <class 'datetime.datetime'>

#把np.datetime64、数值或字符串转为timestamp ，即datetime64
format:
%Y:四位年份；%y:两位年份 (尽量写4位)
%m:2位月份；%d:2位日
%H:2位时；%M:2位分钟
%S:2位秒 
pd.to_datetime(2023040500)
pd.to_datetime(2023040500,format='%Y%m%d%H')
pd.to_datetime('2023040500',format='%Y%m%d%H')
都可以，但是当写成pd.to_datetime('2023040500') 会报错，
不能识别写到'时',且不写format的字符串
#格式化df中的时间格式
df = pd.DataFrame({'date': ['2019-6-10 20:30:0', 
                            '2020-7-1 19:45:30', 
                            '2021-10-12 4:5:1']})
df['date'] = pd.to_datetime(df['date'], format="%Y%d%m_%H%M%S")


#提取timestamp中的day和hour
t = pd.to_datetime('2023040500',format='%Y%m%d%H')
t.day #提取t的day （1-31）
t.hour #提取t的hour (0-23)
t.timetuple().tm_yday #提取t所在的日期在一年中的day (1-366)


#df转成csv文件保存,并且保存时不保存index
df.to_csv('df.csv',index=False)

##### datetime
#timestamp转为datetime.datetime
ts = pd.to_datetime('2023040500',format='%Y%m%d%H')
ts.to_pydatetime()

#把字符串时间按指定格式读取，转成datetime.datetime
from datetime import datetime,timedelta
time_datetime=datetime.strptime('20230405_0000','%Y%m%d_%H%M%S')
#把datetime.datetime转为指定格式字符串
time_str=datetime.strftime(time_datetime,'%Y%m%d%H')
#timedelta可以对datetime格式数据 进行加减
#可选项参见 class datetime.timedelta(days=0, seconds=0, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0)
df['date'] = df['date'] + timedelta(hours=6) 
# dt.timedelta 不能处理年和月，用relativedelta
from dateutil.relativedelta import relativedelta
from datetime import datetime
datetime(2016, 2, 29) + relativedelta(years=1)


pd.Series

#处理离散数据，one-hot编码
pd.get_dummies(all_features,dummy_na=True)
###########################################################################




############################### numpy ####################################
#如果numpy中存放了timestamp等类型，可能报错pickle相关内容(加载对应的变量时会显示ValueError: Object arrays cannot be loaded when allow_pickle=False)，添加allow_pickle=True
np.load(file,allow_pickle=True)

#numpy.random
#生成 与另一个数组大小相同 ，但全为0的数组
np.zeros((2,3)) #生成一个全是0的数组
np.ones((2,3))
np.linspace(1,5,num=5) #生成一个1-5平均取5个值的数组，与torch.linspace(1,5,steps=5)类似
#合并numpy数组
a=np.random.randn(2,3)
b=np.random.randn(2,2,3)
c = np.concatenate((a[None],b),axis=0) #指定维度合并

a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]]) 
c = np.append(a,b) #结果为： [1 2 3 4 5 6] #将二维数组变为了一维数组

#统计一个数组的元素个数
a.size
#统计一个数组中 非nan值的个数
np.count_nonzero(~np.isnan(a))   #先通过 ~np.isnan(a) 获取非nan的布尔索引，在统计非0值的个数（False为0，True为1）


#去除一组元素
nan_idx = np.isnan(prep_all)
prep_all = np.delete(prep_all,nan_idx) #prep_all是个一维数组，去除其中的nan
dbz_all = np.delete(dbz_all,nan_idx,axis=0) #dbz_all是第一维与prep_all维度一致的多维数据，在第一维中去除prep_all去除的index

#np.repeat与tensor的repeat不同
a=np.random.randn(2,2,3)
a.repeat(2,axis=1) 等同于 np.repeat(a,2,axis=1) #对第1维repeat2次，2个参数，第一个是重复次数，第二个是维度

#对比 tensor的repeat
a=torch.randn(2,2,3)
a.repeat(2,1,1) #对第1维repeat两次，第2、3维重复一次（也就是不变），有几个维度就要写几个参数

#裁剪数据范围
#与torch.clip(x,min=0)类似，torch可以只写最大或最小值，
#np.clip必须 把a_min 和 a_max都写上
x=np.random.randn(4,3,2)
x=np.clip(x,a_min=0,a_max=np.inf) 
#把一个由np数组组成的list，合并为一个数组，会新生成一维
a = [np.random.randn(2,3),np.random.randn(2,3),np.random.randn(2,3)]
b = np.stack(a,axis=0) # 指定axis 为新生成维度的位置，b.shape为（3,2,3）
b = np.stack(a,axis=1) # 指定axis 为新生成维度的位置，b.shape为（2,3,3）

#简单计算数组中的非nan数据， np.nanmax() np.nanmin() np.nanmean()
np.nanmax(a) 查看a中非nan的最大值，不可以写成a.nanmax()

#将a中的元素从小到大排列，提取其在排列前对应的index(索引)输出
a = np.array([1,4,3,-1,6,9])
b = np.argsort(a)



#二维数组 怎么按照阈值 进行索引
###只能mask得到布尔形状的数组，不符合的设置为nan
lon = np.linspace(120, 150, 120)
lat = np.linspace(0, 40, 40)
lon_2d, lat_2d = np.meshgrid(lon, lat)
mask = (lon_2d > 134) & (lat_2d > 20)
filtered_lon = np.where(mask, lon_2d, np.nan)
filtered_lat = np.where(mask, lat_2d, np.nan)

#2维的lon怎么取124-138之间的，2维的lat怎么取30-34之间的,截取新的lon和lat
lon = np.linspace(120, 150, 120)
lat = np.linspace(0, 40, 40)
lon_2d, lat_2d = np.meshgrid(lon, lat)
# !!必须要把lon_2d 和 lat_2d恢复成1维数组
#1. 已知lat和lon的一维数组，直接对lat lon截取
   lat = lat[(lat>30)&(lat<=34)]
   lon = lon[(lon>124)&(lon<=138)]
   lon_2d,lat_2d = np.meshgrid(lon,lat)
#2. 只知道lon_2d, lat_2d是等经纬网格的
   lon = lon_2d[0,:]
   lat = lat_2d[:,0]
   lat = lat[(lat>30)&(lat<=34)]
   lon = lon[(lon>124)&(lon<=138)]
   lon_2d,lat_2d = np.meshgrid(lon,lat)

#a对应标签数据b，剔除b中的缺测值以及对应的a
a = np.random.rand(100, 16, 16)
b = np.random.choice([1, 2, 3, np.nan], size=100) #b的形状为100
#找到不含NaN的索引
valid_indices = ~np.isnan(b)
#提取不含NaN的标签和对应的数据
valid_b = b[valid_indices]
valid_a = a[valid_indices]
print("不含NaN的标签形状:", valid_b.shape)
print("对应的数据形状:", valid_a.shape)



#读写npz文件（直接保存numpy数组到文件）
np.savez('val.npz', dbz=dbz_val, dem=dem_val[:,0,:,:], obs=obs_val)
f=np.load('val.npz') #list(f) 可以查看f中的keys
dbz = f['dbz']
#计算相关系数和rmse #只能计算2为数据，三维（time,lat,lon）的请看xskillscore
non_nan_mask = ~np.isnan(y) #先剔除缺测数据
x = x[non_nan_mask]
y = y[non_nan_mask]
r = np.corrcoef(x, y)[0,1]
rmse = np.sqrt(np.nanmean((x - y)**2))

#计算corr、rmse、mae、线性回归

threshold 阈值 : 
    mm/10min  0.1（小雨） 0.6（中雨） 1.6（大雨） 5（暴雨） 10（特大暴雨）
    mm/30min  0.1（小雨） 1  （中雨） 3  （大雨） 7（暴雨） 20（特大暴雨）
    mm/1h     0.1（小雨） 1.6（中雨） 7  （大雨）15（暴雨） 30（特大暴雨）
    mm/3h     0.1（小雨） 3  （中雨） 10 （大雨）20（暴雨） 50（特大暴雨）
    mm/6h     0.1（小雨） 4  （中雨） 13 （大雨）25（暴雨） 60（特大暴雨）
    mm/12h    0.1（小雨） 5  （中雨） 15 （大雨）30（暴雨） 70（特大暴雨）
    mm/天     0.1（小雨） 10 （中雨） 25 （大雨）50（暴雨）100（特大暴雨）
#计算 TS(CSI)、ETS、POD、FAR、MAR、BIAS、HSS、BSS、RMSE、MAE
def prep_clf(obs,pre, threshold=0.1):
    #根据阈值分类为 0, 1
    obs = np.where((obs >= threshold), 1, 0)
    pre = np.where((pre >= threshold), 1, 0)
    # True positive (TP)
    hits = np.sum((obs == 1) & (pre == 1))
    # False negative (FN)
    misses = np.sum((obs == 1) & (pre == 0))
    # False positive (FP)
    falsealarms = np.sum((obs == 0) & (pre == 1))
    # True negative (TN)
    correctnegatives = np.sum((obs == 0) & (pre == 0))

    return hits, misses, falsealarms, correctnegatives
  
def TS(obs, pre, threshold=0.1):
    
    '''
    TS评分 & CSI
    TS：风险评分ThreatScore;
    CSI: critical success index 临界成功指数;
    两者的物理概念完全一致。
    func: 计算TS评分: TS = hits/(hits + falsealarms + misses) 
    	  alias: TP/(TP+FP+FN)
    '''
    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, pre = pre, threshold=threshold)
    return hits/(hits + falsealarms + misses) 

def ETS(obs, pre, threshold=0.1):
    '''
    公平技巧评分(Equitable Threat Score, ETS)用于衡量对流尺度集合预报的预报效果。ETS评分表示在预报区域内满足某降水阈值的降水预报结果相对于满足同样降水阈值的随机预报的预报技巧；
    ETS评分是对TS评分的改进，能对空报或漏报进行惩罚，使评分相对后者更加公平
    Args:
        obs (numpy.ndarray): observations
        pre (numpy.ndarray): prediction
        threshold (float)  : threshold for rainfall values binaryzation
                             (rain/no rain)
    Returns:
        float: ETS value
    '''
    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, pre = pre,
                                                           threshold=threshold)
    num = (hits + falsealarms) * (hits + misses)
    den = hits + misses + falsealarms + correctnegatives
    Dr = num / den
    ETS = (hits - Dr) / (hits + misses + falsealarms - Dr)
    return ETS
       
def POD(obs, pre, threshold=0.1):
    '''
    Probability of Detection，POD命中率，即预测出的实际的降水区域占据全部实际降水区域的比重。
    POD = hits / y_obs_1 = hits / (hits + misses) = 1- MAR
    func : 计算命中率 hits / (hits + misses)
    pod - Probability of Detection
    Args:
        obs (numpy.ndarray): observations
        pre (numpy.ndarray): prediction
        threshold (float)  : threshold for rainfall values binaryzation
                             (rain/no rain)
    Returns:
        float: PDO value
    '''
    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, pre = pre,
                                                           threshold=threshold)
    return hits / (hits + misses)

def FAR(obs, pre, threshold=0.1):
    '''
    False Alarm Rate ，空报率FAR，在预报降水区域中实际没有降水的区域占总预报降水区域的比重。
    FAR = (y_pre_1 - hits)/y_pre_1 = falsealarms / (hits + falsealarms)
    func: 计算误警率。falsealarms / (hits + falsealarms) 
    FAR - false alarm rate
    Args:
        obs (numpy.ndarray): observations
        pre (numpy.ndarray): prediction
        threshold (float)  : threshold for rainfall values binaryzation
                             (rain/no rain)
    Returns:
        float: FAR value
    '''
    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, pre = pre,
                                                           threshold=threshold)
    return falsealarms / (hits + falsealarms)


def MAR(obs, pre, threshold=0.1):
    '''
    Missing Alarm Rate，漏报率实际降水区域中漏报的区域占据全部实际降水区域的比重。
    MAR = (y_obs_1 - hits)/y_obs_1 = misses / (hits + misses)			      
    func : 计算漏报率 misses / (hits + misses)
    MAR - Missing Alarm Rate
    Args:
        obs (numpy.ndarray): observations
        pre (numpy.ndarray): prediction
        threshold (float)  : threshold for rainfall values binaryzation
                             (rain/no rain)
    Returns:
        float: MAR value
    '''
    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, pre = pre,
                                                           threshold=threshold)

    return misses / (hits + misses)

def BIAS(obs, pre, threshold = 0.1):
    '''
    偏差评分(Bias score)主要用来衡量模式对某一量级降水的预报偏差, 该评分在数值上等于预报区域内满足某降水阈值的总格点数与对应实况降水总格点数的比值。用来反映降水总体预报效果的检验方法。
    Bias = y_pred_1/y_obs_1 = (hits + falsealarms)/(hits + misses)
    func: 计算Bias评分: Bias =  (hits + falsealarms)/(hits + misses) 
    	  alias: (TP + FP)/(TP + FN)
    inputs:
        obs: 观测值，即真实值；
        pre: 预测值；
        threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。
    returns:
        dtype: float
    '''    
    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, pre = pre,
                                                           threshold=threshold)

    return (hits + falsealarms) / (hits + misses)

				      
       
def HSS(obs, pre,threshold=0.1):
    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, pre = pre, threshold=threshold)

    HSS_num = 2 * (hits * correctnegatives - misses * falsealarms)
    HSS_den = (misses**2 + falsealarms**2 + 2*hits*correctnegatives +
               (misses + falsealarms)*(hits + correctnegatives))

    return HSS_num / HSS_den

def BSS(obs, pre, threshold=0.1):
    '''
    BSS - Brier skill score
    Args:
        obs (numpy.ndarray): observations
        pre (numpy.ndarray): prediction
        threshold (float)  : threshold for rainfall values binaryzation
                             (rain/no rain)
    Returns:
        float: BSS value
    '''
    obs = np.where(obs >= threshold, 1, 0)
    pre = np.where(pre >= threshold, 1, 0)

    obs = obs.flatten()
    pre = pre.flatten()

    return np.sqrt(np.mean((obs - pre) ** 2))

def MAE(obs, pre):
    """
    Mean absolute error
    Args:
        obs (numpy.ndarray): observations
        pre (numpy.ndarray): prediction
    Returns:
        float: mean absolute error between observed and simulated values
    """
    obs = pre.flatten()
    pre = pre.flatten()

    return np.mean(np.abs(pre - obs))

def RMSE(obs, pre):
    """
    Root mean squared error
    Args:
        obs (numpy.ndarray): observations
        pre (numpy.ndarray): prediction
    Returns:
        float: root mean squared error between observed and simulated values
    """
    obs = obs.flatten()
    pre = pre.flatten()

    return np.sqrt(np.mean((obs - pre) ** 2))


############################ xskillscore #################################
#计算三维（time,lat,lon）的计算相关系数和rmse ，两个数组必须是xr.DataArray
import xskillscore as xs
cc = xs.person_r(a,b,dim='time')
rmse = xs.rmse(a,b,dim='time')
rmse = xs.rmse(a,b,dim=['lat','lon'],skipna=True)
############################# pickl ######################################
#读写pkl文件,pickle中可以存储df、数组等多种格式数据
import pickle
with open("./scaler.pkl", "wb") as f:
        pickle.dump({
            "mean": np.array(Mean),
            "std": np.array(Std),
            "dem_mean": np.array(Mean_dem),
            "dem_std": np.array(Std_dem)
        }, f)

with open("./scaler_256.pkl", "rb") as f:
    pkl = pickle.load(f)
    dbz_mean = pkl["mean"]
    dbz_std = pkl["std"]
    dem_mean = pkl["dem_mean"]
    dem_std = pkl["dem_std"]


################################## torch ##################################
#相关的加载
import torch
from torch import nn
from torch.nn import functional as F

#释放当前脚本所占用的显存
torch.cuda.empty_cache()

#把数组转为tensor
a=np.random.rand(2,3)
b=torch.tensor(a,dtype=torch.float32)
c=torch.linspace(-1,1,steps=10) #生成一个-1到1 平均取10个数的tensor
#随机生成tensor
torch.rand 生成在区间 [0, 1) 内均匀分布的随机数
torch.randn 生成从标准正态分布（均值为0，标准差为1）中采样的随机数
a = torch.randn(1,2,3) 或 a = torch.randn((1,2,3)) 结果一样

#生成与另一个tensor大小相同，但全为0/1的tensor


#torch.normal()函数：返回一个张量；是从一个给定mean（均值），std（方差）的正态分布中抽取随机数。mean和std都是属于张量类型的
x=torch.normal(mean=1,std=2,size=(3,4))

#查看tesnsor的类型
a.dtype   #dtype不要()
a = a.to(dtype=torch.float32) #把a转为float32类型

#快速计算
a = torch.randn(1,2,3)
a.mean(dim=1) #对第1维求平均，shape为(1,3) ；max、min同理 ，且与numpy一样
a.mean(dim=1,keepdims=True) #保持原数组的维度，方便矩阵运算，shape为(1,1,3)

a=torch.clip(a,min=0,max=0.5) #与torch.clamp一样，把a截断到0-0.5，小于0的设为0，大于0.5的设为0.5，可以只写min或者max


#合并tensor
a = torch.randn(3,4)
b = torch.randn(3,4)
c = torch.cat([a,b],dim=0)  #dim指定在哪一纬度合并

#给一个tensor添加一维
b = torch.randn(3,4)
b = b[None]  或 b = b.unsqueeze(dim=0) #变成(1,3,4)

#去除tensor所有只有1的维度
a = torch.randn(1,1,3,4)
a = a.suqeeze() #去除所有只有1的维度,变成(3,4)

#repeat，合并多个tensor，成一个多通道tensor
a = torch.randn(2,3,4)
b = torch.randn(3,4)
b = b[None].repeat(2,1,1)
c = torch.cat([a,b],dim=0) 



#tensor读写
x = torch.arange(4)
torch.save(x, 'x-file')
x2 = torch.load('x-file')

#读写张量列表
y = torch.zeros(4)
torch.save([x,y],'x-files')
x2,y2 = torch.load('x-files') #返回的是包含两个tensor的元组（x2,y2）

#读写从字符串映射到张量的字典
mydict = {'x':x, 'y':y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict') #返回的是包含2个key和value的张量字典 {'x': tensor([0,1,2,3]),'y': tensor([0,0,0,0])}


#torch中device的设置
torch.device('cpu') , torch.device('cuda'), torch.device('cuda:1')
#查询可用gpu的数量
torch.cuda.device_count()
#查询张量所在的设备
x = torch.tensor([1,2,3])
x.device
#在指定设备创建张量
x = torch.tensor([1,2,3], device=torch.device('cuda:1'))
#把x移到'cuda:2'   
z = x.cuda(2)

#在gpu上运行模型
model = model.to(device=torch.device('cuda:1'))
x = x.to(device=torch.device('cuda:1'))
out = model(x)


#怎么把数据转成dataset，再转成dataloader 供模型使用
from torch.utils import data
dataset = data.TensorDataset(*data_arrays) # *表示把列表解开入参，即把列表元素分别当做参数入参
dataloader = data.DataLoader(dataset,batch_size,shuffle=False)  #shuffle为True则是打乱顺序，一般训练集需要打乱，验证和测试不需要

#怎么保存模型
#1.只保存模型的state_dict,推荐文件后缀名是pt或pth
torch.save(model.state_dict(), PATH)  
#加载
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH))	
model.eval()
# - 或者 -
model.train() #继续训练

#2.保存整个模型,不建议，不容易移植使用
torch.save(model, PATH)
加载
'''
模型类必须在别的地方定义，这种保存/加载模型的过程使用了最直观的语法，所用代码量少。
这使用Python的pickle保存所有模块。这种方法的缺点是，保存模型的时候，序列化的数据被绑定到了特定的类和确切的目录。
这是因为pickle不保存模型类本身，而是保存这个类的路径，并且在加载的时候会使用。
因此，当在其他项目里使用或者重构的时候，加载模型的时候会出错。
'''
model = torch.load(PATH)
model.eval()
# - 或者 -
model.train() #继续训练

#3.保存加载用于推理的常规Checkpoint/或继续训练
torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            ...
            }, PATH)
#加载
model = TheModelClass(*args, **kwargs)
optimizer = TheOptimizerClass(*args, **kwargs)

checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.eval()
# - or -
model.train()



#4.保存多个模型到一个文件
torch.save({
            'modelA_state_dict': modelA.state_dict(),
            'modelB_state_dict': modelB.state_dict(),
            'optimizerA_state_dict': optimizerA.state_dict(),
            'optimizerB_state_dict': optimizerB.state_dict(),
            ...
            }, PATH)

#加载
modelA = TheModelAClass(*args, **kwargs)
modelB = TheModelBClass(*args, **kwargs)
optimizerA = TheOptimizerAClass(*args, **kwargs)
optimizerB = TheOptimizerBClass(*args, **kwargs)
 
checkpoint = torch.load(PATH)
modelA.load_state_dict(checkpoint['modelA_state_dict'])
modelB.load_state_dict(checkpoint['modelB_state_dict'])
optimizerA.load_state_dict(checkpoint['optimizerA_state_dict'])
optimizerB.load_state_dict(checkpoint['optimizerB_state_dict'])
modelA.eval()
modelB.eval()

# 查看模型输入输出形状
model = ResUNet_plus(n_channels=11, n_classes = 2)
import torchinfo
torchinfo.summary(model, input_size=(3, 11, 512, 512))


#5.使用不同模型的参数来预训练模型
'''
加载部分模型是迁移学习或训练新的复杂模型时常见的场景。利用经过训练的参数，即使只有少数是可用的，将有助于热身开始训练过程，并有希望帮助您的模型比从零开始的训练更快地收敛。
无论是从缺少一些键的部分 state_dict 加载，还是在拥有更多键的state_dict加载。可以通过设置参数strict = False来忽略与模型不匹配的键。
如果您希望将参数从一个层加载到另一个层，但有些键不匹配，只需更改要加载的state_dict 中的参数键的名称，以匹配要加载到的模型中的键。
'''
# save
torch.save(modelA.state_dict(), PATH)
# load
modelB = TheModelBClass(*args, **kwargs)
modelB.load_state_dict(torch.load(PATH), strict=False)


#######单机多卡

'''
#两种方式：
torch.nn.DataParallel：早期PyTorch 的类，现在已经不推荐使用了,但是很简单；
	简单一行代码，包裹model即可：
		model = DataParallel(model.cuda(), device_ids=[0,1,2,3])
		data = data.cuda()

device = [torch.device('cuda:0'), torch.device('cuda:1')]
net = net.DataParallel(net, device_ids=devices)
trainer = torch.optim.SGD(net.parameters(), lr)
loss = nn.CrossEntropyLoss()
for epoch in range(num_epochs):
    net.train()
    for X, y in train_iter:
	trainer.zero_grad()
        X, y = X.to(devices[0]), y.to(device[0])
	l = loss(net(X), y)
	l.backward()
	trainer.step()

	
torch.nn.parallel.DistributedDataParallel：推荐使用
'''

1.保存并行训练torch.nn.DataParallel模型，这也是pytorch lightning训练及保存的ckpt 的方式
# save
torch.save(model.module.state_dict(), PATH) #注意是调用model.module 在使用多GPU训练并保存模型时，模型的参数名都带上了module前缀










 
#怎么导入模型权重进行推理
import sys 
sys.path.append('model/wjj_model')
#导入模型
from flow.flow_model import *	
#加载权重

#### pth 中只存有state_dict
from collections import OrderedDict
model = Evolution_Network(10, 10)
state_dict = torch.load('./model/wjj_model/flow/1f.pth')	#或使用state_dict = torch.load(PATH, map_location='cpu')，指定读取到cpu上
new_state_dict = OrderedDict()
for k, v in state_dict.items():	########
    name = k[7:]  # remove `module.` 在使用多GPU训练并保存模型时，模型的参数名都带上了module前缀，因此在加载模型时，把key中的这个前缀去掉
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)
model.eval() #推理时要用.eval(),固定模型dropout和批次归一化

#### ckpt（checkpoint）中存有epoch、lr、state_dict等信息，因此要提取其中的state_dict，要记得添加.items() 同时返回key和values，否则返回的是一个元组
sys.path.append('/home/qixiang/sunhh/BACKUP/QPE/2/model')
from UNet_attention import *
model_QPE = UNet_attention(16,11,2)
state_dict = torch.load('/home/qixiang/sunhh/BACKUP/QPE/2/pretrain.ckpt')
new_state_dict = OrderedDict()
for k, v in state_dict['state_dict'].items(): ########区别在这里，提取了key为 'state_dict'
    # 模型训练时 引入的可学习参数，在模型推理时不需要，因此跳过。这个参数要跳过的可能在最上面 也可能在中间，看报错信息 跳过就行
    if 'wp' in k:
        continue
    name = '.'.join(k.split('.')[1:]) #去掉module，跟name=k[7:]一样
    new_state_dict[name] = v
model_QPE.load_state_dict(new_state_dict,strict=True) #strict=True严格要求权重名称匹配，防止出错
model_QPE.eval()	#推理时要用.eval(),固定模型dropout和批次归一化


#用cpu推理(不管用不用gpu，都要写with torch.no_grad()： 来禁止计算梯度，否则算的很慢)
with torch.no_grad():
	prep,_,_,_ = model(torch.randn(3,10,256,452))

	input = np.random.randn(3,10,256,452)
	prep,_,_,_ = model(torch.tensor(input,dtype=torch.float32)) #把np转为tensor进行推理，注意最好写dtype=torch.float32

#用GPU推理
with torch.no_grad():
	device='cuda:1' #指定编号为1的显卡，在python中使用torch.cuda.is_available() 或在命令行中使用nvidia-smi查看显卡编号及使用情况
	a = torch.randn(3,10,256,452)
	#把数据和模型都要放到gpu上
	input = a.to(device) 
	model = model.to(device)	
	prep,_,_,_ = model(input) #把np转为tensor进行推理，注意最好写dtype=torch.float32
	prep = prep.cpu()	#把输出结果从gpu取到cpu中

#并行推理
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2' #指定使用的多张显卡编号
num_gpu = torch.cuda.device_count()
model = torch.nn.DataParallel(model).cuda()  #把模型变成并行的
dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs_per_gpu * num_gpu, **kwargs)   #把batch_size设置为原始的bs乘以gpu数量，推理时会自动分配给每张卡

import torch.distributed as dist
rank = dist.get_rank()
device = torch.device(f'cuda:{opt.local_rank}')  #推理时 把数据加载到这里

#模型格式ckpt、pth、oxxn等的区别


#自定义模型


#自定义层


#定义一个多个层 组成的一个块(多个卷积、ReLU堆叠，最后加一个MaxPool2d)
def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
	layers.append(nn.ReLU())
	in_channels = out_channels #下个循环的输入等于上个循环的输出
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)

#生成一个由5个vgg_block组成，最后加一个MLP的模型
con_arch = ((1,64), (1,128), (2,256), (2,512), (2,512))
def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
	in_channels = out_channels
    return nn.Sequential(
		         *conv_blks, nn.Flatten(),
		         nn.Linear(out_channels *7 *7, 4096), nn.ReLU(),
		         nn.Dropout(0.5), nn.Linear(4096, 4096), nn.ReLU(),
		         nn.Dropout(0.5), nn.Linear(4096,10)
			)
net = vgg(conv_arch)
   
#构建模型训练

#微调，迁移训练


### 把纯pytorch的模型训练 转为 pytorch lightning的代码





### 测试dataloader加载速度
data = MyDataModule(batch_size=64, num_workers=8, data_dir='/home/qixiang/SHARE/us_data', scale_dir=None)
train_dataloader = data.train_dataloader()
for i in tqdm(iter(train_dataloader)):
	pass

#K折教程验证
def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j*fold_size, (j + 1)*fold_size)
        X_part, y_part = X[idx,:], y[idx]
    if j == i:
        X_valid, y_valid = X_part, y_part
    elif X_train is None:
        X_train, y_train = X_part, y_part
    else:
        X_train = torch.cat([X_train, X_part], 0)
        y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid



#自定义一个模型层
class Conv2D(nn.Module):
    def __ini__(self, kernel_size):
        super().__ini__()
        self.weight = nn.Parameter(torch.rand(kernel_size)) #nn.Parameter 构造可学习的
        self.bias = nnp.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias #corr2d是一个定义过的函数 






#nn.init中包含大量初始化相关函数
#各种初始化
def weights_init_normal(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    #print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)




#查看模型中每层输入输出形状的变化
x = torch.rand(size=(1,1,28,28),dtype=torch.float32)
 for layer in net:
      x = layer(x)
      print(layer.__class__.__name__, 'output shape: \t', x.shape)

###########################################################################

############################## sklearn ######################################
#把数据按比例划 随机 分成训练集验证集（也可以是测试集，纯看怎么使用）
from sklearn.model_selection import train_test_split
dbz_train,dbz_val, obs_train,obs_val = train_test_split(dbz,obs,test_size=0.2, random_state=2023)





############################### matplotlib ####################################
plt是模块matplotlib.pyplot(基于fig和axes实现的)；
fig是类matplotlib.figure.Figure的实例；
ax是类matplotlib.axes.Axes的实例，axs是类Axes实例的集合。

plt快速画图和fig，ax画图的区别
plt.plot(dt_list, np.nanmean(acc_all,axis=(1,2)), 'ro-', alpha=0.8, linewidth=1, label='ACC')
-> im_acc = axs[0].plot(dt_list, np.nanmean(acc_all,axis=(1,2)), 'ro-', alpha=0.8, linewidth=1, label='ACC')

plt.legend()  ->  axs[0].legend()   #显示图例label
plt.xlim(0,3,0.1)  ->  axs[0].set_xlim(0,3,0.1)  #设置x轴显示范围
plt.xlabel('leadtime(day)')  ->  axs[0].set_xlabel('leadtime(day)') #设置x轴的label
plt.ylabel('ACC')  ->  axs[0].set_ylabel('ACC')
plt.title('Shandong')  ->  axs[0].set_title('Shandong') #设置标题

#快速画折线图
plt.plot(x轴数据, np.nanmean(acc_all,axis=(1,2)), 'ro-', alpha=0.8, linewidth=1, label='ACC') #'ro-' 为 红色 原点 直线，可以不写

#一个二维数组 快速画空间分布图
data = np.random.rand(10, 10)
plt.imshow(data)

#两个数组，快速画散点图
import matplotlib.pyplot as plt
a = np.random.rand(4,5)
b = np.random.rand(4,5)
plt.scatter(a,b)

#组图快速画图
# plot a frame from each img_type
fig,axs = plt.subplots(1,4,figsize=(10,5))
frame_idx = 30
axs[0].imshow(vis[:,:,frame_idx]), axs[0].set_title('VIS')
axs[1].imshow(ir069[:,:,frame_idx]), axs[1].set_title('IR 6.9')
axs[2].imshow(ir107[:,:,frame_idx]), axs[2].set_title('IR 10.7')
axs[3].imshow(vil[:,:,frame_idx]), axs[3].set_title('VIL')

#保存图片
plt.savefig('./test.png',bbox_inches='tight'）

#中英文字体设置
from mplfonts import use_font
use_font('SimHei') #设置全局使用中文黑体，建议都使用最下面两行
或者
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置全局为中文黑体
plt.rcParams['font.family'] = ['Times New Roman','SimHei'] #设置英文字体为 Times New Roman ,中文为黑体 
plt.rcParams['axes.unicode_minus'] = False  # 设置-号，如果只设置上一行，负号画图会不显示
#添加自己下载的字体
https://blog.csdn.net/HLBoy_happy/article/details/131667829

创建画布画一张图
创建多子图 循环画图
空间分布图、散点图

#画不填色的等高线图
	import numpy as np
	import matplotlib.pyplot as plt
	x = np.arange(-2.0,2.0,0.01)
	y = np.arange(-2.0,2.0,0.01)
	'''meshgrid用于生成三维曲面的分格线座标；产生“格点”矩阵'''
	X,Y = np.meshgrid(x,y)  # 确定x/y的取值范围
	'''定义一个函数，用来计算x/y对应的z值'''
	def f(x,y):
	    return (1-y**5+x**5)*np.exp(-x**2-y**2)
	'''contour()函数可生成三维结构表面的等值线图'''
	C = plt.contour(X,Y,f(X,Y),8,colors='red')
	'''clabel用于标记等高线'''
	plt.clabel(C,inline=1,fontsize=10)
	plt.show()


手动设置colorbar的范围
0值设置为白色
画经纬度刻度
双colorbar
#双y轴（左右各一个） 主要是ax2=ax1.twinx()
	import numpy as np
	import matplotlib.pyplot as plt
	plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置支持中文
	plt.rcParams['axes.unicode_minus'] = False  # 设置-号
	x = np.arange(-2.0,2.0,0.01)
	y1 = np.arange(-2.0,2.0,0.01)
	'''meshgrid用于生成三维曲面的分格线座标；产生“格点”矩阵'''
	X,Y = np.meshgrid(x,y1)  # 确定x/y的取值范围
	'''定义一个函数，用来计算x/y对应的z值'''
	def f(x,y):
	    return (1-y**5+x**5)*np.exp(-x**2-y**2)
	'''contour()函数可生成三维结构表面的等值线图'''
	fig, ax1 = plt.subplots()  
	ax2 = ax1.twinx()  
	line = ax1.contour(X,Y,f(X,Y),8,colors='black')
	'''clabel用于标记等高线'''
	plt.clabel(line,inline=1,fontsize=10)
	
	y2=x*0.8
	bar = ax2.bar(x, y2, label='降水',alpha = 0.05, color='gray', width=0.4)   #alpha控制透明度
	
	ax1.set_xlabel('经度', fontdict={'size': 16})  
	ax1.set_ylabel('等高线',fontdict={'size': 16})  
	ax2.set_ylabel('降水',fontdict={'size': 16}) 
	plt.show()
 
5.cartopy + matplotlib
读取shp文件添加地图
根据经纬度范围绘制
画布的投影与画图的投影
绘制Lambert投影的WRF数据到Lambert
绘制Lambert投影的WRF数据到PlateCarree






dt_list = [30,60,90,120,150,180,210]
# dt_list = [30]
for dt in dt_list:              						#循环画图
    x = t2m_SEAS5_anomaly.t2m_a.loc[:,dt]
    time_obs = t2m_SEAS5_anomaly.time + np.timedelta64(dt,'D')
    y = t2m_CN_anomaly.t2m_a.loc[time_obs]

    x['time'] = y['time'] #时间对其才可以用xs					
    fig,axs = plt.subplots(1,2,figsize=(9,4)) 					#创建组图，2个子图
    acc = xs.pearson_r(x, y,dim='time')
    rmse = xs.rmse(x, y,dim='time')

    im_acc = axs[0].imshow(acc)							#两个子图都要画colorbar以及其他复杂操作，必须把ax存下来后面调用
    axs[0].set_title('acc')							#设置子图的标题
    cbar_acc = plt.colorbar(im_acc, ax=axs[0], orientation='horizontal')	#给im_acc画colorbar，画到axs[0]，horizontal水平的  vertical垂直的
    cbar_acc.set_label('Correlation Coefficient')                               #设置colorbar的标签
    axs[0].set_ylabel(f'lead {dt}d')					        #设置y轴的标签

    im_rmse = axs[1].imshow(rmse)
    axs[1].set_title('rmse')
    cbar_rmse = plt.colorbar(im_rmse, ax=axs[1], orientation='horizontal')
    cbar_rmse.set_label('Root Mean Squared Error')






### dbz常用colorbar


### 一小时累计降水常用colorbar






###########################################################################


############################### 读取png、img文件，转成numpy数组 ####################################
from PIL import Image
file_path = '/dataset/radar_map_2308/2023_8_11_2300.png'  #img文件也一样
f = Image.open(file_path)
dbz = np.array(f)

################################ 多个png 转为gif 动图 ###########################################
import imageio.v2 as imageio
def create_gif(source, name, duration):
    """
    生成gif的函数，原始图片仅支持png
    source: 为png图片列表（排好序）
    name ：生成的文件名称
    duration: 每张图片之间的时间间隔
    """
    frames = []   # 读入缓冲区
    for img in source:
        frames.append(imageio.imread(img))
    imageio.mimsave(name, frames, 'GIF', duration=duration,loop=0)
    print("处理完成")

png_list = glob('figure/png/*.png')
create_gif(png_list,f'figure/gif/test.gif',0.5)




############################### xarray ####################################
import xarray as xr
#读取nc文件
f=xr.open_dataset('test.nc')
#读取grib文件，以GFS为例
  ！！！要先安装cfgirb、ECCODES，并且在脚本最上方写入ECCODES的路径（写到share/eccodes/definitions）,可以用find -name 'definitions' 搜一下
#导入环境变量顶头写
import os
os.environ['ECCODES_DEFINITION_PATH']='/home/qixiang/SHARE/anaconda3/envs/sunhh_usual/share/eccodes/definitions/'
import xarray as xr
import pygrib
import datetime
import numpy as np
import pandas as pd
from scipy.interpolate import griddata


#安装rioxarray 和 xarray后可以直接读取asc和img文件
import xarray as xr
data = xr.open_dataset(f'{dir}/asc/Z_SURF_C_BABJ_P_CMPA_FAST_CHN_0P05_DAY-PRE-2021031520.asc


 
#利用pygrib包，从多个grib2文件中提取相应时间和经纬度范围内，并拼接成一整个xarray数组的代码
import pygrib as pg
import xarray
import numpy as np
import datetime

class grib_decode_helper:
    def __init__(self,grib_paths,lons,lats):
        self.paths = grib_paths
        self.lons = lons
        self.lats = lats

    def get_element_xarray(self,**args):
        all_grb_v = list()
        all_grb_t = list()
        lat1 = float(min(self.lats))
        lat2 = float(max(self.lats))
        lon1 = float(min(self.lons))
        lon2 = float(max(self.lons))
        index = 0
        for p in self.paths:
            index = index+1
            grbs = pg.open(p)
            grb = grbs.select(**args)[0]
            if self.lats is None:
                all_grb_v.append(grb.values)
                lats,lons = grb.latlons()
            else:
                data, lats, lons = grb.data(lat1=lat1,lat2=lat2,lon1=lon1,lon2=lon2)
                all_grb_v.append(data)
            all_grb_t.append(datetime.datetime(grb.year,grb.month,grb.day,grb.hour,grb.minute) )
        return xarray.DataArray(np.array(all_grb_v),coords={'lon':lons[0,:],'lat':lats[:,0],'times':all_grb_t},dims=['times','lat','lon'])

#创建 dataarray 和 dataset
data = np.random.rand(2,3)
data_vars = {  
		"var1":(("x", "y"), np.random.rand(2,3)), 
		"var2":(("x", "y"), np.random.rand(2,3))
		       }  ## 不同对象坐标对应的属性
coords = {  
	  "x":[1,2],  
	  "y":[1,2,3]  
		}  
da = xr.DataArray(data=data,coords=coords)
ds = xr.Dataset(data_vars = data_vars, coords = coords)


#数据输出成nc文件
ds.to_netcdf('output.nc')


#修改nc文件中的数值



#修改grib中的数据



#修改xr.DataArray中的数值
ds['tm'] = tm_new  #不能用ds.tm = tm_new，这样没改
#按时间分类，计算气候态
t2m_SEAS5.t2m.groupby('time.month').mean('time')  #按月分，然后求平均，就是算的月的气候态（1-12月）
#选取指定月份的数据
t2m.sel(time=(t2m.time.dt.month.isin([1,2,3]))) #提取1,2,3月的数据
#选取指定时间、经纬度的数据
f_SEAS5.t2m.loc['2011-01':'2012-01',28.5:21.8,180:220]

#查看da或者ds时间有哪些月份
t2m.time.dt.month


################## netCDF4
import netCDF4 as nc
file = 'test.nc'
dataset =nc.Dataset(file)
variable=dataset.variables['date']


Dataset.variables["varname"][:] = newarr
Dataset.close()


################ wrf-python

    
################## xesmf
#插值
https://zhuanlan.zhihu.com/p/560132563
#wrf的Lambert投影数据插值成站点/等经纬度（相当于2D数组插值到1D数组,即站点或等经纬网格)
#插值到Lambert或其他二维经纬度网格（相当于2D数组插值到2D数组）
https://mp.weixin.qq.com/s/VaIjmxWu3zrKlkRmv63bIQ
		
##############################其他常用代码及库
#查看内存已用情况
import psutil
import os

print('当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024) )
info = psutil.virtual_memory()
print( '电脑总内存：%.4f GB' % (info.total / 1024 / 1024 / 1024) )
print('当前使用的总内存占比：',info.percent)
print('cpu个数：',psutil.cpu_count())

    
#手动释放内存垃圾，多用在jupyter中，jupyter不会及时释放内存
import gc
gc.collect()

      
强制一位的数值字符串变成2位数的字符串
#方法1 使用字符串格式化
number = 5
format_number = '%02d' %number #输出就是05
#方法2 使用zill,把字符串补齐成指定位数
format_number = str(number).zill(2)  #把‘5’ 补齐为2位数
      
#循环
X = [1,2,3]
Y = [3,4,5]
for x,y in zip(X,Y):     # 对X和Y同时循环(可以超过两个变量，例如x,y,z)，效果类似于： for i in range(len(X))
    test = x + y 				          test = X[i] + Y[i]
							      
for i,x in enumerate(X): #循环的同时 也获取索引，i为索引，x为循环值,i常用来作为计数
    print(i,x)
							      
#分裂字符串
f = os.path.abspath('test.csv') # 返回f 为 '/home/qixiang/sunhh/QPE_transfer/test.csv'
f.split('/')  #返回 ['', 'home', 'qixiang', 'sunhh', 'QPE_transfer', 'test.csv']
							      
#链接字符串组成的list
f_list = ['', 'home', 'qixiang', 'sunhh', 'QPE_transfer', 'test.csv']
('.').join(f_list) #以 '.' 来链接f_list中的字符串，返回 '.home.qixiang.sunhh.QPE_transfer.test.csv'
							      
#强制输入范围
assert a in ['train','val','test']  #a不在列表中，则报错退出
assert 0 <= b <= 1   #b不在[0,1]之间则报错退出 	

#mask 可以乘以数组，来使得mask为False 对应的数组中的数为0，其他数不变
a = np.random.rand(3,4)*10 #0-10的随机数
b = deepcopy(a)
mask = a>5
mask * a # 小于等于5的数全都变成0,等价于 a[~mask] = 0


#显示循环进度
from tqdm import tqdm
for i in tqdm(range(3)): 实时显示循环的进度

#保持脚本内的随机种子都固定

#显示目前占用内存较大的变量，显示后 挑选其中变量，用del删除
import sys
print("{}{: >25}{}{: >10}{}".format('|','Variable Name','|','Memory','|'))
print(" ------------------------------------ ")
for var_name in dir():
    if not var_name.startswith("_") and sys.getsizeof(eval(var_name)) > 10000: 
        print("{}{: >25}{}{: >10}{}".format('|',var_name,'|',sys.getsizeof(eval(var_name)),'|'))


### 多线程下载,默认支持断点续传
可以用conda、ubantu(sudo apt install)、centos(yum install)安装，在命令行使用
axel -n 10 https://srtm.csi.cgiar.org/wp-content/uploads/files/srtm_5x5/TIFF/srtm_12_03.zip


############################### conda环境 ####################################
######## conda install一直出现以下内容，无法安装成功或者非常慢
#Solving environment: failed with initial frozen solve. Retrying with flexible solve.
#Solving environment: failed with repodata from current_repodata.json, will retry with next repodata source.
conda update --all
或者 conda update -n base -c defaults conda
######## conda 环境移植
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
conda install ipykernel
python -m ipykernel install --user --name sunhh  #把sunhh环境加入可选项
###########################################################################


################################### WRF_GPU ########################################
#gpu版的wrf，名为AceCast，支持常用的参数化方案，其中虽然有编译好的wps和real部分的exe但是不能使用，只能用wrf.exe（名为 acecast.exe）
#网址:https://acecast-docs.readthedocs.io/en/latest/								  
#从网页上获取许可，（通过邮箱收取），需要把许可放到run路径下
#检查namelist.input 中的参数化方案是否全部支持
./acecast-advisor.sh --tool support-check
#指定单卡运行WRF
mpirun -np 1 ./gpu-launch.sh --gpu-list 2 ./acecast.exe
#指定多卡
mpirun -np 2 ./gpu-launch.sh --gpu-list 0,1 ./acecast.exe
###########################################################################

################################### gdal安装 ########################################
conda create -n gdal python=3.10
conda activate gdal
conda install gdal geotiff libtiff -c conda-forge

#导入gdal报错，搜索后 得知需要重新安装poppler
conda uninstall poppler
conda install poppler
conda install gdal -c conda-forge #ok gdal正常导入

编写.bashrc_gdal
GeoTIFF_PATH=~/SHARE/anaconda3/envs/gdal/
export LD_LIBRARY_PATH=${GeoTIFF_PATH}/lib:$LD_LIBRARY_PATH

source .bashrc_gdal

下载convert_geotiff-0.1.0.tar.gz
tar -xvf convert_geotiff-0.1.0.tar.gz
 
cd convert_geotiff-0.1.0
./configure --prefix=`pwd`/build CPPFLAGS=-I${GeoTIFF_PATH}/include LDFLAGS=-L${GeoTIFF_PATH}/lib
make
make install



#将下载的srtm下的tif合并成一个（srtm dem数据）
gdal_merge.py 是gdal库自带的脚本，安装后在conda环境下的bin里，可以直接命令行使用
# srtm/srtm_china/*tif 下载的srtm的路径
# -o 输出文件，得到srtm_china.tif
# -a_nodata 缺省值，srtm缺省值为-32678
gdal_merge.py srtm/*tif -o out.tif -a_nodata -32768
#把tif转成nc
gdalwarp -t_srs "+proj=longlat +datum=WGS84 +no_defs" -co "FORMAT=NC4" out.tif dem.nc
###########################################################################
















############################ 自定义函数
def delete_nanobs(dbz_np,obs_np,dem_np):
    """
    剔除 标签数据obs为nan的数据 以及其对应索引的其他数据
    """
    print('input shape: ',dbz_np.shape,obs_np.shape,dem_np.shape)
    valid_indices = ~np.isnan(obs_np)

    dbz_sel = dbz_np[valid_indices]
    obs_sel = obs_np[valid_indices]
    dem_sel = dem_np[valid_indices]

    print('out shape: ',dbz_sel.shape,obs_sel.shape,dem_sel.shape)
    return dbz_sel,obs_sel,dem_sel


def delete_wrongdbz(dbz_np,obs_np,dem_np):
    """
    剔除 当观测存在降水，而1个小时雷达回波块的值均为0的情况
    """
    print('input shape: ',dbz_np.shape,obs_np.shape,dem_np.shape)


    dbz_sum = np.sum(dbz_np, axis=(1,2,3))

    # 获取obs大于0且dbz的后三维和大于0的索引
    valid_positive_indices = np.where((obs_np > 0) & (dbz_sum > 0))[0]

    # 获取obs_np等于0的索引
    obs_zero_indices = np.where(obs_np == 0)[0]

    # 使用np.concatenate来合并索引
    valid_indices = np.concatenate([valid_positive_indices, obs_zero_indices])

    # 使用这些索引来挑选满足条件的a和b的数据
    dbz_sel = dbz_np[valid_indices]
    obs_sel = obs_np[valid_indices]
    dem_sel = dem_np[valid_indices]

    print('out shape: ',dbz_sel.shape,obs_sel.shape,dem_sel.shape)
    return dbz_sel,obs_sel,dem_sel


def sel_sample(dbz_all,obs_all,dem_all,a=1):
    print('input shape: ',dbz_all.shape,obs_all.shape,dem_all.shape)
    #挑选出所有 有降水的样本，并添加  a 倍 随机无降水的样本

    # 找出obs大于0的索引
    positive_indices = np.where(obs_all > 0)[0]
    # 找出obs等于0的索引
    zero_indices = np.where(obs_all == 0)[0]

    # 计算需要挑选的obs等于0的数量
    num_zero_indices = int(len(positive_indices) * a)

    # 如果obs等于0的数量少于需要挑选的数量，抛出异常
    if num_zero_indices > len(zero_indices):
        raise ValueError('Not enough obs equal to 0 to satisfy the ratio.')

    # 随机挑选obs等于0的索引
    '''
    replace=False 参数表示在挑选索引时不进行替换，即每个索引只能被挑选一次。
    如果 obs 等于0的数量少于需要挑选的数量，这个函数将抛出异常。
    如果你希望在这种情况下仍然能够挑选出足够数量的索引（即允许一些索引被重复挑选），
    你可以将 replace=False 改为 replace=True。
    '''
    random_zero_indices = np.random.choice(zero_indices, num_zero_indices, replace=False)

    # 合并所有挑选的索引
    selected_indices = np.concatenate([positive_indices, random_zero_indices])
    # 选取对应的dbz
    dbz_sel = dbz_all[selected_indices]
    dem_sel = dem_all[selected_indices]
    obs_sel = obs_all[selected_indices]

    print('output shape: ',dbz_sel.shape,obs_sel.shape,dem_sel.shape)
    return dbz_sel,obs_sel,dem_sel
