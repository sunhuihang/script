import numpy as np
import pandas as pd
import copy
import sys
import os
# import matplotlib.pyplot as plt
import env

'''目标函数'''

f = lambda x: x * x

class genetic_algorithms():

    def Binary2Decimal(self, Binary, include_negative=False):
        '''二进制转为十进制'''
        sum = 0
        bit_flag = -1 if include_negative == True else 1
        for index in range(1, len(Binary)):
            sum = sum + Binary[-index] * 2 ** (index - 1)
        if bit_flag == -1:
            if Binary[0] == 1:
                sum = sum * bit_flag
            else:
                sum = sum + 0
        else:
            sum = sum + Binary[0] * 2 ** (index)

        return sum

    def Binary2Decimal_groups(self, input):
        '''二进制分组'''
        pop = 0
        y_group = np.arange(15).reshape(5, 3)
        for Binary in input:
            y_group[pop, 0] = self.Binary2Decimal(Binary[0:4], False)
            y_group[pop, 1] = self.Binary2Decimal(Binary[4:8], False)
            y_group[pop, 2] = self.Binary2Decimal(Binary[8:12], False)
            pop += 1
        return y_group

    def Creat_Check_Para(self, N, Bit_num):
        '''不断生成一个随机二进制矩阵 new_epochs，并将其转换为十进制组 new_group
        如果生成的矩阵中某一行的某一列的值大于预设的长度值 cu_value 或 pbl_value，
        就会继续生成。否则，函数将返回生成的二进制矩阵和十进制组'''
        while True:
            new_epochs = np.zeros([N, Bit_num], int)
            for x_index in range(0, N):
                for y_index in range(0, Bit_num):
                    new_epochs[x_index, y_index] = np.random.choice([0, 1])
            new_group = self.Binary2Decimal_groups(new_epochs)
            stop_index = 0
            for row in range(N):
                if new_group[row][1] >= len(cu_value) or new_group[row][2] >= len(pbl_value):
                    stop_index += 1
                    break
            if stop_index == 0:
                break
        return new_epochs, new_group


    def run_wrf(self, input, used_para, epochs, inner, fct_initime, bg_type):
        # mp_name = ['Kessler scheme', 'Lin scheme', 'WSM3 scheme', 'WSM5 scheme', 'Ferrier scheme', 'WSM6 scheme', 'Thompson scheme', 'Milbrandt-Yau 2-moment scheme', 'Morrison 2-moment scheme', 'WDM5', 'High-resolution Ferrier microphysics', 'WDM6', 'NSSL two-moment', 'NSSL one-moment', 'P3 two-moment', 'P3-nc']
        # cu_name = ['no cumulus','KF scheme', 'Betts-Miller-Janjic scheme', 'GF ensemble scheme', 'GFS SAS scheme', 'New Grell scheme (G3)', 'Tiedtke scheme', 'Modified KF scheme', 'Multi-scale KF scheme', 'New GFS SAS scheme', 'newer Tiedke scheme', 'GD ensemble scheme','2015 GFS SAS scheme (HWRF)', 'Previous GFS SAS scheme (HWRF)', 'previous KF scheme']
        # pbl_name = ['YSU scheme', 'MYJ scheme', 'QNSE-EDMF', 'MYNN 2.5', 'MYNN 3', 'ACM2 (Pleim) scheme', 'BouLac TKE', 'Bretherton-Park/UW TKE scheme', 'TEMF scheme', 'Shin-Hong scale-aware PBL scheme', 'GBM TKE-type scheme','MRF scheme']

        for para in input:
            if (epochs == 0 and inner == 0) or (para not in used_para):
                mp_para = mp_value[para[0]]
                cu_para = cu_value[para[1]]
                pbl_para = pbl_value[para[2]]
                if pbl_para == 2:
                    sf_sfclay_para = 2
                elif pbl_para == 4:
                    sf_sfclay_para = 4
                elif pbl_para == 6:
                    sf_sfclay_para = 5
                else:
                    sf_sfclay_para = 1
                print('mp: %d  cu: %d  pbl: %d  sf: %d' % (mp_para, cu_para, pbl_para, sf_sfclay_para))
                os.system(f'python run_GAwrf.py {fct_initime} {bg_type} {mp_para} {cu_para} {pbl_para} {sf_sfclay_para} ' + '> log.wrf 2>&1')

    def calculate_suitability(self,input,used_para,epochs,inner,opt_num,fct_initime):
        out_data = input
        out_data_sum = []
        for x_index in range(len(input)):
            mp_para = mp_value[out_data[x_index][0]]
            cu_para = cu_value[out_data[x_index][1]]
            pbl_para = pbl_value[out_data[x_index][2]]
            if (epochs == 0 and inner == 0) or x_index == opt_num or out_data[x_index] not in used_para:
                try:
                    # ts_score = os.popen(f'python cal_ts.py {fct_initime} {mp_para} {cu_para} {pbl_para}').read().strip('\n')
                    ts_score = os.popen(f'python cal_ts.py {fct_initime} {mp_para} {cu_para} {pbl_para}').read().split('\n')[-2] 
                    ts_score = float(ts_score) 
                except (TypeError,ValueError):
                    fp = open('Wrong_comb.txt','a')
                    fp.write(f'Wrong combination:  mp: {mp_para}  cu: {cu_para}  pbl: {pbl_para}\n')
                    fp.close()
                    ts_score = -1
            else: 
                ts_score = 0.0
            out_data_sum.append(ts_score)
            print(out_data_sum)
        return out_data_sum

    def cross_variation(self, init_groups, opt_chose, row=0, all_cross=True):
        opt_chose_data = init_groups[np.argmax(opt_chose)]
        cross_Data = copy.deepcopy(opt_chose_data.tolist())
        if all_cross == True:
            for index in range(0, int(len(init_groups))):
                variation_site = np.random.choice(np.arange(11))
                variation_data = cross_Data[variation_site:variation_site + 2]  # 保存中间变量
                init_groups[index][variation_site:variation_site + 2] = np.array(variation_data)
        else:
            variation_site = np.random.choice(np.arange(11))
            variation_data = cross_Data[variation_site:variation_site + 2]  # 保存中间变量
            init_groups[row][variation_site:variation_site + 2] = np.array(variation_data)
        return init_groups
 
    # def show_evolution(self,data_collect,epochs):
    #     #x_axis_data = range(1,epochs+1)
    #     x_axis_data = range(1,len(data_collect)+1)
    #     y_axis_data = data_collect 
    #     plt.plot(x_axis_data, y_axis_data, alpha=0.5, linewidth=1, label='OPT')
    #     plt.legend()  
    #     plt.xlabel('The number of generation') 
    #     plt.ylabel('Fitness value')
    #     plt.savefig('show_evolution.png')
    #     plt.show()    

    def run(self,fct_initime,bg_type,epochs_num,N,Bit_num,used_para_input):
        data_collect = []
        init_groups, y_group = self.Creat_Check_Para(N, Bit_num)
        # used_para_input为true时代表重启
        if used_para_input == True:
            # 初始化引入最优精英代
            data_collect = np.loadtxt("Epochs_evolution.txt",dtype=float).tolist()
            array1 = np.loadtxt("Elitism_info.txt",dtype=int)
            Elitism_list = array1.tolist()
            new_list = init_groups.tolist()
            new_list[0] = Elitism_list
            init_groups = np.array(new_list)
            # 引入跑过的参数化方案
            data2 = pd.read_csv("Used_para.txt",header=None,index_col=False)
            array2 = np.array(data2)
            para_collect = array2.tolist()
        else:
            os.system('/bin/rm -f *txt')
            os.system(f'python deal_station.py {fct_initime}')
            para_collect = y_group.tolist()
            para_info = pd.DataFrame(para_collect)
            para_info.to_csv("Used_para.txt", index=None, header=False)
        for epochs in range(epochs_num):
            opt_num = 0
            for inner in range(20):
                print(init_groups)
                y_group = self.Binary2Decimal_groups(init_groups)
                self.run_wrf(y_group.tolist(), para_collect, epochs, inner, fct_initime, bg_type)
                print('***stop***')
                opt_chose = self.calculate_suitability(y_group.tolist(),para_collect,epochs,inner,opt_num,fct_initime)
                opt_num = np.argmax(opt_chose)
                for i in y_group.tolist():
                    if i not in para_collect:
                        para_collect.append(i)  # 收集所有跑过的参数化方案，避免重复运行
                para_info = pd.DataFrame(para_collect)
                para_info.to_csv("Used_para.txt", index=None, header=False)
                print(np.argmax(opt_chose))
                print(init_groups[np.argmax(opt_chose)].tolist())
                print('epoch: %d 当前最优值为%f' % (epochs + 1, np.max(opt_chose))) 
                data_collect.append(np.max(opt_chose))
                evolution_info = pd.DataFrame(data_collect)
                evolution_info.to_csv("Epochs_evolution.txt",index=None, header=False)
                Elitism_groups = init_groups[np.argmax(opt_chose)]
                Elitism_info = pd.DataFrame(Elitism_groups)
                Elitism_info.to_csv("Elitism_info.txt", index=None, header=False)
                mp_para = mp_value[y_group[np.argmax(opt_chose)][0]]
                cu_para = cu_value[y_group[np.argmax(opt_chose)][1]]
                pbl_para = pbl_value[y_group[np.argmax(opt_chose)][2]]
                bettergroups_dir1 = os.path.join(wrfout_dir1,f'{mp_para}_{cu_para}_{pbl_para}')
                bettergroups_dir2 = os.path.join(wrfout_dir2,f'{mp_para}_{cu_para}_{pbl_para}')
                os.system('/bin/mv ' + bettergroups_dir2 + ' ' + bettergroups_dir1)
                if os.path.exists(wrfout_dir2): os.system('/bin/rm -rf '+ wrfout_dir2 + '/*')
                os.system('/bin/mv ' + bettergroups_dir1 + ' ' + bettergroups_dir2)
                cross_variation_data = self.cross_variation(init_groups, opt_chose)
                # 若CU超过14或PBL超过11则重新与最优种群进行交叉互换
                while True:
                    z_group = self.Binary2Decimal_groups(cross_variation_data)
                    check_count = 0
                    for row in range(N):
                        if z_group[row][1] >= len(cu_value) or z_group[row][2] >= len(pbl_value):
                            cross_variation_data = self.cross_variation(init_groups, opt_chose, row, False)
                        else:
                            check_count += 1
                            continue
                    if check_count == N:
                        break
                init_groups = copy.deepcopy(cross_variation_data)
                count2 = 0
                # 与最优种群的二进制位数差异小于一位时，内环收敛到最优，退出内层循环（种群相似）
                for x in range(N):
                    count1 = 0
                    for y in range(Bit_num):
                        if init_groups[x][y] == Elitism_groups[y]:
                            count1 += 1
                    if count1 >= (Bit_num-1):
                        count2 += 1
                if count2 == N:
                    print("**inner cycle stop**")
                    break
                if os.path.exists(workdir): os.system('/bin/rm -rf '+ workdir + '/*')
            # 除父代存活其余个体随机生成（外环搜索，设置最大代数实现收敛）
            new_epochs, new_group = self.Creat_Check_Para(N, Bit_num)
            Elitism_groups = init_groups[np.argmax(opt_chose)]
            new_epochs[0] = Elitism_groups
            init_groups = new_epochs
        #self.show_evolution(data_collect,epochs_num)

# if __name__=='__main__':
if (len(sys.argv) != 2 ):
    print("The user input is not correct")
    print("Usage: python " + str(sys.argv[0] + " Initime"))
    print("For example: python " + str(sys.argv[0] + " 2020081800"))
    sys.exit(1)
else:
    fct_initime = str(sys.argv[1])
workdir = os.path.join(env.work_dir, fct_initime, 'tmp.GA4wrf')
# wrfout_dir1 = os.path.join(env.output_dir, fct_initime)
# wrfout_dir2 = os.path.join(env.output_dir, fct_initime, 'WRFOUT')
wrfout_dir1 = os.path.join(env.top,'WRFOUT', )
wrfout_dir2 = os.path.join(env.top,'WRFOUT', fct_initime)
# mp_value = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14, 16, 17, 19, 51]
# cu_value = [0, 1, 2, 3, 5, 6, 10, 11, 14, 16, 93, 94, 95, 99]
# pbl_value = [1, 2, 4, 5, 6, 7, 8, 9, 11, 12, 99]
mp_value = [1, 2, 3, 4, 6, 8, 9, 10, 11, 13, 14, 16, 17, 19, 4, 6]
cu_value = [1, 2, 3, 5, 6, 10, 11, 14, 16]
pbl_value = [1, 2, 4, 5, 6, 7, 8, 9, 11, 12]
bg_type = 'wrfda'
epochs_num = 500
obj = genetic_algorithms()
obj.run(fct_initime,bg_type,epochs_num,N=5,Bit_num=12,used_para_input=True) #输入起报时刻、背景场类别、外循环代数、种群数、二进制位数以及是否restart
#array1 = np.loadtxt("Epochs_evolution.txt",dtype=float)
#data_collect = array1.tolist()
#obj.show_evolution(data_collect,epochs_num)
