模型构建的逻辑及步骤

1.数据清洗，特征分析
要注意
2.数据标准化方法，尽量把数据范围控制到[0,1]或者 均值为0方差为1 ，也有常用控制在[-1,1]
降水常见 取对数 x=np.log10(x+1) , 有上限的降水率 如美国数据集上限为128， x=x/128
RGB 三通道数据 x=x/255.
dem、sat等 进行 均值方差标准化

3.数据采样：尺寸裁剪 及 重要性采样

计算多时空均值，标准化后进行不放回的重要性采样，处理为固定的样本，供后续作为训练集、验证集、测试集
或者再训练过程中 再随机采样。例如在一个时间中 裁剪了多个patch，在其中采样
4.训练集、验证集、测试集的构建

5.模型结构的选择，根据数据及模型的用途和性质确认。
        不同功能块的选择及堆积：常见功能块：
        编码器、解码器、处理器：
        上下采样：
        参数初始化：

6.损失函数的选择
L1损失、L2损失、感知损失、BCE、分类损失、SSIM损失、自适应损失、傅里叶损失
7.训练过程的可视化分析
写一个call_back，在pytorch lightning的train中导入，然后再config文件中写入相关参数
8.训练完成后 测试集上的推荐分析


上下采样特征块参考simvp的ConSC https://github.com/chengtan9907/OpenSTL/blob/OpenSTL-Lightning/openstl/modules/simvp_modules.py

其中下采样的主要实现为 stride=2的Conv2d, self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,stride=stride, padding=padding, dilation=dilation)

上采样的主要实现为 4倍out_channels的Conv2d 接 nn.PixelShuffle(2)

self.conv = nn.Sequential(*[nn.Conv2d(in_channels, out_channels*4, kernel_size=kernel_size,stride=1, padding=padding, dilation=dilation),
                            nn.PixelShuffle(2)])


