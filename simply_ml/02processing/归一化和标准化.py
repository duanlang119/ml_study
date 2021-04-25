# 特征工程
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler,Binarizer

mm = MinMaxScaler()
# 归一化，最典型的就是数据的归一化处理，即将数据统一映射到[0,1]区间上
# 将数据的最大最小值记录下来，并通过Max-Min作为基数，进行数据的归一化处理
#所谓数据归一化处理就是将所有数据都映射到同一尺度。
#最常用的一种数据归一化方法叫最值归一化
#最值归一化(normalization)把所有数据映射到0-1之间，作用于每列，max为当前列最大值
#min为当前列最小值
#x'=(x-min)/(max-min)


g_data = np.array([[3, -1.7, 3.5, -6],
                         [0, 4, -0.3, 2.5],
                         [1, 3.5, -1.8, -4.5]])

def mm_simple():
    mm_data = mm.fit_transform(g_data)

    # 三个样本，四个特征
    print('归一化之后的数据\n', mm_data)
    data = mm.inverse_transform(mm_data)
    # sklearn中transform用来归一化后，可以用inverse_transform还原。
    print('原始数据\n', data)

"""
数据要精确
该方法也有很大的缺点，就是受极端数据值（outlier）影响比较大，
比如工资就不是一个有明显边界的特征，绝大部分人月薪0-3w，而有些人收入极其高，月薪100w甚至更高，这样往0-1之间映射会有很大误差。
"""


# 标准化
"""
利用标准差剔除异常数据 

标准差公式是一种数学公式。标准差也被称为标准偏差，或者实验标准差，
公式如下所示：标准差=方差的算术平方根=s=sqrt(((x1-x)^2 +(x2-x)^2 +......(xn-x)^2)/n)。

简单来说，标准差是一组数值自平均值分散开来的程度的一种测量观念。
一个较大的标准差，代表大部分的数值和其平均值之间差异较大；一个较小的标准差，代表这些数值较接近平均值。

例如，A、B两组各有6位学生参加同一次语文测验，A组的分数为95、85、75、65、55、45，B组的分数为73、72、71、69、68、67。
这两组的平均数都是70，但A组的标准差为17.078分，B组的标准差为2.160分，说明A组学生之间的差距要比B组学生之间的差距大得多。
y=(x-μ)/σ   ，μ \muμ表示对应列的均值，σ \sigmaσ表示对应列的标准差
在变化缓慢的场景中，利用标准差，可以过滤异常变化。
例如在智能手表中，设备经常处于低功耗状态，一般间隔几分钟到几十分钟采集一次电池电压，
如果采集电池电压时，马达震动提醒，由于马达功耗较高，造成电压跌幅比较大（停止震动后，电压会恢复）所以电池电量会出现跳变。
此时利用标准差，可以发现此时标准差过大，抛弃此次数据，从而避免电池电量跳变。
如若采用平均值滤波的方法（去除最高和最低值后求平均值）在采集到多个异常的场景无法剔除异常。

"""

def std_simple():
    print('-' * 100)
    std = StandardScaler()
    data = std.fit_transform(g_data)
    print('每列特征的平均值', std.mean_)

    d_u=g_data.mean(axis=0)
    print('每列特征的平均值',d_u )

    d_xigema=g_data.std(axis=0)
    print('每列特征的标准差', d_xigema)
    first_l=g_data[:,0]
    print('first line: ',first_l)

    new_l=[(x-d_u)/d_xigema for x in first_l]

    print('y value: ',new_l)

    print('归一化之后的数据\n', data)
    data = std.inverse_transform(data)
    print('原始数据\n', data)


def mm_01_scal():
    data_minmaxscaler = MinMaxScaler(feature_range=(0, 1)).fit_transform(g_data)
    print('原始归一化用minmaxscaler：')
    print(data_minmaxscaler)

def binarizer():
    data_binarizer = Binarizer().fit_transform(g_data)
    print('使用binarizer二值化处理')
    print(data_binarizer)

# std_simple()
# mm_01_scal()
binarizer()