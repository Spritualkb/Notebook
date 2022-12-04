---

---

# 机器学习和深度学习的综述

## 人工智能、机器学习、深度学习的关系

![image-20220610134018614](https://img2023.cnblogs.com/blog/2554043/202212/2554043-20221204183927407-182794746.png)

### 机器学习的实现

机器学习的实现可以分成两步：训练和预测，类似于归纳和演绎：

- **归纳：** 从具体案例中抽象一般规律，机器学习中的“训练”亦是如此。从一定数量的样本（已知模型输入XX*X*和模型输出YY*Y*）中，学习输出YY*Y*与输入XX*X*的关系（可以想象成是某种表达式）。
- **演绎：** 从一般规律推导出具体案例的结果，机器学习中的“预测”亦是如此。基于训练得到的YY*Y*与XX*X*之间的关系，如出现新的输入XX*X*，计算出输出YY*Y*。通常情况下，如果通过模型计算的输出和真实场景的输出一致，则说明模型是有效的。

### 机器学习的方法论

**模型有效的基本条件是能够拟合已知的样本**

**衡量模型预测值和真实值差距的评价函数也被称为损失函数（损失Loss）**。

**模型假设、评价函数（损失/优化目标）和优化算法是构成模型的三个关键要素**。

机器执行学习任务的框架体现了其**学习的本质是“参数估计”**（Learning is parameter estimation）。

![image-20220610161018005](https://img2023.cnblogs.com/blog/2554043/202212/2554043-20221204183935605-947591845.png)

## 深度学习

**机器学习和深度学习两者在理论结构上是一致的，即：模型假设、评价函数和优化算法，其根本差别在于假设的复杂度**

### 神经网络的基本概念

人工神经网络包括多个神经网络层，如：卷积层、全连接层、LSTM等，每一层又包括很多神经元，超过三层的非线性神经网络都可以被称为深度神经网络。

- 神经元：

  神经网络中每个节点称为神经元，由两部分组成：

  - 加权和：将所有输入加权求和。
  - 非线性变换（激活函数）：加权和的结果经过一个非线性函数变换，让神经元计算具备非线性的能力。

- **多层连接：** 大量这样的节点按照不同的层次排布，形成多层的结构连接起来，即称为神经网络。

- **前向计算：** 从输入计算输出的过程，顺序从网络前至后。

- **计算图：** 以图形化的方式展现神经网络的计算逻辑又称为计算图，也可以将神经网络的计算图以公式的方式表达：

Y=f3(f2(f1(w1⋅x1+w2⋅x2+w3⋅x3+b)+…)…)…)Y =f_3 ( f_2 ( f_1 ( w_1\cdot x_1+w_2\cdot x_2+w_3\cdot x_3+b ) + … ) … ) … )*Y*=*f*3(*f*2(*f*1(*w*1⋅*x*1+*w*2⋅*x*2+*w*3⋅*x*3+*b*)+…)…)…)

### 深度学习改变了AI应用的研发模式

#### 实现了端到端的学习

![image-20220614225919010](https://img2023.cnblogs.com/blog/2554043/202212/2554043-20221204183939921-591952193.png)

## 波士顿房价预测任务

### 数据处理

#### 1读入数据

```python
# 导入需要用到的package
import numpy as np
import json
# 读入训练数据
datafile = './work/housing.data'
data = np.fromfile(datafile, sep=' ')
data
```

```
array([6.320e-03, 1.800e+01, 2.310e+00, ..., 3.969e+02, 7.880e+00,
       1.190e+01])
```

#### 2 数据形状变换

由于读入的原始数据是1维的，所有数据都连在一起。因此需要我们将数据的形状进行变换，形成一个2维的矩阵，每行为一个数据样本（14个值），每个数据样本包含13个X（影响房价的特征）和一个Y（该类型房屋的均价）。

```python
# 读入之后的数据被转化成1维array，其中array的第0-13项是第一条数据，第14-27项是第二条数据，以此类推.... 
# 这里对原始数据做reshape，变成N x 14的形式
feature_names = [ 'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE','DIS', 
                 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV' ]
feature_num = len(feature_names)
data = data.reshape([data.shape[0] // feature_num, feature_num])
```

In [3]

```
# 查看数据
x = data[0]
print(x.shape)
print(x)
(14,)
[6.320e-03 1.800e+01 2.310e+00 0.000e+00 5.380e-01 6.575e+00 6.520e+01
 4.090e+00 1.000e+00 2.960e+02 1.530e+01 3.969e+02 4.980e+00 2.400e+01]
```

#### 3数据集划分

在本案例中，我们将80%的数据用作训练集，20%用作测试集，实现代码如下。通过打印训练集的形状，可以发现共有404个样本，每个样本含有13个特征和1个预测值。

In [4]

```python
ratio = 0.8
offset = int(data.shape[0] * ratio)
training_data = data[:offset]
training_data.shape
(404, 14)
```

#### 4数据集归一处理

对每一个特征进行归一化处理，使得每个特征的取值缩放到0~1之间，这样做有两个好处：一是模型训练更高效，在本节的后半部分会详细说明；二是特征前的权重大小可以代表该变量对预测结果的贡献度（因为每个特征值本身的范围相同）。

```python
# 计算train数据集的最大值，最小值，平均值
maximums, minimums, avgs = \
                     training_data.max(axis=0), \
                     training_data.min(axis=0), \
     training_data.sum(axis=0) / training_data.shape[0]
# 对数据进行归一化处理
for i in range(feature_num):
    #print(maximums[i], minimums[i], avgs[i])
    data[:, i] = (data[:, i] - avgs[i]) / (maximums[i] - minimums[i])
```

| 属性名字        | 属性解释                 |
| --------------- | ------------------------ |
| ndarry.shape    | 数组维度的元组           |
| ndarry.ndim     | 数组维数                 |
| ndarry.size     | 数组中的元素数量         |
| ndarry.itemsize | 一个数组元素的长度(字节) |
| ndarry.dtype    | 数组元素的类型           |

#### 5封装成load data函数

将上述几个数据处理操作封装成`load data`函数，以便下一步模型的调用

```python
def load_data():
    # 从文件导入数据
    datafile = './work/housing.data'
    data = np.fromfile(datafile, sep=' ')

    # 每条数据包括14项，其中前面13项是影响因素，第14项是相应的房屋价格中位数
    feature_names = [ 'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', \
                      'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV' ]
    feature_num = len(feature_names)

    # 将原始数据进行Reshape，变成[N, 14]这样的形状
    data = data.reshape([data.shape[0] // feature_num, feature_num])

    # 将原数据集拆分成训练集和测试集
    # 这里使用80%的数据做训练，20%的数据做测试
    # 测试集和训练集必须是没有交集的
    ratio = 0.8
    offset = int(data.shape[0] * ratio)
    training_data = data[:offset]

    # 计算训练集的最大值，最小值，平均值,axis 0 代表列
    maximums, minimums, avgs = training_data.max(axis=0), training_data.min(axis=0), \
                                 training_data.sum(axis=0) / training_data.shape[0]

    # 对数据进行归一化处理
    for i in range(feature_num):
        #print(maximums[i], minimums[i], avgs[i])
        data[:, i] = (data[:, i] - avgs[i]) / (maximums[i] - minimums[i])

    # 训练集和测试集的划分比例
    training_data = data[:offset]
    test_data = data[offset:]
    return training_data, test_data
```

```python
# 获取数据
training_data, test_data = load_data()
x = training_data[:, :-1]
y = training_data[:, -1:]
```

```python
# 查看数据
print(x[0])
print(y[0])

[-0.02146321  0.03767327 -0.28552309 -0.08663366  0.01289726  0.04634817
  0.00795597 -0.00765794 -0.25172191 -0.11881188 -0.29002528  0.0519112
 -0.17590923]
[-0.00390539]
```

### 模型设计

模型设计是深度学习模型关键要素之一，也称为网络结构设计，相当于模型的假设空间，即实现模型“前向计算”（从输入到输出）的过程。

如果将输入特征和输出预测值均以向量表示，输入特征x有13个分量，y有1个分量，那么参数权重的形状（shape）是13×1。假设我们以如下任意数字赋值参数做初始化：

*w*=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,−0.1,−0.2,−0.3,−0.4,0.0]

```python
w = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, -0.1, -0.2, -0.3, -0.4, 0.0]
w = np.array(w).reshape([13, 1])
```

取出第1条样本数据，观察样本的特征向量与参数向量相乘的结果。

```python
x1=x[0]
t = np.dot(x1, w)
print(t)

[0.03395597]
```

完整的线性回归公式，还需要初始化偏移量b，同样随意赋初值-0.2。那么，线性回归模型的完整输出是z=t+b，这个从特征和参数计算输出值的过程称为“前向计算”。

```
b = -0.2
z = t + b
print(z)
[-0.16604403]
```

将上述计算预测输出的过程以“类和对象”的方式来描述，类成员变量有参数w和b。通过写一个`forward`函数（代表“前向计算”）完成上述从特征和参数到输出预测值的计算过程，代码如下所示。

```python
class Network(object):
    def __init__(self, num_of_weights):
        # 随机产生w的初始值
        # 为了保持程序每次运行结果的一致性，
        # 此处设置固定的随机数种子
        np.random.seed(0)
        self.w = np.random.randn(num_of_weights, 1)
        self.b = 0.
        
    def forward(self, x):
        z = np.dot(x, self.w) + self.b
        return z
```

基于Network类的定义，模型的计算过程如下所示。

```python
net = Network(13)
x1 = x[0]
y1 = y[0]
z = net.forward(x1)
print(z)

[-0.63182506]
```

从上述前向计算的过程可见，线性回归也可以表示成一种简单的神经网络（只有一个神经元，且激活函数为恒等式）。

### 训练配置

模型设计完成后，需要通过训练配置寻找模型的最优值，即通过损失函数来衡量模型的好坏。训练配置也是深度学习模型关键要素之一。

通过模型计算x1表示的影响因素所对应的房价应该是z, 但实际数据告诉我们房价是y。这时我们需要有某种指标来衡量预测值z跟真实值y之间的差距。对于回归问题，最常采用的衡量方法是使用均方误差作为评价模型好坏的指标，具体定义如下：

Loss=(y−z)**2(公式3)

Loss（简记为: L）通常也被称作损失函数，它是衡量模型好坏的指标。

在回归问题中，均方误差是一种比较常见的形式，分类问题中通常会采用交叉熵作为损失函数，在后续的章节中会更详细的介绍。对一个样本计算损失函数值的实现如下。

```python
Loss = (y1 - z)*(y1 - z)
print(Loss)

[0.39428312]
```

因为计算损失函数时需要把每个样本的损失函数值都考虑到，所以我们需要对单个样本的损失函数进行求和，并除以样本总数N。

![image-20220712105547143](https://img2023.cnblogs.com/blog/2554043/202212/2554043-20221204183948249-1237503452.png)

在Network类下面添加损失函数的计算过程如下

```python
class Network(object):
    def __init__(self, num_of_weights):
        # 随机产生w的初始值
        # 为了保持程序每次运行结果的一致性，此处设置固定的随机数种子
        np.random.seed(0)
        self.w = np.random.randn(num_of_weights, 1)
        self.b = 0.
        
    def forward(self, x):
        z = np.dot(x, self.w) + self.b
        return z
    
    def loss(self, z, y):
        error = z - y
        cost = error * error
        cost = np.mean(cost)
        return cost
```

使用定义的Network类，可以方便的计算预测值和损失函数。需要注意的是，类中的变量x, w，b, z, error等均是向量。以变量x为例，共有两个维度，一个代表特征数量（值为13），一个代表样本数量，代码如下所示。

```python
net = Network(13)
# 此处可以一次性计算多个样本的预测值和损失函数
x1 = x[0:3]
y1 = y[0:3]
z = net.forward(x1)
print('predict: ', z)
loss = net.loss(z, y1)
print('loss:', loss)

predict:  [[-0.63182506]
 [-0.55793096]
 [-1.00062009]]
loss: 0.7229825055441158
```

### 训练过程

上述计算过程描述了如何构建神经网络，通过神经网络完成预测值和损失函数的计算。

求解参数w和b的数值，这个过程也称为模型训练过程。训练过程是深度学习模型的关键要素之一.其目标是让定义的损失函数Loss尽可能的小，也就是说找到一个参数解w和b，使得损失函数取得极小值。

![image-20220712192218680](https://img2023.cnblogs.com/blog/2554043/202212/2554043-20221204183953058-434479416.png)

![image-20220712181634807](https://img2023.cnblogs.com/blog/2554043/202212/2554043-20221204183957520-66176204.png)

处于曲线极值点时的斜率为0，即函数在极值点的导数为0。那么，让损失函数取极小值的w和b应该是下述方程组的解：

![image-20220712181707389](https://img2023.cnblogs.com/blog/2554043/202212/2554043-20221204183955414-234860097.png)

其中L表示的是损失函数的值，w为模型权重，b为偏置项。w和b均为要学习的模型参数。

把损失函数表示成矩阵的形式为

![image-20220712181911116](https://img2023.cnblogs.com/blog/2554043/202212/2554043-20221204183959678-638252100.png)

其中y为N个样本的标签值构成的向量，形状为N×1；**X**为N个样本特征向量构成的矩阵，形状为N×D，D为数据特征长度；**w**为权重向量，形状为D×1；**b**为所有元素都为b的向量，形状为N×1。

计算公式7对参数**b**的偏导数

![image-20220712183050065](https://img2023.cnblogs.com/blog/2554043/202212/2554043-20221204184004890-1398556126.png)

请注意，上述公式忽略了系数2/N，并不影响最后结果。其中**1**为N维的全1向量。

令公式8等于0，得到

![image-20220712183403470](https://img2023.cnblogs.com/blog/2554043/202212/2554043-20221204184002718-1463531149.png)

#### 梯度向下发(盲人下坡法)

![image-20220712191519562](https://img2023.cnblogs.com/blog/2554043/202212/2554043-20221204184009933-1226569764.png)

这种情况特别类似于一位想从山峰走到坡谷的盲人，他看不见坡谷在哪（无法逆向求解出Loss导数为0时的参数值），但可以伸脚探索身边的坡度（当前点的导数值，也称为梯度）。

```python
net = Network(13)
losses = []
#只画出参数w5和w9在区间[-160, 160]的曲线部分，以及包含损失函数的极值
w5 = np.arange(-160.0, 160.0, 1.0)
w9 = np.arange(-160.0, 160.0, 1.0)
losses = np.zeros([len(w5), len(w9)])

#计算设定区域内每个参数取值所对应的Loss
for i in range(len(w5)):
    for j in range(len(w9)):
        net.w[5] = w5[i]
        net.w[9] = w9[j]
        z = net.forward(x)
        loss = net.loss(z, y)
        losses[i, j] = loss

#使用matplotlib将两个变量和对应的Loss作3D图
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)

w5, w9 = np.meshgrid(w5, w9)

ax.plot_surface(w5, w9, losses, rstride=1, cstride=1, cmap='rainbow')
plt.show()
```

![image-20220712191620448](https://img2023.cnblogs.com/blog/2554043/202212/2554043-20221204184014767-1082764838.png)

均方误差表现的“圆滑”的坡度有两个好处：

- 曲线的最低点是可导的。
- 越接近最低点，曲线的坡度逐渐放缓，有助于通过当前的梯度来判断接近最低点的程度（是否逐渐减少步长，以免错过最低点）。

而绝对值误差是不具备这两个特性的，这也是损失函数的设计不仅仅要考虑“合理性”，还要追求“易解性”的原因。

![image-20220712192140185](https://img2023.cnblogs.com/blog/2554043/202212/2554043-20221204184018567-1371643396.png)

梯度下降方向示意图

#### 梯度计算

##### 计算梯度的公式推导

![image-20220712192457976](https://img2023.cnblogs.com/blog/2554043/202212/2554043-20221204184019761-1012401401.png)

![image-20220712192546590](https://img2023.cnblogs.com/blog/2554043/202212/2554043-20221204184022407-931898552.png)



可以通过具体的程序查看每个变量的数据和维度。

```
x1 = x[0]
y1 = y[0]
z1 = net.forward(x1)
print('x1 {}, shape {}'.format(x1, x1.shape))
print('y1 {}, shape {}'.format(y1, y1.shape))
print('z1 {}, shape {}'.format(z1, z1.shape))

x1 [-0.02146321  0.03767327 -0.28552309 -0.08663366  0.01289726  0.04634817
  0.00795597 -0.00765794 -0.25172191 -0.11881188 -0.29002528  0.0519112
 -0.17590923], shape (13,)
y1 [-0.00390539], shape (1,)
z1 [-12.05947643], shape (1,)
```

按上面的公式，当只有一个样本时，可以计算某个wj，比如w0的梯度。

In [19]

```
gradient_w0 = (z1 - y1) * x1[0]
print('gradient_w0 {}'.format(gradient_w0))
gradient_w0 [0.25875126]
```

同样我们可以计算w1的梯度。

In [20]

```
gradient_w1 = (z1 - y1) * x1[1]
print('gradient_w1 {}'.format(gradient_w1))
gradient_w1 [-0.45417275]
```

依次计算w2的梯度。

In [21]

```
gradient_w2= (z1 - y1) * x1[2]
print('gradient_w1 {}'.format(gradient_w2))
gradient_w1 [3.44214394]
```

![image-20220712193357081](https://img2023.cnblogs.com/blog/2554043/202212/2554043-20221204184026215-1588899498.png)

- 基于Numpy的广播机制，扩展参数的维度

使用Numpy矩阵操作，计算梯度的代码中直接用(z1-y1)*x1，得到一个13维的向量，每个分量分别代表该维度的梯度

```python
gradient_w = (z1 - y1) * x1
print('gradient_w_by_sample1 {}, gradient.shape {}'.format(gradient_w, gradient_w.shape))
```

```
gradient_w_by_sample1 [ 0.25875126 -0.45417275  3.44214394  1.04441828 -0.15548386 -0.55875363
 -0.09591377  0.09232085  3.03465138  1.43234507  3.49642036 -0.62581917
  2.12068622], gradient.shape (13,)
```

输入数据中有多个样本，每个样本都对梯度有贡献。如上代码计算了只有样本1时的梯度值，同样的计算方法也可以计算样本2和样本3对梯度的贡献。

```python
x2 = x[1]
y2 = y[1]
z2 = net.forward(x2)
gradient_w = (z2 - y2) * x2
print('gradient_w_by_sample2 {}, gradient.shape {}'.format(gradient_w, gradient_w.shape))
```

```
gradient_w_by_sample2 [ 0.7329239   4.91417754  3.33394253  2.9912385   4.45673435 -0.58146277
 -5.14623287 -2.4894594   7.19011988  7.99471607  0.83100061 -1.79236081
  2.11028056], gradient.shape (13,)
```

```python
x3 = x[2]
y3 = y[2]
z3 = net.forward(x3)
gradient_w = (z3 - y3) * x3
print('gradient_w_by_sample3 {}, gradient.shape {}'.format(gradient_w, gradient_w.shape))
```

```
gradient_w_by_sample3 [ 0.25138584  1.68549775  1.14349809  1.02595515  1.5286008  -1.93302947
  0.4058236  -0.85385157  2.46611579  2.74208162  0.28502219 -0.46695229
  2.39363651], gradient.shape (13,)
```

可能有的读者再次想到可以使用`for`循环把每个样本对梯度的贡献都计算出来，然后再作平均。但是我们不需要这么做，仍然可以使用NumPy的矩阵操作来简化运算，如3个样本的情况。

```
# 注意这里是一次取出3个样本的数据，不是取出第3个样本
x3samples = x[0:3]
y3samples = y[0:3]
z3samples = net.forward(x3samples)

print('x {}, shape {}'.format(x3samples, x3samples.shape))
print('y {}, shape {}'.format(y3samples, y3samples.shape))
print('z {}, shape {}'.format(z3samples, z3samples.shape))

x [[-0.02146321  0.03767327 -0.28552309 -0.08663366  0.01289726  0.04634817
   0.00795597 -0.00765794 -0.25172191 -0.11881188 -0.29002528  0.0519112
  -0.17590923]
 [-0.02122729 -0.14232673 -0.09655922 -0.08663366 -0.12907805  0.0168406
   0.14904763  0.0721009  -0.20824365 -0.23154675 -0.02406783  0.0519112
  -0.06111894]
 [-0.02122751 -0.14232673 -0.09655922 -0.08663366 -0.12907805  0.1632288
  -0.03426854  0.0721009  -0.20824365 -0.23154675 -0.02406783  0.03943037
  -0.20212336]], shape (3, 13)
y [[-0.00390539]
 [-0.05723872]
 [ 0.23387239]], shape (3, 1)
z [[-12.05947643]
 [-34.58467747]
 [-11.60858134]], shape (3, 1)
```



x3samples、 y3samples 和 z3samples的第一维大小均为3，表示有3个样本,下面计算这3个样本对梯度的贡献。

```python
gradient_w = (z3samples - y3samples) * x3samples
print('gradient_w {}, gradient.shape {}'.format(gradient_w, gradient_w.shape))


gradient_w [[ 0.25875126 -0.45417275  3.44214394  1.04441828 -0.15548386 -0.55875363
  -0.09591377  0.09232085  3.03465138  1.43234507  3.49642036 -0.62581917
   2.12068622]
 [ 0.7329239   4.91417754  3.33394253  2.9912385   4.45673435 -0.58146277
  -5.14623287 -2.4894594   7.19011988  7.99471607  0.83100061 -1.79236081
   2.11028056]
 [ 0.25138584  1.68549775  1.14349809  1.02595515  1.5286008  -1.93302947
   0.4058236  -0.85385157  2.46611579  2.74208162  0.28502219 -0.46695229
   2.39363651]], gradient.shape (3, 13)
```

此处可见，计算梯度`gradient_w`的维度是3×13，并且其第1行与上面第1个样本计算的梯度gradient_w_by_sample1一致，第2行与上面第2个样本计算的梯度gradient_w_by_sample2一致，第3行与上面第3个样本计算的梯度gradient_w_by_sample3一致。这里使用矩阵操作，可以更加方便的对3个样本分别计算各自对梯度的贡献。

那么对于有N个样本的情形，我们可以直接使用如下方式计算出所有样本对梯度的贡献，这就是使用NumPy库广播功能带来的便捷。 小结一下这里使用NumPy库的广播功能：

- 一方面可以扩展参数的维度，代替for循环来计算1个样本对从w0到w1 2的所有参数的梯度。
- 另一方面可以扩展样本的维度，代替for循环来计算样本0到样本403对参数的梯度。

```
z = net.forward(x)
gradient_w = (z - y) * x
print('gradient_w shape {}'.format(gradient_w.shape))
print(gradient_w)

gradient_w shape (404, 13)
[[  0.25875126  -0.45417275   3.44214394 ...   3.49642036  -0.62581917
    2.12068622]
 [  0.7329239    4.91417754   3.33394253 ...   0.83100061  -1.79236081
    2.11028056]
 [  0.25138584   1.68549775   1.14349809 ...   0.28502219  -0.46695229
    2.39363651]
 ...
 [ 14.70025543 -15.10890735  36.23258734 ...  24.54882966   5.51071122
   26.26098922]
 [  9.29832217 -15.33146159  36.76629344 ...  24.91043398  -1.27564923
   26.61808955]
 [ 19.55115919 -10.8177237   25.94192351 ...  17.5765494    3.94557661
   17.64891012]]
```

上面gradient_w的每一行代表了一个样本对梯度的贡献。根据梯度的计算公式，总梯度是对每个样本对梯度贡献的平均值。



可以使用NumPy的均值函数来完成此过程，代码实现如下。

In [28]

```
# axis = 0 表示把每一行做相加然后再除以总的行数
gradient_w = np.mean(gradient_w, axis=0)
print('gradient_w ', gradient_w.shape)
print('w ', net.w.shape)
print(gradient_w)
print(net.w)

gradient_w  (13,)
w  (13, 1)
[ 1.59697064 -0.92928123  4.72726926  1.65712204  4.96176389  1.18068454
  4.55846519 -3.37770889  9.57465893 10.29870662  1.3900257  -0.30152215
  1.09276043]
[[ 1.76405235e+00]
 [ 4.00157208e-01]
 [ 9.78737984e-01]
 [ 2.24089320e+00]
 [ 1.86755799e+00]
 [ 1.59000000e+02]
 [ 9.50088418e-01]
 [-1.51357208e-01]
 [-1.03218852e-01]
 [ 1.59000000e+02]
 [ 1.44043571e-01]
 [ 1.45427351e+00]
 [ 7.61037725e-01]]
```

使用NumPy的矩阵操作方便地完成了gradient的计算，但引入了一个问题，`gradient_w`的形状是(13,)，而w的维度是(13, 1)。导致该问题的原因是使用`np.mean`函数时消除了第0维。为了加减乘除等计算方便，`gradient_w`和w必须保持一致的形状。因此我们将`gradient_w`的维度也设置为(13,1)，代码如下：

```
gradient_w = gradient_w[:, np.newaxis]
print('gradient_w shape', gradient_w.shape)
gradient_w shape (13, 1)
```

综合上面的剖析，计算梯度的代码如下所示。

In [30]

```python
z = net.forward(x)
gradient_w = (z - y) * x
gradient_w = np.mean(gradient_w, axis=0)
gradient_w = gradient_w[:, np.newaxis]
gradient_w
array([[ 1.59697064],
       [-0.92928123],
       [ 4.72726926],
       [ 1.65712204],
       [ 4.96176389],
       [ 1.18068454],
       [ 4.55846519],
       [-3.37770889],
       [ 9.57465893],
       [10.29870662],
       [ 1.3900257 ],
       [-0.30152215],
       [ 1.09276043]])
```

上述代码非常简洁地完成了w的梯度计算。同样，计算b的梯度的代码也是类似的原理。

In [31]

```python
gradient_b = (z - y)
gradient_b = np.mean(gradient_b)
# 此处b是一个数值，所以可以直接用np.mean得到一个标量
gradient_b
-1.0918438870293816e-13
```

将上面计算w和b的梯度的过程，写成Network类的`gradient`函数，实现方法如下所示。

In [32]

```python
class Network(object):
    def __init__(self, num_of_weights):
        # 随机产生w的初始值
        # 为了保持程序每次运行结果的一致性，此处设置固定的随机数种子
        np.random.seed(0)
        self.w = np.random.randn(num_of_weights, 1)
        self.b = 0.
        
    def forward(self, x):
        z = np.dot(x, self.w) + self.b
        return z
    
    def loss(self, z, y):
        error = z - y
        num_samples = error.shape[0]
        cost = error * error
        cost = np.sum(cost) / num_samples
        return cost
    
    def gradient(self, x, y):
        z = self.forward(x)
        gradient_w = (z-y)*x
        gradient_w = np.mean(gradient_w, axis=0)
        gradient_w = gradient_w[:, np.newaxis]
        gradient_b = (z - y)
        gradient_b = np.mean(gradient_b)
        
        return gradient_w, gradient_b
```

In [33]

```python
# 调用上面定义的gradient函数，计算梯度
# 初始化网络
net = Network(13)
# 设置[w5, w9] = [-100., -100.]
net.w[5] = -100.0
net.w[9] = -100.0

z = net.forward(x)
loss = net.loss(z, y)
gradient_w, gradient_b = net.gradient(x, y)
gradient_w5 = gradient_w[5][0]
gradient_w9 = gradient_w[9][0]
print('point {}, loss {}'.format([net.w[5][0], net.w[9][0]], loss))
print('gradient {}'.format([gradient_w5, gradient_w9]))
point [-100.0, -100.0], loss 686.3005008179159
gradient [-0.850073323995813, -6.138412364807849]
```

#### 梯度更新

下面研究更新梯度的方法，确定损失函数更小的点。首先沿着梯度的反方向移动一小步，找到下一个点P1，观察损失函数的变化。

In [34]

```python
# 在[w5, w9]平面上，沿着梯度的反方向移动到下一个点P1
# 定义移动步长 eta
eta = 0.1
# 更新参数w5和w9
net.w[5] = net.w[5] - eta * gradient_w5
net.w[9] = net.w[9] - eta * gradient_w9
# 重新计算z和loss
z = net.forward(x)
loss = net.loss(z, y)
gradient_w, gradient_b = net.gradient(x, y)
gradient_w5 = gradient_w[5][0]
gradient_w9 = gradient_w[9][0]
print('point {}, loss {}'.format([net.w[5][0], net.w[9][0]], loss))
print('gradient {}'.format([gradient_w5, gradient_w9]))
point [-99.91499266760042, -99.38615876351922], loss 678.6472185028845
gradient [-0.8556356178645292, -6.0932268634065805]
```

运行上面的代码，可以发现沿着梯度反方向走一小步，下一个点的损失函数的确减少了。感兴趣的话，大家可以尝试不停的点击上面的代码块，观察损失函数是否一直在变小。

在上述代码中，每次更新参数使用的语句： `net.w[5] = net.w[5] - eta * gradient_w5`

- 相减：参数需要向梯度的反方向移动。
- eta：控制每次参数值沿着梯度反方向变动的大小，即每次移动的步长，又称为学习率。

大家可以思考下，为什么之前我们要做输入特征的归一化，保持尺度一致？这是为了让统一的步长更加合适，使训练更加高效。

如 **图8** 所示，特征输入归一化后，不同参数输出的Loss是一个比较规整的曲线，学习率可以设置成统一的值 ；特征输入未归一化时，不同特征对应的参数所需的步长不一致，尺度较大的参数需要大步长，尺寸较小的参数需要小步长，导致无法设置统一的学习率。

![image-20220714090958798](https://img2023.cnblogs.com/blog/2554043/202212/2554043-20221204184053591-14179715.png)

#### 封装Train函数

将上面的循环计算过程封装在`train`和`update`函数中，实现方法如下所示

```python
class Network(object):
    def __init__(self, num_of_weights):
        # 随机产生w的初始值
        # 为了保持程序每次运行结果的一致性，此处设置固定的随机数种子
        np.random.seed(0)
        self.w = np.random.randn(num_of_weights,1)
        self.w[5] = -100.
        self.w[9] = -100.
        self.b = 0.
        
    def forward(self, x):
        z = np.dot(x, self.w) + self.b
        return z
    
    def loss(self, z, y):
        error = z - y
        num_samples = error.shape[0]
        cost = error * error
        cost = np.sum(cost) / num_samples
        return cost
    
    def gradient(self, x, y):
        z = self.forward(x)
        gradient_w = (z-y)*x
        gradient_w = np.mean(gradient_w, axis=0)
        gradient_w = gradient_w[:, np.newaxis]
        gradient_b = (z - y)
        gradient_b = np.mean(gradient_b)        
        return gradient_w, gradient_b
    
    def update(self, gradient_w5, gradient_w9, eta=0.01):
        net.w[5] = net.w[5] - eta * gradient_w5
        net.w[9] = net.w[9] - eta * gradient_w9
        
    def train(self, x, y, iterations=100, eta=0.01):
        points = []
        losses = []
        for i in range(iterations):
            points.append([net.w[5][0], net.w[9][0]])
            z = self.forward(x)
            L = self.loss(z, y)
            gradient_w, gradient_b = self.gradient(x, y)
            gradient_w5 = gradient_w[5][0]
            gradient_w9 = gradient_w[9][0]
            self.update(gradient_w5, gradient_w9, eta)
            losses.append(L)
            if i % 50 == 0:
                print('iter {}, point {}, loss {}'.format(i, [net.w[5][0], net.w[9][0]], L))
        return points, losses

# 获取数据
train_data, test_data = load_data()
x = train_data[:, :-1]
y = train_data[:, -1:]
# 创建网络
net = Network(13)
num_iterations=2000
# 启动训练
points, losses = net.train(x, y, iterations=num_iterations, eta=0.01)

# 画出损失函数的变化趋势
plot_x = np.arange(num_iterations)
plot_y = np.array(losses)
plt.plot(plot_x, plot_y)
plt.show()
```

```
iter 0, point [-99.99149926676004, -99.93861587635192], loss 686.3005008179159
iter 50, point [-99.55950291459486, -96.92630620094545], loss 649.2144359737484
iter 100, point [-99.1143836791974, -94.0227231684414], loss 614.6583805120641
iter 150, point [-98.65689507661327, -91.22377715643974], loss 582.4474015283215
iter 200, point [-98.18775974718623, -88.52553386435612], loss 552.4103147331971
iter 250, point [-97.70767065020402, -85.92420840138672], loss 524.388658297806
iter 300, point [-97.21729221288223, -83.41615959961561], loss 498.23574335719394
iter 350, point [-96.71726143542814, -80.99788454368958], loss 473.81577544420406
iter 400, point [-96.20818895385739, -78.66601330881193], loss 451.0030415538293
iter 450, point [-95.69066006217508, -76.41730389912176], loss 429.6811579341994
iter 500, point [-95.16523569546983, -74.24863737882666], loss 409.74237406676986
iter 550, point [-94.63245337541025, -72.1570131887469], loss 391.0869286373302
iter 600, point [-94.09282811957833, -70.13954464121011], loss 373.62245361317457
iter 650, point [-93.5468533160177, -68.19345458650403], loss 357.26342283205247
iter 700, point [-92.99500156432298, -66.31607124435298], loss 341.9306417770852
iter 750, point [-92.43772548454575, -64.50482419413264], loss 327.55077546036
iter 800, point [-91.87545849514491, -62.757240517778385], loss 314.05591156785766
iter 850, point [-91.30861556116066, -61.07094108957052], loss 301.3831562311184
iter 900, point [-90.73759391374827, -59.44363700720379], loss 289.47425998793403
iter 950, point [-90.16277374216338, -57.873126158759185], loss 278.2752716764845
iter 1000, point [-89.58451885924933, -56.35728992040244], loss 267.73621817589355
iter 1050, point [-89.00317734143711, -54.894089979830845], loss 257.8108080621227
iter 1100, point [-88.41908214422965, -53.481565280678495], loss 248.45615739241018
iter 1150, point [-87.83255169410579, -52.11782908327363], loss 239.63253596498384
iter 1200, point [-87.24389045774262, -50.801066137316646], loss 231.30313252430267
iter 1250, point [-86.65338948942181, -49.52952996221624], loss 223.43383749639403
iter 1300, point [-86.06132695745193, -48.301540230983434], loss 215.9930419446124
iter 1350, point [-85.46796865040659, -47.11548025373949], loss 208.95145153400557
iter 1400, point [-84.87356846394941, -45.969794557044004], loss 202.28191438302676
iter 1450, point [-84.27836886898521, -44.86298655539359], loss 195.95926176510483
iter 1500, point [-83.68260136185022, -43.793616311381186], loss 189.96016070011396
iter 1550, point [-83.08648689722693, -42.76029838113894], loss 184.26297754750686
iter 1600, point [-82.49023630444216, -41.76169974181719], loss 178.84765177925001
iter 1650, point [-81.89405068778211, -40.796537797974494], loss 173.6955791720998
iter 1700, point [-81.29812181143511, -39.86357846387417], loss 168.789503715592
iter 1750, point [-80.70263246964815, -38.961634318795845], loss 164.11341758467972
iter 1800, point [-80.10775684266093, -38.0895628325817], loss 159.65246857460858
iter 1850, point [-79.51366083896096, -37.246264658742334], loss 155.39287444062649
iter 1900, point [-78.92050242438167, -36.43068199254937], loss 151.32184362677512
iter 1950, point [-78.32843193854492, -35.64179699163965], loss 147.42750190654235
```

![img]()

#### 训练过程扩展到全部参数

为了能给读者直观的感受，上文演示的梯度下降的过程仅包含w5和w9两个参数。但房价预测的模型必须要对所有参数w和b进行求解，这需要将Network中的`update`和`train`函数进行修改。由于不再限定参与计算的参数（所有参数均参与计算），修改之后的代码反而更加简洁。

实现逻辑：“前向计算输出、根据输出和真实值计算Loss、基于Loss和输入计算梯度、根据梯度更新参数值”四个部分反复执行，直到到损失函数最小。

```python
class Network(object):
    def __init__(self, num_of_weights):
        # 随机产生w的初始值
        # 为了保持程序每次运行结果的一致性，此处设置固定的随机数种子
        np.random.seed(0)
        self.w = np.random.randn(num_of_weights, 1)
        self.b = 0.
        
    def forward(self, x):
        z = np.dot(x, self.w) + self.b
        return z
    
    def loss(self, z, y):
        error = z - y
        num_samples = error.shape[0]
        cost = error * error
        cost = np.sum(cost) / num_samples
        return cost
    
    def gradient(self, x, y):
        z = self.forward(x)
        gradient_w = (z-y)*x
        gradient_w = np.mean(gradient_w, axis=0)
        gradient_w = gradient_w[:, np.newaxis]
        gradient_b = (z - y)
        gradient_b = np.mean(gradient_b)        
        return gradient_w, gradient_b
    
    def update(self, gradient_w, gradient_b, eta = 0.01):
        self.w = self.w - eta * gradient_w
        self.b = self.b - eta * gradient_b
        
    def train(self, x, y, iterations=100, eta=0.01):
        losses = []
        for i in range(iterations):
            z = self.forward(x)
            L = self.loss(z, y)
            gradient_w, gradient_b = self.gradient(x, y)
            self.update(gradient_w, gradient_b, eta)
            losses.append(L)
            if (i+1) % 10 == 0:
                print('iter {}, loss {}'.format(i, L))
        return losses

# 获取数据
train_data, test_data = load_data()
x = train_data[:, :-1]
y = train_data[:, -1:]
# 创建网络
net = Network(13)
num_iterations=1000
# 启动训练
losses = net.train(x,y, iterations=num_iterations, eta=0.01)

# 画出损失函数的变化趋势
plot_x = np.arange(num_iterations)
plot_y = np.array(losses)
plt.plot(plot_x, plot_y)
plt.show()
```



```
iter 9, loss 1.8984947314576224
iter 19, loss 1.8031783384598725
iter 29, loss 1.7135517565541092
iter 39, loss 1.6292649416831264
iter 49, loss 1.5499895293373231
iter 59, loss 1.4754174896452612
iter 69, loss 1.4052598659324693
iter 79, loss 1.3392455915676864
iter 89, loss 1.2771203802372917
iter 99, loss 1.218645685090292
iter 109, loss 1.1635977224791534
iter 119, loss 1.111766556287068
iter 129, loss 1.0629552390811503
iter 139, loss 1.0169790065644477
iter 149, loss 0.9736645220185994
iter 159, loss 0.9328491676343147
iter 169, loss 0.8943803798194309
iter 179, loss 0.8581150257549611
iter 189, loss 0.8239188186389669
iter 199, loss 0.7916657692169988
iter 209, loss 0.761237671346902
iter 219, loss 0.7325236194855752
iter 229, loss 0.7054195561163928
iter 239, loss 0.6798278472589763
iter 249, loss 0.6556568843183528
iter 259, loss 0.6328207106387195
iter 269, loss 0.6112386712285092
iter 279, loss 0.59083508421862
iter 289, loss 0.5715389327049418
iter 299, loss 0.5532835757100347
iter 309, loss 0.5360064770773406
iter 319, loss 0.5196489511849665
iter 329, loss 0.5041559244351538
iter 339, loss 0.48947571154034963
iter 349, loss 0.4755598056875569
iter 359, loss 0.46236268171965056
iter 369, loss 0.44984161152579916
iter 379, loss 0.43795649088328303
iter 389, loss 0.4266696770400226
iter 399, loss 0.4159458363712466
iter 409, loss 0.4057518014851036
iter 419, loss 0.3960564371908221
iter 429, loss 0.38683051477942226
iter 439, loss 0.3780465941011246
iter 449, loss 0.36967891295560856
iter 459, loss 0.3617032833413179
iter 469, loss 0.3540969941381647
iter 479, loss 0.3468387198244131
iter 489, loss 0.3399084348532937
iter 499, loss 0.33328733333814486
iter 509, loss 0.3269577537166779
iter 519, loss 0.32090310808539985
iter 529, loss 0.31510781591441284
iter 539, loss 0.30955724187078903
iter 549, loss 0.3042376374955925
iter 559, loss 0.2991360864954391
iter 569, loss 0.2942404534243286
iter 579, loss 0.2895393355454012
iter 589, loss 0.28502201767532415
iter 599, loss 0.2806784298262616
iter 609, loss 0.27649910747186535
iter 619, loss 0.2724751542744919
iter 629, loss 0.2685982071209627
iter 639, loss 0.26486040332365085
iter 649, loss 0.2612543498525749
iter 659, loss 0.2577730944725093
iter 669, loss 0.2544100986669443
iter 679, loss 0.2511592122380609
iter 689, loss 0.2480146494787638
iter 699, loss 0.24497096681926708
iter 709, loss 0.2420230418567802
iter 719, loss 0.23916605368251415
iter 729, loss 0.23639546442555454
iter 739, loss 0.23370700193813698
iter 749, loss 0.23109664355154746
iter 759, loss 0.2285606008362593
iter 769, loss 0.22609530530403904
iter 779, loss 0.22369739499361888
iter 789, loss 0.22136370188515422
iter 799, loss 0.21909124009208833
iter 809, loss 0.21687719478222933
iter 819, loss 0.2147189117828403
iter 829, loss 0.21261388782734392
iter 839, loss 0.2105597614038757
iter 849, loss 0.20855430416838638
iter 859, loss 0.2065954128873093
iter 869, loss 0.20468110187697833
iter 879, loss 0.2028094959090178
iter 889, loss 0.20097882355283644
iter 899, loss 0.19918741092814596
iter 909, loss 0.19743367584210875
iter 919, loss 0.1957161222872899
iter 929, loss 0.19403333527807182
iter 939, loss 0.19238397600456975
iter 949, loss 0.19076677728439412
iter 959, loss 0.1891805392938162
iter 969, loss 0.18762412556104593
iter 979, loss 0.18609645920539716
iter 989, loss 0.18459651940712488
iter 999, loss 0.18312333809366155
```

![img](data:image/png;base64,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)

#### 随机梯度下降法

##### mini-batch、batch_size、epoch

- mini-batch：每次迭代时抽取出来的一批数据被称为一个mini-batch。
- batch_size：一个mini-batch所包含的样本数目称为batch_size。
- epoch：当程序迭代的时候，按mini-batch逐渐抽取出样本，当把整个数据集都遍历到了的时候，则完成了一轮训练，也叫一个epoch。启动训练时，可以将训练的轮数num_epochs和batch_size作为参数传入。



- **数据处理代码修改**

数据处理需要实现拆分数据批次和样本乱序（为了实现随机抽样的效果）两个功能。

```python
# 获取数据
train_data, test_data = load_data()
train_data.shape
(404, 14)
```



train_data中一共包含404条数据，如果batch_size=10，即取前0-9号样本作为第一个mini-batch，命名train_data1。

```python
train_data1 = train_data[0:10]
train_data1.shape
(10, 14)
```



使用train_data1的数据（0-9号样本）计算梯度并更新网络参数。

```python
net = Network(13)
x = train_data1[:, :-1]
y = train_data1[:, -1:]
loss = net.train(x, y, iterations=1, eta=0.01)
loss
[0.9001866101467375]
```



再取出10-19号样本作为第二个mini-batch，计算梯度并更新网络参数。

```python
train_data2 = train_data[10:20]
x = train_data2[:, :-1]
y = train_data2[:, -1:]
loss = net.train(x, y, iterations=1, eta=0.01)
loss
[0.1171681170130832]
```

按此方法不断的取出新的mini-batch，并逐渐更新网络参数。

接下来，将train_data分成大小为batch_size的多个mini_batch，如下代码所示：将train_data分成404/10+1=41个 mini_batch，其中前40个mini_batch，每个均含有10个样本，最后一个mini_batch只含有4个样本。

```python
batch_size = 10
n = len(train_data)
mini_batches = [train_data[k:k+batch_size] for k in range(0, n, batch_size)]
print('total number of mini_batches is ', len(mini_batches))
print('first mini_batch shape ', mini_batches[0].shape)
print('last mini_batch shape ', mini_batches[-1].shape)
total number of mini_batches is  41
first mini_batch shape  (10, 14)
last mini_batch shape  (4, 14)
```

另外，这里是按顺序读取mini_batch，而SGD里面是随机抽取一部分样本代表总体。为了实现随机抽样的效果，我们先将train_data里面的样本顺序随机打乱，然后再抽取mini_batch。随机打乱样本顺序，需要用到`np.random.shuffle`函数，下面先介绍它的用法。

****

**说明：**

通过大量实验发现，模型受训练后期的影响更大，类似于人脑总是对近期发生的事情记忆的更加清晰。为了避免数据样本集合的顺序干扰模型的训练效果，需要进行样本乱序操作。当然，如果训练样本的顺序就是样本产生的顺序，而我们期望模型更重视近期产生的样本（预测样本会和近期的训练样本分布更接近），则不需要乱序这个步骤。

*****



```python
# 新建一个array
a = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
print('before shuffle', a)
np.random.shuffle(a)
print('after shuffle', a)

before shuffle [ 1  2  3  4  5  6  7  8  9 10 11 12]
after shuffle [ 7  2 11  3  8  6 12  1  4  5 10  9]
```

多次运行上面的代码，可以发现每次执行shuffle函数后的数字顺序均不同。 上面举的是一个1维数组乱序的案例，我们再观察下2维数组乱序后的效果。



```
# 新建一个array
a = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
a = a.reshape([6, 2])
print('before shuffle\n', a)
np.random.shuffle(a)
print('after shuffle\n', a)
before shuffle
 [[ 1  2]
 [ 3  4]
 [ 5  6]
 [ 7  8]
 [ 9 10]
 [11 12]]
after shuffle
 [[ 1  2]
 [ 3  4]
 [ 5  6]
 [ 9 10]
 [11 12]
 [ 7  8]]
```



观察运行结果可发现，数组的元素在第0维被随机打乱，但第1维的顺序保持不变。例如数字2仍然紧挨在数字1的后面，数字8仍然紧挨在数字7的后面，而第二维的[3, 4]并不排在[1, 2]的后面。将这部分实现SGD算法的代码集成到Network类中的`train`函数中，最终的完整代码如下。



```python
# 获取数据
train_data, test_data = load_data()

# 打乱样本顺序
np.random.shuffle(train_data)

# 将train_data分成多个mini_batch
batch_size = 10
n = len(train_data)
mini_batches = [train_data[k:k+batch_size] for k in range(0, n, batch_size)]

# 创建网络
net = Network(13)

# 依次使用每个mini_batch的数据
for mini_batch in mini_batches:
    x = mini_batch[:, :-1]
    y = mini_batch[:, -1:]
    loss = net.train(x, y, iterations=1)
```



- **训练过程代码修改**

将每个随机抽取的mini-batch数据输入到模型中用于参数训练。训练过程的核心是两层循环：

1. 第一层循环，代表样本集合要被训练遍历几次，称为“epoch”，代码如下：

```
for epoch_id in range(num_epochs):
```



1. 第二层循环，代表每次遍历时，样本集合被拆分成的多个批次，需要全部执行训练，称为“iter (iteration)”，代码如下：

```
for iter_id,mini_batch in emumerate(mini_batches):
```

在两层循环的内部是经典的四步训练流程：前向计算->计算损失->计算梯度->更新参数，这与大家之前所学是一致的，代码如下：

```
 			x = mini_batch[:, :-1]
            y = mini_batch[:, -1:]
            a = self.forward(x)  #前向计算
            loss = self.loss(a, y)  #计算损失
            gradient_w, gradient_b = self.gradient(x, y)  #计算梯度
            self.update(gradient_w, gradient_b, eta)  #更新参数
```

将两部分改写的代码集成到Network类中的`train`函数中，最终的实现如下。

```python
import numpy as np

class Network(object):
    def __init__(self, num_of_weights):
        # 随机产生w的初始值
        # 为了保持程序每次运行结果的一致性，此处设置固定的随机数种子
        #np.random.seed(0)
        self.w = np.random.randn(num_of_weights, 1)
        self.b = 0.
        
    def forward(self, x):
        z = np.dot(x, self.w) + self.b
        return z
    
    def loss(self, z, y):
        error = z - y
        num_samples = error.shape[0]
        cost = error * error
        cost = np.sum(cost) / num_samples
        return cost
    
    def gradient(self, x, y):
        z = self.forward(x)
        N = x.shape[0]
        gradient_w = 1. / N * np.sum((z-y) * x, axis=0)
        gradient_w = gradient_w[:, np.newaxis]
        gradient_b = 1. / N * np.sum(z-y)
        return gradient_w, gradient_b
    
    def update(self, gradient_w, gradient_b, eta = 0.01):
        self.w = self.w - eta * gradient_w
        self.b = self.b - eta * gradient_b
            
                
    def train(self, training_data, num_epochs, batch_size=10, eta=0.01):
        n = len(training_data)
        losses = []
        for epoch_id in range(num_epochs):
            # 在每轮迭代开始之前，将训练数据的顺序随机打乱
            # 然后再按每次取batch_size条数据的方式取出
            np.random.shuffle(training_data)
            # 将训练数据进行拆分，每个mini_batch包含batch_size条的数据
            mini_batches = [training_data[k:k+batch_size] for k in range(0, n, batch_size)]
            for iter_id, mini_batch in enumerate(mini_batches):
                #print(self.w.shape)
                #print(self.b)
                x = mini_batch[:, :-1]
                y = mini_batch[:, -1:]
                a = self.forward(x)
                loss = self.loss(a, y)
                gradient_w, gradient_b = self.gradient(x, y)
                self.update(gradient_w, gradient_b, eta)
                losses.append(loss)
                print('Epoch {:3d} / iter {:3d}, loss = {:.4f}'.
                                 format(epoch_id, iter_id, loss))
        
        return losses

# 获取数据
train_data, test_data = load_data()

# 创建网络
net = Network(13)
# 启动训练
losses = net.train(train_data, num_epochs=50, batch_size=100, eta=0.1)

# 画出损失函数的变化趋势
plot_x = np.arange(len(losses))
plot_y = np.array(losses)
plt.plot(plot_x, plot_y)
plt.show()
```



```
Epoch   0 / iter   0, loss = 0.6273
Epoch   0 / iter   1, loss = 0.4835
Epoch   0 / iter   2, loss = 0.5830
Epoch   0 / iter   3, loss = 0.5466
Epoch   0 / iter   4, loss = 0.2147
Epoch   1 / iter   0, loss = 0.6645
Epoch   1 / iter   1, loss = 0.4875
Epoch   1 / iter   2, loss = 0.4707
Epoch   1 / iter   3, loss = 0.4153
Epoch   1 / iter   4, loss = 0.1402
Epoch   2 / iter   0, loss = 0.5897
Epoch   2 / iter   1, loss = 0.4373
Epoch   2 / iter   2, loss = 0.4631
Epoch   2 / iter   3, loss = 0.3960
Epoch   2 / iter   4, loss = 0.2340
Epoch   3 / iter   0, loss = 0.4139
Epoch   3 / iter   1, loss = 0.5635
Epoch   3 / iter   2, loss = 0.3807
Epoch   3 / iter   3, loss = 0.3975
Epoch   3 / iter   4, loss = 0.1207
Epoch   4 / iter   0, loss = 0.3786
Epoch   4 / iter   1, loss = 0.4474
Epoch   4 / iter   2, loss = 0.4019
Epoch   4 / iter   3, loss = 0.4352
Epoch   4 / iter   4, loss = 0.0435
Epoch   5 / iter   0, loss = 0.4387
Epoch   5 / iter   1, loss = 0.3886
Epoch   5 / iter   2, loss = 0.3182
Epoch   5 / iter   3, loss = 0.4189
Epoch   5 / iter   4, loss = 0.1741
Epoch   6 / iter   0, loss = 0.3191
Epoch   6 / iter   1, loss = 0.3601
Epoch   6 / iter   2, loss = 0.4199
Epoch   6 / iter   3, loss = 0.3289
Epoch   6 / iter   4, loss = 1.2691
Epoch   7 / iter   0, loss = 0.3202
Epoch   7 / iter   1, loss = 0.2855
Epoch   7 / iter   2, loss = 0.4129
Epoch   7 / iter   3, loss = 0.3331
Epoch   7 / iter   4, loss = 0.2218
Epoch   8 / iter   0, loss = 0.2368
Epoch   8 / iter   1, loss = 0.3457
Epoch   8 / iter   2, loss = 0.3339
Epoch   8 / iter   3, loss = 0.3812
Epoch   8 / iter   4, loss = 0.0534
Epoch   9 / iter   0, loss = 0.3567
Epoch   9 / iter   1, loss = 0.4033
Epoch   9 / iter   2, loss = 0.1926
Epoch   9 / iter   3, loss = 0.2803
Epoch   9 / iter   4, loss = 0.1557
Epoch  10 / iter   0, loss = 0.3435
Epoch  10 / iter   1, loss = 0.2790
Epoch  10 / iter   2, loss = 0.3456
Epoch  10 / iter   3, loss = 0.2076
Epoch  10 / iter   4, loss = 0.0935
Epoch  11 / iter   0, loss = 0.3024
Epoch  11 / iter   1, loss = 0.2517
Epoch  11 / iter   2, loss = 0.2797
Epoch  11 / iter   3, loss = 0.2989
Epoch  11 / iter   4, loss = 0.0301
Epoch  12 / iter   0, loss = 0.2507
Epoch  12 / iter   1, loss = 0.2563
Epoch  12 / iter   2, loss = 0.2971
Epoch  12 / iter   3, loss = 0.2833
Epoch  12 / iter   4, loss = 0.0597
Epoch  13 / iter   0, loss = 0.2827
Epoch  13 / iter   1, loss = 0.2094
Epoch  13 / iter   2, loss = 0.2417
Epoch  13 / iter   3, loss = 0.2985
Epoch  13 / iter   4, loss = 0.4036
Epoch  14 / iter   0, loss = 0.3085
Epoch  14 / iter   1, loss = 0.2015
Epoch  14 / iter   2, loss = 0.1830
Epoch  14 / iter   3, loss = 0.2978
Epoch  14 / iter   4, loss = 0.0630
Epoch  15 / iter   0, loss = 0.2342
Epoch  15 / iter   1, loss = 0.2780
Epoch  15 / iter   2, loss = 0.2571
Epoch  15 / iter   3, loss = 0.1838
Epoch  15 / iter   4, loss = 0.0627
Epoch  16 / iter   0, loss = 0.1896
Epoch  16 / iter   1, loss = 0.1966
Epoch  16 / iter   2, loss = 0.2018
Epoch  16 / iter   3, loss = 0.3257
Epoch  16 / iter   4, loss = 0.1268
Epoch  17 / iter   0, loss = 0.1990
Epoch  17 / iter   1, loss = 0.2031
Epoch  17 / iter   2, loss = 0.2662
Epoch  17 / iter   3, loss = 0.2128
Epoch  17 / iter   4, loss = 0.0133
Epoch  18 / iter   0, loss = 0.1780
Epoch  18 / iter   1, loss = 0.1575
Epoch  18 / iter   2, loss = 0.2547
Epoch  18 / iter   3, loss = 0.2544
Epoch  18 / iter   4, loss = 0.2007
Epoch  19 / iter   0, loss = 0.1657
Epoch  19 / iter   1, loss = 0.2000
Epoch  19 / iter   2, loss = 0.2045
Epoch  19 / iter   3, loss = 0.2524
Epoch  19 / iter   4, loss = 0.0632
Epoch  20 / iter   0, loss = 0.1629
Epoch  20 / iter   1, loss = 0.1895
Epoch  20 / iter   2, loss = 0.2523
Epoch  20 / iter   3, loss = 0.1896
Epoch  20 / iter   4, loss = 0.0918
Epoch  21 / iter   0, loss = 0.1583
Epoch  21 / iter   1, loss = 0.2322
Epoch  21 / iter   2, loss = 0.1567
Epoch  21 / iter   3, loss = 0.2089
Epoch  21 / iter   4, loss = 0.2035
Epoch  22 / iter   0, loss = 0.2273
Epoch  22 / iter   1, loss = 0.1427
Epoch  22 / iter   2, loss = 0.1712
Epoch  22 / iter   3, loss = 0.1826
Epoch  22 / iter   4, loss = 0.2878
Epoch  23 / iter   0, loss = 0.1685
Epoch  23 / iter   1, loss = 0.1622
Epoch  23 / iter   2, loss = 0.1499
Epoch  23 / iter   3, loss = 0.2329
Epoch  23 / iter   4, loss = 0.1486
Epoch  24 / iter   0, loss = 0.1617
Epoch  24 / iter   1, loss = 0.2083
Epoch  24 / iter   2, loss = 0.1442
Epoch  24 / iter   3, loss = 0.1740
Epoch  24 / iter   4, loss = 0.1641
Epoch  25 / iter   0, loss = 0.1159
Epoch  25 / iter   1, loss = 0.2064
Epoch  25 / iter   2, loss = 0.1690
Epoch  25 / iter   3, loss = 0.1778
Epoch  25 / iter   4, loss = 0.0159
Epoch  26 / iter   0, loss = 0.1730
Epoch  26 / iter   1, loss = 0.1861
Epoch  26 / iter   2, loss = 0.1387
Epoch  26 / iter   3, loss = 0.1486
Epoch  26 / iter   4, loss = 0.1090
Epoch  27 / iter   0, loss = 0.1393
Epoch  27 / iter   1, loss = 0.1775
Epoch  27 / iter   2, loss = 0.1564
Epoch  27 / iter   3, loss = 0.1245
Epoch  27 / iter   4, loss = 0.7611
Epoch  28 / iter   0, loss = 0.1470
Epoch  28 / iter   1, loss = 0.1211
Epoch  28 / iter   2, loss = 0.1285
Epoch  28 / iter   3, loss = 0.1854
Epoch  28 / iter   4, loss = 0.5240
Epoch  29 / iter   0, loss = 0.1740
Epoch  29 / iter   1, loss = 0.0898
Epoch  29 / iter   2, loss = 0.1392
Epoch  29 / iter   3, loss = 0.1842
Epoch  29 / iter   4, loss = 0.0251
Epoch  30 / iter   0, loss = 0.0978
Epoch  30 / iter   1, loss = 0.1529
Epoch  30 / iter   2, loss = 0.1640
Epoch  30 / iter   3, loss = 0.1503
Epoch  30 / iter   4, loss = 0.0975
Epoch  31 / iter   0, loss = 0.1399
Epoch  31 / iter   1, loss = 0.1595
Epoch  31 / iter   2, loss = 0.1209
Epoch  31 / iter   3, loss = 0.1203
Epoch  31 / iter   4, loss = 0.2008
Epoch  32 / iter   0, loss = 0.1501
Epoch  32 / iter   1, loss = 0.1310
Epoch  32 / iter   2, loss = 0.1065
Epoch  32 / iter   3, loss = 0.1489
Epoch  32 / iter   4, loss = 0.0818
Epoch  33 / iter   0, loss = 0.1401
Epoch  33 / iter   1, loss = 0.1367
Epoch  33 / iter   2, loss = 0.0970
Epoch  33 / iter   3, loss = 0.1481
Epoch  33 / iter   4, loss = 0.0711
Epoch  34 / iter   0, loss = 0.1157
Epoch  34 / iter   1, loss = 0.1050
Epoch  34 / iter   2, loss = 0.1378
Epoch  34 / iter   3, loss = 0.1505
Epoch  34 / iter   4, loss = 0.0429
Epoch  35 / iter   0, loss = 0.1096
Epoch  35 / iter   1, loss = 0.1279
Epoch  35 / iter   2, loss = 0.1715
Epoch  35 / iter   3, loss = 0.0888
Epoch  35 / iter   4, loss = 0.0473
Epoch  36 / iter   0, loss = 0.1350
Epoch  36 / iter   1, loss = 0.0781
Epoch  36 / iter   2, loss = 0.1458
Epoch  36 / iter   3, loss = 0.1288
Epoch  36 / iter   4, loss = 0.0421
Epoch  37 / iter   0, loss = 0.1083
Epoch  37 / iter   1, loss = 0.0972
Epoch  37 / iter   2, loss = 0.1513
Epoch  37 / iter   3, loss = 0.1236
Epoch  37 / iter   4, loss = 0.0366
Epoch  38 / iter   0, loss = 0.1204
Epoch  38 / iter   1, loss = 0.1341
Epoch  38 / iter   2, loss = 0.1109
Epoch  38 / iter   3, loss = 0.0905
Epoch  38 / iter   4, loss = 0.3906
Epoch  39 / iter   0, loss = 0.0923
Epoch  39 / iter   1, loss = 0.1094
Epoch  39 / iter   2, loss = 0.1295
Epoch  39 / iter   3, loss = 0.1239
Epoch  39 / iter   4, loss = 0.0684
Epoch  40 / iter   0, loss = 0.1188
Epoch  40 / iter   1, loss = 0.0984
Epoch  40 / iter   2, loss = 0.1067
Epoch  40 / iter   3, loss = 0.1057
Epoch  40 / iter   4, loss = 0.4602
Epoch  41 / iter   0, loss = 0.1478
Epoch  41 / iter   1, loss = 0.0980
Epoch  41 / iter   2, loss = 0.0921
Epoch  41 / iter   3, loss = 0.1020
Epoch  41 / iter   4, loss = 0.0430
Epoch  42 / iter   0, loss = 0.0991
Epoch  42 / iter   1, loss = 0.0994
Epoch  42 / iter   2, loss = 0.1270
Epoch  42 / iter   3, loss = 0.0988
Epoch  42 / iter   4, loss = 0.1176
Epoch  43 / iter   0, loss = 0.1286
Epoch  43 / iter   1, loss = 0.1013
Epoch  43 / iter   2, loss = 0.1066
Epoch  43 / iter   3, loss = 0.0779
Epoch  43 / iter   4, loss = 0.1481
Epoch  44 / iter   0, loss = 0.0840
Epoch  44 / iter   1, loss = 0.0858
Epoch  44 / iter   2, loss = 0.1388
Epoch  44 / iter   3, loss = 0.1000
Epoch  44 / iter   4, loss = 0.0313
Epoch  45 / iter   0, loss = 0.0896
Epoch  45 / iter   1, loss = 0.1173
Epoch  45 / iter   2, loss = 0.0916
Epoch  45 / iter   3, loss = 0.1043
Epoch  45 / iter   4, loss = 0.0074
Epoch  46 / iter   0, loss = 0.1008
Epoch  46 / iter   1, loss = 0.0915
Epoch  46 / iter   2, loss = 0.0877
Epoch  46 / iter   3, loss = 0.1139
Epoch  46 / iter   4, loss = 0.0292
Epoch  47 / iter   0, loss = 0.0679
Epoch  47 / iter   1, loss = 0.0987
Epoch  47 / iter   2, loss = 0.0929
Epoch  47 / iter   3, loss = 0.1098
Epoch  47 / iter   4, loss = 0.4838
Epoch  48 / iter   0, loss = 0.0693
Epoch  48 / iter   1, loss = 0.1095
Epoch  48 / iter   2, loss = 0.1128
Epoch  48 / iter   3, loss = 0.0890
Epoch  48 / iter   4, loss = 0.1008
Epoch  49 / iter   0, loss = 0.0724
Epoch  49 / iter   1, loss = 0.0804
Epoch  49 / iter   2, loss = 0.0919
Epoch  49 / iter   3, loss = 0.1233
Epoch  49 / iter   4, loss = 0.1849
```

![img](data:image/png;base64,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)

观察上述Loss的变化，随机梯度下降加快了训练过程，但由于每次仅基于少量样本更新参数和计算损失，所以损失下降曲线会出现震荡。

------

**说明：**

由于房价预测的数据量过少，所以难以感受到随机梯度下降带来的性能提升。

------

### 模型保存

Numpy提供了save接口，可直接将模型权重数组保存为.npy格式的文件。

In [46]

```
np.save('w.npy', net.w)
np.save('b.npy', net.b)
```

### 小结

本节我们详细介绍了如何使用NumPy实现梯度下降算法，构建并训练了一个简单的线性模型实现波士顿房价预测，可以总结出，使用神经网络建模房价预测有三个要点：

- 构建网络，初始化参数ww*w*和bb*b*，定义预测和损失函数的计算方法。
- 随机选择初始点，建立梯度的计算方法和参数更新方式。
- 从总的数据集中抽取部分数据作为一个mini_batch，计算梯度并更新参数，不断迭代直到损失函数几乎不再下降。









