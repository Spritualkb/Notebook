通过极简方案构建手写数字识别模型

上一节介绍了创新性的“横纵式”教学法，有助于深度学习初学者快速掌握深度学习理论知识，并在过程中让读者获得真实建模的实战体验。在“横纵式”教学法中，纵向概要介绍模型的基本代码结构和极简实现方案，如 **图1** 所示。本节将使用这种极简实现方案快速完成手写数字识别的建模。

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/762c127363684c32832cb61b5d6deaa013023131a36948b6b695cec2df72f791" width="1000" hegiht="" ></center>
<center><br>图1：“横纵式”教学法—纵向极简实现方案</br></center>
<br></br>

### 前提条件

在数据处理前，首先要加载飞桨平台与“手写数字识别”模型相关的类库，实现方法如下。


```python
#加载飞桨和相关类库
import paddle
from paddle.nn import Linear
import paddle.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt
```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/__init__.py:107: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      from collections import MutableMapping
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/rcsetup.py:20: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      from collections import Iterable, Mapping
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/colors.py:53: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      from collections import Sized


# 数据处理

飞桨提供了多个封装好的数据集API，涵盖计算机视觉、自然语言处理、推荐系统等多个领域，帮助读者快速完成深度学习任务。如在手写数字识别任务中，通过[paddle.vision.datasets.MNIST](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/vision/datasets/mnist/MNIST_cn.html)可以直接获取处理好的MNIST训练集、测试集，飞桨API支持如下常见的学术数据集：

* mnist
* cifar
* Conll05
* imdb
* imikolov
* movielens
* sentiment
* uci_housing
* wmt14
* wmt16

通过paddle.vision.datasets.MNIST API设置数据读取器，代码如下所示。


```python
# 设置数据读取器，API自动读取MNIST数据训练集
train_dataset = paddle.vision.datasets.MNIST(mode='train')
```

 通过如下代码读取任意一个数据内容，观察打印结果。


```python
train_data0 = np.array(train_dataset[0][0])
train_label_0 = np.array(train_dataset[0][1])

# 显示第一batch的第一个图像
import matplotlib.pyplot as plt
plt.figure("Image") # 图像窗口名称
plt.figure(figsize=(2,2))
plt.imshow(train_data0, cmap=plt.cm.binary)
plt.axis('on') # 关掉坐标轴为 off
plt.title('image') # 图像题目
plt.show()

print("图像数据形状和对应数据为:", train_data0.shape)
print("图像标签形状和对应数据为:", train_label_0.shape, train_label_0)
print("\n打印第一个batch的第一个图像，对应标签数字为{}".format(train_label_0))
```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/cbook/__init__.py:2349: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      if isinstance(obj, collections.Iterator):
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/cbook/__init__.py:2366: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      return list(data) if isinstance(data, collections.MappingView) else data
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/numpy/lib/type_check.py:546: DeprecationWarning: np.asscalar(a) is deprecated since NumPy v1.16, use a.item() instead
      'a.item() instead', DeprecationWarning, stacklevel=1)



    <Figure size 432x288 with 0 Axes>



![png](output_5_2.png)


    图像数据形状和对应数据为: (28, 28)
    图像标签形状和对应数据为: (1,) [5]
    
    打印第一个batch的第一个图像，对应标签数字为[5]


使用matplotlib工具包将其显示出来，如**图2** 所示。可以看到图片显示的数字是5，和对应标签数字一致。

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/a07d9b3b5839434e98afe05a298d3ce1c9b6cbc02124488a9bd8b7c2efeb42c4" width="300" hegiht="" ></center>
<center><br>图2：matplotlib打印结果示意图</br></center>
<br></br>

```python
img = np.arry(img_data[0]+1)*127.5
#图像是数据经过归一化的，展示之前需要缩放回原始数据
```



------
**说明：**

飞桨将维度是28×28的手写数字图像转成向量形式存储，因此使用飞桨数据加载器读取到的手写数字图像是长度为784（28×28）的向量。

------

## 飞桨API的使用方法

熟练掌握飞桨API的使用方法，是使用飞桨完成各类深度学习任务的基础，也是开发者必须掌握的技能。

**飞桨API文档获取方式及目录结构**

登录“[飞桨官网->文档->API文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.0-rc1/api/index_cn.html)”，可以获取飞桨API文档。在飞桨最新的版本中，对API做了许多优化，目录结构与说明，如 **图3** 所示。

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/316984568d8e4e189fe3449108fa1d76a7d82330834f41139f2aaba8f745d49a" width="900" hegiht="" ></center>
<center><br>图3：飞桨API文档目录</br></center>
<br></br>

**API文档使用方法**

飞桨每个API的文档结构一致，包含接口形式、功能说明和计算公式、参数和返回值、代码示例四个部分。 以Relu函数为例，API文档结构如 **图4** 所示。通过飞桨API文档，读者不仅可以详细查看函数功能，还可以通过可运行的代码示例来实践API的使用。

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/badc3b56be924955b97cc30d253eb4c850582d8e94004d5c876cf17eac8aee15" width="700" hegiht="" ></center>
<center><br>图4：Relu的API文档</br></center>
<br></br>

# 模型设计

在房价预测深度学习任务中，我们使用了单层且没有非线性变换的模型，取得了理想的预测效果。在手写数字识别中，我们依然使用这个模型预测输入的图形数字值。其中，模型的输入为784维（28×28）数据，输出为1维数据，如 **图5** 所示。

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/9c146e7d9c4a4119a8cd09f7c8b5ee61f2ac1820a221429a80430291728b9c4a" width="400" hegiht="" ></center>
<center><br>图5：手写数字识别网络模型</br></center>
<br></br>

输入像素的位置排布信息对理解图像内容非常重要（如将原始尺寸为28×28图像的像素按照7×112的尺寸排布，那么其中的数字将不可识别），因此网络的输入设计为28×28的尺寸，而不是1×784，以便于模型能够正确处理像素之间的空间信息。

------
**说明：**

事实上，采用只有一层的简单网络（对输入求加权和）时并没有处理位置关系信息，因此可以猜测出此模型的预测效果可能有限。在后续优化环节介绍的卷积神经网络则更好的考虑了这种位置关系信息，模型的预测效果也会有显著提升。

------

下面以类的方式组建手写数字识别的网络，实现方法如下所示。


```python
# 定义mnist数据识别网络结构，同房价预测网络
class MNIST(paddle.nn.Layer):
    def __init__(self):
        super(MNIST, self).__init__()
        
        # 定义一层全连接层，输出维度是1
        self.fc = paddle.nn.Linear(in_features=784, out_features=1)
        
    # 定义网络结构的前向计算过程
    def forward(self, inputs):
        outputs = self.fc(inputs)
        return outputs
```

# 训练配置

训练配置需要先生成模型实例（设为“训练”状态），再设置优化算法和学习率（使用随机梯度下降SGD，学习率设置为0.001），实现方法如下所示。


```python
# 声明网络结构
model = MNIST()

def train(model):
    # 启动训练模式
    model.train()
    # 加载训练集 batch_size 设为 16
    train_loader = paddle.io.DataLoader(paddle.vision.datasets.MNIST(mode='train'), 
                                        batch_size=16, 
                                        shuffle=True)
    # 定义优化器，使用随机梯度下降SGD优化器，学习率设置为0.001
    opt = paddle.optimizer.SGD(learning_rate=0.001, parameters=model.parameters())
```

# 训练过程

训练过程采用二层循环嵌套方式，训练完成后需要保存模型参数，以便后续使用。

- 内层循环：负责整个数据集的一次遍历，遍历数据集采用分批次（batch）方式。
- 外层循环：定义遍历数据集的次数，本次训练中外层循环10次，通过参数EPOCH_NUM设置。


```python
# 图像归一化函数，将数据范围为[0, 255]的图像归一化到[0, 1]
def norm_img(img):
    # 验证传入数据格式是否正确，img的shape为[batch_size, 28, 28]
    assert len(img.shape) == 3
    batch_size, img_h, img_w = img.shape[0], img.shape[1], img.shape[2]
    # 归一化图像数据
    img = img / 255
    # 将图像形式reshape为[batch_size, 784]
    img = paddle.reshape(img, [batch_size, img_h*img_w])
    
    return img
```


```python
import paddle
# 确保从paddle.vision.datasets.MNIST中加载的图像数据是np.ndarray类型
paddle.vision.set_image_backend('cv2')

# 声明网络结构
model = MNIST()

def train(model):
    # 启动训练模式
    model.train()
    # 加载训练集 batch_size 设为 16
    train_loader = paddle.io.DataLoader(paddle.vision.datasets.MNIST(mode='train'), 
                                        batch_size=16, 
                                        shuffle=True)
    # 定义优化器，使用随机梯度下降SGD优化器，学习率设置为0.001
    opt = paddle.optimizer.SGD(learning_rate=0.001, parameters=model.parameters())
    EPOCH_NUM = 10
    for epoch in range(EPOCH_NUM):
        for batch_id, data in enumerate(train_loader()):
            images = norm_img(data[0]).astype('float32')
            labels = data[1].astype('float32')
            
            #前向计算的过程
            predicts = model(images)
            
            # 计算损失
            loss = F.square_error_cost(predicts, labels)
            avg_loss = paddle.mean(loss)
            
            #每训练了1000批次的数据，打印下当前Loss的情况
            if batch_id % 1000 == 0:
                print("epoch_id: {}, batch_id: {}, loss is: {}".format(epoch, batch_id, avg_loss.numpy()))
            
            #后向传播，更新参数的过程
            avg_loss.backward()
            opt.step()
            opt.clear_grad()
            
train(model)
paddle.save(model.state_dict(), './mnist.pdparams')
```

    epoch_id: 0, batch_id: 0, loss is: [35.525185]
    epoch_id: 0, batch_id: 1000, loss is: [7.4399786]
    epoch_id: 0, batch_id: 2000, loss is: [2.0210705]
    epoch_id: 0, batch_id: 3000, loss is: [2.325027]
    epoch_id: 1, batch_id: 0, loss is: [2.4414306]
    epoch_id: 1, batch_id: 1000, loss is: [4.6318164]
    epoch_id: 1, batch_id: 2000, loss is: [4.6807127]
    epoch_id: 1, batch_id: 3000, loss is: [5.7014084]
    epoch_id: 2, batch_id: 0, loss is: [3.4229655]
    epoch_id: 2, batch_id: 1000, loss is: [2.1136832]
    epoch_id: 2, batch_id: 2000, loss is: [2.3517294]
    epoch_id: 2, batch_id: 3000, loss is: [6.7515297]
    epoch_id: 3, batch_id: 0, loss is: [4.119179]
    epoch_id: 3, batch_id: 1000, loss is: [4.4800296]
    epoch_id: 3, batch_id: 2000, loss is: [3.4902763]
    epoch_id: 3, batch_id: 3000, loss is: [3.631486]
    epoch_id: 4, batch_id: 0, loss is: [6.123066]
    epoch_id: 4, batch_id: 1000, loss is: [2.8558893]
    epoch_id: 4, batch_id: 2000, loss is: [2.6112337]
    epoch_id: 4, batch_id: 3000, loss is: [2.0097098]
    epoch_id: 5, batch_id: 0, loss is: [3.9023933]
    epoch_id: 5, batch_id: 1000, loss is: [2.1165676]
    epoch_id: 5, batch_id: 2000, loss is: [3.2067215]
    epoch_id: 5, batch_id: 3000, loss is: [2.4574804]
    epoch_id: 6, batch_id: 0, loss is: [1.8463242]
    epoch_id: 6, batch_id: 1000, loss is: [3.4741895]
    epoch_id: 6, batch_id: 2000, loss is: [2.057652]
    epoch_id: 6, batch_id: 3000, loss is: [2.0860665]
    epoch_id: 7, batch_id: 0, loss is: [3.90655]
    epoch_id: 7, batch_id: 1000, loss is: [2.5527935]
    epoch_id: 7, batch_id: 2000, loss is: [3.239427]
    epoch_id: 7, batch_id: 3000, loss is: [6.7344103]
    epoch_id: 8, batch_id: 0, loss is: [1.6209174]
    epoch_id: 8, batch_id: 1000, loss is: [2.686802]
    epoch_id: 8, batch_id: 2000, loss is: [7.759363]
    epoch_id: 8, batch_id: 3000, loss is: [3.1380877]
    epoch_id: 9, batch_id: 0, loss is: [3.1067057]
    epoch_id: 9, batch_id: 1000, loss is: [2.864774]
    epoch_id: 9, batch_id: 2000, loss is: [2.528369]
    epoch_id: 9, batch_id: 3000, loss is: [4.1854725]


另外，从训练过程中损失所发生的变化可以发现，虽然损失整体上在降低，但到训练的最后一轮，损失函数值依然较高。可以猜测手写数字识别完全复用房价预测的代码，训练效果并不好。接下来我们通过模型测试，获取模型训练的真实效果。

# 模型测试

模型测试的主要目的是验证训练好的模型是否能正确识别出数字，包括如下四步：

* 声明实例
* 加载模型：加载训练过程中保存的模型参数，
* 灌入数据：将测试样本传入模型，模型的状态设置为校验状态（eval），显式告诉框架我们接下来只会使用前向计算的流程，不会计算梯度和梯度反向传播。
* 获取预测结果，取整后作为预测标签输出。

在模型测试之前，需要先从'./work/example_0.png'文件中读取样例图片，并进行归一化处理。


```python
# 导入图像读取第三方库
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

img_path = './work/example_0.jpg'
# 读取原始图像并显示
im = Image.open('./work/example_0.jpg')
plt.imshow(im)
plt.show()
# 将原始图像转为灰度图
im = im.convert('L')
print('原始图像shape: ', np.array(im).shape)
# 使用Image.ANTIALIAS方式采样原始图片
im = im.resize((28, 28), Image.ANTIALIAS)
plt.imshow(im)
plt.show()
print("采样后图片shape: ", np.array(im).shape)
```


![png](output_16_0.png)


    原始图像shape:  (28, 28)



![png](output_16_2.png)


    采样后图片shape:  (28, 28)



```python
# 读取一张本地的样例图片，转变成模型输入的格式
def load_image(img_path):
    # 从img_path中读取图像，并转为灰度图
    im = Image.open(img_path).convert('L')
    # print(np.array(im))
    im = im.resize((28, 28), Image.ANTIALIAS)
    im = np.array(im).reshape(1, -1).astype(np.float32)
    # 图像归一化，保持和数据集的数据范围一致
    im = 1 - im / 255
    return im

# 定义预测过程
model = MNIST()
params_file_path = 'mnist.pdparams'
img_path = './work/example_0.jpg'
# 加载模型参数
param_dict = paddle.load(params_file_path)
model.load_dict(param_dict)
# 灌入数据
model.eval()
tensor_img = load_image(img_path)
result = model(paddle.to_tensor(tensor_img))
print('result',result)
#  预测输出取整，即为预测的数字，打印结果
print("本次预测的数字是", result.numpy().astype('int32'))
```

    result Tensor(shape=[1, 1], dtype=float32, place=CPUPlace, stop_gradient=False,
           [[1.11405170]])
    本次预测的数字是 [[1]]


从打印结果来看，模型预测出的数字是与实际输出的图片的数字不一致。这里只是验证了一个样本的情况，如果我们尝试更多的样本，可发现许多数字图片识别结果是错误的。因此完全复用房价预测的实验并不适用于手写数字识别任务！

接下来我们会对手写数字识别实验模型进行逐一改进，直到获得令人满意的结果。

## 作业 2-1：

1. 使用飞桨API [paddle.vision.datasets.MNIST](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/vision/datasets/mnist/MNIST_cn.html)的mode函数获得测试集数据，计算当前模型的准确率。

2. 怎样进一步提高模型的准确率？可以在接下来内容开始前，写出你想到的优化思路。



# 2.【手写数字识别】之数据处理

上一节我们使用“横纵式”教学法中的纵向极简方案快速完成手写数字识别任务的建模，但模型测试效果并未达成预期。我们换个思路，从横向展开，如 **图1** 所示，逐个环节优化，以达到最优训练效果。本节主要介绍手写数字识别模型中，数据处理的优化方法。

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/257c74a23cef401c9db75fd2841bb93cdec28f756c6049b7b3e5ec1bf0ed058d" width="1000" hegiht="" ></center>
<center><br>图1：“横纵式”教学法 — 数据处理优化 </br></center>

<br></br>


上一节，我们通过调用飞桨提供的[paddle.vision.datasets.MNIST](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.0-rc1/api/paddle/vision/datasets/mnist/MNIST_cn.html) API加载MNIST数据集。但在工业实践中，我们面临的任务和数据环境千差万别，通常需要自己编写适合当前任务的数据处理程序，一般涉及如下五个环节：

* 读入数据
* 划分数据集
* 生成批次数据
* 训练样本集乱序
* 校验数据有效性

### 前提条件

在数据读取与处理前，首先要加载飞桨平台和数据处理库，代码如下。


```python
#数据处理部分之前的代码，加入部分数据处理的库
import paddle
from paddle.nn import Linear
import paddle.nn.functional as F
import os
import gzip
import json
import random
import numpy as np
```

# 读入数据并划分数据集

在实际应用中，保存到本地的数据存储格式多种多样，如MNIST数据集以json格式存储在本地，其数据存储结构如 **图2** 所示。

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/7075f5ca75c54e4e8553c10b696913a1a178dad37c5c460a899cd75635cd7961" width="500" hegiht="" ></center>
<center><br>图2：MNIST数据集的存储结构</br></center>

<br></br>

**data**包含三个元素的列表：train_set、val_set、 test_set，包括50 000条训练样本、10 000条验证样本、10 000条测试样本。每个样本包含手写数字图片和对应的标签。

* **train_set（训练集）**：用于确定模型参数。
* **val_set（验证集）**：用于调节模型超参数（如多个网络结构、正则化权重的最优选择）。
* **test_set（测试集）**：用于估计应用效果（没有在模型中应用过的数据，更贴近模型在真实场景应用的效果）。

**train_set**包含两个元素的列表：train_images、train_labels。

* **train_images**：[50 000, 784]的二维列表，包含50 000张图片。每张图片用一个长度为784的向量表示，内容是28\*28尺寸的像素灰度值（黑白图片）。
* **train_labels**：[50 000, ]的列表，表示这些图片对应的分类标签，即0~9之间的一个数字。

在本地`./work/`目录下读取文件名称为`mnist.json.gz`的MNIST数据，并拆分成训练集、验证集和测试集，实现方法如下所示。



```python
# 声明数据集文件位置
datafile = './work/mnist.json.gz'
print('loading mnist dataset from {} ......'.format(datafile))
# 加载json数据文件
data = json.load(gzip.open(datafile))
print('mnist dataset load done')
# 读取到的数据区分训练集，验证集，测试集
train_set, val_set, eval_set = data

# 观察训练集数据
imgs, labels = train_set[0], train_set[1]
print("训练数据集数量: ", len(imgs))

# 观察验证集数量
imgs, labels = val_set[0], val_set[1]
print("验证数据集数量: ", len(imgs))

# 观察测试集数量
imgs, labels = val= eval_set[0], eval_set[1]
print("测试数据集数量: ", len(imgs))
```

    loading mnist dataset from ./work/mnist.json.gz ......
    mnist dataset load done
    训练数据集数量:  50000
    验证数据集数量:  10000
    测试数据集数量:  10000


## 扩展阅读：为什么学术界的模型总在不断精进呢？

通常某组织发布一个新任务的训练集和测试集数据后，全世界的科学家都针对该数据集进行创新研究，随后大量针对该数据集的论文会陆续发表。论文1的A模型声称在测试集的准确率70%，论文2的B模型声称在测试集的准确率提高到72%，论文N的X模型声称在测试集的准确率提高到90% ...

然而这些论文中的模型在测试集上准确率提升真实有效么？我们不妨大胆猜测一下。

假设所有论文共产生1000个模型，这些模型使用的是测试数据集来评判模型效果，并最终选出效果最优的模型。这相当于把原始的测试集当作了验证集，使得测试集失去了真实评判模型效果的能力，正如机器学习领域非常流行的一句话：“拷问数据足够久，它终究会招供”。

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/61ff2532a33346cb9641db0e5b97b9f30f2ad9d17bad444193c07191cdacd189" width="600" hegiht="" ></center>
<center><br>图3：拷问数据足够久，它总会招供</br></center>

<br></br>


那么当我们需要将学术界研发的模型复用于工业项目时，应该如何选择呢？给读者一个小建议：当几个模型的准确率在测试集上差距不大时，尽量选择网络结构相对简单的模型。往往越精巧设计的模型和方法，越不容易在不同的数据集之间迁移。

# 训练样本乱序、生成批次数据

* **训练样本乱序：** 先将样本按顺序进行编号，建立ID集合index_list。然后将index_list乱序，最后按乱序后的顺序读取数据。

------

**说明：**

通过大量实验发现，模型对最后出现的数据印象更加深刻。训练数据导入后，越接近模型训练结束，最后几个批次数据对模型参数的影响越大。为了避免模型记忆影响训练效果，需要进行样本乱序操作。

------

* **生成批次数据：** 先设置合理的batch_size，再将数据转变成符合模型输入要求的np.array格式返回。同时，在返回数据时将Python生成器设置为``yield``模式，以减少内存占用。

在执行如上两个操作之前，需要先将数据处理代码封装成load_data函数，方便后续调用。load_data有三种模型：``train``、``valid``、``eval``，分为对应返回的数据是训练集、验证集、测试集。


```python
imgs, labels = train_set[0], train_set[1]
print("训练数据集数量: ", len(imgs))
# 获得数据集长度
imgs_length = len(imgs)
# 定义数据集每个数据的序号，根据序号读取数据
index_list = list(range(imgs_length))
# 读入数据时用到的批次大小
BATCHSIZE = 100

# 随机打乱训练数据的索引序号
random.shuffle(index_list)

# 定义数据生成器，返回批次数据
def data_generator():
    imgs_list = []
    labels_list = []
    for i in index_list:
        # 将数据处理成希望的类型
        img = np.array(imgs[i]).astype('float32')
        label = np.array(labels[i]).astype('float32')
        imgs_list.append(img) 
        labels_list.append(label)
        if len(imgs_list) == BATCHSIZE:
            # 获得一个batchsize的数据，并返回
            yield np.array(imgs_list), np.array(labels_list)
            # 清空数据读取列表
            imgs_list = []
            labels_list = []

    # 如果剩余数据的数目小于BATCHSIZE，
    # 则剩余数据一起构成一个大小为len(imgs_list)的mini-batch
    if len(imgs_list) > 0:
        yield np.array(imgs_list), np.array(labels_list)
    return data_generator
```

    训练数据集数量:  50000



```python
# 声明数据读取函数，从训练集中读取数据
train_loader = data_generator
# 以迭代的形式读取数据
for batch_id, data in enumerate(train_loader()):
    image_data, label_data = data
    if batch_id == 0:
        # 打印数据shape和类型
        print("打印第一个batch数据的维度:")
        print("图像维度: {}, 标签维度: {}".format(image_data.shape, label_data.shape))
    break
```

    打印第一个batch数据的维度:
    图像维度: (100, 784), 标签维度: (100,)


# 校验数据有效性

在实际应用中，原始数据可能存在标注不准确、数据杂乱或格式不统一等情况。因此在完成数据处理流程后，还需要进行数据校验，一般有两种方式：

* 机器校验：加入一些校验和清理数据的操作。
* 人工校验：先打印数据输出结果，观察是否是设置的格式。再从训练的结果验证数据处理和读取的有效性。

##  机器校验

如下代码所示，如果数据集中的图片数量和标签数量不等，说明数据逻辑存在问题，可使用assert语句校验图像数量和标签数据是否一致。


```python
    imgs_length = len(imgs)

    assert len(imgs) == len(labels), \
          "length of train_imgs({}) should be the same as train_labels({})".format(len(imgs), len(label))
```

## 人工校验

人工校验是指打印数据输出结果，观察是否是预期的格式。实现数据处理和加载函数后，我们可以调用它读取一次数据，观察数据的shape和类型是否与函数中设置的一致。


```python
# 声明数据读取函数，从训练集中读取数据
train_loader = data_generator
# 以迭代的形式读取数据
for batch_id, data in enumerate(train_loader()):
    image_data, label_data = data
    if batch_id == 0:
        # 打印数据shape和类型
        print("打印第一个batch数据的维度，以及数据的类型:")
        print("图像维度: {}, 标签维度: {}, 图像数据类型: {}, 标签数据类型: {}".format(image_data.shape, label_data.shape, type(image_data), type(label_data)))
    break
```

    打印第一个batch数据的维度，以及数据的类型:
    图像维度: (100, 784), 标签维度: (100,), 图像数据类型: <class 'numpy.ndarray'>, 标签数据类型: <class 'numpy.ndarray'>


# 封装数据读取与处理函数

上文，我们从读取数据、划分数据集、到打乱训练数据、构建数据读取器以及数据数据校验，完成了一整套一般性的数据处理流程，下面将这些步骤放在一个函数中实现，方便在神经网络训练时直接调用。


```python
def load_data(mode='train'):
    datafile = './work/mnist.json.gz'
    print('loading mnist dataset from {} ......'.format(datafile))
    # 加载json数据文件
    data = json.load(gzip.open(datafile))
    print('mnist dataset load done')
   
    # 读取到的数据区分训练集，验证集，测试集
    train_set, val_set, eval_set = data
    if mode=='train':
        # 获得训练数据集
        imgs, labels = train_set[0], train_set[1]
    elif mode=='valid':
        # 获得验证数据集
        imgs, labels = val_set[0], val_set[1]
    elif mode=='eval':
        # 获得测试数据集
        imgs, labels = eval_set[0], eval_set[1]
    else:
        raise Exception("mode can only be one of ['train', 'valid', 'eval']")
    print("训练数据集数量: ", len(imgs))
    
    # 校验数据
    imgs_length = len(imgs)

    assert len(imgs) == len(labels), \
          "length of train_imgs({}) should be the same as train_labels({})".format(len(imgs), len(labels))
    
    # 获得数据集长度
    imgs_length = len(imgs)
    
    # 定义数据集每个数据的序号，根据序号读取数据
    index_list = list(range(imgs_length))
    # 读入数据时用到的批次大小
    BATCHSIZE = 100
    
    # 定义数据生成器
    def data_generator():
        if mode == 'train':
            # 训练模式下打乱数据
            random.shuffle(index_list)
        imgs_list = []
        labels_list = []
        for i in index_list:
            # 将数据处理成希望的类型
            img = np.array(imgs[i]).astype('float32')
            label = np.array(labels[i]).astype('float32')
            imgs_list.append(img) 
            labels_list.append(label)
            if len(imgs_list) == BATCHSIZE:
                # 获得一个batchsize的数据，并返回
                yield np.array(imgs_list), np.array(labels_list)
                # 清空数据读取列表
                imgs_list = []
                labels_list = []
    
        # 如果剩余数据的数目小于BATCHSIZE，
        # 则剩余数据一起构成一个大小为len(imgs_list)的mini-batch
        if len(imgs_list) > 0:
            yield np.array(imgs_list), np.array(labels_list)
    return data_generator
```

下面定义一层神经网络，利用定义好的数据处理函数，完成神经网络的训练。


```python
#数据处理部分之后的代码，数据读取的部分调用Load_data函数
# 定义网络结构，同上一节所使用的网络结构
class MNIST(paddle.nn.Layer):
    def __init__(self):
        super(MNIST, self).__init__()
        # 定义一层全连接层，输出维度是1
        self.fc = paddle.nn.Linear(in_features=784, out_features=1)

    def forward(self, inputs):
        outputs = self.fc(inputs)
        return outputs

# 训练配置，并启动训练过程
def train(model):
    model = MNIST()
    model.train()
    #调用加载数据的函数
    train_loader = load_data('train')
    opt = paddle.optimizer.SGD(learning_rate=0.001, parameters=model.parameters())
    EPOCH_NUM = 10
    for epoch_id in range(EPOCH_NUM):
        for batch_id, data in enumerate(train_loader()):
            #准备数据，变得更加简洁
            images, labels = data
            images = paddle.to_tensor(images)
            labels = paddle.to_tensor(labels) 

            #前向计算的过程
            predits = model(images)
            
            #计算损失，取一个批次样本损失的平均值
            loss = F.square_error_cost(predits, labels)
            avg_loss = paddle.mean(loss)      
            
            #每训练了200批次的数据，打印下当前Loss的情况
            if batch_id % 200 == 0:
                print("epoch: {}, batch: {}, loss is: {}".format(epoch_id, batch_id, avg_loss.numpy()))
            
            #后向传播，更新参数的过程
            avg_loss.backward()
            opt.step()
            opt.clear_grad()

    # 保存模型
    paddle.save(model.state_dict(), './mnist.pdparams')
# 创建模型           
model = MNIST()
# 启动训练过程
train(model)

```

    loading mnist dataset from ./work/mnist.json.gz ......
    mnist dataset load done
    训练数据集数量:  50000
    epoch: 0, batch: 0, loss is: [33.513912]
    epoch: 0, batch: 200, loss is: [7.5029764]
    epoch: 0, batch: 400, loss is: [9.005349]
    epoch: 1, batch: 0, loss is: [8.789618]
    epoch: 1, batch: 200, loss is: [8.182622]
    epoch: 1, batch: 400, loss is: [8.046784]
    epoch: 2, batch: 0, loss is: [7.9067063]
    epoch: 2, batch: 200, loss is: [9.436364]
    epoch: 2, batch: 400, loss is: [7.1310873]
    epoch: 3, batch: 0, loss is: [8.077651]
    epoch: 3, batch: 200, loss is: [8.172729]
    epoch: 3, batch: 400, loss is: [8.880512]
    epoch: 4, batch: 0, loss is: [8.070876]
    epoch: 4, batch: 200, loss is: [9.199152]
    epoch: 4, batch: 400, loss is: [8.222003]
    epoch: 5, batch: 0, loss is: [7.08266]
    epoch: 5, batch: 200, loss is: [8.696235]
    epoch: 5, batch: 400, loss is: [10.446534]
    epoch: 6, batch: 0, loss is: [9.002313]
    epoch: 6, batch: 200, loss is: [8.411713]
    epoch: 6, batch: 400, loss is: [7.0788198]
    epoch: 7, batch: 0, loss is: [9.4499]
    epoch: 7, batch: 200, loss is: [10.016479]
    epoch: 7, batch: 400, loss is: [7.0367813]
    epoch: 8, batch: 0, loss is: [9.276244]
    epoch: 8, batch: 200, loss is: [8.044475]
    epoch: 8, batch: 400, loss is: [7.26185]
    epoch: 9, batch: 0, loss is: [9.196164]
    epoch: 9, batch: 200, loss is: [9.965027]
    epoch: 9, batch: 400, loss is: [6.9533315]



# 异步数据读取

上面提到的数据读取采用的是同步数据读取方式。对于样本量较大、数据读取较慢的场景，建议采用异步数据读取方式。异步读取数据时，数据读取和模型训练并行执行，从而加快了数据读取速度，牺牲一小部分内存换取数据读取效率的提升，二者关系如 **图4** 所示。

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/a5fd990c5355426183a71b95aa28a59f979014f6905144ddb415c5a4fe647441" width="500" ></center>
<center><br>图4：同步数据读取和异步数据读取示意图</br></center>

<br></br>

* **同步数据读取**：数据读取与模型训练串行。当模型需要数据时，才运行数据读取函数获得当前批次的数据。在读取数据期间，模型一直等待数据读取结束才进行训练，数据读取速度相对较慢。
* **异步数据读取**：数据读取和模型训练并行。读取到的数据不断的放入缓存区，无需等待模型训练就可以启动下一轮数据读取。当模型训练完一个批次后，不用等待数据读取过程，直接从缓存区获得下一批次数据进行训练，从而加快了数据读取速度。
* **异步队列**：数据读取和模型训练交互的仓库，二者均可以从仓库中读取数据，它的存在使得两者的工作节奏可以解耦。

使用飞桨实现异步数据读取非常简单，只需要两个步骤：

1. 构建一个继承paddle.io.Dataset类的数据读取器。
2. 通过paddle.io.DataLoader创建异步数据读取的迭代器。

首先，我们创建定义一个paddle.io.Dataset类，使用随机函数生成一个数据读取器，代码如下：


```python
import numpy as np
from paddle.io import Dataset
# 构建一个类，继承paddle.io.Dataset，创建数据读取器
class RandomDataset(Dataset):
    def __init__(self, num_samples):
        # 样本数量
        self.num_samples = num_samples

    def __getitem__(self, idx):
        # 随机产生数据和label
        image = np.random.random([784]).astype('float32')
        label = np.random.randint(0, 9, (1, )).astype('float32')
        return image, label

    def __len__(self):
        # 返回样本总数量
        return self.num_samples
        
# 测试数据读取器
dataset = RandomDataset(10)
for i in range(len(dataset)):
    print(dataset[i])
```

           0.25203583, 0.5808543 , 0.75596815, 0.40311828], dtype=float32), array([1.], dtype=float32))

在定义完paddle.io.Dataset后，使用[paddle.io.DataLoader](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.0-rc1/api/paddle/io/DataLoader_cn.html) API即可实现异步数据读取，数据会由Python线程预先读取，并异步送入一个队列中。

> *class* paddle.io.DataLoader(dataset, batch_size=100, shuffle=True, num_workers=2)

DataLoader支持单进程和多进程的数据加载方式。当 num_workers=0时，使用单进程方式异步加载数据；当 num_workers=n(n>0)时，主进程将会开启n个子进程异步加载数据。
DataLoader返回一个迭代器，迭代的返回dataset中的数据内容；

dataset是支持 map-style 的数据集(可通过下标索引样本)， map-style 的数据集请参考 [paddle.io.Dataset](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.0-rc1/api/paddle/io/Dataset_cn.html) API。

使用paddle.io.DataLoader API以batch的方式进行迭代数据，代码如下：


```python
loader = paddle.io.DataLoader(dataset, batch_size=3, shuffle=True, drop_last=True, num_workers=2)
for i, data in enumerate(loader()):
    images, labels = data[0], data[1]
    print("batch_id: {}, 训练数据shape: {}, 标签数据shape: {}".format(i, images.shape, labels.shape))
```

    batch_id: 0, 训练数据shape: [3, 784], 标签数据shape: [3, 1]
    batch_id: 1, 训练数据shape: [3, 784], 标签数据shape: [3, 1]
    batch_id: 2, 训练数据shape: [3, 784], 标签数据shape: [3, 1]


通过上面的学习，我们指导了如何自定义、使用paddle.io.Dataset和paddle.io.DataLoader，下面我们以MNIST数据为例，生成对应的Dataset和DataLoader。


```python
import paddle
import json
import gzip
import numpy as np

# 创建一个类MnistDataset，继承paddle.io.Dataset 这个类
# MnistDataset的作用和上面load_data()函数的作用相同，均是构建一个迭代器
class MnistDataset(paddle.io.Dataset):
    def __init__(self, mode):
        datafile = './work/mnist.json.gz'
        data = json.load(gzip.open(datafile))
        # 读取到的数据区分训练集，验证集，测试集
        train_set, val_set, eval_set = data
        if mode=='train':
            # 获得训练数据集
            imgs, labels = train_set[0], train_set[1]
        elif mode=='valid':
            # 获得验证数据集
            imgs, labels = val_set[0], val_set[1]
        elif mode=='eval':
            # 获得测试数据集
            imgs, labels = eval_set[0], eval_set[1]
        else:
            raise Exception("mode can only be one of ['train', 'valid', 'eval']")
        
        # 校验数据
        imgs_length = len(imgs)
        assert len(imgs) == len(labels), \
            "length of train_imgs({}) should be the same as train_labels({})".format(len(imgs), len(labels))
        
        self.imgs = imgs
        self.labels = labels

    def __getitem__(self, idx):
        img = np.array(self.imgs[idx]).astype('float32')
        label = np.array(self.labels[idx]).astype('float32')
        
        return img, label

    def __len__(self):
        return len(self.imgs)

```


```python
# 声明数据加载函数，使用训练模式，MnistDataset构建的迭代器每次迭代只返回batch=1的数据
train_dataset = MnistDataset(mode='train')
# 使用paddle.io.DataLoader 定义DataLoader对象用于加载Python生成器产生的数据，
# DataLoader 返回的是一个批次数据迭代器，并且是异步的；
data_loader = paddle.io.DataLoader(train_dataset, batch_size=100, shuffle=True)
# 迭代的读取数据并打印数据的形状
for i, data in enumerate(data_loader()):
    images, labels = data
    print(i, images.shape, labels.shape)
    if i>=2:
        break
```

    0 [100, 784] [100]
    1 [100, 784] [100]
    2 [100, 784] [100]


异步数据读取并训练的完整案例代码如下所示。


```python
def train(model):
    model = MNIST()
    model.train()
    opt = paddle.optimizer.SGD(learning_rate=0.001, parameters=model.parameters())
    EPOCH_NUM = 3
    for epoch_id in range(EPOCH_NUM):
        for batch_id, data in enumerate(data_loader()):
            images, labels = data
            images = paddle.to_tensor(images)
            labels = paddle.to_tensor(labels).astype('float32')
            
            #前向计算的过程  
            predicts = model(images)

            #计算损失，取一个批次样本损失的平均值
            loss = F.square_error_cost(predicts, labels)
            avg_loss = paddle.mean(loss)       
            
            #每训练了200批次的数据，打印下当前Loss的情况
            if batch_id % 200 == 0:
                print("epoch: {}, batch: {}, loss is: {}".format(epoch_id, batch_id, avg_loss.numpy()))
            
            #后向传播，更新参数的过程
            avg_loss.backward()
            opt.step()
            opt.clear_grad()

    #保存模型参数
    paddle.save(model.state_dict(), 'mnist')

#创建模型
model = MNIST()
#启动训练过程
train(model)
```

    epoch: 0, batch: 0, loss is: [24.027695]
    epoch: 0, batch: 200, loss is: [8.518488]
    epoch: 0, batch: 400, loss is: [8.920794]
    epoch: 1, batch: 0, loss is: [8.67727]
    epoch: 1, batch: 200, loss is: [8.308073]
    epoch: 1, batch: 400, loss is: [8.633681]
    epoch: 2, batch: 0, loss is: [8.490253]
    epoch: 2, batch: 200, loss is: [8.408561]
    epoch: 2, batch: 400, loss is: [8.509353]


从异步数据读取的训练结果来看，损失函数下降与同步数据读取训练结果一致。注意，异步读取数据只在数据量规模巨大时会带来显著的性能提升，对于多数场景采用同步数据读取的方式已经足够。

# 3.【手写数字识别】之网络结构

前几节我们尝试使用与房价预测相同的简单神经网络解决手写数字识别问题，但是效果并不理想。原因是手写数字识别的输入是28 × 28的像素值，输出是0-9的数字标签，而线性回归模型无法捕捉二维图像数据中蕴含的复杂信息，如 **图1** 所示。无论是牛顿第二定律任务，还是房价预测任务，输入特征和输出预测值之间的关系均可以使用“直线”刻画（使用线性方程来表达）。但手写数字识别任务的输入像素和输出数字标签之间的关系显然不是线性的，甚至这个关系复杂到我们靠人脑难以直观理解的程度。

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/959776f4cd9c4b77b380c7d29f59df1cf47be626cd8b4bd1ac1af2a7d8e3c1cf" width="800" hegiht="" ></center>
<center><br>图1：数字识别任务的输入和输出不是线性关系 </br></center>

<br></br>

因此，我们需要尝试使用其他更复杂、更强大的网络来构建手写数字识别任务，观察一下训练效果，即将“横纵式”教学法从横向展开，如 **图2** 所示。本节主要介绍两种常见的网络结构：经典的多层全连接神经网络和卷积神经网络。

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/0820ed2841474d0a87ef72d05111e37e4b5287473dc344e084596896ee552c5b" width="900" hegiht="" ></center>
<center><br>图2：“横纵式”教学法 — 网络结构优化 </br></center>

<br></br>


### 数据处理

在介绍网络结构前，需要先进行数据处理，代码与上一节保持一致。


```python
#数据处理部分之前的代码，保持不变
import os
import random
import paddle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import gzip
import json

# 定义数据集读取器
def load_data(mode='train'):

    # 加载数据
    datafile = './work/mnist.json.gz'
    print('loading mnist dataset from {} ......'.format(datafile))
    data = json.load(gzip.open(datafile))
    print('mnist dataset load done')

    # 读取到的数据区分训练集，验证集，测试集
    train_set, val_set, eval_set = data

    # 数据集相关参数，图片高度IMG_ROWS, 图片宽度IMG_COLS
    IMG_ROWS = 28
    IMG_COLS = 28

    if mode == 'train':
        # 获得训练数据集
        imgs, labels = train_set[0], train_set[1]
    elif mode == 'valid':
        # 获得验证数据集
        imgs, labels = val_set[0], val_set[1]
    elif mode == 'eval':
        # 获得测试数据集
        imgs, labels = eval_set[0], eval_set[1]
    else:
        raise Exception("mode can only be one of ['train', 'valid', 'eval']")

    #校验数据
    imgs_length = len(imgs)
    assert len(imgs) == len(labels), \
          "length of train_imgs({}) should be the same as train_labels({})".format(
                  len(imgs), len(labels))

    # 定义数据集每个数据的序号， 根据序号读取数据
    index_list = list(range(imgs_length))
    # 读入数据时用到的batchsize
    BATCHSIZE = 100
    
    # 定义数据生成器
    def data_generator():
        if mode == 'train':
            random.shuffle(index_list)
        imgs_list = []
        labels_list = []
        for i in index_list:
            img = np.array(imgs[i]).astype('float32')
            label = np.array(labels[i]).astype('float32')
            # 在使用卷积神经网络结构时，uncomment 下面两行代码
            img = np.reshape(imgs[i], [1, IMG_ROWS, IMG_COLS]).astype('float32')
            label = np.reshape(labels[i], [1]).astype('float32')
            
            imgs_list.append(img) 
            labels_list.append(label)
            if len(imgs_list) == BATCHSIZE:
                yield np.array(imgs_list), np.array(labels_list)
                imgs_list = []
                labels_list = []

        # 如果剩余数据的数目小于BATCHSIZE，
        # 则剩余数据一起构成一个大小为len(imgs_list)的mini-batch
        if len(imgs_list) > 0:
            yield np.array(imgs_list), np.array(labels_list)

    return data_generator
```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/__init__.py:107: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      from collections import MutableMapping
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/rcsetup.py:20: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      from collections import Iterable, Mapping
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/colors.py:53: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      from collections import Sized


# 经典的全连接神经网络


经典的全连接神经网络来包含四层网络：输入层、两个隐含层和输出层，将手写数字识别任务通过全连接神经网络表示，如 **图3** 所示。

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/2173259df0704335b230ec158be0427677b9c77fd42348a28f2f8adf1ac1c706" width="600" hegiht="" ></center>
<center><br>图3：手写数字识别任务的全连接神经网络结构</br></center>

<br></br>

* 输入层：将数据输入给神经网络。在该任务中，输入层的尺度为28×28的像素值。
* 隐含层：增加网络深度和复杂度，隐含层的节点数是可以调整的，节点数越多，神经网络表示能力越强，参数量也会增加。在该任务中，中间的两个隐含层为10×10的结构，通常隐含层会比输入层的尺寸小，以便对关键信息做抽象，激活函数使用常见的Sigmoid函数。
* 输出层：输出网络计算结果，输出层的节点数是固定的。如果是回归问题，节点数量为需要回归的数字数量。如果是分类问题，则是分类标签的数量。在该任务中，模型的输出是回归一个数字，输出层的尺寸为1。

------

**说明：**

隐含层引入非线性激活函数Sigmoid是为了增加神经网络的非线性能力。

举例来说，如果一个神经网络采用线性变换，有四个输入$x_1$~$x_4$，一个输出$y$。假设第一层的变换是$z_1=x_1-x_2$和$z_2=x_3+x_4$，第二层的变换是$y=z_1+z_2$，则将两层的变换展开后得到$y=x_1-x_2+x_3+x_4$。也就是说，无论中间累积了多少层线性变换，原始输入和最终输出之间依然是线性关系。

------

Sigmoid是早期神经网络模型中常见的非线性变换函数，通过如下代码，绘制出Sigmoid的函数曲线。


```python
def sigmoid(x):
    # 直接返回sigmoid函数
    return 1. / (1. + np.exp(-x))
 
# param:起点，终点，间距
x = np.arange(-8, 8, 0.2)
y = sigmoid(x)
plt.plot(x, y)
plt.show()
```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/cbook/__init__.py:2349: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      if isinstance(obj, collections.Iterator):
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/cbook/__init__.py:2366: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      return list(data) if isinstance(data, collections.MappingView) else data

![output_3_1-16596100085161](photo/output_3_1-16596100085161.png)


针对手写数字识别的任务，网络层的设计如下：

* 输入层的尺度为28×28，但批次计算的时候会统一加1个维度（大小为batch size）。
* 中间的两个隐含层为10×10的结构，激活函数使用常见的Sigmoid函数。
* 与房价预测模型一样，模型的输出是回归一个数字，输出层的尺寸设置成1。

下述代码为经典全连接神经网络的实现。完成网络结构定义后，即可训练神经网络。


```python
import paddle.nn.functional as F
from paddle.nn import Linear

# 定义多层全连接神经网络
class MNIST(paddle.nn.Layer):
    def __init__(self):
        super(MNIST, self).__init__()
        # 定义两层全连接隐含层，输出维度是10，当前设定隐含节点数为10，可根据任务调整
        self.fc1 = Linear(in_features=784, out_features=10)
        self.fc2 = Linear(in_features=10, out_features=10)
        # 定义一层全连接输出层，输出维度是1
        self.fc3 = Linear(in_features=10, out_features=1)
    
    # 定义网络的前向计算，隐含层激活函数为sigmoid，输出层不使用激活函数
    def forward(self, inputs):
        # inputs = paddle.reshape(inputs, [inputs.shape[0], 784])
        outputs1 = self.fc1(inputs)
        outputs1 = F.sigmoid(outputs1)
        outputs2 = self.fc2(outputs1)
        outputs2 = F.sigmoid(outputs2)
        outputs_final = self.fc3(outputs2)
        return outputs_final
```

# 卷积神经网络

虽然使用经典的全连接神经网络可以提升一定的准确率，但其输入数据的形式导致丢失了图像像素间的空间信息，这影响了网络对图像内容的理解。对于计算机视觉问题，效果最好的模型仍然是卷积神经网络。卷积神经网络针对视觉问题的特点进行了网络结构优化，可以直接处理原始形式的图像数据，保留像素间的空间信息，因此更适合处理视觉问题。

卷积神经网络由多个卷积层和池化层组成，如 **图4** 所示。卷积层负责对输入进行扫描以生成更抽象的特征表示，池化层对这些特征表示进行过滤，保留最关键的特征信息。

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/91f3755dfd47461aa04567e73474a3ca56107402feb841a592ddaa7dfcbc67c2" width="800" hegiht="" ></center>

<center><br>图4：在处理计算机视觉任务中大放异彩的卷积神经网络</br></center>

<br></br>


-------

**说明：**

本节只简单介绍用卷积神经网络实现手写数字识别任务，以及它带来的效果提升。读者可以将卷积神经网络先简单的理解成是一种比经典的全连接神经网络更强大的模型即可，更详细的原理和实现在接下来的《计算机视觉-卷积神经网络基础》中讲述。

------

两层卷积和池化的神经网络实现如下所示。


```python
# 定义 SimpleNet 网络结构
import paddle
from paddle.nn import Conv2D, MaxPool2D, Linear
import paddle.nn.functional as F
# 多层卷积神经网络实现
class MNIST(paddle.nn.Layer):
     def __init__(self):
         super(MNIST, self).__init__()
         
         # 定义卷积层，输出特征通道out_channels设置为20，卷积核的大小kernel_size为5，卷积步长stride=1，padding=2
         self.conv1 = Conv2D(in_channels=1, out_channels=20, kernel_size=5, stride=1, padding=2)
         # 定义池化层，池化核的大小kernel_size为2，池化步长为2
         self.max_pool1 = MaxPool2D(kernel_size=2, stride=2)
         # 定义卷积层，输出特征通道out_channels设置为20，卷积核的大小kernel_size为5，卷积步长stride=1，padding=2
         self.conv2 = Conv2D(in_channels=20, out_channels=20, kernel_size=5, stride=1, padding=2)
         # 定义池化层，池化核的大小kernel_size为2，池化步长为2
         self.max_pool2 = MaxPool2D(kernel_size=2, stride=2)
         # 定义一层全连接层，输出维度是1
         self.fc = Linear(in_features=980, out_features=1)
         
    # 定义网络前向计算过程，卷积后紧接着使用池化层，最后使用全连接层计算最终输出
    # 卷积层激活函数使用Relu，全连接层不使用激活函数
     def forward(self, inputs):
         x = self.conv1(inputs)
         x = F.relu(x)
         x = self.max_pool1(x)
         x = self.conv2(x)
         x = F.relu(x)
         x = self.max_pool2(x)
         x = paddle.reshape(x, [x.shape[0], -1])
         x = self.fc(x)
         return x
```


使用MNIST数据集训练定义好的卷积神经网络，如下所示。  

------

**说明：**  
以上数据加载函数load_data返回一个数据迭代器train_loader，该train_loader在每次迭代时的数据shape为[batch_size, 784]，因此需要将该数据形式reshape为图像数据形式[batch_size, 1, 28, 28]，其中第二维代表图像的通道数（在MNIST数据集中每张图片的通道数为1，传统RGB图片通道数为3）。


```python
#网络结构部分之后的代码，保持不变
def train(model):
    model.train()
    #调用加载数据的函数，获得MNIST训练数据集
    train_loader = load_data('train')
    # 使用SGD优化器，learning_rate设置为0.01
    opt = paddle.optimizer.SGD(learning_rate=0.01, parameters=model.parameters())
    # 训练5轮
    EPOCH_NUM = 10
    # MNIST图像高和宽
    IMG_ROWS, IMG_COLS = 28, 28

    for epoch_id in range(EPOCH_NUM):
        for batch_id, data in enumerate(train_loader()):
            #准备数据
            images, labels = data
            images = paddle.to_tensor(images)
            labels = paddle.to_tensor(labels)
            
            #前向计算的过程
            predicts = model(images)
            
            #计算损失，取一个批次样本损失的平均值
            loss = F.square_error_cost(predicts, labels)
            avg_loss = paddle.mean(loss)

            #每训练200批次的数据，打印下当前Loss的情况
            if batch_id % 200 == 0:
                print("epoch: {}, batch: {}, loss is: {}".format(epoch_id, batch_id, avg_loss.numpy()))
            
            #后向传播，更新参数的过程
            avg_loss.backward()
            # 最小化loss,更新参数
            opt.step()
            # 清除梯度
            opt.clear_grad()

    #保存模型参数
    paddle.save(model.state_dict(), 'mnist.pdparams')

model = MNIST()
train(model)
```

    loading mnist dataset from ./work/mnist.json.gz ......
    mnist dataset load done
    epoch: 0, batch: 0, loss is: [29.018454]
    epoch: 0, batch: 200, loss is: [8.702309]
    epoch: 0, batch: 400, loss is: [3.5032523]
    epoch: 1, batch: 0, loss is: [2.4425797]
    epoch: 1, batch: 200, loss is: [2.676876]
    epoch: 1, batch: 400, loss is: [2.1995175]
    epoch: 2, batch: 0, loss is: [3.0969505]
    epoch: 2, batch: 200, loss is: [2.8459315]
    epoch: 2, batch: 400, loss is: [2.0763125]
    epoch: 3, batch: 0, loss is: [2.0255857]
    epoch: 3, batch: 200, loss is: [3.3289883]
    epoch: 3, batch: 400, loss is: [1.6783986]
    epoch: 4, batch: 0, loss is: [3.3174667]
    epoch: 4, batch: 200, loss is: [1.7719826]
    epoch: 4, batch: 400, loss is: [1.3528117]
    epoch: 5, batch: 0, loss is: [1.9914469]
    epoch: 5, batch: 200, loss is: [2.5105848]
    epoch: 5, batch: 400, loss is: [1.2759887]
    epoch: 6, batch: 0, loss is: [1.9961126]
    epoch: 6, batch: 200, loss is: [1.4300797]
    epoch: 6, batch: 400, loss is: [1.8186872]
    epoch: 7, batch: 0, loss is: [1.5185082]
    epoch: 7, batch: 200, loss is: [1.4938874]
    epoch: 7, batch: 400, loss is: [1.228914]
    epoch: 8, batch: 0, loss is: [2.3227534]
    epoch: 8, batch: 200, loss is: [1.5087241]
    epoch: 8, batch: 400, loss is: [1.315771]
    epoch: 9, batch: 0, loss is: [1.604739]
    epoch: 9, batch: 200, loss is: [1.9549865]
    epoch: 9, batch: 400, loss is: [0.98003715]


比较经典全连接神经网络和卷积神经网络的损失变化，可以发现卷积神经网络的损失值下降更快，且最终的损失值更小。

# 4.【手写数字识别】之损失函数

上一节我们尝试通过更复杂的模型（经典的全连接神经网络和卷积神经网络），提升手写数字识别模型训练的准确性。本节我们继续将“横纵式”教学法从横向展开，如 **图1** 所示，探讨损失函数的优化对模型训练效果的影响。

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/8ed4007a16f24549affef4549ba4e2228729b94bc69c479f91c57d0cf3a8bf52" width="1000" hegiht="" ></center>
<center><br>图1：“横纵式”教学法 — 损失函数优化 </br></center>

<br></br>

损失函数是模型优化的目标，用于在众多的参数取值中，识别最理想的取值。损失函数的计算在训练过程的代码中，每一轮模型训练的过程都相同，分如下三步：

1. 先根据输入数据正向计算预测输出。
1. 再根据预测值和真实值计算损失。
1. 最后根据损失反向传播梯度并更新参数。

# 分类任务的损失函数

在之前的方案中，我们复用了房价预测模型的损失函数-均方误差。从预测效果来看，虽然损失不断下降，模型的预测值逐渐逼近真实值，但模型的最终效果不够理想。究其根本，不同的深度学习任务需要有各自适宜的损失函数。我们以房价预测和手写数字识别两个任务为例，详细剖析其中的缘由如下：

1. 房价预测是回归任务，而手写数字识别是分类任务，使用均方误差作为分类任务的损失函数存在逻辑和效果上的缺欠。
1. 房价可以是大于0的任何浮点数，而手写数字识别的输出只可能是0~9之间的10个整数，相当于一种标签。
1. 在房价预测的案例中，由于房价本身是一个连续的实数值，因此以模型输出的数值和真实房价差距作为损失函数（Loss）是符合道理的。但对于分类问题，真实结果是分类标签，而模型输出是实数值，导致以两者相减作为损失不具备物理含义。

那么，什么是分类任务的合理输出呢？分类任务本质上是“某种特征组合下的分类概率”，下面以一个简单案例说明，如 **图2** 所示。

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/c9f479c2960140839b259ca7ab2256a0dcd7a714e76a4edfb5377f1566796460" width="700" hegiht="" ></center>
<center><br>图2：观测数据和背后规律之间的关系 </br></center>

<br></br>


在本案例中，医生根据肿瘤大小$x$作为肿瘤性质$y$的参考判断（判断的因素有很多，肿瘤大小只是其中之一），那么我们观测到该模型判断的结果是$x$和$y$的标签（1为恶性，0为良性）。而这个数据背后的规律是不同大小的肿瘤，属于恶性肿瘤的概率。观测数据是真实规律抽样下的结果，分类模型应该拟合这个真实规律，输出属于该分类标签的概率。

## Softmax函数

如果模型能输出10个标签的概率，对应真实标签的概率输出尽可能接近100%，而其他标签的概率输出尽可能接近0%，且所有输出概率之和为1。这是一种更合理的假设！与此对应，真实的标签值可以转变成一个10维度的one-hot向量，在对应数字的位置上为1，其余位置为0，比如标签“6”可以转变成[0,0,0,0,0,0,1,0,0,0]。

为了实现上述思路，需要引入Softmax函数，它可以将原始输出转变成对应标签的概率，公式如下，其中$C$是标签类别个数。

$$softmax(x_i) = \frac {e^{x_i}}{\sum_{j=0}^N{e^{x_j}}}, i=0, ..., C-1$$ 

从公式的形式可见，每个输出的范围均在0~1之间，且所有输出之和等于1，这是这种变换后可被解释成概率的基本前提。对应到代码上，需要在前向计算中，对全连接网络的输出层增加一个Softmax运算，`outputs = F.softmax(outputs)`。

**图3** 是一个三个标签的分类模型（三分类）使用的Softmax输出层，从中可见原始输出的三个数字3、1、-3，经过Softmax层后转变成加和为1的三个概率值0.88、0.12、0。

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/ef129caf64254318821e9410bb71ab1f45fff20e4282482986081d44a1e3bcbb" width="600" hegiht="" ></center>
<center><br>图3：网络输出层改为softmax函数 </br></center>

<br></br>


上文解释了为何让分类模型的输出拟合概率的原因，但为何偏偏用Softmax函数完成这个职能？ 下面以二分类问题（只输出两个标签）进行原理的探讨。

对于二分类问题，使用两个输出接入Softmax作为输出层，等价于使用单一输出接入Sigmoid函数。如 **图4** 所示，利用两个标签的输出概率之和为1的条件，Softmax输出0.6和0.4两个标签概率，从数学上等价于输出一个标签的概率0.6。

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/4dbdf378438f42b0bc6de6f11955834b7063cc6916544017b0af2ccf1f730984" width="400" hegiht="" ></center>
<center><br>图4：对于二分类问题，等价于单一输出接入Sigmoid函数 </br></center>

<br></br>

在这种情况下，只有一层的模型为$S(w^{T}x_i)$，$S$为Sigmoid函数。模型预测为1的概率为$S(w^{T}x_i)$，模型预测为0的概率为$1-S(w^{T}x_i)$。

**图5** 是肿瘤大小和肿瘤性质的数据图。从图中可发现，往往尺寸越大的肿瘤几乎全部是恶性，尺寸极小的肿瘤几乎全部是良性。只有在中间区域，肿瘤的恶性概率会从0逐渐到1（绿色区域），这种数据的分布是符合多数现实问题的规律。如果我们直接线性拟合，相当于红色的直线，会发现直线的纵轴0-1的区域会拉的很长，而我们期望拟合曲线0-1的区域与真实的分类边界区域重合。那么，观察下Sigmoid的曲线趋势可以满足我们对个问题的一切期望，它的概率变化会集中在一个边界区域，有助于模型提升边界区域的分辨率。

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/bbf5e0eda62c44bb84528dbfd8642ef901b2dc42c6f541bc8cd0b75b967dc934" width="900" hegiht="" ></center>
<center><br>图5：使用Sigmoid拟合输出可提高分类模型对边界的分辨率 </br></center>

<br></br>

这就类似于公共区域使用的带有恒温装置的热水器温度阀门，如 **图6** 所示。由于人体适应的水温在34度-42度之间，我们更期望阀门的水温条件集中在这个区域，而不是在0-100度之间线性分布。

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/9d05d75c9db44d95b8cdec6fe1615e24d9a24c0ce4f64954a2a5659aaaa7437b" width="400" hegiht="" ></center>
<center><br>图6：热水器水温控制 </br></center>

<br></br>



## 交叉熵

在模型输出为分类标签的概率时，直接以标签和概率做比较也不够合理，人们更习惯使用交叉熵误差作为分类问题的损失衡量。

交叉熵损失函数的设计是基于最大似然思想：最大概率得到观察结果的假设是真的。如何理解呢？举个例子来说，如 **图7** 所示。有两个外形相同的盒子，甲盒中有99个白球，1个蓝球；乙盒中有99个蓝球，1个白球。一次试验取出了一个蓝球，请问这个球应该是从哪个盒子中取出的？

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/13a942e5ec7f4e91badb2f4613c6f71a00e51c8afb6a435e94a0b47cedac9515" width="400" hegiht="" ></center>
<center><br>图7：体会最大似然的思想 </br></center>

<br></br>


相信大家简单思考后均会得出更可能是从乙盒中取出的，因为从乙盒中取出一个蓝球的概率更高$（P(D|h)）$，所以观察到一个蓝球更可能是从乙盒中取出的$(P(h|D))$。$D$是观测的数据，即蓝球白球；$h$是模型，即甲盒乙盒。这就是贝叶斯公式所表达的思想：

$$P(h|D) ∝ P(h) \cdot P(D|h)$$

依据贝叶斯公式，某二分类模型“生成”$n$个训练样本的概率：

$$P(x_1)\cdot S(w^{T}x_1)\cdot P(x_2)\cdot(1-S(w^{T}x_2))\cdot … \cdot P(x_n)\cdot S(w^{T}x_n)$$

------

**说明：**

对于二分类问题，模型为$S(w^{T}x_i)$，$S$为Sigmoid函数。当$y_i$=1，概率为$S(w^{T}x_i)$；当$y_i$=0，概率为$1-S(w^{T}x_i)$。

------

经过公式推导，使得上述概率最大等价于最小化交叉熵，得到交叉熵的损失函数。交叉熵的公式如下：

$$ L = -[\sum_{k=1}^{n} t_k\log y_k +(1- t_k)\log(1-y_k)] $$

其中，$\log$表示以$e$为底数的自然对数。$y_k$代表模型输出，$t_k$代表各个标签。$t_k$中只有正确解的标签为1，其余均为0（one-hot表示）。

因此，交叉熵只计算对应着“正确解”标签的输出的自然对数。比如，假设正确标签的索引是“2”，与之对应的神经网络的输出是0.6，则交叉熵误差是$−\log 0.6 = 0.51$；若“2”对应的输出是0.1，则交叉熵误差为$−\log 0.1 = 2.30$。由此可见，交叉熵误差的值是由正确标签所对应的输出结果决定的。

自然对数的函数曲线可由如下代码实现。


```python
import matplotlib.pyplot as plt
import numpy as np
x = np.arange(0.01,1,0.01)
y = np.log(x)
plt.title("y=log(x)") 
plt.xlabel("x") 
plt.ylabel("y") 
plt.plot(x,y)
plt.show()
plt.figure()
```


    <Figure size 640x480 with 1 Axes>





    <Figure size 640x480 with 0 Axes>



如自然对数的图形所示，当$x$等于1时，$y$为0；随着$x$向0靠近，$y$逐渐变小。因此，正确解标签对应的输出越大，交叉熵的值越接近0；当输出为1时，交叉熵误差为0。反之，如果正确解标签对应的输出越小，则交叉熵的值越大。

## 交叉熵的代码实现

在手写数字识别任务中，仅改动三行代码，就可以将在现有模型的损失函数替换成交叉熵（Cross_entropy）。

* 在读取数据部分，将标签的类型设置成int，体现它是一个标签而不是实数值（飞桨框架默认将标签处理成int64）。
* 在网络定义部分，将输出层改成“输出十个标签的概率”的模式。
* 在训练过程部分，将损失函数从均方误差换成交叉熵。

在数据处理部分，需要修改标签变量Label的格式，代码如下所示。

- 从：label = np.reshape(labels[i], [1]).astype('float32')
- 到：label = np.reshape(labels[i], [1]).astype('int64')


```python
#数据处理部分之前的代码，保持不变
import os
import random
import paddle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import gzip
import json

# 创建一个类MnistDataset，继承paddle.io.Dataset 这个类
# MnistDataset的作用和上面load_data()函数的作用相同，均是构建一个迭代器
class MnistDataset(paddle.io.Dataset):
    def __init__(self, mode):
        datafile = './work/mnist.json.gz'
        data = json.load(gzip.open(datafile))
        # 读取到的数据区分训练集，验证集，测试集
        train_set, val_set, eval_set = data
        
        # 数据集相关参数，图片高度IMG_ROWS, 图片宽度IMG_COLS
        self.IMG_ROWS = 28
        self.IMG_COLS = 28

        if mode=='train':
            # 获得训练数据集
            imgs, labels = train_set[0], train_set[1]
        elif mode=='valid':
            # 获得验证数据集
            imgs, labels = val_set[0], val_set[1]
        elif mode=='eval':
            # 获得测试数据集
            imgs, labels = eval_set[0], eval_set[1]
        else:
            raise Exception("mode can only be one of ['train', 'valid', 'eval']")
        
        # 校验数据
        imgs_length = len(imgs)
        assert len(imgs) == len(labels), \
            "length of train_imgs({}) should be the same as train_labels({})".format(len(imgs), len(labels))
        
        self.imgs = imgs
        self.labels = labels

    def __getitem__(self, idx):
        # img = np.array(self.imgs[idx]).astype('float32')
        # label = np.array(self.labels[idx]).astype('int64')
        img = np.reshape(self.imgs[idx], [1, self.IMG_ROWS, self.IMG_COLS]).astype('float32')
        label = np.reshape(self.labels[idx], [1]).astype('int64')

        return img, label

    def __len__(self):
        return len(self.imgs)
# 声明数据加载函数，使用训练模式，MnistDataset构建的迭代器每次迭代只返回batch=1的数据
train_dataset = MnistDataset(mode='train')
# 使用paddle.io.DataLoader 定义DataLoader对象用于加载Python生成器产生的数据，
# DataLoader 返回的是一个批次数据迭代器，并且是异步的；
train_loader = paddle.io.DataLoader(train_dataset, batch_size=100, shuffle=True, drop_last=True)
val_dataset = MnistDataset(mode='valid')
val_loader = paddle.io.DataLoader(val_dataset, batch_size=128,drop_last=True)

```

在网络定义部分，需要修改输出层结构，代码如下所示。

- 从：self.fc = Linear(in_features=980, out_features=1)
- 到：self.fc = Linear(in_features=980, out_features=10)


```python
# 定义 SimpleNet 网络结构
import paddle
from paddle.nn import Conv2D, MaxPool2D, Linear
import paddle.nn.functional as F
# 多层卷积神经网络实现
class MNIST(paddle.nn.Layer):
     def __init__(self):
         super(MNIST, self).__init__()
         
         # 定义卷积层，输出特征通道out_channels设置为20，卷积核的大小kernel_size为5，卷积步长stride=1，padding=2
         self.conv1 = Conv2D(in_channels=1, out_channels=20, kernel_size=5, stride=1, padding=2)
         # 定义池化层，池化核的大小kernel_size为2，池化步长为2
         self.max_pool1 = MaxPool2D(kernel_size=2, stride=2)
         # 定义卷积层，输出特征通道out_channels设置为20，卷积核的大小kernel_size为5，卷积步长stride=1，padding=2
         self.conv2 = Conv2D(in_channels=20, out_channels=20, kernel_size=5, stride=1, padding=2)
         # 定义池化层，池化核的大小kernel_size为2，池化步长为2
         self.max_pool2 = MaxPool2D(kernel_size=2, stride=2)
         # 定义一层全连接层，输出维度是10
         self.fc = Linear(in_features=980, out_features=10)
         
    # 定义网络前向计算过程，卷积后紧接着使用池化层，最后使用全连接层计算最终输出
    # 卷积层激活函数使用Relu，全连接层激活函数使用softmax
     def forward(self, inputs):
         x = self.conv1(inputs)
         x = F.relu(x)
         x = self.max_pool1(x)
         x = self.conv2(x)
         x = F.relu(x)
         x = self.max_pool2(x)
         x = paddle.reshape(x, [x.shape[0], 980])
         x = self.fc(x)
         return x
```

修改计算损失的函数，从均方误差（常用于回归问题）到交叉熵误差（常用于分类问题），代码如下所示。

- 从：loss = paddle.nn.functional.square_error_cost(predict, label)
- 到：loss = paddle.nn.functional.cross_entropy(predict, label)



```python
def evaluation(model, datasets):
    model.eval()

    acc_set = list()
    for batch_id, data in enumerate(datasets()):
        images, labels = data
        images = paddle.to_tensor(images)
        labels = paddle.to_tensor(labels)
        pred = model(images)   # 获取预测值
        acc = paddle.metric.accuracy(input=pred, label=labels)
        acc_set.extend(acc.numpy())
    
    # #计算多个batch的准确率
    acc_val_mean = np.array(acc_set).mean()
    return acc_val_mean
```


```python
#仅修改计算损失的函数，从均方误差（常用于回归问题）到交叉熵误差（常用于分类问题）
def train(model):
    model.train()
    #调用加载数据的函数
    # train_loader = load_data('train')
    # val_loader = load_data('valid')
    opt = paddle.optimizer.SGD(learning_rate=0.01, parameters=model.parameters())
    EPOCH_NUM = 10
    for epoch_id in range(EPOCH_NUM):
        for batch_id, data in enumerate(train_loader()):
            #准备数据
            images, labels = data
            images = paddle.to_tensor(images)
            labels = paddle.to_tensor(labels)
            #前向计算的过程
            predicts = model(images)
            
            #计算损失，使用交叉熵损失函数，取一个批次样本损失的平均值
            loss = F.cross_entropy(predicts, labels)
            avg_loss = paddle.mean(loss)
            
            #每训练了200批次的数据，打印下当前Loss的情况
            if batch_id % 200 == 0:
                print("epoch: {}, batch: {}, loss is: {}".format(epoch_id, batch_id, avg_loss.numpy()))
            
            #后向传播，更新参数的过程
            avg_loss.backward()
            # 最小化loss,更新参数
            opt.step()
            # 清除梯度
            opt.clear_grad()
        # acc_train_mean = evaluation(model, train_loader)
        # acc_val_mean = evaluation(model, val_loader)
        # print('train_acc: {}, val acc: {}'.format(acc_train_mean, acc_val_mean))   
    #保存模型参数
    paddle.save(model.state_dict(), 'mnist.pdparams')
    
#创建模型    
model = MNIST()
#启动训练过程
train(model)
```

虽然上述训练过程的损失明显比使用均方误差算法要小，但因为损失函数量纲的变化，我们无法从比较两个不同的Loss得出谁更加优秀。怎么解决这个问题呢？我们可以回归到问题的本质，谁的分类准确率更高来判断。在后面介绍完计算准确率和作图的内容后，读者可以自行测试采用不同损失函数下，模型准确率的高低。

至此，大家阅读论文中常见的一些分类任务模型图就清晰明了，如全连接神经网络、卷积神经网络，在模型的最后阶段，都是使用Softmax进行处理。

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/309156e1b2f545ef88102fce6b4685fa3a12cf6334db47fa8fe0a281ecd89947" width="1000" hegiht="" ></center>
<center><br>图8：常见的分类任务模型图</br></center>

<br></br>

由于我们修改了模型的输出格式，因此使用模型做预测时的代码也需要做相应的调整。从模型输出10个标签的概率中选择最大的，将其标签编号输出。


```python
# 读取一张本地的样例图片，转变成模型输入的格式
def load_image(img_path):
    # 从img_path中读取图像，并转为灰度图
    im = Image.open(img_path).convert('L')
    im = im.resize((28, 28), Image.ANTIALIAS)
    im = np.array(im).reshape(1, 1, 28, 28).astype(np.float32)
    # 图像归一化
    im = 1.0 - im / 255.
    return im

# 定义预测过程
model = MNIST()
params_file_path = 'mnist.pdparams'
img_path = 'work/example_0.jpg'
# 加载模型参数
param_dict = paddle.load(params_file_path)
model.load_dict(param_dict)
# 灌入数据
model.eval()
tensor_img = load_image(img_path)
#模型反馈10个分类标签的对应概率
results = model(paddle.to_tensor(tensor_img))
#取概率最大的标签作为预测输出
lab = np.argsort(results.numpy())
print("本次预测的数字是: ", lab[0][-1])
```

## 作业 2-2

预习下对于计算机视觉任务，有哪些常见的卷积神经网络（如LeNet-5、AlexNet等）？



*****

# 5.【手写数字识别】之优化算法

上一节我们明确了分类任务的损失函数（优化目标）的相关概念和实现方法，本节我们依旧横向展开"横纵式"教学法，如 **图1** 所示，本节主要探讨在手写数字识别任务中，使得损失达到最小的参数取值的实现方法。

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/af41e7b72180495c96e3ed4a370e9e030addebdfd16d42bc9035c53ca5883cd9" width="1000" hegiht="" ></center>
<center>图1：“横纵式”教学法 — 优化算法</center>

<br></br>

### 前提条件

在优化算法之前，需要进行数据处理、设计神经网络结构，代码与上一节保持一致，如下所示。如果读者已经掌握了这部分内容，可以直接阅读正文部分。


```python
# 加载相关库
import os
import random
import paddle
from paddle.nn import Conv2D, MaxPool2D, Linear
import numpy as np
from PIL import Image
import gzip
import json

# 定义数据集读取器
def load_data(mode='train'):
    # 读取数据文件
    datafile = './work/mnist.json.gz'
    print('loading mnist dataset from {} ......'.format(datafile))
    data = json.load(gzip.open(datafile))
    # 读取数据集中的训练集，验证集和测试集
    train_set, val_set, eval_set = data

    # 数据集相关参数，图片高度IMG_ROWS, 图片宽度IMG_COLS
    IMG_ROWS = 28
    IMG_COLS = 28
    # 根据输入mode参数决定使用训练集，验证集还是测试
    if mode == 'train':
        imgs = train_set[0]
        labels = train_set[1]
    elif mode == 'valid':
        imgs = val_set[0]
        labels = val_set[1]
    elif mode == 'eval':
        imgs = eval_set[0]
        labels = eval_set[1]
    # 获得所有图像的数量
    imgs_length = len(imgs)
    # 验证图像数量和标签数量是否一致
    assert len(imgs) == len(labels), \
          "length of train_imgs({}) should be the same as train_labels({})".format(
                  len(imgs), len(labels))

    index_list = list(range(imgs_length))

    # 读入数据时用到的batchsize
    BATCHSIZE = 100

    # 定义数据生成器
    def data_generator():
        # 训练模式下，打乱训练数据
        if mode == 'train':
            random.shuffle(index_list)
        imgs_list = []
        labels_list = []
        # 按照索引读取数据
        for i in index_list:
            # 读取图像和标签，转换其尺寸和类型
            img = np.reshape(imgs[i], [1, IMG_ROWS, IMG_COLS]).astype('float32')
            label = np.reshape(labels[i], [1]).astype('int64')
            imgs_list.append(img) 
            labels_list.append(label)
            # 如果当前数据缓存达到了batch size，就返回一个批次数据
            if len(imgs_list) == BATCHSIZE:
                yield np.array(imgs_list), np.array(labels_list)
                # 清空数据缓存列表
                imgs_list = []
                labels_list = []

        # 如果剩余数据的数目小于BATCHSIZE，
        # 则剩余数据一起构成一个大小为len(imgs_list)的mini-batch
        if len(imgs_list) > 0:
            yield np.array(imgs_list), np.array(labels_list)

    return data_generator

# 定义模型结构
import paddle.nn.functional as F
# 多层卷积神经网络实现
class MNIST(paddle.nn.Layer):
     def __init__(self):
         super(MNIST, self).__init__()
         
         # 定义卷积层，输出特征通道out_channels设置为20，卷积核的大小kernel_size为5，卷积步长stride=1，padding=2
         self.conv1 = Conv2D(in_channels=1, out_channels=20, kernel_size=5, stride=1, padding=2)
         # 定义池化层，池化核的大小kernel_size为2，池化步长为2
         self.max_pool1 = MaxPool2D(kernel_size=2, stride=2)
         # 定义卷积层，输出特征通道out_channels设置为20，卷积核的大小kernel_size为5，卷积步长stride=1，padding=2
         self.conv2 = Conv2D(in_channels=20, out_channels=20, kernel_size=5, stride=1, padding=2)
         # 定义池化层，池化核的大小kernel_size为2，池化步长为2
         self.max_pool2 = MaxPool2D(kernel_size=2, stride=2)
         # 定义一层全连接层，输出维度是10
         self.fc = Linear(in_features=980, out_features=10)
         
   # 定义网络前向计算过程，卷积后紧接着使用池化层，最后使用全连接层计算最终输出
   # 卷积层激活函数使用Relu，全连接层激活函数使用softmax
     def forward(self, inputs):
         x = self.conv1(inputs)
         x = F.relu(x)
         x = self.max_pool1(x)
         x = self.conv2(x)
         x = F.relu(x)
         x = self.max_pool2(x)
         x = paddle.reshape(x, [x.shape[0], -1])
         x = self.fc(x)
         return x
```

# 设置学习率

在深度学习神经网络模型中，通常使用标准的随机梯度下降算法更新参数，学习率代表参数更新幅度的大小，即步长。当学习率最优时，模型的有效容量最大，最终能达到的效果最好。学习率和深度学习任务类型有关，合适的学习率往往需要大量的实验和调参经验。探索学习率最优值时需要注意如下两点：

- **学习率不是越小越好**。学习率越小，损失函数的变化速度越慢，意味着我们需要花费更长的时间进行收敛，如 **图2** 左图所示。
- **学习率不是越大越好**。只根据总样本集中的一个批次计算梯度，抽样误差会导致计算出的梯度不是全局最优的方向，且存在波动。在接近最优解时，过大的学习率会导致参数在最优解附近震荡，损失难以收敛，如 **图2** 右图所示。

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/1e0f066dc9fa4e2bbc942447bdc0578c2ffc6afc15684154ae84bcf31b298d7b" width="500" hegiht="" ></center>
<center><br>图2: 不同学习率（步长过大/过小）的示意图</br></center>

<br></br>

在训练前，我们往往不清楚一个特定问题设置成怎样的学习率是合理的，因此在训练时可以尝试调小或调大，通过观察Loss下降的情况判断合理的学习率，设置学习率的代码如下所示。


```python
#仅优化算法的设置有所差别
def train(model):
    model.train()
    #调用加载数据的函数
    train_loader = load_data('train')

    #设置不同初始学习率
    opt = paddle.optimizer.SGD(learning_rate=0.001, parameters=model.parameters())
    # opt = paddle.optimizer.SGD(learning_rate=0.0001, parameters=model.parameters())
    # opt = paddle.optimizer.SGD(learning_rate=0.01, parameters=model.parameters())
    
    EPOCH_NUM = 10
    for epoch_id in range(EPOCH_NUM):
        for batch_id, data in enumerate(train_loader()):
            #准备数据
            images, labels = data
            images = paddle.to_tensor(images)
            labels = paddle.to_tensor(labels)
            
            #前向计算的过程
            predicts = model(images)
            
            #计算损失，取一个批次样本损失的平均值
            loss = F.cross_entropy(predicts, labels)
            avg_loss = paddle.mean(loss)
            
            #每训练了100批次的数据，打印下当前Loss的情况
            if batch_id % 200 == 0:
                print("epoch: {}, batch: {}, loss is: {}".format(epoch_id, batch_id, avg_loss.numpy()))
            
            #后向传播，更新参数的过程
            avg_loss.backward()
            # 最小化loss,更新参数
            opt.step()
            # 清除梯度
            opt.clear_grad()
   
    #保存模型参数
    paddle.save(model.state_dict(), 'mnist.pdparams')
    
#创建模型    
model = MNIST()
#启动训练过程
train(model)
```


# 学习率的主流优化算法

学习率是优化器的一个参数，调整学习率看似是一件非常麻烦的事情，需要不断的调整步长，观察训练时间和Loss的变化。经过研究员的不断的实验，当前已经形成了四种比较成熟的优化算法：SGD、Momentum、AdaGrad和Adam，效果如 **图3** 所示。

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/f4cf80f95424411a85ad74998433317e721f56ddb4f64e6f8a28a27b6a1baa6b" width="800" hegiht="" ></center>
<center><br>图3: 不同学习率算法效果示意图</br></center>

<br></br>

* **[SGD：](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.0-beta/api/paddle/optimizer/SGD_cn.html)** 随机梯度下降算法，每次训练少量数据，抽样偏差导致的参数收敛过程中震荡。

* **[Momentum：](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.0-beta/api/paddle/optimizer/Momentum_cn.html)** 引入物理“动量”的概念，累积速度，减少震荡，使参数更新的方向更稳定。

每个批次的数据含有抽样误差，导致梯度更新的方向波动较大。如果我们引入物理动量的概念，给梯度下降的过程加入一定的“惯性”累积，就可以减少更新路径上的震荡，即每次更新的梯度由“历史多次梯度的累积方向”和“当次梯度”加权相加得到。历史多次梯度的累积方向往往是从全局视角更正确的方向，这与“惯性”的物理概念很像，也是为何其起名为“Momentum”的原因。类似不同品牌和材质的篮球有一定的重量差别，街头篮球队中的投手（擅长中远距离投篮）喜欢稍重篮球的比例较高。一个很重要的原因是，重的篮球惯性大，更不容易受到手势的小幅变形或风吹的影响。

* **[AdaGrad：](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.0-beta/api/paddle/optimizer/AdagradOptimizer_cn.html)** 根据不同参数距离最优解的远近，动态调整学习率。学习率逐渐下降，依据各参数变化大小调整学习率。

通过调整学习率的实验可以发现：当某个参数的现值距离最优解较远时（表现为梯度的绝对值较大），我们期望参数更新的步长大一些，以便更快收敛到最优解。当某个参数的现值距离最优解较近时（表现为梯度的绝对值较小），我们期望参数的更新步长小一些，以便更精细的逼近最优解。类似于打高尔夫球，专业运动员第一杆开球时，通常会大力打一个远球，让球尽量落在洞口附近。当第二杆面对离洞口较近的球时，他会更轻柔而细致的推杆，避免将球打飞。与此类似，参数更新的步长应该随着优化过程逐渐减少，减少的程度与当前梯度的大小有关。根据这个思想编写的优化算法称为“AdaGrad”，Ada是Adaptive的缩写，表示“适应环境而变化”的意思。[RMSProp](https://www.paddlepaddle.org.cn/documentation/docs/zh/api_cn/optimizer_cn/RMSPropOptimizer_cn.html#rmspropoptimizer)是在AdaGrad基础上的改进，学习率随着梯度变化而适应，解决AdaGrad学习率急剧下降的问题。

* **[Adam：](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.0-beta/api/paddle/optimizer/Adam_cn.html)**  由于动量和自适应学习率两个优化思路是正交的，因此可以将两个思路结合起来，这就是当前广泛应用的算法。



------

**说明：**

每种优化算法均有更多的参数设置，详情可查阅[飞桨的官方API文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/index_cn.html)。理论最合理的未必在具体案例中最有效，所以模型调参是很有必要的，最优的模型配置往往是在一定“理论”和“经验”的指导下实验出来的。

------

我们可以尝试选择不同的优化算法训练模型，观察训练时间和损失变化的情况，代码实现如下。


```python
#仅优化算法的设置有所差别
def train(model):
    model.train()
    #调用加载数据的函数
    train_loader = load_data('train')
    
    #四种优化算法的设置方案，可以逐一尝试效果
    opt = paddle.optimizer.SGD(learning_rate=0.01, parameters=model.parameters())
    # opt = paddle.optimizer.Momentum(learning_rate=0.01, momentum=0.9, parameters=model.parameters())
    # opt = paddle.optimizer.Adagrad(learning_rate=0.01, parameters=model.parameters())
    # opt = paddle.optimizer.Adam(learning_rate=0.01, parameters=model.parameters())
    
    EPOCH_NUM = 3
    for epoch_id in range(EPOCH_NUM):
        for batch_id, data in enumerate(train_loader()):
            #准备数据
            images, labels = data
            images = paddle.to_tensor(images)
            labels = paddle.to_tensor(labels)
            
            #前向计算的过程
            predicts = model(images)
            
            #计算损失，取一个批次样本损失的平均值
            loss = F.cross_entropy(predicts, labels)
            avg_loss = paddle.mean(loss)
            
            #每训练了100批次的数据，打印下当前Loss的情况
            if batch_id % 200 == 0:
                print("epoch: {}, batch: {}, loss is: {}".format(epoch_id, batch_id, avg_loss.numpy()))
            
            #后向传播，更新参数的过程
            avg_loss.backward()
            # 最小化loss,更新参数
            opt.step()
            # 清除梯度
            opt.clear_grad()

    #保存模型参数
    paddle.save(model.state_dict(), 'mnist.pdparams')
    
#创建模型    
model = MNIST()
#启动训练过程
train(model)
```

## 作业 2-3

在手写数字识别任务上，哪种优化算法的效果最好？多大的学习率最优？（可通过Loss的下降趋势来判断）



# 7.【手写数字识别】之资源配置

从前几节的训练看，无论是房价预测任务还是MNIST手写字数字识别任务，训练好一个模型不会超过10分钟，主要原因是我们所使用的神经网络比较简单。但实际应用时，常会遇到更加复杂的机器学习或深度学习任务，需要运算速度更高的硬件（如GPU、NPU），甚至同时使用多个机器共同训练一个任务（多卡训练和多机训练）。本节我们依旧横向展开"横纵式"教学方法，如 **图1** 所示，探讨在手写数字识别任务中，通过资源配置的优化，提升模型训练效率的方法。

<br></br>

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/917f13e9f6e042efb7041912a79266660db45dbbd98c4613867dff6fe8789d55" width="1000" hegiht="" ></center>
<center>图1：“横纵式”教学法 — 资源配置</center>

<br></br>

### 前提条件

需要先进行数据处理、设计神经网络结构，代码与上一节保持一致，如下所示。如果读者已经掌握了这部分内容，可以直接阅读正文部分。


```python
# 加载相关库
import os
import random
import paddle
from paddle.nn import Conv2D, MaxPool2D, Linear
import numpy as np
from PIL import Image
import gzip
import json

# 定义数据集读取器
def load_data(mode='train'):

    # 读取数据文件
    datafile = './work/mnist.json.gz'
    print('loading mnist dataset from {} ......'.format(datafile))
    data = json.load(gzip.open(datafile))
    # 读取数据集中的训练集，验证集和测试集
    train_set, val_set, eval_set = data

    # 数据集相关参数，图片高度IMG_ROWS, 图片宽度IMG_COLS
    IMG_ROWS = 28
    IMG_COLS = 28
    # 根据输入mode参数决定使用训练集，验证集还是测试
    if mode == 'train':
        imgs = train_set[0]
        labels = train_set[1]
    elif mode == 'valid':
        imgs = val_set[0]
        labels = val_set[1]
    elif mode == 'eval':
        imgs = eval_set[0]
        labels = eval_set[1]
    # 获得所有图像的数量
    imgs_length = len(imgs)
    # 验证图像数量和标签数量是否一致
    assert len(imgs) == len(labels), \
          "length of train_imgs({}) should be the same as train_labels({})".format(
                  len(imgs), len(labels))

    index_list = list(range(imgs_length))

    # 读入数据时用到的batchsize
    BATCHSIZE = 100

    # 定义数据生成器
    def data_generator():
        # 训练模式下，打乱训练数据
        if mode == 'train':
            random.shuffle(index_list)
        imgs_list = []
        labels_list = []
        # 按照索引读取数据
        for i in index_list:
            # 读取图像和标签，转换其尺寸和类型
            img = np.reshape(imgs[i], [1, IMG_ROWS, IMG_COLS]).astype('float32')
            label = np.reshape(labels[i], [1]).astype('int64')
            imgs_list.append(img) 
            labels_list.append(label)
            # 如果当前数据缓存达到了batch size，就返回一个批次数据
            if len(imgs_list) == BATCHSIZE:
                yield np.array(imgs_list), np.array(labels_list)
                # 清空数据缓存列表
                imgs_list = []
                labels_list = []

        # 如果剩余数据的数目小于BATCHSIZE，
        # 则剩余数据一起构成一个大小为len(imgs_list)的mini-batch
        if len(imgs_list) > 0:
            yield np.array(imgs_list), np.array(labels_list)

    return data_generator

# 定义模型结构
import paddle.nn.functional as F
# 多层卷积神经网络实现
class MNIST(paddle.nn.Layer):
     def __init__(self):
         super(MNIST, self).__init__()
         
         # 定义卷积层，输出特征通道out_channels设置为20，卷积核的大小kernel_size为5，卷积步长stride=1，padding=2
         self.conv1 = Conv2D(in_channels=1, out_channels=20, kernel_size=5, stride=1, padding=2)
         # 定义池化层，池化核的大小kernel_size为2，池化步长为2
         self.max_pool1 = MaxPool2D(kernel_size=2, stride=2)
         # 定义卷积层，输出特征通道out_channels设置为20，卷积核的大小kernel_size为5，卷积步长stride=1，padding=2
         self.conv2 = Conv2D(in_channels=20, out_channels=20, kernel_size=5, stride=1, padding=2)
         # 定义池化层，池化核的大小kernel_size为2，池化步长为2
         self.max_pool2 = MaxPool2D(kernel_size=2, stride=2)
         # 定义一层全连接层，输出维度是10
         self.fc = Linear(in_features=980, out_features=10)
         
   # 定义网络前向计算过程，卷积后紧接着使用池化层，最后使用全连接层计算最终输出
   # 卷积层激活函数使用Relu，全连接层激活函数使用softmax
     def forward(self, inputs):
         x = self.conv1(inputs)
         x = F.relu(x)
         x = self.max_pool1(x)
         x = self.conv2(x)
         x = F.relu(x)
         x = self.max_pool2(x)
         x = paddle.reshape(x, [x.shape[0], 980])
         x = self.fc(x)
         return x
```

# 单GPU训练

通过paddle.device.set_device API，设置在GPU上训练还是CPU上训练。

> paddle.device.set_device *(device)*

参数 
device (str)：此参数确定特定的运行设备，可以是`cpu`、 `gpu:x`或者是`xpu:x`。其中，`x`是GPU或XPU的编号。当device是`cpu`时， 程序在CPU上运行；当device是`gpu:x`时，程序在GPU上运行。


```python
#仅优化算法的设置有所差别
def train(model):
    #开启GPU
    use_gpu = True
    paddle.device.set_device('gpu:0') if use_gpu else paddle.device.set_device('cpu')
    model.train()
    #调用加载数据的函数
    train_loader = load_data('train')
    
    #设置不同初始学习率
    opt = paddle.optimizer.Adam(learning_rate=0.01, parameters=model.parameters())
    
    EPOCH_NUM = 5
    for epoch_id in range(EPOCH_NUM):
        for batch_id, data in enumerate(train_loader()):
            #准备数据，变得更加简洁
            images, labels = data
            images = paddle.to_tensor(images)
            labels = paddle.to_tensor(labels)
            
            #前向计算的过程
            predicts = model(images)
            
            #计算损失，取一个批次样本损失的平均值
            loss = F.cross_entropy(predicts, labels)
            avg_loss = paddle.mean(loss)
            
            #每训练了100批次的数据，打印下当前Loss的情况
            if batch_id % 200 == 0:
                print("epoch: {}, batch: {}, loss is: {}".format(epoch_id, batch_id, avg_loss.numpy()))
            
            #后向传播，更新参数的过程
            avg_loss.backward()
            opt.step()
            opt.clear_grad()

    #保存模型参数
    paddle.save(model.state_dict(), 'mnist.pdparams')
#创建模型    
model = MNIST()
#启动训练过程
train(model)
```


    ---------------------------------------------------------------------------
    
    ValueError                                Traceback (most recent call last)
    
    /tmp/ipykernel_165/4007146619.py in <module>
         40 model = MNIST()
         41 #启动训练过程
    ---> 42 train(model)


    /tmp/ipykernel_165/4007146619.py in train(model)
          3     #开启GPU
          4     use_gpu = True
    ----> 5     paddle.device.set_device('gpu:0') if use_gpu else paddle.device.set_device('cpu')
          6     model.train()
          7     #调用加载数据的函数


    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/device/__init__.py in set_device(device)
        311         data = paddle.stack([x1,x2], axis=1)
        312     """
    --> 313     place = _convert_to_place(device)
        314     framework._set_expected_place(place)
        315     return place


    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/device/__init__.py in _convert_to_place(device)
        254                 raise ValueError(
        255                     "The device should not be {}, since PaddlePaddle is "
    --> 256                     "not compiled with CUDA".format(avaliable_gpu_device))
        257             device_info_list = device.split(':', 1)
        258             device_id = device_info_list[1]


    ValueError: The device should not be <re.Match object; span=(0, 5), match='gpu:0'>, since PaddlePaddle is not compiled with CUDA


# 分布式训练

在工业实践中，很多较复杂的任务需要使用更强大的模型。强大模型加上海量的训练数据，经常导致模型训练耗时严重。比如在计算机视觉分类任务中，训练一个在ImageNet数据集上精度表现良好的模型，大概需要一周的时间，因为过程中我们需要不断尝试各种优化的思路和方案。如果每次训练均要耗时1周，这会大大降低模型迭代的速度。在机器资源充沛的情况下，建议采用分布式训练，大部分模型的训练时间可压缩到小时级别。

分布式训练有两种实现模式：模型并行和数据并行。


## 模型并行

模型并行是将一个网络模型拆分为多份，拆分后的模型分到多个设备上（GPU）训练，每个设备的训练数据是相同的。模型并行的实现模式可以节省内存，但是应用较为受限。

模型并行的方式一般适用于如下两个场景：

1. **模型架构过大：** 完整的模型无法放入单个GPU。如2012年ImageNet大赛的冠军模型AlexNet是模型并行的典型案例，由于当时GPU内存较小，单个GPU不足以承担AlexNet，因此研究者将AlexNet拆分为两部分放到两个GPU上并行训练。

2. **网络模型的结构设计相对独立：** 当网络模型的设计结构可以并行化时，采用模型并行的方式。如在计算机视觉目标检测任务中，一些模型（如YOLO9000）的边界框回归和类别预测是独立的，可以将独立的部分放到不同的设备节点上完成分布式训练。


## 数据并行

数据并行与模型并行不同，数据并行每次读取多份数据，读取到的数据输入给多个设备（GPU）上的模型，每个设备上的模型是完全相同的，飞桨采用的就是这种方式。

------

**说明：**

当前GPU硬件技术快速发展，深度学习使用的主流GPU的内存已经足以满足大多数的网络模型需求，所以大多数情况下使用数据并行的方式。

------

数据并行的方式与众人拾柴火焰高的道理类似，如果把训练数据比喻为砖头，把一个设备（GPU）比喻为一个人，那单GPU训练就是一个人在搬砖，多GPU训练就是多个人同时搬砖，每次搬砖的数量倍数增加，效率呈倍数提升。值得注意的是，每个设备的模型是完全相同的，但是输入数据不同，因此每个设备的模型计算出的梯度是不同的。如果每个设备的梯度只更新当前设备的模型，就会导致下次训练时，每个模型的参数都不相同。因此我们还需要一个梯度同步机制，保证每个设备的梯度是完全相同的。

梯度同步有两种方式：PRC通信方式和NCCL2通信方式（Nvidia Collective multi-GPU Communication Library）。

### PRC通信方式

PRC通信方式通常用于CPU分布式训练，它有两个节点：参数服务器Parameter server和训练节点Trainer，结构如 **图2** 所示。

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/560af46fd88140e8bc357dfad0d21e547e4703073c834c6c99c342b79e5076e4" width="400" hegiht="" ></center>
<center><br>图2：Pserver通信方式的结构</br></center>

<br></br>

parameter server收集来自每个设备的梯度更新信息，并计算出一个全局的梯度更新。Trainer用于训练，每个Trainer上的程序相同，但数据不同。当Parameter server收到来自Trainer的梯度更新请求时，统一更新模型的梯度。

### NCCL2通信方式（Collective）

当前飞桨的GPU分布式训练使用的是基于NCCL2的通信方式，结构如 **图3** 所示。

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/a27f1873e4934a0f8cda436b33830268ef4621cf6b994deb839db0a272e75de1" width="500" hegiht="" ></center>
<center><br>图3：NCCL2通信方式的结构</br></center>

<br></br>

相比PRC通信方式，使用NCCL2（Collective通信方式）进行分布式训练，不需要启动Parameter server进程，每个Trainer进程保存一份完整的模型参数，在完成梯度计算之后通过Trainer之间的相互通信，Reduce梯度数据到所有节点的所有设备，然后每个节点在各自完成参数更新。

飞桨提供了便利的数据并行训练方式，用户只需要对程序进行简单修改，即可实现在多GPU上并行训练。接下来将讲述如何将一个单机程序通过简单的改造，变成单机多卡程序。

单机多卡程序通过如下两步改动即可完成：

1. 初始化并行环境。
2. 使用paddle.DataParallel封装模型。

> 注意：由于我们的数据是通过手动构造批次的方式输入给模型的，没有针对多卡情况进行划分，因此每个卡上会基于全量数据迭代训练。可通过继承[paddle.io.Dataset](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/io/Dataset_cn.html#dataset)的方式准备自己的数据，再通过[DistributedBatchSampler](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/io/DistributedBatchSampler_cn.html)实现分布式批采样器加载数据的一个子集。这样，每个进程可以传递给DataLoader一个DistributedBatchSampler的实例，每个进程加载原始数据的一个子集。


```python
import paddle 
import paddle.distributed as dist

def train_multi_gpu(model):
    # 修改1- 初始化并行环境
    dist.init_parallel_env()
    # 修改2- 增加paddle.DataParallel封装
    model = paddle.DataParallel(model)
    model.train()
    #调用加载数据的函数
    train_loader = load_data('train')
    opt = paddle.optimizer.Adam(learning_rate=0.01, parameters=model.parameters())
    EPOCH_NUM = 5
    for epoch_id in range(EPOCH_NUM):
        for batch_id, data in enumerate(train_loader()):
            #准备数据，变得更加简洁
            images, labels = data
            images = paddle.to_tensor(images)
            labels = paddle.to_tensor(labels)
            #前向计算的过程
            predicts = model(images)
            #计算损失，取一个批次样本损失的平均值
            loss = F.cross_entropy(predicts, labels)
            avg_loss = paddle.mean(loss)
            #每训练了100批次的数据，打印下当前Loss的情况
            if batch_id % 200 == 0:
                print("epoch: {}, batch: {}, loss is: {}".format(epoch_id, batch_id, avg_loss.numpy()))
            #后向传播，更新参数的过程
            avg_loss.backward()
            opt.step()
            opt.clear_grad()

    #保存模型参数
    paddle.save(model.state_dict(), 'mnist.pdparams')

paddle.set_device('gpu')
#创建模型    
model = MNIST()
#启动训练过程
train_multi_gpu(model)

```

启动多GPU的训练，有两种方式：

1. 基于launch启动；
1. 基于spawn方式启动。

------

**说明：**

AI Studio当前仅支持单卡GPU，因此本案例需要在本地GPU上执行，无法在AI Studio上演示。

------

### 1. 基于launch方式启动

需要在命令行中设置参数变量。打开终端，运行如下命令： 

#### 单机单卡启动，默认使用第0号卡。

```
$ python train.py
```

#### 单机多卡启动，默认使用当前可见的所有卡。

```
$ python -m paddle.distributed.launch train.py
```

#### 单机多卡启动，设置当前使用的第0号和第1号卡。

```
$ python -m paddle.distributed.launch --gpus '0,1' --log_dir ./mylog train.py
```

```
$ export CUDA_VISIABLE_DEVICES='0,1'
$ python -m paddle.distributed.launch train.py
```

相关参数含义如下：

* paddle.distributed.launch：启动分布式运行。
* gpus：设置使用的GPU的序号（需要是多GPU卡的机器，通过命令watch nvidia-smi查看GPU的序号）。
* log_dir：存放训练的log，若不设置，每个GPU上的训练信息都会打印到屏幕。
* train.py：多GPU训练的程序，包含修改过的train_multi_gpu()函数。

训练完成后，在指定的./mylog文件夹下会产生四个日志文件，其中worklog.0的内容如下：

```
grep: warning: GREP_OPTIONS is deprecated; please use an alias or script
dev_id 0
I1104 06:25:04.377323 31961 nccl_context.cc:88] worker: 127.0.0.1:6171 is not ready, will retry after 3 seconds...
I1104 06:25:07.377645 31961 nccl_context.cc:127] init nccl context nranks: 3 local rank: 0 gpu id: 1↩
W1104 06:25:09.097079 31961 device_context.cc:235] Please NOTE: device: 1, CUDA Capability: 61, Driver API Version: 10.1, Runtime API Version: 9.0
W1104 06:25:09.104460 31961 device_context.cc:243] device: 1, cuDNN Version: 7.5.
start data reader (trainers_num: 3, trainer_id: 0)
epoch: 0, batch_id: 10, loss is: [0.47507238]
epoch: 0, batch_id: 20, loss is: [0.25089613]
epoch: 0, batch_id: 30, loss is: [0.13120805]
epoch: 0, batch_id: 40, loss is: [0.12122715]
epoch: 0, batch_id: 50, loss is: [0.07328521]
epoch: 0, batch_id: 60, loss is: [0.11860339]
epoch: 0, batch_id: 70, loss is: [0.08205047]
epoch: 0, batch_id: 80, loss is: [0.08192863]
epoch: 0, batch_id: 90, loss is: [0.0736289]
epoch: 0, batch_id: 100, loss is: [0.08607423]
start data reader (trainers_num: 3, trainer_id: 0)
epoch: 1, batch_id: 10, loss is: [0.07032011]
epoch: 1, batch_id: 20, loss is: [0.09687119]
epoch: 1, batch_id: 30, loss is: [0.0307216]
epoch: 1, batch_id: 40, loss is: [0.03884467]
epoch: 1, batch_id: 50, loss is: [0.02801813]
epoch: 1, batch_id: 60, loss is: [0.05751991]
epoch: 1, batch_id: 70, loss is: [0.03721186]
.....
```

### 2. 基于spawn方式启动

launch方式启动训练，是以文件为单位启动多进程，需要用户在启动时调用`paddle.distributed.launch`，对于进程的管理要求较高；飞桨最新版本中，增加了spawn启动方式，可以更好地控制进程，在日志打印、训练和退出时更加友好。spawn方式和launch方式仅在启动上有所区别。

```
# 启动train多进程训练，默认使用所有可见的GPU卡。
if __name__ == '__main__':
    dist.spawn(train)

# 启动train函数2个进程训练，默认使用当前可见的前2张卡。
if __name__ == '__main__':
    dist.spawn(train, nprocs=2)

# 启动train函数2个进程训练，默认使用第4号和第5号卡。
if __name__ == '__main__':
    dist.spawn(train, nprocs=2, selelcted_gpus='4,5')

```



***

# 8.【手写数字识别】之训练调试与优化

上一节我们研究了资源部署优化的方法，通过使用单GPU和分布式部署，提升模型训练的效率。本节我们依旧横向展开"横纵式"，如 **图1** 所示，探讨在手写数字识别任务中，为了保证模型的真实效果，在模型训练部分，对模型进行一些调试和优化的方法。

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/0f94e16e03f34223903319eb50c8c09687d203f0e718491bbc7c4fc1b2a88585" width="1000" hegiht="" ></center>
<center>图1：“横纵式”教学法 — 训练过程</center>

<br></br>

训练过程优化思路主要有如下五个关键环节：

**1. 计算分类准确率，观测模型训练效果。**

交叉熵损失函数只能作为优化目标，无法直接准确衡量模型的训练效果。准确率可以直接衡量训练效果，但由于其离散性质，不适合做为损失函数优化神经网络。
    
**2. 检查模型训练过程，识别潜在问题。**

如果模型的损失或者评估指标表现异常，通常需要打印模型每一层的输入和输出来定位问题，分析每一层的内容来获取错误的原因。
    
**3. 加入校验或测试，更好评价模型效果。**

理想的模型训练结果是在训练集和验证集上均有较高的准确率，如果训练集的准确率低于验证集，说明网络训练程度不够；如果训练集的准确率高于验证集，可能是发生了过拟合现象。通过在优化目标中加入正则化项的办法，解决过拟合的问题。
    
**4. 加入正则化项，避免模型过拟合。**

飞桨框架支持为整体参数加入正则化项，这是通常的做法。此外，飞桨框架也支持为某一层或某一部分的网络单独加入正则化项，以达到精细调整参数训练的效果。

**5. 可视化分析。**

用户不仅可以通过打印或使用matplotlib库作图，飞桨还提供了更专业的可视化分析工具VisualDL，提供便捷的可视化分析方法。

# 计算模型的分类准确率

准确率是一个直观衡量分类模型效果的指标，由于这个指标是离散的，因此不适合作为损失来优化。通常情况下，交叉熵损失越小的模型，分类的准确率也越高。基于分类准确率，我们可以公平地比较两种损失函数的优劣，例如在【手写数字识别】之损失函数章节中均方误差和交叉熵的比较。

使用飞桨提供的计算分类准确率API，可以直接计算准确率。

> *class* paddle.metric.Accuracy

该API的输入参数input为预测的分类结果predict，输入参数label为数据真实的label。飞桨还提供了更多衡量模型效果的计算指标，详细可以查看paddle.meric包下面的API。

在下述代码中，我们在模型前向计算过程forward函数中计算分类准确率，并在训练时打印每个批次样本的分类准确率。


```python
# 加载相关库
import os
import random
import paddle
import numpy as np
from PIL import Image
import gzip
import json


# 定义数据集读取器
def load_data(mode='train'):

    # 读取数据文件
    datafile = './work/mnist.json.gz'
    print('loading mnist dataset from {} ......'.format(datafile))
    data = json.load(gzip.open(datafile))
    # 读取数据集中的训练集，验证集和测试集
    train_set, val_set, eval_set = data

    # 数据集相关参数，图片高度IMG_ROWS, 图片宽度IMG_COLS
    IMG_ROWS = 28
    IMG_COLS = 28
    # 根据输入mode参数决定使用训练集，验证集还是测试
    if mode == 'train':
        imgs = train_set[0]
        labels = train_set[1]
    elif mode == 'valid':
        imgs = val_set[0]
        labels = val_set[1]
    elif mode == 'eval':
        imgs = eval_set[0]
        labels = eval_set[1]
    # 获得所有图像的数量
    imgs_length = len(imgs)
    # 验证图像数量和标签数量是否一致
    assert len(imgs) == len(labels), \
          "length of train_imgs({}) should be the same as train_labels({})".format(
                  len(imgs), len(labels))

    index_list = list(range(imgs_length))

    # 读入数据时用到的batchsize
    BATCHSIZE = 100

    # 定义数据生成器
    def data_generator():
        # 训练模式下，打乱训练数据
        if mode == 'train':
            random.shuffle(index_list)
        imgs_list = []
        labels_list = []
        # 按照索引读取数据
        for i in index_list:
            # 读取图像和标签，转换其尺寸和类型
            img = np.reshape(imgs[i], [1, IMG_ROWS, IMG_COLS]).astype('float32')
            label = np.reshape(labels[i], [1]).astype('int64')
            imgs_list.append(img) 
            labels_list.append(label)
            # 如果当前数据缓存达到了batch size，就返回一个批次数据
            if len(imgs_list) == BATCHSIZE:
                yield np.array(imgs_list), np.array(labels_list)
                # 清空数据缓存列表
                imgs_list = []
                labels_list = []

        # 如果剩余数据的数目小于BATCHSIZE，
        # 则剩余数据一起构成一个大小为len(imgs_list)的mini-batch
        if len(imgs_list) > 0:
            yield np.array(imgs_list), np.array(labels_list)

    return data_generator

```


```python
# 定义模型结构
import paddle.nn.functional as F
from paddle.nn import Conv2D, MaxPool2D, Linear

# 多层卷积神经网络实现
class MNIST(paddle.nn.Layer):
     def __init__(self):
         super(MNIST, self).__init__()
         
         # 定义卷积层，输出特征通道out_channels设置为20，卷积核的大小kernel_size为5，卷积步长stride=1，padding=2
         self.conv1 = Conv2D(in_channels=1, out_channels=20, kernel_size=5, stride=1, padding=2)
         # 定义池化层，池化核的大小kernel_size为2，池化步长为2
         self.max_pool1 = MaxPool2D(kernel_size=2, stride=2)
         # 定义卷积层，输出特征通道out_channels设置为20，卷积核的大小kernel_size为5，卷积步长stride=1，padding=2
         self.conv2 = Conv2D(in_channels=20, out_channels=20, kernel_size=5, stride=1, padding=2)
         # 定义池化层，池化核的大小kernel_size为2，池化步长为2
         self.max_pool2 = MaxPool2D(kernel_size=2, stride=2)
         # 定义一层全连接层，输出维度是10
         self.fc = Linear(in_features=980, out_features=10)
         
   # 定义网络前向计算过程，卷积后紧接着使用池化层，最后使用全连接层计算最终输出
   # 卷积层激活函数使用Relu，全连接层激活函数使用softmax
     def forward(self, inputs, label):
         x = self.conv1(inputs)
         x = F.relu(x)
         x = self.max_pool1(x)
         x = self.conv2(x)
         x = F.relu(x)
         x = self.max_pool2(x)
         x = paddle.reshape(x, [x.shape[0], 980])
         x = self.fc(x)
         if label is not None:
             acc = paddle.metric.accuracy(input=x, label=label)
             return x, acc
         else:
             return x

#调用加载数据的函数
train_loader = load_data('train')
    
#在使用GPU机器时，可以将use_gpu变量设置成True
use_gpu = True
paddle.set_device('gpu:0') if use_gpu else paddle.set_device('cpu')

#仅优化算法的设置有所差别
def train(model):
    model = MNIST()
    model.train()
    
    #四种优化算法的设置方案，可以逐一尝试效果
    # opt = paddle.optimizer.SGD(learning_rate=0.01, parameters=model.parameters())
    # opt = paddle.optimizer.Momentum(learning_rate=0.01, momentum=0.9, parameters=model.parameters())
    # opt = paddle.optimizer.Adagrad(learning_rate=0.01, parameters=model.parameters())
    opt = paddle.optimizer.Adam(learning_rate=0.01, parameters=model.parameters())
    
    EPOCH_NUM = 5
    for epoch_id in range(EPOCH_NUM):
        for batch_id, data in enumerate(train_loader()):
            #准备数据
            images, labels = data
            images = paddle.to_tensor(images)
            labels = paddle.to_tensor(labels)
            
            #前向计算的过程
            predicts, acc = model(images, labels)
            
            #计算损失，取一个批次样本损失的平均值
            loss = F.cross_entropy(predicts, labels)
            avg_loss = paddle.mean(loss)
            
            #每训练了100批次的数据，打印下当前Loss的情况
            if batch_id % 200 == 0:
                print("epoch: {}, batch: {}, loss is: {}, acc is {}".format(epoch_id, batch_id, avg_loss.numpy(), acc.numpy()))
                
            #后向传播，更新参数，消除梯度的过程
            avg_loss.backward()
            opt.step()
            opt.clear_grad()

    #保存模型参数
    paddle.save(model.state_dict(), 'mnist.pdparams')
    
#创建模型    
model = MNIST()
#启动训练过程
train(model)
```

    loading mnist dataset from ./work/mnist.json.gz ......


# 检查模型训练过程，识别潜在训练问题

使用飞桨动态图编程可以方便的查看和调试训练的执行过程。在网络定义的Forward函数中，可以打印每一层输入输出的尺寸，以及每层网络的参数。通过查看这些信息，不仅可以更好地理解训练的执行过程，还可以发现潜在问题，或者启发继续优化的思路。

在下述程序中，使用``check_shape``变量控制是否打印“尺寸”，验证网络结构是否正确。使用``check_content``变量控制是否打印“内容值”，验证数据分布是否合理。假如在训练中发现中间层的部分输出持续为0，说明该部分的网络结构设计存在问题，没有充分利用。


```python
import paddle.nn.functional as F
# 定义模型结构
class MNIST(paddle.nn.Layer):
     def __init__(self):
         super(MNIST, self).__init__()
         
         # 定义卷积层，输出特征通道out_channels设置为20，卷积核的大小kernel_size为5，卷积步长stride=1，padding=2
         self.conv1 = Conv2D(in_channels=1, out_channels=20, kernel_size=5, stride=1, padding=2)
         # 定义池化层，池化核的大小kernel_size为2，池化步长为2
         self.max_pool1 = MaxPool2D(kernel_size=2, stride=2)
         # 定义卷积层，输出特征通道out_channels设置为20，卷积核的大小kernel_size为5，卷积步长stride=1，padding=2
         self.conv2 = Conv2D(in_channels=20, out_channels=20, kernel_size=5, stride=1, padding=2)
         # 定义池化层，池化核的大小kernel_size为2，池化步长为2
         self.max_pool2 = MaxPool2D(kernel_size=2, stride=2)
         # 定义一层全连接层，输出维度是10
         self.fc = Linear(in_features=980, out_features=10)
     
     #加入对每一层输入和输出的尺寸和数据内容的打印，根据check参数决策是否打印每层的参数和输出尺寸
     # 卷积层激活函数使用Relu，全连接层激活函数使用softmax
     def forward(self, inputs, label=None, check_shape=False, check_content=False):
         # 给不同层的输出不同命名，方便调试
         outputs1 = self.conv1(inputs)
         outputs2 = F.relu(outputs1)
         outputs3 = self.max_pool1(outputs2)
         outputs4 = self.conv2(outputs3)
         outputs5 = F.relu(outputs4)
         outputs6 = self.max_pool2(outputs5)
         outputs6 = paddle.reshape(outputs6, [outputs6.shape[0], -1])
         outputs7 = self.fc(outputs6)
         
         # 选择是否打印神经网络每层的参数尺寸和输出尺寸，验证网络结构是否设置正确
         if check_shape:
             # 打印每层网络设置的超参数-卷积核尺寸，卷积步长，卷积padding，池化核尺寸
             print("\n########## print network layer's superparams ##############")
             print("conv1-- kernel_size:{}, padding:{}, stride:{}".format(self.conv1.weight.shape, self.conv1._padding, self.conv1._stride))
             print("conv2-- kernel_size:{}, padding:{}, stride:{}".format(self.conv2.weight.shape, self.conv2._padding, self.conv2._stride))
             #print("max_pool1-- kernel_size:{}, padding:{}, stride:{}".format(self.max_pool1.pool_size, self.max_pool1.pool_stride, self.max_pool1._stride))
             #print("max_pool2-- kernel_size:{}, padding:{}, stride:{}".format(self.max_pool2.weight.shape, self.max_pool2._padding, self.max_pool2._stride))
             print("fc-- weight_size:{}, bias_size_{}".format(self.fc.weight.shape, self.fc.bias.shape))
             
             # 打印每层的输出尺寸
             print("\n########## print shape of features of every layer ###############")
             print("inputs_shape: {}".format(inputs.shape))
             print("outputs1_shape: {}".format(outputs1.shape))
             print("outputs2_shape: {}".format(outputs2.shape))
             print("outputs3_shape: {}".format(outputs3.shape))
             print("outputs4_shape: {}".format(outputs4.shape))
             print("outputs5_shape: {}".format(outputs5.shape))
             print("outputs6_shape: {}".format(outputs6.shape))
             print("outputs7_shape: {}".format(outputs7.shape))
             # print("outputs8_shape: {}".format(outputs8.shape))
             
         # 选择是否打印训练过程中的参数和输出内容，可用于训练过程中的调试
         if check_content:
            # 打印卷积层的参数-卷积核权重，权重参数较多，此处只打印部分参数
             print("\n########## print convolution layer's kernel ###############")
             print("conv1 params -- kernel weights:", self.conv1.weight[0][0])
             print("conv2 params -- kernel weights:", self.conv2.weight[0][0])

             # 创建随机数，随机打印某一个通道的输出值
             idx1 = np.random.randint(0, outputs1.shape[1])
             idx2 = np.random.randint(0, outputs4.shape[1])
             # 打印卷积-池化后的结果，仅打印batch中第一个图像对应的特征
             print("\nThe {}th channel of conv1 layer: ".format(idx1), outputs1[0][idx1])
             print("The {}th channel of conv2 layer: ".format(idx2), outputs4[0][idx2])
             print("The output of last layer:", outputs7[0], '\n')
            
        # 如果label不是None，则计算分类精度并返回
         if label is not None:
             acc = paddle.metric.accuracy(input=F.softmax(outputs7), label=label)
             return outputs7, acc
         else:
             return outputs7

#在使用GPU机器时，可以将use_gpu变量设置成True
use_gpu = True
paddle.set_device('gpu:0') if use_gpu else paddle.set_device('cpu')    

def train(model):
    model = MNIST()
    model.train()
    
    #四种优化算法的设置方案，可以逐一尝试效果
    opt = paddle.optimizer.SGD(learning_rate=0.01, parameters=model.parameters())
    # opt = paddle.optimizer.Momentum(learning_rate=0.01, momentum=0.9, parameters=model.parameters())
    # opt = paddle.optimizer.Adagrad(learning_rate=0.01, parameters=model.parameters())
    # opt = paddle.optimizer.Adam(learning_rate=0.01, parameters=model.parameters())
    
    EPOCH_NUM = 1
    for epoch_id in range(EPOCH_NUM):
        for batch_id, data in enumerate(train_loader()):
            #准备数据，变得更加简洁
            images, labels = data
            images = paddle.to_tensor(images)
            labels = paddle.to_tensor(labels)
            
            #前向计算的过程，同时拿到模型输出值和分类准确率
            if batch_id == 0 and epoch_id==0:
                # 打印模型参数和每层输出的尺寸
                predicts, acc = model(images, labels, check_shape=True, check_content=False)
            elif batch_id==401:
                # 打印模型参数和每层输出的值
                predicts, acc = model(images, labels, check_shape=False, check_content=True)
            else:
                predicts, acc = model(images, labels)
            
            #计算损失，取一个批次样本损失的平均值
            loss = F.cross_entropy(predicts, labels)
            avg_loss = paddle.mean(loss)
            
            #每训练了100批次的数据，打印下当前Loss的情况
            if batch_id % 200 == 0:
                print("epoch: {}, batch: {}, loss is: {}, acc is {}".format(epoch_id, batch_id, avg_loss.numpy(), acc.numpy()))
            
            #后向传播，更新参数的过程
            avg_loss.backward()
            opt.step()
            opt.clear_grad()

    #保存模型参数
    paddle.save(model.state_dict(), 'mnist_test.pdparams')
    
#创建模型    
model = MNIST()
#启动训练过程
train(model)

print("Model has been saved.")
```

# 加入校验或测试，更好评价模型效果 

在训练过程中，我们会发现模型在训练样本集上的损失在不断减小。但这是否代表模型在未来的应用场景上依然有效？为了验证模型的有效性，通常将样本集合分成三份，训练集、校验集和测试集。

- **训练集** ：用于训练模型的参数，即训练过程中主要完成的工作。
- **校验集** ：用于对模型超参数的选择，比如网络结构的调整、正则化项权重的选择等。
- **测试集** ：用于模拟模型在应用后的真实效果。因为测试集没有参与任何模型优化或参数训练的工作，所以它对模型来说是完全未知的样本。在不以校验数据优化网络结构或模型超参数时，校验数据和测试数据的效果是类似的，均更真实的反映模型效果。

如下程序读取上一步训练保存的模型参数，读取校验数据集，并测试模型在校验数据集上的效果。


```python
def evaluation(model):
    print('start evaluation .......')
    # 定义预测过程
    params_file_path = 'mnist.pdparams'
    # 加载模型参数
    param_dict = paddle.load(params_file_path)
    model.load_dict(param_dict)

    model.eval()
    eval_loader = load_data('eval')

    acc_set = []
    avg_loss_set = []
    for batch_id, data in enumerate(eval_loader()):
        images, labels = data
        images = paddle.to_tensor(images)
        labels = paddle.to_tensor(labels)
        predicts, acc = model(images, labels)
        loss = F.cross_entropy(input=predicts, label=labels)
        avg_loss = paddle.mean(loss)
        acc_set.append(float(acc.numpy()))
        avg_loss_set.append(float(avg_loss.numpy()))
    
    #计算多个batch的平均损失和准确率
    acc_val_mean = np.array(acc_set).mean()
    avg_loss_val_mean = np.array(avg_loss_set).mean()

    print('loss={}, acc={}'.format(avg_loss_val_mean, acc_val_mean))

model = MNIST()
evaluation(model)
```

从测试的效果来看，模型在验证集上依然有98.6%的准确率，证明它是有预测效果的。

# 加入正则化项，避免模型过拟合

## 过拟合现象

对于样本量有限、但需要使用强大模型的复杂任务，模型很容易出现过拟合的表现，即在训练集上的损失小，在验证集或测试集上的损失较大，如 **图2** 所示。

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/99b879c21113494a9d7315eeda74bc4c8fea07f984824a03bf8411e946c75f1b" width="400" hegiht="" ></center>
<center><br>图2：过拟合现象，训练误差不断降低，但测试误差先降后增</br></center>

<br></br>

反之，如果模型在训练集和测试集上均损失较大，则称为欠拟合。过拟合表示模型过于敏感，学习到了训练数据中的一些误差，而这些误差并不是真实的泛化规律（可推广到测试集上的规律）。欠拟合表示模型还不够强大，还没有很好的拟合已知的训练样本，更别提测试样本了。因为欠拟合情况容易观察和解决，只要训练loss不够好，就不断使用更强大的模型即可，因此实际中我们更需要处理好过拟合的问题。

## 导致过拟合原因

造成过拟合的原因是模型过于敏感，而训练数据量太少或其中的噪音太多。

如**图3** 所示，理想的回归模型是一条坡度较缓的抛物线，欠拟合的模型只拟合出一条直线，显然没有捕捉到真实的规律，但过拟合的模型拟合出存在很多拐点的抛物线，显然是过于敏感，也没有正确表达真实规律。

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/53c389bb3c824706bd2fbc05f83ab0c6dd6b5b2fdedb4150a17e16a1b64c243e" width="700" hegiht="" ></center>
<center><br>图3：回归模型的过拟合，理想和欠拟合状态的表现</br></center>

<br></br>

如**图4** 所示，理想的分类模型是一条半圆形的曲线，欠拟合用直线作为分类边界，显然没有捕捉到真实的边界，但过拟合的模型拟合出很扭曲的分类边界，虽然对所有的训练数据正确分类，但对一些较为个例的样本所做出的妥协，高概率不是真实的规律。

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/b5a46f7e0fbe4f8686a71d9a2d330ed09f23bca565a44e0d941148729fd2f7d7" width="700" hegiht="" ></center>
<center><br>图4：分类模型的欠拟合，理想和过拟合状态的表现</br></center>

<br></br>


## 过拟合的成因与防控

为了更好的理解过拟合的成因，可以参考侦探定位罪犯的案例逻辑，如 **图5** 所示。

<center><img src="https://ai-studio-static-online.cdn.bcebos.com/34de60a675b64468a2c3fee0844a168d53e891eaacf643fd8c1c9ba8e3812bcc" width="700" hegiht="" ></center>
<center><br>图5：侦探定位罪犯与模型假设示意</br></center>

<br></br>

**对于这个案例，假设侦探也会犯错，通过分析发现可能的原因：**

1. 情况1：罪犯证据存在错误，依据错误的证据寻找罪犯肯定是缘木求鱼。

2. 情况2：搜索范围太大的同时证据太少，导致符合条件的候选（嫌疑人）太多，无法准确定位罪犯。

**那么侦探解决这个问题的方法有两种：或者缩小搜索范围（比如假设该案件只能是熟人作案），或者寻找更多的证据。**

**归结到深度学习中，假设模型也会犯错，通过分析发现可能的原因：**

1. 情况1：训练数据存在噪音，导致模型学到了噪音，而不是真实规律。

2. 情况2：使用强大模型（表示空间大）的同时训练数据太少，导致在训练数据上表现良好的候选假设太多，锁定了一个“虚假正确”的假设。

**对于情况1，我们使用数据清洗和修正来解决。 对于情况2，我们或者限制模型表示能力，或者收集更多的训练数据。**

而清洗训练数据中的错误，或收集更多的训练数据往往是一句“正确的废话”，在任何时候我们都想获得更多更高质量的数据。在实际项目中，更快、更低成本可控制过拟合的方法，只有限制模型的表示能力。

## 正则化项

为了防止模型过拟合，在没有扩充样本量的可能下，只能降低模型的复杂度，可以通过限制参数的数量或可能取值（参数值尽量小）实现。

具体来说，在模型的优化目标（损失）中人为加入对参数规模的惩罚项。当参数越多或取值越大时，该惩罚项就越大。通过调整惩罚项的权重系数，可以使模型在“尽量减少训练损失”和“保持模型的泛化能力”之间取得平衡。泛化能力表示模型在没有见过的样本上依然有效。正则化项的存在，增加了模型在训练集上的损失。

飞桨支持为所有参数加上统一的正则化项，也支持为特定的参数添加正则化项。前者的实现如下代码所示，仅在优化器中设置``weight_decay``参数即可实现。使用参数``coeff``调节正则化项的权重，权重越大时，对模型复杂度的惩罚越高。


```python
def train(model):
    model.train() 

    #各种优化算法均可以加入正则化项，避免过拟合，参数regularization_coeff调节正则化项的权重
    opt = paddle.optimizer.Adam(learning_rate=0.01, weight_decay=paddle.regularizer.L2Decay(coeff=1e-5), parameters=model.parameters())           

    EPOCH_NUM = 5
    for epoch_id in range(EPOCH_NUM):
        for batch_id, data in enumerate(train_loader()):
            #准备数据，变得更加简洁
            images, labels = data
            images = paddle.to_tensor(images)
            labels = paddle.to_tensor(labels)
            
            #前向计算的过程，同时拿到模型输出值和分类准确率
            predicts, acc = model(images, labels)
            #计算损失，取一个批次样本损失的平均值
            loss = F.cross_entropy(predicts, labels)
            avg_loss = paddle.mean(loss)
            
            #每训练了100批次的数据，打印下当前Loss的情况
            if batch_id % 200 == 0:
                print("epoch: {}, batch: {}, loss is: {}, acc is {}".format(epoch_id, batch_id, avg_loss.numpy(), acc.numpy()))
            
            #后向传播，更新参数的过程
            avg_loss.backward()
            opt.step()
            opt.clear_grad()

    #保存模型参数
    paddle.save(model.state_dict(), 'mnist_regul.pdparams')

model = MNIST()
train(model)
```


```python
def evaluation(model):
    print('start evaluation .......')
    # 定义预测过程
    params_file_path = 'mnist_regul.pdparams'
    # 加载模型参数
    param_dict = paddle.load(params_file_path)
    model.load_dict(param_dict)

    model.eval()
    eval_loader = load_data('eval')

    acc_set = []
    avg_loss_set = []
    for batch_id, data in enumerate(eval_loader()):
        images, labels = data
        images = paddle.to_tensor(images)
        labels = paddle.to_tensor(labels)
        predicts, acc = model(images, labels)
        loss = F.cross_entropy(input=predicts, label=labels)
        avg_loss = paddle.mean(loss)
        acc_set.append(float(acc.numpy()))
        avg_loss_set.append(float(avg_loss.numpy()))
    
    #计算多个batch的平均损失和准确率
    acc_val_mean = np.array(acc_set).mean()
    avg_loss_val_mean = np.array(avg_loss_set).mean()

    print('loss={}, acc={}'.format(avg_loss_val_mean, acc_val_mean))

model = MNIST()
evaluation(model)
```

# 可视化分析

训练模型时，经常需要观察模型的评价指标，分析模型的优化过程，以确保训练是有效的。可选用这两种工具：Matplotlib库和VisualDL。

* **Matplotlib库**：Matplotlib库是Python中使用的最多的2D图形绘图库，它有一套完全仿照MATLAB的函数形式的绘图接口，使用轻量级的PLT库（Matplotlib）作图是非常简单的。
* **VisualDL**：如果期望使用更加专业的作图工具，可以尝试VisualDL，飞桨可视化分析工具。VisualDL能够有效地展示飞桨在运行过程中的计算图、各种指标变化趋势和数据信息。

## 使用Matplotlib库绘制损失随训练下降的曲线图

将训练的批次编号作为X轴坐标，该批次的训练损失作为Y轴坐标。

1. 训练开始前，声明两个列表变量存储对应的批次编号(iters=[])和训练损失(losses=[])。

```
iters=[]
losses=[]
for epoch_id in range(EPOCH_NUM):
	"""start to training"""
```

2. 随着训练的进行，将iter和losses两个列表填满。

```python
import paddle.nn.functional as F

iters=[]
losses=[]
for epoch_id in range(EPOCH_NUM):
	for batch_id, data in enumerate(train_loader()):
        images, labels = data
        predicts, acc = model(images, labels)
        loss = F.cross_entropy(predicts, label = labels.astype('int64'))
        avg_loss = paddle.mean(loss)
        # 累计迭代次数和对应的loss
   	iters.append(batch_id + epoch_id*len(list(train_loader()))
	losses.append(avg_loss)
```

3. 训练结束后，将两份数据以参数形式导入PLT的横纵坐标。

```
plt.xlabel("iter", fontsize=14)，plt.ylabel("loss", fontsize=14)
```

4. 最后，调用plt.plot()函数即可完成作图。

```
plt.plot(iters, losses,color='red',label='train loss') 
```

详细代码如下：


```python
#引入matplotlib库
import matplotlib.pyplot as plt

def train(model):
    model.train()
    
    opt = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
    
    EPOCH_NUM = 10
    iter=0
    iters=[]
    losses=[]
    for epoch_id in range(EPOCH_NUM):
        for batch_id, data in enumerate(train_loader()):
            #准备数据，变得更加简洁
            images, labels = data
            images = paddle.to_tensor(images)
            labels = paddle.to_tensor(labels)
            
            #前向计算的过程，同时拿到模型输出值和分类准确率
            predicts, acc = model(images, labels)
            #计算损失，取一个批次样本损失的平均值
            loss = F.cross_entropy(predicts, labels)
            avg_loss = paddle.mean(loss)
            
            #每训练了100批次的数据，打印下当前Loss的情况
            if batch_id % 100 == 0:
                print("epoch: {}, batch: {}, loss is: {}, acc is {}".format(epoch_id, batch_id, avg_loss.numpy(), acc.numpy()))
                iters.append(iter)
                losses.append(avg_loss.numpy())
                iter = iter + 100
            
            #后向传播，更新参数的过程
            avg_loss.backward()
            opt.step()
            opt.clear_grad()
            
    #保存模型参数
    paddle.save(model.state_dict(), 'mnist.pdparams')
    
    return iters, losses
    
model = MNIST()
iters, losses = train(model)
```


```python
#画出训练过程中Loss的变化曲线
plt.figure()
plt.title("train loss", fontsize=24)
plt.xlabel("iter", fontsize=14)
plt.ylabel("loss", fontsize=14)
plt.plot(iters, losses,color='red',label='train loss') 
plt.grid()
plt.show()
```

## 使用VisualDL可视化分析

VisualDL是飞桨可视化分析工具，以丰富的图表呈现训练参数变化趋势、模型结构、数据样本、高维数据分布等。帮助用户清晰直观地理解深度学习模型训练过程及模型结构，进而实现高效的模型调优，具体代码实现如下。

* 步骤1：引入VisualDL库，定义作图数据存储位置（供第3步使用），本案例的路径是“log”。

```
from visualdl import LogWriter
log_writer = LogWriter("./log")
```

* 步骤2：在训练过程中插入作图语句。当每100个batch训练完成后，将当前损失作为一个新增的数据点(iter和acc的映射对)存储到第一步设置的文件中。使用变量iter记录下已经训练的批次数，作为作图的X轴坐标。

```
log_writer.add_scalar(tag = 'acc', step = iter, value = avg_acc.numpy())
log_writer.add_scalar(tag = 'loss', step = iter, value = avg_loss.numpy())
iter = iter + 100
```


```python
# 安装VisualDL
!pip install --upgrade --pre visualdl
```


```python
#引入VisualDL库，并设定保存作图数据的文件位置
from visualdl import LogWriter
log_writer = LogWriter(logdir="./log")

def train(model):
    model.train()
    
    opt = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
    
    EPOCH_NUM = 10
    iter = 0
    for epoch_id in range(EPOCH_NUM):
        for batch_id, data in enumerate(train_loader()):
            #准备数据，变得更加简洁
            images, labels = data
            images = paddle.to_tensor(images)
            labels = paddle.to_tensor(labels)
            
            #前向计算的过程，同时拿到模型输出值和分类准确率
            predicts, avg_acc = model(images, labels)
            #计算损失，取一个批次样本损失的平均值
            loss = F.cross_entropy(predicts, labels)
            avg_loss = paddle.mean(loss)
            
            #每训练了100批次的数据，打印下当前Loss的情况
            if batch_id % 100 == 0:
                print("epoch: {}, batch: {}, loss is: {}, acc is {}".format(epoch_id, batch_id, avg_loss.numpy(), avg_acc.numpy()))
                log_writer.add_scalar(tag = 'acc', step = iter, value = avg_acc.numpy())
                log_writer.add_scalar(tag = 'loss', step = iter, value = avg_loss.numpy())
                iter = iter + 100

            #后向传播，更新参数的过程
            avg_loss.backward()
            opt.step()
            opt.clear_grad()

    #保存模型参数
    paddle.save(model.state_dict(), 'mnist.pdparams')
    
model = MNIST()
train(model)
```

* 步骤3：命令行启动VisualDL。

使用“visualdl --logdir [数据文件所在文件夹路径] 的命令启动VisualDL。在VisualDL启动后，命令行会打印出可用浏览器查阅图形结果的网址。

``` 
$ visualdl --logdir ./log --port 8080
```

* 步骤4：打开浏览器，查看作图结果，如 **图6** 所示。

查阅的网址在第三步的启动命令后会打印出来（如http://127.0.0.1:8080/），将该网址输入浏览器地址栏刷新页面的效果如下图所示。除了右侧对数据点的作图外，左侧还有一个控制板，可以调整诸多作图的细节。


<center><img src="https://ai-studio-static-online.cdn.bcebos.com/c29fdeed4f404eeb846c8c864497f939b5920c3e6fea4f60a9e4ebe4503b6a6b" width="600" hegiht="" ></center>
<center><br>图6：VisualDL作图示例</br></center>

<br></br>

## 作业题 2-4

* 将普通神经网络模型的每层输出打印，观察内容。
* 将分类准确率的指标 用PLT库画图表示。
* 通过分类准确率，判断以采用不同损失函数训练模型的效果优劣。
* 作图比较：随着训练进行，模型在训练集和测试集上的Loss曲线。
* 调节正则化权重，观察4的作图曲线的变化，并分析原因。

# 9.【手写数字识别】之恢复训练

# 模型加载及恢复训练

在快速入门中，我们已经介绍了将训练好的模型保存到磁盘文件的方法。应用程序可以随时加载模型，完成预测任务。但是在日常训练工作中我们会遇到一些突发情况，导致训练过程主动或被动的中断。如果训练一个模型需要花费几天的训练时间，中断后从初始状态重新训练是不可接受的。

万幸的是，飞桨支持从上一次保存状态开始训练，只要我们随时保存训练过程中的模型状态，就不用从初始状态重新训练。

下面介绍恢复训练的实现方法，依然使用手写数字识别的案例，网络定义的部分保持不变。


```python
#数据处理部分之前的代码，保持不变
import os
import random
import paddle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import gzip
import json
import warnings 
warnings.filterwarnings('ignore')

import paddle.nn as nn
from paddle.nn import Conv2D, MaxPool2D, Linear
import paddle.nn.functional as F

# 创建一个类MnistDataset，继承paddle.io.Dataset 这个类
# MnistDataset的作用和上面load_data()函数的作用相同，均是构建一个迭代器
class MnistDataset(paddle.io.Dataset):
    def __init__(self, mode):
        datafile = './work/mnist.json.gz'
        data = json.load(gzip.open(datafile))
        # 读取到的数据区分训练集，验证集，测试集
        train_set, val_set, eval_set = data
        
        # 数据集相关参数，图片高度IMG_ROWS, 图片宽度IMG_COLS
        self.IMG_ROWS = 28
        self.IMG_COLS = 28

        if mode=='train':
            # 获得训练数据集
            imgs, labels = train_set[0], train_set[1]
        elif mode=='valid':
            # 获得验证数据集
            imgs, labels = val_set[0], val_set[1]
        elif mode=='eval':
            # 获得测试数据集
            imgs, labels = eval_set[0], eval_set[1]
        else:
            raise Exception("mode can only be one of ['train', 'valid', 'eval']")
        
        # 校验数据
        imgs_length = len(imgs)
        assert len(imgs) == len(labels), \
            "length of train_imgs({}) should be the same as train_labels({})".format(len(imgs), len(labels))
        
        self.imgs = imgs
        self.labels = labels

    def __getitem__(self, idx):
        # MLP
        # img = np.array(self.imgs[idx]).astype('float32')
        # label = np.array(self.labels[idx]).astype('int64')
        # CNN
        img = np.reshape(self.imgs[idx], [1, self.IMG_ROWS, self.IMG_COLS]).astype('float32')
        label = np.reshape(self.labels[idx], [1]).astype('int64')

        return img, label

    def __len__(self):
        return len(self.imgs)

# 定义模型结构
class MNIST(paddle.nn.Layer):
     def __init__(self):
         super(MNIST, self).__init__()
         nn.initializer.set_global_initializer(nn.initializer.Uniform(), nn.initializer.Constant())
         # 定义卷积层，输出特征通道out_channels设置为20，卷积核的大小kernel_size为5，卷积步长stride=1，padding=2
         self.conv1 = Conv2D(in_channels=1, out_channels=20, kernel_size=5, stride=1, padding=2)
         # 定义池化层，池化层卷积核kernel_size为2，池化步长为2
         self.max_pool1 = MaxPool2D(kernel_size=2, stride=2)
         # 定义卷积层，输出特征通道out_channels设置为20，卷积核的大小kernel_size为5，卷积步长stride=1，padding=2
         self.conv2 = Conv2D(in_channels=20, out_channels=20, kernel_size=5, stride=1, padding=2)
         # 定义池化层，池化层卷积核kernel_size为2，池化步长为2
         self.max_pool2 = MaxPool2D(kernel_size=2, stride=2)
         # 定义一层全连接层，输出维度是10
         self.fc = Linear(in_features=980, out_features=10)
         
    # 定义网络前向计算过程，卷积后紧接着使用池化层，最后使用全连接层计算最终输出
     def forward(self, inputs):
         x = self.conv1(inputs)
         x = F.relu(x)
         x = self.max_pool1(x)
         x = self.conv2(x)
         x = F.relu(x)
         x = self.max_pool2(x)
         x = paddle.reshape(x, [x.shape[0], 980])
         x = self.fc(x)
        #  x = F.softmax(x)
         return x         
```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/__init__.py:107: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      from collections import MutableMapping
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/rcsetup.py:20: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      from collections import Iterable, Mapping
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/colors.py:53: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      from collections import Sized


定义训练Trainer，包含训练过程和模型保存


```python
class Trainer(object):
    def __init__(self, model_path, model, optimizer):
        self.model_path = model_path   # 模型存放路径 
        self.model = model             # 定义的模型
        self.optimizer = optimizer     # 优化器

    def save(self):
        # 保存模型
        paddle.save(self.model.state_dict(), self.model_path)

    def train_step(self, data):
        images, labels = data
        # 前向计算的过程
        predicts = self.model(images)
        # 计算损失
        loss = F.cross_entropy(predicts, labels)
        avg_loss = paddle.mean(loss)
        # 后向传播，更新参数的过程
        avg_loss.backward()
        self.optimizer.step()
        self.optimizer.clear_grad()
        return avg_loss

    def train_epoch(self, datasets, epoch):
        self.model.train()
        for batch_id, data in enumerate(datasets()):
            loss = self.train_step(data)
            # 每训练了1000批次的数据，打印下当前Loss的情况
            if batch_id % 500 == 0:
                print("epoch_id: {}, batch_id: {}, loss is: {}".format(epoch, batch_id, loss.numpy()))

    def train(self, train_datasets, start_epoch, end_epoch, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for i in range(start_epoch, end_epoch):
            self.train_epoch(train_datasets, i)
            paddle.save(opt.state_dict(), './{}/mnist_epoch{}'.format(save_path,i)+'.pdopt')
            paddle.save(model.state_dict(), './{}/mnist_epoch{}'.format(save_path,i)+'.pdparams')
        self.save()
```

在开始介绍使用飞桨恢复训练前，先正常训练一个模型，优化器使用Adam，使用动态变化的学习率，学习率从0.01衰减到0.001。每训练一轮后保存一次模型，之后将采用其中某一轮的模型参数进行恢复训练，验证一次性训练和中断再恢复训练的模型表现是否相差不多（训练loss的变化）。

注意进行恢复训练的程序不仅要保存模型参数，还要保存优化器参数。这是因为某些优化器含有一些随着训练过程变换的参数，例如Adam、Adagrad等优化器采用可变学习率的策略，随着训练进行会逐渐减少学习率。这些优化器的参数对于恢复训练至关重要。

为了演示这个特性，训练程序使用PolynomialDecay API、Adam优化器、学习率以多项式曲线从0.01衰减到0.001。

> *class* paddle.optimizer.lr.PolynomialDecay *(learningrate, decaysteps, endlr=0.0001, power=1.0, cycle=False, lastepoch=- 1, verbose=False)*

参数说明如下：

* learning_rate (float)：初始学习率，数据类型为Python float。
* decay_steps (int)：进行衰减的步长，这个决定了衰减周期。
* end_lr (float，可选）：最小的最终学习率，默认值为0.0001。
* power (float，可选)：多项式的幂，默认值为1.0。
* last_epoch (int，可选)：上一轮的轮数，重启训练时设置为上一轮的epoch数。默认值为-1，则为初始学习率。
* verbose (bool，可选)：如果是 True，则在每一轮更新时在标准输出stdout输出一条信息，默认值为False。
* cycle (bool，可选)：学习率下降后是否重新上升。若为True，则学习率衰减到最低学习率值时会重新上升。若为False，则学习率单调递减。默认值为False。

PolynomialDecay的变化曲线下图所示：

<img src="photo/4877393d22a445098adc002b53ebbbcef82f8b0ac46d409ca7e3a19e2622fe8a.png" width=300> 
<br></br>	


```python
#在使用GPU机器时，可以将use_gpu变量设置成True
use_gpu = True
paddle.set_device('gpu:0') if use_gpu else paddle.set_device('cpu')

import warnings
warnings.filterwarnings('ignore')
paddle.seed(1024)


epochs = 3
BATCH_SIZE = 32
model_path = './mnist.pdparams'

train_dataset = MnistDataset(mode='train')
# 这里为了使每次的训练精度都保持一致，因此先选择了shuffle=False，真正训练时应改为shuffle=True
train_loader = paddle.io.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0) 

model = MNIST()
# lr = 0.01
total_steps = (int(50000//BATCH_SIZE) + 1) * epochs
lr = paddle.optimizer.lr.PolynomialDecay(learning_rate=0.01, decay_steps=total_steps, end_lr=0.001)
opt = paddle.optimizer.Momentum(learning_rate=lr, parameters=model.parameters())

trainer = Trainer(
    model_path=model_path,
    model=model,
    optimizer=opt
)

trainer.train(train_datasets=train_loader, start_epoch = 0, end_epoch = epochs, save_path='checkpoint')
```


    ---------------------------------------------------------------------------
    
    ValueError                                Traceback (most recent call last)
    
    /tmp/ipykernel_94/598531791.py in <module>
          1 #在使用GPU机器时，可以将use_gpu变量设置成True
          2 use_gpu = True
    ----> 3 paddle.set_device('gpu:0') if use_gpu else paddle.set_device('cpu')
          4 
          5 import warnings


    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/device/__init__.py in set_device(device)
        311         data = paddle.stack([x1,x2], axis=1)
        312     """
    --> 313     place = _convert_to_place(device)
        314     framework._set_expected_place(place)
        315     return place


    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/device/__init__.py in _convert_to_place(device)
        254                 raise ValueError(
        255                     "The device should not be {}, since PaddlePaddle is "
    --> 256                     "not compiled with CUDA".format(avaliable_gpu_device))
        257             device_info_list = device.split(':', 1)
        258             device_id = device_info_list[1]


    ValueError: The device should not be <re.Match object; span=(0, 5), match='gpu:0'>, since PaddlePaddle is not compiled with CUDA


# 恢复训练

**模型恢复训练，需要重新组网，所以我们需要重启AIStudio，运行`MnistDataset`数据读取和`MNIST`网络定义、`Trainer`部分代码，再执行模型恢复代码**

在上述训练代码中，我们训练了五轮（epoch）。在每轮结束时，我们均保存了模型参数和优化器相关的参数。

- 使用``model.state_dict()``获取模型参数。
- 使用``opt.state_dict``获取优化器和学习率相关的参数。
- 调用``paddle.save``将参数保存到本地。

比如第一轮训练保存的文件是mnist_epoch0.pdparams，mnist_epoch0.pdopt，分别存储了模型参数和优化器参数。

使用``paddle.load``分别加载模型参数和优化器参数，如下代码所示。

``` 
paddle.load(params_path+'.pdparams')
paddle.load(params_path+'.pdopt')
```

如何判断模型是否准确的恢复训练呢？

理想的恢复训练是模型状态回到训练中断的时刻，恢复训练之后的梯度更新走向是和恢复训练前的梯度走向完全相同的。基于此，我们可以通过恢复训练后的损失变化，判断上述方法是否能准确的恢复训练。即从epoch 0结束时保存的模型参数和优化器状态恢复训练，校验其后训练的损失变化（epoch 1）是否和不中断时的训练相差不多。

------

**说明：**

恢复训练有如下两个要点：

* 保存模型时分别保存模型参数和优化器参数。
* 恢复参数时分别恢复模型参数和优化器参数。

------

下面的代码将展示恢复训练的过程，并验证恢复训练是否成功。加载模型参数并从第一个epoch开始训练，以便读者可以校验恢复训练后的损失变化。


```python
import warnings
warnings.filterwarnings('ignore')
# MLP继续训练
paddle.seed(1024)

epochs = 3
BATCH_SIZE = 32
model_path = './mnist_retrain.pdparams'

train_dataset = MnistDataset(mode='train')
# 这里为了使每次的训练精度都保持一致，因此先选择了shuffle=False，真正训练时应改为shuffle=True
train_loader = paddle.io.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4) 

model = MNIST()
# lr = 0.01
total_steps = (int(50000//BATCH_SIZE) + 1) * epochs
lr = paddle.optimizer.lr.PolynomialDecay(learning_rate=0.01, decay_steps=total_steps, end_lr=0.001)
opt = paddle.optimizer.Momentum(learning_rate=lr, parameters=model.parameters())

params_dict = paddle.load('./checkpoint/mnist_epoch0.pdparams')
opt_dict = paddle.load('./checkpoint/mnist_epoch0.pdopt')

# 加载参数到模型
model.set_state_dict(params_dict)
opt.set_state_dict(opt_dict)

trainer = Trainer(
    model_path=model_path,
    model=model,
    optimizer=opt
)
# 前面训练模型都保存了，这里save_path设置为新路径，实际训练中保存在同一目录就可以
trainer.train(train_datasets=train_loader,start_epoch = 1, end_epoch = epochs, save_path='checkpoint_con')
```

从恢复训练的损失变化来看，加载模型参数继续训练的损失函数值和正常训练损失函数值是相差不多的，可见使用飞桨实现恢复训练是极其简单的。
总结一下：

* 保存模型时同时保存模型参数和优化器参数；

```
paddle.save(opt.state_dict(), 'model.pdopt')
paddle.save(model.state_dict(), 'model.pdparams')
```

* 恢复参数时同时恢复模型参数和优化器参数。

```
model_dict = paddle.load("model.pdparams")
opt_dict = paddle.load("model.pdopt")

model.set_state_dict(model_dict)
opt.set_state_dict(opt_dict)
```



****

# 10 .【手写数字识别】之动转静部署

## 动静转换

动态图有诸多优点，比如易用的接口、Python风格的编程体验、友好的调试交互机制等。在动态图模式下，代码可以按照我们编写的顺序依次执行。这种机制更符合Python程序员的使用习惯，可以很方便地将脑海中的想法快速地转化为实际代码，也更容易调试。

但在性能方面，由于Python执行开销较大，与C++有一定差距，因此在工业界的许多部署场景中（如大型推荐系统、移动端）都倾向于直接使用C++进行提速。相比动态图，静态图在部署方面更具有性能的优势。静态图程序在编译执行时，先搭建模型的神经网络结构，然后再对神经网络执行计算操作。预先搭建好的神经网络可以脱离Python依赖，在C++端被重新解析执行，而且拥有整体网络结构也能进行一些网络结构的优化。

那么，有没有可能，深度学习框架实现一个新的模式，同时具备动态图高易用性与静态图高性能的特点呢？飞桨从2.0版本开始，新增新增支持动静转换功能，编程范式的选择更加灵活。用户依然使用动态图编写代码，只需添加一行装饰器 @paddle.jit.to_static，即可实现动态图转静态图模式运行，进行模型训练或者推理部署。在本章节中，将介绍飞桨动态图转静态图的基本用法和相关原理。

## 动态图转静态图训练

飞桨的动转静方式是基于源代码级别转换的ProgramTranslator实现，其原理是通过分析Python代码，将动态图代码转写为静态图代码，并在底层自动使用静态图执行器运行。其基本使用方法十分简便，只需要在要转化的函数（该函数也可以是用户自定义动态图Layer的forward函数）前添加一个装饰器 @paddle.jit.to_static。这种转换方式使得用户可以灵活使用Python语法及其控制流来构建神经网络模型。下面通过一个例子说明如何使用飞桨实现动态图转静态图训练。


```python
import paddle

# 定义手写数字识别模型
class MNIST(paddle.nn.Layer):
    def __init__(self):
        super(MNIST, self).__init__()
        
        # 定义一层全连接层，输出维度是1
        self.fc = paddle.nn.Linear(in_features=784, out_features=10)

    # 定义网络结构的前向计算过程
    @paddle.jit.to_static  # 添加装饰器，使动态图网络结构在静态图模式下运行
    def forward(self, inputs):
        outputs = self.fc(inputs)
        return outputs

```

上述代码构建了仅有一层全连接层的手写字符识别网络。特别注意，在forward函数之前加了装饰器`@paddle.jit.to_static`，要求模型在静态图模式下运行。下面是模型的训练代码，由于飞桨实现动转静的功能是在内部完成的，对使用者来说，动态图的训练代码和动转静模型的训练代码是完全一致的。训练代码如下：


```python
import paddle
import paddle.nn.functional as F
# 确保从paddle.vision.datasets.MNIST中加载的图像数据是np.ndarray类型
paddle.vision.set_image_backend('cv2')

# 图像归一化函数，将数据范围为[0, 255]的图像归一化到[-1, 1]
def norm_img(img):
    batch_size = img.shape[0]
    # 归一化图像数据
    img = img/127.5 - 1
    # 将图像形式reshape为[batch_size, 784]
    img = paddle.reshape(img, [batch_size, 784])
    
    return img

def train(model):
    model.train()
    # 加载训练集 batch_size 设为 16
    train_loader = paddle.io.DataLoader(paddle.vision.datasets.MNIST(mode='train'), 
                                        batch_size=16, 
                                        shuffle=True)
    opt = paddle.optimizer.SGD(learning_rate=0.001, parameters=model.parameters())
    EPOCH_NUM = 10
    for epoch in range(EPOCH_NUM):
        for batch_id, data in enumerate(train_loader()):
            images = norm_img(data[0]).astype('float32')
            labels = data[1].astype('int64')
            
            #前向计算的过程
            predicts = model(images)
            
            # 计算损失
            loss = F.cross_entropy(predicts, labels)
            avg_loss = paddle.mean(loss)
            
            #每训练了1000批次的数据，打印下当前Loss的情况
            if batch_id % 1000 == 0:
                print("epoch_id: {}, batch_id: {}, loss is: {}".format(epoch, batch_id, avg_loss.numpy()))
            
            #后向传播，更新参数的过程
            avg_loss.backward()
            opt.step()
            opt.clear_grad()


model = MNIST() 

train(model)

paddle.save(model.state_dict(), './mnist.pdparams')
print("==>Trained model saved in ./mnist.pdparams")
```

    item  121/2421 [>.............................] - ETA: 2s - 1ms/it
    
    Cache file /home/aistudio/.cache/paddle/dataset/mnist/train-images-idx3-ubyte.gz not found, downloading https://dataset.bj.bcebos.com/mnist/train-images-idx3-ubyte.gz 
    Begin to download


    item  242/2421 [=>............................] - ETA: 2s - 1ms/itemitem  331/2421 [===>..........................] - ETA: 2s - 1ms/
    item 8/8 [============================>.] - ETA: 0s - 2ms/it


​    

    Download finished
    Cache file /home/aistudio/.cache/paddle/dataset/mnist/train-labels-idx1-ubyte.gz not found, downloading https://dataset.bj.bcebos.com/mnist/train-labels-idx1-ubyte.gz 
    Begin to download
    
    Download finished


    epoch_id: 0, batch_id: 0, loss is: [2.211587]
    epoch_id: 0, batch_id: 1000, loss is: [0.81444955]
    epoch_id: 0, batch_id: 2000, loss is: [0.51870286]
    epoch_id: 0, batch_id: 3000, loss is: [0.7179581]
    epoch_id: 1, batch_id: 0, loss is: [0.34819442]
    epoch_id: 1, batch_id: 1000, loss is: [0.3227439]
    epoch_id: 1, batch_id: 2000, loss is: [0.2903515]
    epoch_id: 1, batch_id: 3000, loss is: [0.46440566]
    epoch_id: 2, batch_id: 0, loss is: [0.48201334]


我们可以观察到，动转静的训练方式与动态图训练代码是完全相同的。因此，在动转静训练的时候，开发者只需要在动态图的组网前向计算函数上添加一个装饰器即可实现动转静训练。
在模型构建和训练中，飞桨更希望借用动态图的易用性优势，实际上，在加上@to_static装饰器运行的时候，飞桨内部是在静态图模式下执行OP的，但是展示给开发者的依然是动态图的使用方式。

动转静更能体现静态图的方面在于模型部署上。下面将介绍动态图转静态图的部署方式。



## 动态图转静态图模型保存

在推理&部署场景中，需要同时保存推理模型的结构和参数，但是动态图是即时执行即时得到结果，并不会记录模型的结构信息。动态图在保存推理模型时，需要先将动态图模型转换为静态图写法，编译得到对应的模型结构再保存，而飞桨框架2.0版本推出paddle.jit.save和paddle.jit.load接口，无需重新实现静态图网络结构，直接实现动态图模型转成静态图模型格式。paddle.jit.save接口会自动调用飞桨框架2.0推出的动态图转静态图功能，使得用户可以做到使用动态图编程调试，自动转成静态图训练部署。

这两个接口的基本关系如下图所示： 

<br><center><img src="photo/018ac3d24c22423a8a263dfd0f0f7f49898b29e707c14dbdb8f9f5abdde75449.png" width="700" hegiht="" ></center></br>

<center>图1：动转静模型保存接口 </center>

<br></br>
当用户使用paddle.jit.save保存Layer对象时，飞桨会自动将用户编写的动态图Layer模型转换为静态图写法，并编译得到模型结构，同时将模型结构与参数保存。paddle.jit.save需要适配飞桨沿用已久的推理模型与参数格式，做到前向完全兼容，因此其保存格式与paddle.save有所区别，具体包括三种文件：保存模型结构的*.pdmodel文件；保存推理用参数的*.pdiparams文件和保存兼容变量信息的*.pdiparams.info文件，这几个文件后缀均为paddle.jit.save保存时默认使用的文件后缀。

比如，如果保存上述手写字符识别的inference模型用于部署，可以直接用下面代码实现：



```python
# save inference model
from paddle.static import InputSpec
# 加载训练好的模型参数
state_dict = paddle.load("./mnist.pdparams")
# 将训练好的参数读取到网络中
model.set_state_dict(state_dict)
# 设置模型为评估模式
model.eval()

# 保存inference模型
paddle.jit.save(
    layer=model,
    path="inference/mnist",
    input_spec=[InputSpec(shape=[None, 784], dtype='float32')])

print("==>Inference model saved in inference/mnist.")

```

其中，paddle.jit.save API 将输入的网络存储为 paddle.jit.TranslatedLayer 格式的模型，载入后可用于预测推理或者fine-tune训练。
该接口会将输入网络转写后的模型结构 Program 和所有必要的持久参数变量存储至输入路径 path 。

path 是存储目标的前缀，存储的模型结构 Program 文件的后缀为 .pdmodel ，存储的持久参数变量文件的后缀为 .pdiparams ，同时这里也会将一些变量描述信息存储至文件，文件后缀为 .pdiparams.info。

通过调用对应的paddle.jit.load接口，可以把存储的模型载入为 paddle.jit.TranslatedLayer格式，用于预测推理或者fine-tune训练。



```python
import numpy as np
import paddle
import paddle.nn.functional as F
# 确保从paddle.vision.datasets.MNIST中加载的图像数据是np.ndarray类型
paddle.vision.set_image_backend('cv2')

# 读取mnist测试数据，获取第一个数据
mnist_test = paddle.vision.datasets.MNIST(mode='test')
test_image, label = mnist_test[0]
# 获取读取到的图像的数字标签
print("The label of readed image is : ", label)

# 将测试图像数据转换为tensor，并reshape为[1, 784]
test_image = paddle.reshape(paddle.to_tensor(test_image), [1, 784])
# 然后执行图像归一化
test_image = norm_img(test_image)
# 加载保存的模型
loaded_model = paddle.jit.load("./inference/mnist")
# 利用加载的模型执行预测
preds = loaded_model(test_image)
pred_label = paddle.argmax(preds)
# 打印预测结果
print("The predicted label is : ", pred_label.numpy())
```

paddle.jit.save API 可以把输入的网络结构和参数固化到一个文件中，所以通过加载保存的模型，可以不用重新构建网络结构而直接用于预测，易于模型部署。


## 总结：

本章节中，介绍了飞桨动转静的功能和基本原理，并以一个例子介绍了如何将一个动态图模型转换到静态图模式下训练，并将训练好的模型转换成更易于部署的inference模型，有关更多动转静的功能介绍，可以参考[官方文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/04_dygraph_to_static/index_cn.html)。

