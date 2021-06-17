# 参考博客地址：https://blog.csdn.net/weixin_43718786/article/details/115252637

import time
import torch
from torch import nn, optim

# import sys
# sys.path.append("..")

class LeNet(nn.Module):  # 我们定义网络时一般是继承的torch.nn.Module创建新的子类
    def __init__(self):
        super(LeNet, self).__init__()   # 第二、三行都是python类继承的基本操作,此写法应该是python2.7的继承格式,但python3里写这个好像也可以
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5),  # in_channels, out_channels, kernel_size # 添加第一个卷积层,调用了nn里面的Conv2d（）
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2),  # kernel_size, stride # 最大池化层
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16*4*4, 120),  # 接着三个全连接层
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )

    def forward(self, img):   # 这里定义前向传播的方法，为什么没有定义反向传播的方法呢？这其实就涉及到torch.autograd模块了，
        feature = self.conv(img)
        # 把120张大小为1的图像当成一个长度为120的一维tensor?
        # .view( )是一个tensor的方法，使得tensor改变size但是元素的总数是不变的。
        #  第一个参数-1是说这个参数由另一个参数确定， 比如矩阵在元素总数一定的情况下，确定列数就能确定行数。
        #  那么为什么这里只关心列数不关心行数呢，因为马上就要进入全连接层了，而全连接层说白了就是矩阵乘法，
        #  你会发现第一个全连接层的首参数是16*5*5，所以要保证能够相乘，在矩阵乘法之前就要把x调到正确的size
        # 更多的Tensor方法参考Tensor: http://pytorch.org/docs/0.3.0/tensors.html
        output = self.fc(feature.view(img.shape[0], -1))
        return output




