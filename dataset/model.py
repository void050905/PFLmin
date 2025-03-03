# 导入pytorch库
import torch
# 导入torch.nn模块
from torch import nn


# 定义LeNet网络模型
# MyLeNet5（子类）继承nn.Module（父类）
class MyLeNet5(nn.Module):
    # 子类继承中重新定义Module类的__init__()和forward()函数
    # init()函数：进行初始化，申明模型中各层的定义
    def __init__(self):
        # super：引入父类的初始化方法给子类进行初始化
        super(MyLeNet5, self).__init__()
        # 卷积层，输入大小为28*28，输出大小为28*28，输入通道为1，输出为6，卷积核为5，扩充边缘为2
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        # 使用sigmoid作为激活函数
        self.Sigmoid = nn.Sigmoid()
        # AvgPool2d：二维平均池化操作
        # 池化层，输入大小为28*28，输出大小为14*14，输入通道为6，输出为6，卷积核为2，步长为2
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)
        # 卷积层，输入大小为14*14，输出大小为10*10，输入通道为6，输出为16，卷积核为5
        self.c3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        # 池化层，输入大小为10*10，输出大小为5*5，输入通道为16，输出为16，卷积核为2，步长为2
        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)
        # 卷积层，输入大小为5*5，输出大小为1*1，输入通道为16，输出为120，卷积核为5
        self.c5 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)
        # Flatten()：将张量（多维数组）平坦化处理，张量的第0维表示的是batch_size（数量），所以Flatten()默认从第二维开始平坦化
        self.flatten = nn.Flatten()
        # 全连接层
        # Linear（in_features，out_features）
        # in_features指的是[batch_size, size]中的size,即样本的大小
        # out_features指的是[batch_size，output_size]中的output_size，样本输出的维度大小，也代表了该全连接层的神经元个数
        self.f6 = nn.Linear(120, 84)
        # 全连接层&输出层
        self.output = nn.Linear(84, 10)

    # forward()：定义前向传播过程,描述了各层之间的连接关系
    def forward(self, x):
        # x输入为28*28*1， 输出为28*28*6
        x = self.Sigmoid(self.c1(x))
        # x输入为28*28*6，输出为14*14*6
        x = self.s2(x)
        # x输入为14*14*6，输出为10*10*16
        x = self.Sigmoid(self.c3(x))
        # x输入为10*10*16，输出为5*5*16
        x = self.s4(x)
        # x输入为5*5*16，输出为1*1*120
        x = self.c5(x)
        x = self.flatten(x)
        # x输入为120，输出为84
        x = self.f6(x)
        # x输入为84，输出为10
        x = self.output(x)
        return x


# 测试代码
# 每个python模块（python文件）都包含内置的变量 __name__，当该模块被直接执行的时候，__name__ 等于文件名（包含后缀 .py ）
# 如果该模块 import 到其他模块中，则该模块的 __name__ 等于模块名称（不包含后缀.py）
# “__main__” 始终指当前执行模块的名称（包含后缀.py）
# if确保只有单独运行该模块时，此表达式才成立，才可以进入此判断语法，执行其中的测试代码，反之不行
if __name__ == "__main__":
    # rand:返回一个张量，包含了从区间[0, 1)的均匀分布中抽取的一组随机数，此处为四维张量
    x = torch.rand([1, 1, 28, 28])
    # 模型实例化
    model = MyLeNet5()
    y = model(x)