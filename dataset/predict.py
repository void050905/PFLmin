import torch
from model import MyLeNet5
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage

# Compose()：将多个transforms的操作整合在一起
data_transform = transforms.Compose([
    # ToTensor()：数据转化为Tensor格式
    transforms.ToTensor()
])

# 加载训练数据集
train_dataset = datasets.MNIST(root='./data', train=True, transform=data_transform, download=True)
# 给训练集创建一个数据加载器, shuffle=True用于打乱数据集，每次都会以不同的顺序返回
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)

# 加载测试数据集
test_dataset = datasets.MNIST(root='./data', train=False, transform=data_transform, download=True)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=16, shuffle=True)

# 如果有NVIDA显卡，转到GPU训练，否则用CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 模型实例化，将模型转到device
model = MyLeNet5().to(device)

# 加载train.py里训练好的模型
model.load_state_dict(torch.load("D:/LeNet/save_model/best_model.pth"))

# 结果类型
classes = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
]

# 把Tensor转化为图片，方便可视化
show = ToPILImage()

# 进入验证阶段
for i in range(10):
    x, y = test_dataset[i][0], test_dataset[i][1]
    # show()：显示图片
    show(x).show()
    # unsqueeze(input, dim)，input(Tensor)：输入张量，dim (int)：插入维度的索引，最终将张量维度扩展为4维
    x = Variable(torch.unsqueeze(x, dim=0).float(), requires_grad=False).to(device)
    with torch.no_grad():
        pred = model(x)
        # argmax(input)：返回指定维度最大值的序号
        # 得到验证类别中数值最高的那一类，再对应classes中的那一类
        predicted, actual = classes[torch.argmax(pred[0])], classes[y]
        # 输出预测值与真实值
        print(f'predicted: "{predicted}", actual:"{actual}"')