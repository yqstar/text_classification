# from torch.utils.tensorboard import SummaryWriter
# import numpy as np
#
# writer = SummaryWriter()
#
# for n_iter in range(1000):
#     writer.add_scalar('Loss/train', np.random.random(), n_iter)
#     writer.add_scalar('Loss/test', np.random.random(), n_iter)
#     writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
#     writer.add_scalar('Accuracy/test', np.random.random(), n_iter)
#
#
# # 'runs\\Feb15_00-14-45_DESKTOP-6MABOH6'
from torch.utils.tensorboard import SummaryWriter
import torch
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable


class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Sequential(  # 输入torch.Size([64, 1, 28, 28])
            torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            # 用于搭建卷积神经网络的卷积层，主要的输入参数有输入通道数、
            # 输出通道数、卷积核大小、卷积核移动步长和Padding值。
            # 输出维度 = 1+(输入维度-卷积核大小+2*padding)/卷积核步长
            # 输出torch.Size([64, 64, 28, 28])

            torch.nn.ReLU(),  # 输出torch.Size([64, 64, 28, 28])
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            # 输出torch.Size([64, 128, 28, 28])

            torch.nn.ReLU(),
            torch.nn.MaxPool2d(stride=2, kernel_size=2)
            # 主要的输入参数是池化窗口大小、池化窗口移动步长和Padding值
            # 输出torch.Size([64, 128, 14, 14])
        )

        self.dense = torch.nn.Sequential(  # 输入torch.Size([64, 14*14*128])
            torch.nn.Linear(14 * 14 * 128, 1024),
            # class torch.nn.Linear(in_features，out_features，bias = True)
            # 输出torch.Size([64, 1024])
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            # torch.nn.Dropout类用于防止卷积神经网络在训练的过程中
            # 发生过拟合，其工作原理简单来说就是在模型训练的过程中，
            # 以一定的随机概率将卷积神经网络模型的部分参数归零，以达
            # 到减少相邻两层神经连接的目的。这样做是为了让我们最后训
            # 练出来的模型对各部分的权重参数不产生过度依赖，从而防止
            # 过拟合。对于torch.nn.Dropout类，我们可以对随机概率值
            # 的大小进行设置，如果不做任何设置，就使用默认的概率值0.5。
            torch.nn.Linear(1024, 10)
            # 输出torch.Size([64, 10])
        )

    def forward(self, x):  # torch.Size([64, 1, 28, 28])
        print("x shape {}".format(x.shape))
        x = self.conv1(x)  # 输出torch.Size([64, 128, 14, 14])
        x = x.view(-1, 14 * 14 * 128)
        # view()函数作用是将一个多行的Tensor,拼接成一行，torch.Size([64, 14*14*128])
        x = self.dense(x)  # 输出torch.Size([64, 10])
        return x


transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5],
                                                     std=[0.5])])
data_train = datasets.MNIST(root="./data/",
                            transform=transform,
                            train=True,
                            download=True)
data_loader_train = torch.utils.data.DataLoader(
    dataset=data_train,
    batch_size=64,
    shuffle=True)
# images, labels = next(iter(data_loader_train))#迭代器
# torch.Size([64, 1, 28, 28])
images = torch.randn(64, 1, 28, 28)

model = Model()

writer = SummaryWriter()
for i in range(5):
    images = torch.randn(64, 1, 28, 28)
    writer.add_graph(model, input_to_model=images, verbose=False)

writer.flush()
writer.close()

# tensorboard --logdir=runs