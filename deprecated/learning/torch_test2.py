import torch
from torch import nn

# N是批大小；D是输入维度
# H是隐藏层维度；D_out是输出维度
from torch.nn import Module

N, D_in, H, D_out = 64, 1000, 100, 10

# 产生随机输入和输出张量

n_data = torch.ones(100, 1000)
x0 = torch.normal(2 * n_data, 1)
y0 = torch.zeros(100)
x1 = torch.normal(-2 * n_data, 1)
y1 = torch.ones(100)
x = torch.cat((x0, x1), 0).type(torch.FloatTensor).requires_grad_()
y = torch.cat((y0, y1), 0).type(torch.FloatTensor).view((-1, 1))


# net=nn.Sequential(  OrderedDict(     [
#                   ('conv1', nn.Linear(1000,30)),
#                   ('relu1', nn.ReLU()),
#                   ('conv2', nn.Linear(30,1)),
#                   ('relu2', nn.Sigmoid())
#                 ]))

class Net(Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Linear(1000, 30)
        self.conv2 = nn.Linear(30, 1)
        self.relu1 = nn.Sigmoid()

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)

        x = self.relu1(x)
        return x


net = Net()

opitmizer = torch.optim.SGD(net.parameters(), lr=0.03)
loss_fun = nn.BCELoss()
feature_result = []
feature_result2 = []


def hook(self, input, output):
    feature_result.append(output[0].data.detach())


def hook2(self, input, output):
    feature_result2.append(output[0].data.detach())
    print(input)


for i in range(4):
    net.relu1.register_backward_hook(hook)
    net.register_backward_hook(hook2)
    predict = net(x)
    # print(predict)
    loss = loss_fun(predict, y)
    print(float(loss))

    opitmizer.zero_grad()
    loss.backward()
    opitmizer.step()
