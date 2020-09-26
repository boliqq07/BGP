# -*- coding: utf-8 -*-
# import numpy as np
#
# # N是批尺寸参数；D_in是输入维度
# # H是隐藏层维度；D_out是输出维度
# N, D_in, H, D_out = 64, 1000, 100, 10
#
# # 产生随机输入和输出数据
# x = np.random.randn(N, D_in)
# y = np.random.randn(N, D_out)
#
# # 随机初始化权重
# w1 = np.random.randn(D_in, H)
# w2 = np.random.randn(H, D_out)
#
# learning_rate = 1e-6
# for t in range(500):
#     # 前向传播：计算预测值y
#     h = x.dot(w1)
#     h_relu = np.maximum(h, 0)
#     y_pred = h_relu.dot(w2)
#
#     # 计算并显示loss(损失）
#     loss = np.square(y_pred - y).sum()
#     print(t, loss)
#
#     # 反向传播，计算w1、w2对loss的梯度
# #     grad_y_pred = 2.0 * (y_pred - y) #计算平方导数
# #     grad_w2 = h_relu.T.dot(grad_y_pred) #计算w2导数
# #     grad_h_relu = grad_y_pred.dot(w2.T) #计算对relu导数
# #     grad_h = grad_h_relu.copy()
# #     grad_h[h < 0] = 0      #relu 如何求导方式
# #     grad_w1 = x.T.dot(grad_h) #计算对w1导数
# #
# #     # 更新权重
# #     w1 -= learning_rate * grad_w1
# #     w2 -= learning_rate * grad_w2
# # -*- coding: utf-8 -*-
#
# import torch
#
#
# dtype = torch.float
# device = torch.device("cpu")
# # device = torch.device("cuda:0") # Uncomment this to run on GPU
#
# # N是批尺寸大小； D_in 是输入维度；
# # H 是隐藏层维度； D_out 是输出维度
# N, D_in, H, D_out = 64, 1000, 100, 10
#
#
# x = torch.randn(N, D_in, device=device, dtype=dtype)
# y = torch.randn(N, D_out, device=device, dtype=dtype)
#
# # 产生随机权重tensor，将requires_grad设置为True，意味着我们希望在反向传播时候计算这些值的梯度
# w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
# w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)
#
# learning_rate = 1e-6
# for t in range(500):
#     # 前向传播：计算预测值y
#     h = x.mm(w1)
#     h_relu = h.clamp(min=0)
#     y_pred = h_relu.mm(w2)
#
#     y_pred = x.mm(w1).clamp(min=0).mm(w2)
#
#     # 计算并输出loss
#     # loss是一个形状为(1,)的张量
#     # loss.item()是这个张量对应的python数值
#     loss = (y_pred - y).pow(2).sum()
#
#     print(t, loss.item())
#
#     # 使用autograd计算反向传播,这个调用将计算loss对所有requires_grad=True的tensor的梯度。
#     # 这次调用后，w1.grad和w2.grad将分别是loss对w1和w2的梯度张量。
#     loss.backward()
#
#     with torch.no_grad():
#         w1 -= learning_rate * w1.grad
#         w2 -= learning_rate * w2.grad
#
#         # 反向传播之后手动将梯度置零
#         w1.grad.zero_()
#         w2.grad.zero_()


# -*- coding: utf-8 -*-
import torch
from torch import nn

# N是样本数量大小；D_in是输入维度,D_out是输出维度
# 其他是隐藏层维度；
N, D_in, D_out = 64, 1000, 100
D1, D2 = 100, 100
dense1, dense2, dense3 = 32, 32, 16
# 产生随机输入和输出张量
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# 使用nn包定义模型和损失函数
model = nn.Sequential(
    nn.Conv2d(D_in, D1, 3),
    nn.ReLU(),
    nn.Conv2d(D1, D2, 3),
    nn.ReLU(),
    nn.Conv2d(D1, dense1, 3),
    nn.ReLU(),
    nn.Linear(dense1, dense2),
    nn.ReLU(True),
    nn.Dropout(),
    nn.Linear(dense2, dense3),
    nn.ReLU(True),
    nn.Dropout(),
    nn.Linear(dense3, D_out),
)

# 使用optim包定义优化器(Optimizer）。Optimizer将会为我们更新模型的权重
# 这里我们使用Adam优化方法；optim包还包含了许多别的优化算法
# Adam构造函数的第一个参数告诉优化器应该更新哪些张量
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.MSELoss(reduction='sum')

for t in range(500):
    # 前向传播：通过像模型输入x计算预测的y
    y_pred = model(x)

    # 计算并输出loss
    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    # 在反向传播之前，使用optimizer将它要更新的所有张量的梯度清零(这些张量是模型可学习的权重)。
    # 这是因为默认情况下，每当调用.backward(）时，渐变都会累积在缓冲区中(即不会被覆盖）
    # 有关更多详细信息，请查看torch.autograd.backward的文档。
    optimizer.zero_grad()
    # 反向传播：根据模型的参数计算loss的梯度
    loss.backward()
    # 调用Optimizer的step函数使它所有参数更新
    optimizer.step()
