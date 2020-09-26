# -*- coding: utf-8 -*-

# @Time    : 2019/12/28 22:48
# @Email   : 986798607@qq.com
# @Software: PyCharm
# @License: BSD 3-Clause
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset


class RandomDataset(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size),
                         batch_size=batch_size, shuffle=True)
