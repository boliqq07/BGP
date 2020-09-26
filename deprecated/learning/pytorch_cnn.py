import torch
from torch import nn


def line_scatter(x, y_pre, y_true, strx='x', stry='y_pre', stry2="y_true"):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    # ax = fig.add_subplot(111)
    ly_pre, = plt.plot(x, y_pre, '^-', ms=5, lw=2, alpha=0.7, color='black')
    ly_true, = plt.plot(y_true, 'o-', ms=5, lw=2, alpha=0.7, color='red')
    plt.xlabel(strx)
    plt.ylabel("y")

    plt.legend(handles=(ly_pre, ly_true), labels=(stry, stry2),
               )
    plt.show()


def spilt_x(x, n):
    x_i = []
    for i in range(int(x.shape[0] / n)):
        xi = x[n * i:n * (i + 1)]
        x_i.append(xi)
    x = np.array(x_i)
    return x


class VGG(nn.Module):

    def __init__(self, conv_set, linear_set, batch_norm=False, init_weights=True):
        super(VGG, self).__init__()

        self.features = self.conv_layers(conv_set, batch_norm=batch_norm)
        self.connect = self.linear_layers(linear_set)

        if init_weights:
            self._initialize_weights()

    @staticmethod
    def conv_layers(conv_set, batch_norm=False):
        layers = []
        in_channels = 1
        for v in conv_set:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, stride=2)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v

        return nn.Sequential(*layers)

    @staticmethod
    def linear_layers(linear_set):
        layers = []
        layers += [nn.Linear(linear_set[0], linear_set[1])]
        channels = linear_set[1]
        for v in linear_set[1:]:
            layers += [nn.ReLU(True), nn.Dropout(), nn.Linear(channels, v), ]
            channels = v

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.connect(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    from scipy import ndimage as ndi

    ######################################################################################
    # import #数据输入
    ######################################################################################
    com_data = pd.read_excel(r'/data/home/wangchangxin/data/pytorch_cnn/pytorch_cnn.xlsx',
                             sheet_name=0, index_col=None, header=0)

    x = com_data[
        ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "s12", "s13", "s14", "s15", "s16"]].values
    y = com_data.drop(
        labels=["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "s12", "s13", "s14", "s15", "s16"],
        axis=1)

    x = spilt_x(x, 16)
    y = y.dropna(axis=0).values

    ######################################################################################
    # interpolate to size*size #把初始图片大小插值变大
    ######################################################################################

    size = 64
    scale = size / 16
    x = ndi.zoom(x, (1, scale, scale), order=3)

    # defination
    t_x = torch.from_numpy(x)
    t_y = torch.from_numpy(y)
    t_x = t_x.unsqueeze(dim=1)
    t_x = t_x.to(torch.float32)
    t_y = t_y.to(torch.float32)

    ######################################################################################
    # model
    ######################################################################################
    # ToDo parameter
    model = VGG(conv_set=(8, 64, "M"), linear_set=(4096, 300, 61), batch_norm=True, init_weights=True)
    print(model)
    # run
    opitmizer = torch.optim.SGD(model.parameters(), lr=0.01)  # ToDo parameter
    loss_fun = nn.MSELoss()
    for i in range(20):  # ToDo parameter
        predict = model(t_x)
        loss = loss_fun(predict, t_y)
        loss = loss.to(dtype=torch.float32)

        print(float(loss))

        opitmizer.zero_grad()
        loss.backward()
        opitmizer.step()

    ########################################################################################
    # plot 第一条线
    ########################################################################################
    tx0 = t_x[0]
    tx0 = tx0.unsqueeze(dim=1)
    predict = model(tx0)
    predict = predict.detach().numpy()
    predict_y = predict.ravel()

    y_true = t_y[0].detach().numpy()

    x = np.arange(predict.shape[1])

    line_scatter(x, predict_y, y_true, )
