import torch
from torch import nn


class VGG(nn.Module):

    def __init__(self, features, dense1=256, dense2=128, dense3=64, out=41, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(dense1, dense2),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(dense2, dense3),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(dense3, out),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
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


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 1
    for v in cfg:
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


def vgg_self(cfgs=None, batch_norm=False, dense1=256, dense2=128, dense3=64, out=41):
    if cfgs is None:
        cfgs = [64]

    model = VGG(make_layers(cfgs, batch_norm=batch_norm),
                dense1=dense1, dense2=dense2, dense3=dense3, out=out)

    return model


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


if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    from scipy import ndimage as ndi

    # import
    com_data = pd.read_excel(r'C:\Users\Administrator\Desktop\dg.xlsx',
                             sheet_name=0, index_col=None, header=0)

    x = com_data[["s1", "s2", "s3", "s4", "s5"]].values
    x_i = []
    for i in range(int(x.shape[0] / 5)):
        xi = x[5 * i:5 * (i + 1)]
        x_i.append(xi)
    x = np.array(x_i)

    y = com_data.drop(labels=["s1", "s2", "s3", "s4", "s5"], axis=1)
    y = y.dropna(axis=0).values

    # interpolate to 40*40
    size = 40
    scale = size / 5
    x = ndi.zoom(x, (1, scale, scale), order=3)

    # defination
    t_x = torch.from_numpy(x)
    t_y = torch.from_numpy(y)
    t_x = t_x.unsqueeze(dim=1)
    t_x = t_x.to(torch.float32)
    t_y = t_y.to(torch.float32)

    # ToDo parameter
    cov = [8, 16, 32]
    dense1 = int(cov[-1] * (size / (2 ** len(cov))) ** 2)
    model = vgg_self(cov, batch_norm=True, dense1=dense1, dense2=128, dense3=64, out=41)

    # run
    opitmizer = torch.optim.SGD(model.parameters(), lr=0.01)  # ToDo parameter
    loss_fun = nn.MSELoss()
    for i in range(5):  # ToDo parameter
        predict = model(t_x)
        loss = loss_fun(predict, t_y)
        loss = loss.to(dtype=torch.float32)

        print(float(loss))

        opitmizer.zero_grad()
        loss.backward()
        opitmizer.step()

    # 第一条线
    tx0 = t_x[0]
    tx0 = tx0.unsqueeze(dim=1)
    predict = model(tx0)
    predict = predict.detach().numpy()
    predict_y = predict.ravel()

    y_true = t_y[0].detach().numpy()

    x = np.arange(predict.shape[1])

    line_scatter(x, predict_y, y_true, )
