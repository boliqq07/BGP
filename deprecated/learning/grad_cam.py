# # -*- coding: utf-8 -*-
#
# # @Time    : 2019/12/19 13:50
# # @Email   : 986798607@qq.com
# # @Software: PyCharm
# # @License: BSD 3-Clause
# from __future__ import print_function
# from __future__ import print_function, division
#

#
# class Net(nn.Module):
#
#     def __init__(self):
#         super(Net, self).__init__()
#         # 1 input0 image channel, 6 output channels, 5x5 square convolution
#         # kernel
#         self.conv1 = nn.Conv2d(1, 6, 5)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         # an affine operation: y = Wx + b
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)
#
#     def forward(self, x):
#         # Max pooling over a (2, 2) window
#         x = self.conv1(x)
#         x = F.relu(x)
#         x = F.max_pool2d(x, (2, 2))
#         # If the size is a square you can only specify a single number
#         x = self.conv2(x)
#         x = F.relu(x)
#         x = F.max_pool2d(x, 2)
#         x = x.view(-1, self.num_flat_features(x))
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
#
#     def num_flat_features(self, x):
#         size = x.size()[1:]  # all dimensions except the batch dimension
#         num_features = 1
#         for s in size:
#             num_features *= s
#         return num_features
#
#
# net = Net()
# #
# input0 = torch.randn(1, 1, 32, 32)
# #
# target = torch.randn(10)  # 随机值作为样例
# target = target.view(1, -1)  # 使target和output的shape相同
# criterion = nn.MSELoss()
#
# out = net(input0)
# out.backward(torch.randn(1, 10))
# # create your optimizer
# optimizer = optim.SGD(net.parameters(), lr=0.01)
# for i in range(60):
#     # in your training loop:
#     optimizer.zero_grad()  # zero the gradient buffers
#     output = net(input0)
#     loss = criterion(output, target)
#
#     loss.backward()
#     optimizer.step()

# from tensorboardX import SummaryWriter
#
# with SummaryWriter(log_dir=r'C:\Users\Administrator\Desktop/logs', comment='vgg161') as writer:
#     writer.add_graph(net, input0)

# from torchvision import transforms
# from tensorboardX import SummaryWriter
#
# # from torch.utils.tensorboard import SummaryWriter
# cat_img = Image.open(r'C:\Users\Administrator\Desktop\图片1.png')
# cat_img.size
#
# transform_224 = transforms.Compose([
#     transforms.Resize(224),  # 这里要说明下 Scale 已经过期了，使用Resize
#     transforms.CenterCrop(224),
#     transforms.Totensor(),
# ])
# cat_img_224 = transform_224(cat_img)
# writer = SummaryWriter(log_dir=r'C:\Users\Administrator\Desktop/logs', comment='cat image')  # 这里的logs要与--logdir的参数一样
# writer.add_image("cat", cat_img_224)
# writer.close()  # 执行close立即刷新，否则将每120秒自动刷新

# x = torch.Floattensor([100])
# y = torch.Floattensor([500])

# import numpy as np
# for epoch in range(30):
#     x = x * 1.2
#     y = y / 1.1
#     loss = np.random.random()
#     with SummaryWriter(log_dir=r'C:\Users\Administrator\Desktop/logs', comment='train') as writer: #可以直接使用python的with语法，自动调用close方法
#         writer.add_histogram('his/x', x, epoch)
#         writer.add_histogram('his/y', y, epoch)
#         writer.add_scalar('datamnist/x', x, epoch)
#         writer.add_scalar('datamnist/y', y, epoch)
#         writer.add_scalar('datamnist/loss', loss, epoch)
#         writer.add_scalars('datamnist/data_group', {'x': x,

# net2 = torch.nn.Sequential(
#     torch.nn.Linear(2, 10),
#     torch.nn.ReLU(),
#     torch.nn.Linear(10, 2),
# )
# print('方法2：\n', net2)
# with SummaryWriter(log_dir=r'C:\Users\Administrator\Desktop/logs', comment='seq') as writer:
#     input0 = torch.randn(100, 2)
#     writer.add_graph(net2, input0)

import numpy as np
import torch
from matplotlib import cm
from matplotlib.cm import ScalarMappable
from torch.autograd import Function
from torchvision import models


class FeatureExtractor():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers
    提取激活和梯度，从目标中间层
     """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)  # 累计计算新图片，每一层顺序叠加
            if name in self.target_layers:
                x.register_hook(self.save_gradient)  # 保留 target层的梯度
                outputs += [x]  # 添加
        return outputs, x  # 获取多少层输出列表output，以及及最后输出x


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output. #总输出
    2. Activations from intermeddiate targetted layers. #中间目标层
    3. Gradients from intermeddiate targetted layers. """  # 中间目标层梯度

    def __init__(self, model, target_layers):
        self.model = model
        self.feature_extractor = FeatureExtractor(self.model.features, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients  # 梯度列表

    def __call__(self, x):
        target_activations, output = self.feature_extractor(x)  # 分类层之前的目标层
        output = output.view(output.size(0), -1)
        output = self.model.connect(output)
        return target_activations, output


class GradCam:
    def __init__(self, model, target_layer_names, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())  # 最终目标

        # 看index影响的部分，其他置0
        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.model.features.zero_grad()
        self.model.connect.zero_grad()
        one_hot.backward()

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()  # 1 512,14,14 收敛的最后一个结果的target层梯度

        target = features[-1]  # 取了最后一层feature层，在分类层之前
        target = target.cpu().data.numpy()[0, :]  # 512,14,14

        weights = np.mean(grads_val, axis=(2, 3))[0, :]  # 512个特征的2，3维度取平均梯度（临近平均），相当于衡量512个特征中哪个特征更重要！！

        cam = np.zeros(target.shape[1:], dtype=np.float32)  # 14,14

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
            # 每一个feature*其对应的权值 然后加和！！！！
            # 梯度考察哪个特征更重要
            # 然后乘上相应的特征层
            # 此重要性映射到原先图片，肯定会模糊
            # 利用了平移不变性质，注意此网络中不应该对数据转置等变换，
            # 否则破坏平移不变性！

        cam = np.maximum(cam, 0)
        from skimage.transform import resize
        cam = resize(cam, (224, 224))
        cam -= np.min(cam)
        cam /= np.max(cam)
        torch.cuda.empty_cache()
        return cam


class GuidedBackpropReLU(Function):

    @staticmethod
    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input),
                                   torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output,
                                                 positive_mask_1), positive_mask_2)

        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        # replace ReLU with GuidedBackpropReLU
        for idx, module in self.model.features._modules.items():
            if module.__class__.__name__ == 'ReLU':
                self.model.features._modules[idx] = GuidedBackpropReLU.apply

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            output = self.forward(input.cuda())
        else:
            output = self.forward(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        # self.model.features.zero_grad()
        # self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)

        output = input.grad.cpu().data.numpy()
        output = output[0, :, :, :]

        return output


def show_cam_on_image(img, mask):
    sc = ScalarMappable(cmap=cm.jet)
    mask = sc.to_rgba(mask)[:, :, :-1]
    heatmap = np.float32(mask)
    cam = heatmap + np.float32(img)
    cam /= np.max(cam)
    cam = np.nan_to_num(cam)
    imshow(cam)


def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = preprocessed_img.requires_grad_(True)
    return input


if __name__ == '__main__':
    """ python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """

    # Can work with any model, but it assumes that the model has a
    # feature method, and a classifier method,
    # as in the VGG models in torchvision.
    from skimage.io import imshow, imsave
    from skimage import io
    from skimage.transform import resize

    img1 = io.imread(r"C:\Users\Administrator\Desktop\fig\dog.png") / 255
    img2 = io.imread(r"C:\Users\Administrator\Desktop\fig\cat.png") / 255
    img = np.float32(resize(img2, (224, 224)))
    input0 = preprocess_image(img)
    #
    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested index.
    target_index = None
    grad_cam = GradCam(model=models.vgg19(pretrained=True), target_layer_names=["35"], use_cuda=False)
    mask = grad_cam(input0, target_index)

    show_cam_on_image(img, mask)
    # del grad_cam
    # import gc
    # gc.collect()

    gb_model = GuidedBackpropReLUModel(model=models.vgg19(pretrained=True), use_cuda=False)
    #
    gb = gb_model(input0, index=target_index)
    gb = gb.transpose((1, 2, 0))
    # from skimage.io import imshow
    # imshow(gb)
    # #
    cam_mask = np.zeros(gb.shape)
    for i in range(0, gb.shape[2]):
        cam_mask[:, :, i] = mask
    # #
    cam_gb = np.multiply(cam_mask, gb)

    imsave('cam_gb.jpg', cam_gb)

    # del gb_model
    # import gc
    # gc.collect()
