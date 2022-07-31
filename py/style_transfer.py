from __future__ import print_function

import sys

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import py2og


def style_change(style_img_path, content_img_path, sid):
    # print(style_img_path, content_img_path)
    # 选择要运行代码的设备,torch.cuda相较于cpu拥有更佳的运行速度
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 图像大小(读入后都转换为imsize*imsize大小的图片）
    imsize = 512 if torch.cuda.is_available() else 128

    # 用于转换读入图片的大小，并将图片转换为张量(tensor)
    loader = transforms.Compose([
        transforms.Resize([imsize, imsize]),
        transforms.ToTensor()])

    # 加载图片
    def image_loader(image_name):
        image = Image.open(image_name)
        image = loader(image).unsqueeze(0)
        return image.to(device, torch.float)

    # 读入风格图片和内容图片
    style_img = image_loader(style_img_path)
    content_img = image_loader(content_img_path)

    # 判断风格图片和内容图片大小是否一致
    assert style_img.size() == content_img.size(), \
        "we need to import style and content images of the same size"

    # load的逆操作，用于将tensor转化为图片
    unloader = transforms.ToPILImage()

    # plt的动态绘图模式
    plt.ion()

    # 打印图片,传入一个tensor后unload成图片并打印
    def imshow(tensor, title=None):
        image = tensor.cpu().clone()
        image = image.squeeze(0)
        image = unloader(image)
        plt.imshow(image)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)

    # plt.figure()
    # imshow(style_img, title='Style Image')
    #
    # plt.figure()
    # imshow(content_img, title='Content Image')

    # 计算输入图片与内容图片之间差异的损失函数
    class ContentLoss(nn.Module):

        def __init__(self, target, ):
            super(ContentLoss, self).__init__()
            self.target = target.detach()

        def forward(self, input):
            # 使用torch中定义好的均方差损失函数
            self.loss = F.mse_loss(input, self.target)
            return input

    # 计算输入的gram矩阵
    def gram_matrix(input):
        a, b, c, d = input.size()  # 对于一张图片，a为1，b为卷积核数量，c、d为图片宽和高
        features = input.view(a * b, c * d)
        # gram矩阵即为features乘以features的转置
        G = torch.mm(features, features.t())
        return G.div(a * b * c * d)

    # 利用上面的gram矩阵，计算输入图片与风格图片之间差异的损失函数
    class StyleLoss(nn.Module):

        def __init__(self, target_feature):
            super(StyleLoss, self).__init__()
            self.target = gram_matrix(target_feature).detach()

        def forward(self, input):
            G = gram_matrix(input)
            self.loss = F.mse_loss(G, self.target)
            return input

    # 预定义好的VGG19卷积神经网络
    cnn = models.vgg19(pretrained=True).features.to(device).eval()
    # 归一化平均值和方差
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    # 创建一个模块来规范化输入图像，以便轻松地将其放入模型进行训练
    class Normalization(nn.Module):
        def __init__(self, mean, std):
            super(Normalization, self).__init__()
            self.mean = torch.tensor(mean).view(-1, 1, 1)
            self.std = torch.tensor(std).view(-1, 1, 1)

        def forward(self, img):
            # normalize img
            return (img - self.mean) / self.std

    # 使用输入图像与内容图像的conv_4层结果来计算损失
    content_layers_default = ['conv_4']
    # 使用输入图像与风格图像的以下五层结果来计算损失
    style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    # 计算损失
    def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                                   style_img, content_img,
                                   content_layers=content_layers_default,
                                   style_layers=style_layers_default):
        # 模型归一化
        normalization = Normalization(normalization_mean, normalization_std).to(device)
        # 损失值
        content_losses = []
        style_losses = []

        model = nn.Sequential(normalization)

        i = 0  # 记录卷积层的数量

        # 为每一层命名，便于通过名字拿到每一层的结果
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)
            # 使用输入图像与内容图像的conv_4层结果来计算损失
            if name in content_layers:
                target = model(content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)
            # 使用输入图像与风格图像的对应层结果来计算损失
            if name in style_layers:
                target_feature = model(style_img).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)

        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break

        model = model[:(i + 1)]

        return model, style_losses, content_losses

    input_img = content_img.clone()
    # plt.figure()
    # imshow(input_img, title='Input Image')

    # 得到优化器
    def get_input_optimizer(input_img):
        optimizer = optim.LBFGS([input_img])
        return optimizer

    def run_style_transfer(cnn, normalization_mean, normalization_std,
                           content_img, style_img, input_img, num_steps=250,
                           style_weight=1000000, content_weight=1):
        print('Building the style transfer model..')
        model, style_losses, content_losses = get_style_model_and_losses(cnn,
                                                                         normalization_mean, normalization_std,
                                                                         style_img, content_img)

        # 优化输入图片而不是模型参数，所以需要更新所有的requires_grad字段
        input_img.requires_grad_(True)
        model.requires_grad_(False)

        optimizer = get_input_optimizer(input_img)

        print('Optimizing..')
        # 训练迭代次数
        run = [0]
        while run[0] <= num_steps:

            def closure():
                with torch.no_grad():
                    input_img.clamp_(0, 1)

                optimizer.zero_grad()
                model(input_img)
                style_score = 0
                content_score = 0

                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss

                style_score *= style_weight
                content_score *= content_weight

                loss = style_score + content_score
                # 根据损失值反馈
                loss.backward()

                run[0] += 1
                # 每训练50次输出损失值
                if run[0] % 10 == 0:
                    print("run {}:".format(run))
                    print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                        style_score.item(), content_score.item()))
                    print()

                return style_score + content_score

            optimizer.step(closure)

        with torch.no_grad():
            input_img.clamp_(0, 1)

        return input_img

    output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                content_img, style_img, input_img)
    plt.figure()
    imshow(output, title='Output Image')
    plt.savefig("./output/"+ sid + "_out.jpg")
    # output = unloader(output)
    # cv2.imwrite("./output/cv2result.jpg", output)


'''
argv[1]: 风格图路径
argv[2]: 内容图路径
argv[3]：提交编号
风格图和内容图尺寸应一致
'''
if __name__ == "__main__":
    style_change(py2og.inputFilePath + sys.argv[1], py2og.inputFilePath + sys.argv[2], sys.argv[3])
    py2og.updateDBwithFinishSignal(sys.argv[3])
