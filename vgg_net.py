import torch

from torch import nn

architecture = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, "M", 512, 512, 512, "M"]


class VGGNet(nn.Module):
    def __init__(self, input_channels, class_num=1000):
        super(VGGNet, self).__init__()
        self.input_channels = input_channels
        self.conv_layers = self.conv_constractor()
        self.FC = nn.Sequential(nn.Linear(512*7*7, 4096),
                                nn.ReLU(),
                                nn.Dropout(p=0.5),
                                nn.Linear(4096, 4096),
                                nn.ReLU(),
                                nn.Dropout(p=0.5),
                                nn.Linear(4096, class_num))

    def forward(self, x):

        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.FC(x)
        return x

    def conv_constractor(self):

        input_c = self.input_channels
        conv_layers = []

        for layer in architecture:

            if isinstance(layer, int):
                conv_layers += [nn.Conv2d(in_channels=input_c, out_channels=layer, stride=(1, 1),
                                          kernel_size=(3, 3), padding=(1, 1)),
                                nn.BatchNorm2d(layer),
                                nn.ReLU()]
                input_c = layer
            else:
                conv_layers += [nn.MaxPool2d(stride=(2, 2), kernel_size=(2, 2))]
        return nn.Sequential(*conv_layers)