import torch.nn as nn
import torch.nn.functional as F
from types import FunctionType
import torch

activations = {
    'relu': nn.ReLU(),
    'leaky_relu': nn.LeakyReLU(),
    'tanh': nn.Tanh()
}

class ResidualBlock(nn.Module):
    def __init__(self, kernel_size: int, filters: int, inChannels: int, strides: int, conv_nonlinearity: str):
        '''

        :param kernel_size:
        :param filters:
        :param inChannels:
        :param input_shape: 3 size (channels, height, width)
        :param strides:
        :param conv_nonlinearity:
        '''
        super(ResidualBlock, self).__init__()


        self.conv1 = nn.Conv2d(in_channels=inChannels, out_channels=filters, kernel_size=(1,1), stride=(strides, strides), padding=(0,0))
        self.ln1 = nn.BatchNorm2d(filters)
        self.activation1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=(kernel_size,kernel_size), stride=(1, 1), padding=(1,1))
        self.ln2 = nn.BatchNorm2d(filters)
        self.activation2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=filters, out_channels=4*filters, kernel_size=(1,1), stride=(1, 1), padding=(0,0))
        self.ln3 = nn.BatchNorm2d(4*filters)

        if strides > 1 or (inChannels != 4 * filters):
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=inChannels, out_channels=4*filters, kernel_size=(1,1), stride=(strides,strides), padding=(0,0)),
                nn.BatchNorm2d(4*filters)
            )
        else:
            self.downsample = lambda x : x

        self.out_activation = nn.ReLU()

    def forward(self, inputs):
        X = self.conv1(inputs)
        X = self.ln1(X)
        X = self.activation1(X)

        X = self.conv2(X)
        X = self.ln2(X)
        X = self.activation2(X)

        X = self.conv3(X)
        X = self.ln3(X)

        residual = self.downsample(inputs)
        output = X + residual
        output = self.out_activation(output)
        return output

class Resnet(nn.Module):
    def __init__(self, kernel_size: int, filters: int, inChannels: int, input_shape: tuple, conv_nonlinearity: str, num_class: int):
        super(Resnet, self).__init__()

        # Output (64, 117, 157)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=inChannels, out_channels=filters, kernel_size=(7, 7), stride=(2, 2), padding=(0, 0)),
            nn.BatchNorm2d(filters),
            nn.ReLU()
        )

        # Output (64, 58, 78)
        self.conv2_block1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=(0,0))
        )
        # Output (256, 58, 78)
        self.conv2_block2 = ResidualBlock(kernel_size=kernel_size, filters=filters,
                                          inChannels=filters, strides=1, conv_nonlinearity=conv_nonlinearity)
        # Output (256, 58, 78)
        self.conv2_block3 = ResidualBlock(kernel_size=kernel_size, filters=filters,
                                          inChannels=filters*4, strides=1, conv_nonlinearity=conv_nonlinearity)
        # Output (256, 58, 78)
        self.conv2_block4 = ResidualBlock(kernel_size=kernel_size, filters=filters,
                                          inChannels=filters*4, strides=1, conv_nonlinearity=conv_nonlinearity)

        # Output (512, 29, 39)
        self.conv3_block1 = ResidualBlock(kernel_size=kernel_size, filters=filters*2,
                                          inChannels=filters*4, strides=2, conv_nonlinearity=conv_nonlinearity)
        # Output (512, 29, 39)
        self.conv3_block2 = ResidualBlock(kernel_size=kernel_size, filters=filters*2,
                                          inChannels=filters*8, strides=1, conv_nonlinearity=conv_nonlinearity)
        # Output (512, 29, 39)
        self.conv3_block3 = ResidualBlock(kernel_size=kernel_size, filters=filters*2,
                                          inChannels=filters*8, strides=1, conv_nonlinearity=conv_nonlinearity)
        # Output (512, 29, 39)
        self.conv3_block4 = ResidualBlock(kernel_size=kernel_size, filters=filters * 2,
                                          inChannels=filters * 8, strides=1, conv_nonlinearity=conv_nonlinearity)

        # Output (1024, 15, 20)
        self.conv4_block1 = ResidualBlock(kernel_size=kernel_size, filters=filters*4,
                                          inChannels=filters*8, strides=2, conv_nonlinearity=conv_nonlinearity)
        # Output (1024, 15, 20)
        self.conv4_block2 = ResidualBlock(kernel_size=kernel_size, filters=filters*4,
                                          inChannels=filters*16, strides=1, conv_nonlinearity=conv_nonlinearity)
        # Output (1024, 15, 20)
        self.conv4_block3 = ResidualBlock(kernel_size=kernel_size, filters=filters*4,
                                          inChannels=filters*16, strides=1, conv_nonlinearity=conv_nonlinearity)
        # Output (1024, 15, 20)
        self.conv4_block4 = ResidualBlock(kernel_size=kernel_size, filters=filters * 4,
                                          inChannels=filters * 16, strides=1, conv_nonlinearity=conv_nonlinearity)
        # Output (1024, 15, 20)
        self.conv4_block5 = ResidualBlock(kernel_size=kernel_size, filters=filters * 4,
                                          inChannels=filters * 16, strides=1, conv_nonlinearity=conv_nonlinearity)
        # Output (1024, 15, 20)
        self.conv4_block6 = ResidualBlock(kernel_size=kernel_size, filters=filters * 4,
                                          inChannels=filters * 16, strides=1, conv_nonlinearity=conv_nonlinearity)

        # Output (2048, 8, 10)
        self.conv5_block1 = ResidualBlock(kernel_size=kernel_size, filters=filters*8,
                                          inChannels=filters*16, strides=2, conv_nonlinearity=conv_nonlinearity)
        # Output (2048, 8, 10)
        self.conv5_block2 = ResidualBlock(kernel_size=kernel_size, filters=filters*8,
                                          inChannels=filters*32, strides=1, conv_nonlinearity=conv_nonlinearity)
        # Output (2048, 8, 10)
        self.conv5_block3 = ResidualBlock(kernel_size=kernel_size, filters=filters*8,
                                          inChannels=filters*32, strides=1, conv_nonlinearity=conv_nonlinearity)

        self.fc = nn.Sequential(
            nn.AvgPool2d(kernel_size=(8,10)),
            nn.Flatten(),
            nn.Linear(in_features=2048, out_features=1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=num_class)
        )

    def forward(self, inputs):
        X = self.conv1(inputs)

        X = self.conv2_block1(X)
        X = self.conv2_block2(X)
        X = self.conv2_block3(X)
        X = self.conv2_block4(X)

        X = self.conv3_block1(X)
        X = self.conv3_block2(X)
        X = self.conv3_block3(X)
        X = self.conv3_block4(X)

        X = self.conv4_block1(X)
        X = self.conv4_block2(X)
        X = self.conv4_block3(X)
        X = self.conv4_block4(X)
        X = self.conv4_block5(X)
        X = self.conv4_block6(X)

        X = self.conv5_block1(X)
        X = self.conv5_block2(X)
        X = self.conv5_block3(X)

        X = self.fc(X)
        scores = F.log_softmax(X, dim=-1)

        return scores



# (240, 320, 3)
# model = Resnet(kernel_size=3, filters=64, inChannels=3, input_shape=(3, 240, 320), conv_nonlinearity='relu', num_class=25)
# x = torch.randn((2, 3, 240, 320))
# y = model(x)
# print(y.shape)