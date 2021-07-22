import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(
            self,
            in_planes: int,
            out_planes: int,
            kernel_size: int,
            stride: int,
            padding: int,
    ):
        super(BasicBlock, self).__init__()
        self.layers = nn.Sequential()
        self.layers.add_module(
            "Conv",
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, bias=False),
        )
        self.layers.add_module("BatchNorm", nn.BatchNorm2d(out_planes))
        self.layers.add_module("ReLU", nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        return self.layers(x)


class TransposeBlock(nn.Module):
    def __init__(
            self,
            in_planes: int,
            out_planes: int,
            kernel_size: int,
            stride: int,
            padding: int,
    ):
        super(TransposeBlock, self).__init__()
        self.layers = nn.Sequential()
        self.layers.add_module(
            "TransposeConv",
            nn.ConvTranspose2d(
                in_planes, out_planes, kernel_size, stride, padding, bias=False
            ),
        )
        self.layers.add_module("BatchNorm", nn.BatchNorm2d(out_planes))
        self.layers.add_module("ReLU", nn.ReLU(inplace=True))

    def forward(self, x):
        return self.layers(x)


class ContextEncoder(nn.Module):
    def __init__(self, bottleneck_dim=2048, in_channels=3, img_size=64):
        super(ContextEncoder, self).__init__()
        assert img_size in (
            32,
            64,
            128,
        ), "Only image sizes of 32, 64 or 128 are supported"
        len_channel_sizes = 4 if img_size in (64, 128) else 3
        nChannels = [64 * 2 ** i for i in range(len_channel_sizes)]

        self.conv1 = nn.Conv2d(
            in_channels, nChannels[0], kernel_size=4, stride=2, padding=1, bias=False
        )
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)

        blocks = []
        if img_size == 128:
            blocks.append(
                BasicBlock(
                    nChannels[0], nChannels[0], kernel_size=4, stride=2, padding=1
                )
            )

        blocks.extend(
            [
                BasicBlock(
                    nChannels[i], nChannels[i + 1], kernel_size=4, stride=2, padding=1
                )
                for i in range(len_channel_sizes - 1)
            ]
        )
        self.blocks = nn.Sequential(*blocks)

        self.conv_bottleneck = nn.Conv2d(nChannels[-1], bottleneck_dim, kernel_size=4)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.blocks(x)
        return self.conv_bottleneck(x)


class ContextDecoder(nn.Module):
    def __init__(self, bottleneck_dim=2048, out_channels=3, out_size: int = 64):
        super(ContextDecoder, self).__init__()
        assert out_size in (
            16,
            32,
            64,
            128,
        ), "Only output sizes of 32, 64 or 128 are supported. For image size of 32 only random masking is supported."
        len_channel_sizes = 4 if out_size in (64, 128) else 3
        nChannels = [64 * 2 ** i for i in range(len_channel_sizes - 1, -1, -1)]

        self.bn1 = nn.BatchNorm2d(bottleneck_dim)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.bottleneck_block = TransposeBlock(
            bottleneck_dim, nChannels[0], kernel_size=4, stride=1, padding=0
        )

        blocks = [
            TransposeBlock(
                nChannels[i], nChannels[i + 1], kernel_size=4, stride=2, padding=1
            )
            for i in range(len_channel_sizes - 1)
        ]
        if out_size == 128:
            blocks.append(
                TransposeBlock(
                    nChannels[-1], nChannels[-1], kernel_size=4, stride=2, padding=1
                )
            )
        self.blocks = nn.Sequential(*blocks)

        self.final_conv = nn.ConvTranspose2d(
            nChannels[-1], out_channels, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu1(self.bn1(x))
        x = self.bottleneck_block(x)
        x = self.blocks(x)
        return self.tanh(self.final_conv(x))


class ContextGenerator(nn.Module):
    def __init__(
            self,
            bottleneck_dim: int = 2048,
            channels: int = 3,
            img_size: int = 64,
            out_size: int = 64,
    ):
        super(ContextGenerator, self).__init__()
        self.encoder = ContextEncoder(
            bottleneck_dim=bottleneck_dim, in_channels=channels, img_size=img_size
        )
        self.decoder = ContextDecoder(
            bottleneck_dim=bottleneck_dim, out_channels=channels, out_size=out_size
        )

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)


class ContextDiscriminator(nn.Module):
    def __init__(self, in_channels=3, input_size: int = 64):
        super(ContextDiscriminator, self).__init__()
        assert input_size in (
            32,
            64,
            128,
        ), "Only image sizes of 32, 64 or 128 are supported"
        len_channel_sizes = 4 if input_size in (64, 128) else 3
        nChannels = [64 * 2 ** i for i in range(len_channel_sizes)]

        self.conv1 = nn.Conv2d(
            in_channels, nChannels[0], kernel_size=4, stride=2, padding=1, bias=False
        )
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)

        blocks = []
        if input_size == 128:
            blocks.append(
                BasicBlock(
                    nChannels[0], nChannels[0], kernel_size=4, stride=2, padding=1
                )
            )

        blocks.extend(
            [
                BasicBlock(
                    nChannels[i], nChannels[i + 1], kernel_size=4, stride=2, padding=1
                )
                for i in range(len_channel_sizes - 1)
            ]
        )
        self.blocks = nn.Sequential(*blocks)

        self.final = nn.Conv2d(nChannels[-1], 1, kernel_size=4, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.blocks(x)
        x = self.sigmoid(self.final(x))
        return x.squeeze()
