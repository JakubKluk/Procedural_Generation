import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, gpu: int, color_channels: int, latent_vector_size: int, feature_maps_size: int):
        super(Generator, self).__init__()
        self.gpu = gpu
        self.color_channels = color_channels
        self.latent_vector_size = latent_vector_size
        self.feature_maps_size = feature_maps_size
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(latent_vector_size, 128 * feature_maps_size, 4, 1, 0, bias=False),
            nn.BatchNorm2d(128 * feature_maps_size),
            nn.ReLU(True),
            # state size. (feature_maps_size*8) x 4 x 4
            nn.ConvTranspose2d(128 * feature_maps_size, 64 * feature_maps_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * feature_maps_size),
            nn.ReLU(True),
            # state size. (feature_maps_size*4) x 8 x 8
            nn.ConvTranspose2d(64 * feature_maps_size, 32 * feature_maps_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32 * feature_maps_size),
            nn.ReLU(True),
            # state size. (feature_maps_size*2) x 16 x 16
            nn.ConvTranspose2d(32 * feature_maps_size, 16 * feature_maps_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(16 * feature_maps_size),
            nn.ReLU(True),
            # tup tup tup
            nn.ConvTranspose2d(16 * feature_maps_size, 8 * feature_maps_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(8 * feature_maps_size),
            nn.ReLU(True),
            # tap tap tap
            nn.ConvTranspose2d(8 * feature_maps_size, 4 * feature_maps_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(4 * feature_maps_size),
            nn.ReLU(True),
            # czlap czlap czlap
            nn.ConvTranspose2d(4 * feature_maps_size, 2 * feature_maps_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(2 * feature_maps_size),
            nn.ReLU(True),
            # end end end
            nn.ConvTranspose2d(2 * feature_maps_size, feature_maps_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps_size),
            nn.ReLU(True),
            # state size. (feature_maps_size) x 32 x 32
            nn.ConvTranspose2d(feature_maps_size, color_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (color_channels) x 64 x 64
        )

    def forward(self, input_data):
        return self.main(input_data)


class Discriminator(nn.Module):
    def __init__(self, gpu: int, color_channels: int, feature_maps_size: int):
        super(Discriminator, self).__init__()
        self.gpu = gpu
        self.color_channels = color_channels
        self.feature_maps_size = feature_maps_size
        self.main = nn.Sequential(
            # input is (color_channels) x 64 x 64
            nn.Conv2d(color_channels, feature_maps_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (feature_maps_size) x 32 x 32
            nn.Conv2d(feature_maps_size, 2 * feature_maps_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(2 * feature_maps_size),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (feature_maps_size*2) x 16 x 16
            nn.Conv2d(2 * feature_maps_size, 4 * feature_maps_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(4 * feature_maps_size),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (feature_maps_size*4) x 8 x 8
            nn.Conv2d(4 * feature_maps_size, 8 * feature_maps_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(8 * feature_maps_size),
            nn.LeakyReLU(0.2, inplace=True),
            # tup tup tup
            nn.Conv2d(8 * feature_maps_size, 16 * feature_maps_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(16 * feature_maps_size),
            nn.LeakyReLU(0.2, inplace=True),
            # czla czlap czlap
            nn.Conv2d(16 * feature_maps_size, 32 * feature_maps_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32 * feature_maps_size),
            nn.LeakyReLU(0.2, inplace=True),
            # tap tap tap
            nn.Conv2d(32 * feature_maps_size, 64 * feature_maps_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * feature_maps_size),
            nn.LeakyReLU(0.2, inplace=True),
            # end end
            nn.Conv2d(64 * feature_maps_size, 128 * feature_maps_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128 * feature_maps_size),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(128 * feature_maps_size, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input_data):
        return self.main(input_data)
