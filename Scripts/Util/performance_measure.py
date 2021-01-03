from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from Scripts.NeuralNetwork.GANModel import MapGAN
from Scripts.Util.ConfigFile import GENERATOR_LEARNING_RATE, DISCRIMINATOR_LEARNING_RATE, BETAS
from Scripts.InputGenerator.Generators import PerlinNoiseGenerator
import numpy as np
from datetime import datetime

import torch.optim as optim
import torch.nn as nn
import torch

# Make data.
device = torch.device("cuda:0")
# optimisers
opt_g = optim.Adam
opt_d = optim.Adam
opt_g_params = {"lr": GENERATOR_LEARNING_RATE, "betas": BETAS}
opt_d_params = {"lr": DISCRIMINATOR_LEARNING_RATE, "betas": BETAS}
# Initialize BCELoss function
criterion = nn.BCELoss()

gan = MapGAN()
gan.load_gan()

# torch.manual_seed(np.random.randint(1000))
torch.manual_seed(314)
np.random.seed(127)
noise = torch.randn(100, gan.generator.latent_vector_size, 1, 1, device=device)
start_gan = datetime.now()
Z = gan.generator(noise).cpu().detach().numpy()
end_gan = datetime.now()
gen = PerlinNoiseGenerator((512, 512), octaves=8)
start_perl = datetime.now()
for i in range(100):
    temp = gen.generate_input_data(land_percentage=0.4)
end_perl = datetime.now()
print("The time needed by the GAN network to generate 100 maps was: " + str((end_gan - start_gan).total_seconds()
                                                                            * 1000) + "ms")
print("The time needed to generate 100 maps using Perlin's Noise was: " + str((end_perl - start_perl
                                                                               ).total_seconds() * 1000) + "ms")
