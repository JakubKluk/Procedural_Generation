from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from Scripts.NeuralNetwork.GANModel import MapGAN
from Scripts.Util.ConfigFile import GENERATOR_LEARNING_RATE, DISCRIMINATOR_LEARNING_RATE, BETAS
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

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
noise = torch.randn(1, gan.generator.latent_vector_size, 1, 1, device=device)
Z = gan.generator(noise).cpu().detach().numpy().reshape(gan.generator.latent_vector_size, gan.generator.latent_vector_size)
Z = (255 * (Z - Z.min()) / (Z.max() - Z.min())).astype(np.uint8)

f2 = plt.figure()
plt.imshow(Z, cmap='gray')
plt.axis('off')
plt.show()
plt.close(f2)

fig = plt.figure()
ax = fig.gca(projection='3d')

X, Y = np.meshgrid(range(Z.shape[0]), range(Z.shape[1]))
Y = Y[::-1]

# make the panes transparent
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# make the grid lines transparent
ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
ax.set_axis_off()

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, antialiased=False)

# Customize the z axis.
ax.set_zlim(0, 255)
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()