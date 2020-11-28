from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from multiprocessing import freeze_support
from pathlib import Path
import json
import pickle

from typing import Union, Tuple, Dict

from Scripts.NeuralNetwork.NetworkUtils import create_data_loader, weights_init
from Scripts.Util.ConfigFile import BATCH_SIZE, WORKERS, IMAGE_ROOT_PATH, IMAGE_SIZE, SEED, COLOR_CHANNELS, GPU, \
    LEARNING_RATE, BETAS, LATENT_VECTOR_SIZE, DISCRIMINATOR_FEATURE_MAPS_SIZE, GENERATOR_FEATURE_MAPS_SIZE, \
    SAVE_MODEL_PATH
from Scripts.NeuralNetwork.NetworkModels import Generator, Discriminator


class MapGAN:
    def __init__(self, dataroot: str = IMAGE_ROOT_PATH, batch_size: int = BATCH_SIZE, workers: int = WORKERS,
                 seed: int = SEED, image_size: Union[Tuple[int, int], int] = IMAGE_SIZE,
                 color_channels: int = COLOR_CHANNELS, gpu: int = GPU):
        # defining data loading settings and some basic processing settings
        self.dataroot = dataroot
        self.batch_size = batch_size
        self.workers = workers
        self.image_size = image_size
        self.color_channels = color_channels
        self.gpu = gpu
        # determining the device
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and gpu > 0) else "cpu")
        # creating discriminator and generator networks as None in order to define them later
        self.generator = None
        self.discriminator = None
        # creating optimizers as None in order to define them later
        self.gen_optimizer = None
        self.dis_optimizer = None
        # initialization of program seed
        seed = seed if seed is not None else random.randint(1, 10000)
        random.seed(seed)
        torch.manual_seed(seed)

    def init_generator(self, color_channels: int, latent_vector_size: int, feature_maps_size: int,
                       optimizer: torch.optim.Optimizer, optimizer_params: Dict):
        self.generator = Generator(gpu=self.gpu, color_channels=color_channels, latent_vector_size=latent_vector_size,
                                   feature_maps_size=feature_maps_size).to(self.device)
        self.generator.apply(weights_init)
        self.gen_optimizer = optimizer(self.generator.parameters(), **optimizer_params)

    def init_discriminator(self, color_channels: int, feature_maps_size: int, optimizer: torch.optim.Optimizer,
                           optimizer_params: Dict):
        self.discriminator = Discriminator(gpu=self.gpu, color_channels=color_channels,
                                           feature_maps_size=feature_maps_size).to(self.device)
        self.discriminator.apply(weights_init)
        self.dis_optimizer = optimizer(self.discriminator.parameters(), **optimizer_params)

    def save_gan(self, path: Path):
        if not path.exists():
            path.mkdir(parents=True)
        torch.save(self.generator.state_dict(), path / "generator.pth")
        generator_init = {"gpu": self.generator.gpu, "color_channels": self.generator.color_channels,
                          "latent_vector_size": self.generator.latent_vector_size,
                          "feature_maps_size": self.generator.feature_maps_size}
        with open(path / "generator_init.json", "w") as fp:
            json.dump(generator_init, fp)
        torch.save(self.discriminator.state_dict(), path / "discriminator.pth")
        discriminator_init = {"gpu": self.discriminator.gpu, "color_channels": self.discriminator.color_channels,
                              "feature_maps_size": self.discriminator.feature_maps_size}
        with open(path / "discriminator_init.json", "w") as fp:
            json.dump(discriminator_init, fp)
        with open(path / "discriminator_optimizator.pkl", "wb") as fp:
            pickle.dump(self.dis_optimizer, fp)
        with open(path / "generator_optimizator.pkl", "wb") as fp:
            pickle.dump(self.gen_optimizer, fp)

    def load_gan(self, path: Path):
        with open(path / "generator_init.json", "r") as fp:
            generator_init = json.load(fp)
        with open(path / "discriminator_init.json", "r") as fp:
            discriminator_init = json.load(fp)
        self.generator = Generator(**generator_init).to(self.device)
        self.generator.load_state_dict(torch.load(path / "generator.pth"))
        self.discriminator = Discriminator(**discriminator_init).to(self.device)
        self.discriminator.load_state_dict(torch.load(path / "discriminator.pth"))
        with open(path / "generator_optimizator.pkl", "rb") as fp:
            self.gen_optimizer = pickle.load(fp)
        with open(path / "discriminator_optimizator.pkl", "rb") as fp:
            self.dis_optimizer = pickle.load(fp)

    def train(self, epochs: int, criterion, print_training_stats: int = 50, save_generator_output: int = 500):
        # Create batch of latent vectors that we will use to visualize
        #  the progression of the generator
        fixed_noise = torch.randn(64, self.generator.latent_vector_size, 1, 1, device=self.device)
        # TODO check this 64 value

        # Establish convention for real and fake labels during training
        real_label = 1.
        fake_label = 0.

        # keeping the progress of the training
        img_list = []
        G_losses = []
        D_losses = []
        iters = 0

        # initializing dataloader
        dataloader = create_data_loader(self.dataroot, self.batch_size, self.workers)
        print("Starting the training loop.")
        for epoch in range(epochs):
            for i, data in enumerate(dataloader):
                # ====================================================================
                # Updating Discriminator
                # clearing gradients
                self.discriminator.zero_grad()
                # Format batch
                real_cpu = data[0].to(self.device)
                b_size = real_cpu.size(0)
                label = torch.full((b_size,), real_label, dtype=torch.float, device=self.device)
                # Forward pass real batch through D
                output = self.discriminator(real_cpu).view(-1)
                # Calculate loss on all-real batch
                errD_real = criterion(output, label)
                # Calculate gradients for D in backward pass
                errD_real.backward()
                D_x = output.mean().item()

                # Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(b_size, self.generator.latent_vector_size, 1, 1, device=self.device)
                # Generate fake image batch with G
                fake = self.generator(noise)
                label.fill_(fake_label)
                # Classify all fake batch with D
                # .detach() is used not to update generator's weights
                output = self.discriminator(fake.detach()).view(-1)
                # Calculate D's loss on the all-fake batch
                errD_fake = criterion(output, label)
                # Calculate the gradients for this batch
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                # Add the gradients from the all-real and all-fake batches
                errD = errD_real + errD_fake
                # Update D
                self.dis_optimizer.step()

                # ====================================================================
                # (2) Updating Generator
                # clearing gradients
                self.generator.zero_grad()
                label.fill_(real_label)  # fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = self.discriminator(fake).view(-1)
                # Calculate G's loss based on this output
                errG = criterion(output, label)
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()
                # Update G
                self.gen_optimizer.step()

                # ====================================================================
                # (3) Output training stats
                if i % print_training_stats == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                          % (epoch, epochs, i, len(dataloader),
                             errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

                # Save Losses for plotting later
                G_losses.append(errG.item())
                D_losses.append(errD.item())

                # Check how the generator is doing by saving G's output on fixed_noise
                if (iters % save_generator_output == 0) or ((epoch == epochs - 1) and (i == len(dataloader) - 1)):
                    with torch.no_grad():
                        fake = self.generator(fixed_noise).detach().cpu()
                    img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

                iters += 1


if __name__ == '__main__':

    # optimisers
    opt_g = optim.Adam
    opt_d = optim.Adam
    opt_g_params = {"lr": LEARNING_RATE, "betas": BETAS}
    opt_d_params = {"lr": LEARNING_RATE, "betas": BETAS}
    # Initialize BCELoss function
    criterion = nn.BCELoss()

    gan = MapGAN()
    gan.init_generator(COLOR_CHANNELS, LATENT_VECTOR_SIZE, GENERATOR_FEATURE_MAPS_SIZE, opt_g, opt_g_params)
    gan.init_discriminator(COLOR_CHANNELS, DISCRIMINATOR_FEATURE_MAPS_SIZE, opt_d, opt_d_params)
    gan.save_gan(Path(SAVE_MODEL_PATH))
    gan.train(5, criterion)
    print('xd')

    # # Plot some training images
    # real_batch = next(iter(dataloader))
    # plt.figure(figsize=(8,8))
    # plt.axis("off")
    # plt.title("Training Images")
    # plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1,2,0)))
    # plt.savefig(dataroot + "\\xd.png")
    #
    #
    # plt.figure(figsize=(10, 5))
    # plt.title("Generator and Discriminator Loss During Training")
    # plt.plot(G_losses, label="G")
    # plt.plot(D_losses, label="D")
    # plt.xlabel("iterations")
    # plt.ylabel("Loss")
    # plt.legend()
    # plt.show()
    #
    # # Grab a batch of real images from the dataloader
    # real_batch = next(iter(dataloader))
    #
    # # Plot the real images
    # plt.figure(figsize=(15, 15))
    # plt.subplot(1, 2, 1)
    # plt.axis("off")
    # plt.title("Real Images")
    # plt.imshow(
    #     np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))
    #
    # # Plot the fake images from the last epoch
    # plt.subplot(1, 2, 2)
    # plt.axis("off")
    # plt.title("Fake Images")
    # plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
    # plt.show()
