from __future__ import print_function

import json
import pickle
import random
from pathlib import Path
from typing import Union, Tuple, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils

from Scripts.NeuralNetwork.NetworkModels import Generator, Discriminator
from Scripts.NeuralNetwork.NetworkUtils import create_data_loader, weights_init
from Scripts.Util.ConfigFile import BATCH_SIZE, WORKERS, IMAGE_ROOT_PATH, IMAGE_SIZE, SEED, COLOR_CHANNELS, GPU, \
    GENERATOR_LEARNING_RATE, DISCRIMINATOR_LEARNING_RATE, BETAS, LATENT_VECTOR_SIZE, DISCRIMINATOR_FEATURE_MAPS_SIZE, \
    GENERATOR_FEATURE_MAPS_SIZE, SAVE_MODEL_PATH, WAIT_FOR_UPDATE, NUMBER_OF_EPOCHS


class MapGAN:
    def __init__(self, logging_path: Union[Path, str] = SAVE_MODEL_PATH, dataroot: str = IMAGE_ROOT_PATH,
                 batch_size: int = BATCH_SIZE, workers: int = WORKERS, seed: int = SEED,
                 image_size: Union[Tuple[int, int], int] = IMAGE_SIZE, color_channels: int = COLOR_CHANNELS,
                 gpu: int = GPU, batch_waiting: int = WAIT_FOR_UPDATE):
        # defining data loading settings and some basic processing settings
        self.logging_path = logging_path if isinstance(logging_path, Path) else Path(logging_path)
        self.dataroot = dataroot
        self.batch_size = batch_size
        self.workers = workers
        self.image_size = image_size
        self.color_channels = color_channels
        self.gpu = gpu
        self.batch_waiting = batch_waiting
        # determining the device
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and gpu > 0) else "cpu")
        # creating discriminator and generator networks as None in order to define them later
        self.generator = None
        self.discriminator = None
        # creating optimizers as None in order to define them later
        self.gen_optimizer = None
        self.dis_optimizer = None
        self.G_losses = []
        self.D_losses = []
        self.G_perfo = []
        self.D_perfo = []
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

    def save_gan(self, path: Optional[Path] = None):
        if path is None:
            path = self.logging_path
        path = path if isinstance(path, Path) else Path(path)
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

    def load_gan(self, path: Optional[Path] = None):
        if path is None:
            path = self.logging_path
        path = path if isinstance(path, Path) else Path(path)
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

    def log_training_progress(self, iteration: int, data_length: int, epoch: int, epochs: int, D_loss: torch.Tensor,
                              G_loss: torch.Tensor, D_accu: float, G_accu_1: float, G_accu_2: float):
        print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
              % (epoch, epochs, iteration, data_length,
                 D_loss.item(), G_loss.item(), D_accu, G_accu_1, G_accu_2))

        # Save Losses for plotting later
        if not (self.logging_path / "training_log.csv").exists():
            with open(self.logging_path / "training_log.csv", "a") as log:
                log.write("Iteration,DataLength,Epoch,AllEpochs,DiscriminatorLoss,GeneratorLoss,DiscriminatorAccuracy,"
                          "GeneratorAccuracy_1,GeneratorAccuracy_2\n")
                log.write(f"{iteration},{data_length},{epoch},{epochs},{D_loss},{G_loss},{D_accu},{G_accu_1},{G_accu_2}"
                          + "\n")
        else:
            with open(self.logging_path / "training_log.csv", "a") as log:
                log.write(f"{iteration},{data_length},{epoch},{epochs},{D_loss},{G_loss},{D_accu},{G_accu_1},{G_accu_2}"
                          + "\n")

    def log_training_results(self, epoch: int, iteration: int, fixed_noise: torch.Tensor):
        # Check how the generator is doing by saving G's output on fixed_noise
        with torch.no_grad():
            fake = self.generator(fixed_noise).detach().cpu()

            imags = vutils.make_grid(fake, padding=2, normalize=True, nrow=4)
            f = plt.figure()
            plt.axis("off")
            plt.title("Fake Images")
            plt.imshow(np.transpose(imags, (1, 2, 0)))
            f.savefig(self.logging_path / "generated_epoch_{0}_iteration_{1}.png".format(epoch, iteration))
            plt.close(f)

        G_average = [0] * 11
        D_average = [0] * 11
        G_average += list(np.convolve(self.G_losses, np.ones(12), 'valid') / 12)
        D_average += list(np.convolve(self.D_losses, np.ones(12), 'valid') / 12)
        f = plt.figure()
        plt.title("Generator and Discriminator Average Loss During Training")
        plt.plot(G_average, label="G")
        plt.plot(D_average, label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        f.savefig(self.logging_path / "average_losses_epoch_{0}_iteration_{1}.png".format(epoch, iteration))
        plt.close(f)


        f = plt.figure()
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(self.G_losses, label="G")
        plt.plot(self.D_losses, label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        f.savefig(self.logging_path / "losses_epoch_{0}_iteration_{1}.png".format(epoch, iteration))
        plt.close(f)

        f = plt.figure()
        plt.title("Generator and Discriminator Performance During Training")
        plt.plot(self.G_perfo, label="G")
        plt.plot(self.D_perfo, label="D")
        plt.xlabel("iterations")
        plt.ylabel("Performance")
        plt.legend()
        f.savefig(self.logging_path / "performance_epoch_{0}_iteration_{1}.png".format(epoch, iteration))
        plt.close(f)

        G_average = [0] * 11
        D_average = [0] * 11
        G_average += list(np.convolve(self.G_perfo, np.ones(12), 'valid') / 12)
        D_average += list(np.convolve(self.D_perfo, np.ones(12), 'valid') / 12)
        f = plt.figure()
        plt.title("Generator and Discriminator Average Performance During Training")
        plt.plot(G_average, label="G")
        plt.plot(D_average, label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        f.savefig(self.logging_path / "average_performance_epoch_{0}_iteration_{1}.png".format(epoch, iteration))
        plt.close(f)

    def train(self, epochs: int, criterion, print_training_stats: int = 50, save_generator_output: int = 500):
        # Create batch of latent vectors that we will use to visualize
        #  the progression of the generator
        fixed_noise = torch.randn(16, self.generator.latent_vector_size, 1, 1, device=self.device)
        # number of fixed noise images is independent from the batch size, because it is used only for a progress
        # visualization

        # Establish convention for real and fake labels during training
        real_label = 1.
        fake_label = 0.

        # initializing dataloader
        dataloader = create_data_loader(self.dataroot, self.batch_size, self.workers, self.image_size)
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
                # label = torch.full((b_size,), real_label, dtype=torch.float, device=self.device)
                label = 0.08 * torch.randn((b_size,), dtype=torch.float, device=self.device) + 0.9
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
                # label = 0.2 * torch.randn((b_size,), dtype=torch.float, device=self.device) + 0.3
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
                # if ((i % self.batch_waiting) == (self.batch_waiting - 1)) or (i == (len(dataloader) - 1)):
                #     self.dis_optimizer.step()
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
                # if ((i % self.batch_waiting) == (self.batch_waiting - 1)) or (i == (len(dataloader) - 1)):
                #     self.gen_optimizer.step()
                self.gen_optimizer.step()

                # ====================================================================
                # (3) Output training stats

                if (i % print_training_stats) == 0 and i != 0:
                    self.log_training_progress(i, len(dataloader), epoch, epochs, errD, errG, D_x, D_G_z1, D_G_z2)
                    self.G_losses.append(errG.item())
                    self.D_losses.append(errD.item())
                    self.G_perfo.append(D_G_z2)
                    self.D_perfo.append(D_x)
                if ((i % save_generator_output == 0) and i != 0) or ((epoch == epochs - 1) and (i == len(dataloader) - 1)):
                    self.log_training_results(epoch, i, fixed_noise)


if __name__ == '__main__':

    # optimisers
    opt_g = optim.Adam
    opt_d = optim.Adam
    opt_g_params = {"lr": GENERATOR_LEARNING_RATE, "betas": BETAS}
    opt_d_params = {"lr": DISCRIMINATOR_LEARNING_RATE, "betas": BETAS}
    # Initialize BCELoss function
    criterion = nn.BCELoss()

    gan = MapGAN()
    gan.init_generator(COLOR_CHANNELS, LATENT_VECTOR_SIZE, GENERATOR_FEATURE_MAPS_SIZE, opt_g, opt_g_params)
    gan.init_discriminator(COLOR_CHANNELS, DISCRIMINATOR_FEATURE_MAPS_SIZE, opt_d, opt_d_params)
    gan.train(NUMBER_OF_EPOCHS, criterion, print_training_stats=100, save_generator_output=2500)
    gan.save_gan(Path(SAVE_MODEL_PATH))
    # gan.load_gan(Path(SAVE_MODEL_PATH))
    print("End")
