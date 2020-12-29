from Scripts.NeuralNetwork.GANModel import MapGAN
from Scripts.Util.ConfigFile import GENERATOR_LEARNING_RATE, DISCRIMINATOR_LEARNING_RATE, BETAS
from Scripts.InputGenerator.Generators import PerlinNoiseGenerator
from Scripts.DataProcessing.colorMaps import color_map
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.utils as vutils

if __name__ == "__main__":
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
    torch.manual_seed(127)
    np.random.seed(127)
    noise = torch.randn(9, gan.generator.latent_vector_size, 1, 1, device=device)
    image = gan.generator(noise).cpu().detach()
    colored_image = torch.empty((image.shape[0], 3, image.shape[2], image.shape[3]))
    for i in range(image.shape[0]):
        temp = image[i].numpy()
        temp = (255 * (temp - temp.min()) / (temp.max() - temp.min())).astype(np.uint8)
        tud = color_map(temp).astype(np.float32)
        temp = torch.tensor(np.transpose(color_map(temp).astype(np.float32).reshape(128, 128, 3), (2, 0, 1)))
        # temp = torch.transpose(temp, 1, 3).rot90(k=2, dims=(2, 3))
        colored_image[i] = temp
    imags = vutils.make_grid(image, padding=4, normalize=True, nrow=3)
    colored_imags = vutils.make_grid(colored_image, padding=4, normalize=True, nrow=3)
    f = plt.figure()
    plt.axis("off")
    plt.title("GAN images")
    plt.imshow(np.transpose(imags, (1, 2, 0)))
    plt.savefig("D:\\Praca_inzynierska_projekt\\imags.png")
    plt.close(f)

    # ==============================================================================================================
    # Generating Perlin's Noise
    generator = PerlinNoiseGenerator((1024, 1024), octaves=8)
    perlin_noises = []
    colored_perlin = torch.empty((image.shape[0], 3, image.shape[2], image.shape[3]))
    for i in range(9):
        if i > 4:
            temp = generator.generate_continento(land_percentage=(np.random.randint(30, 75)) / 100)
        else:
            temp = generator.generate_input_data(land_percentage=(np.random.randint(30, 75)) / 100)
        temp = Image.fromarray(temp).resize((128, 128))
        perlin_noises.append(np.array(temp).astype(np.float32).reshape(1, 128, 128))
        color_temp = np.transpose(color_map(np.array(temp).reshape(1, 128, 128)).astype(np.float32).reshape(128, 128, 3), (2, 0, 1))
        colored_perlin[i] = torch.tensor(color_temp)
    perlin_noises = torch.tensor(perlin_noises)
    perlin_imags = vutils.make_grid(perlin_noises, padding=2, normalize=True, nrow=3)
    colored_perlin_imags = vutils.make_grid(colored_perlin, padding=2, normalize=True, nrow=3)
    f = plt.figure()
    plt.axis("off")
    plt.title("Perlin's Noise images")
    plt.imshow(np.transpose(perlin_imags, (1, 2, 0)))
    plt.savefig("D:\\Praca_inzynierska_projekt\\perlin_imags.png")
    plt.close(f)

    # ================================================================================================================
    # Creating merged image
    f = plt.figure()
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("GAN images")
    plt.imshow(np.transpose(imags, (1, 2, 0)))

    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Perlin's Noise images")
    plt.imshow(np.transpose(perlin_imags, (1, 2, 0)))
    plt.savefig("D:\\Praca_inzynierska_projekt\\merged.png")
    plt.close(f)

    # Merged and colored
    f = plt.figure()
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("GAN images")
    plt.imshow(np.transpose(colored_imags, (1, 2, 0)))

    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Perlin's Noise images")
    plt.imshow(np.transpose(colored_perlin_imags, (1, 2, 0)))
    plt.savefig("D:\\Praca_inzynierska_projekt\\merged_color.png")
    plt.close(f)


    # image_np = image.numpy().reshape(128, 128)
    # img_min = image_np.min()
    # img_max = image_np.max()
    # image_np = (255 * (image_np - img_min) / (img_max - img_min + 1e-5)).astype(np.uint8)
    # image = Image.fromarray(image_np)
    #
    #
    # # plt.show()
    # image.save("D:\\Praca_inzynierska_projekt\\test_image_before_resize.png")
    # image = image.resize((1024, 1024))
    # image.save("D:\\Praca_inzynierska_projekt\\test_image_after_resize.png")
