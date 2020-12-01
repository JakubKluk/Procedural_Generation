from pathlib import Path
import numpy as np

JUMP_SIZE = 128
MAP_HEIGHT = 1024
MAP_WIDTH = 1024

DATA_PATH = Path("../../OriginalData")
PROCESSED_PATH = Path("../../ProcessedData/")

MIN_LAND_PERCENTAGE = 30

# DEFAULT_COLORING = {"breakpoints": [0, 1, 128, 220, 255], "colors": [[0, 0, 255], [92, 214, 92], [0, 153, 51],
#                                                                      [102, 102, 0], [153, 102, 51]]}

# DEFAULT_COLORING = {"breakpoints": [0, 1, 40, 80, 128, 180, 220, 255], "colors": [[0, 0, 255],
#                                                                                   [92, 214, 92],
#                                                                                   [38, 77, 0],
#                                                                                   [34, 51, 0],
#                                                                                   [51, 51, 26],
#                                                                                   [77, 51, 25],
#                                                                                   [102, 51, 0],
#                                                                                   [51, 26, 0]]}


DEFAULT_COLORING = {"breakpoints": [0, 1, 128, 180, 255], "colors": [[0, 0, 255],
                                                                [92, 214, 92],
                                                                # [38, 77, 0],
                                                                [34, 51, 0],
                                                                # [51, 51, 26],
                                                                [77, 51, 25],
                                                                # [102, 51, 0],
                                                                [51, 26, 0]]}


# ======================================================================================================================
# Neural Network Configuration

# Data Loading
IMAGE_ROOT_PATH = "..\\..\\ProcessedData\\filtered_h1024_w1024_j128_m30"
# IMAGE_ROOT_PATH = "..\\..\\OriginalData"
WORKERS = 4
BATCH_SIZE = 16
IMAGE_SIZE = (1024, 1024)
SEED = 999                  # set a number for reproducibility. Use "None" to go random
COLOR_CHANNELS = 1
GPU = 1                     # 1 - use GPU, 0 - use CPU

# Optimizers
LEARNING_RATE = 0.0002
BETAS = (0.5, 0.999)
WAIT_FOR_UPDATE = 16         # how many batches has to pass before weight update (due to the high memory costs)

# Generator
GENERATOR_FEATURE_MAPS_SIZE = 8
LATENT_VECTOR_SIZE = 128

# Discriminator
DISCRIMINATOR_FEATURE_MAPS_SIZE = 8

# Training
NUMBER_OF_EPOCHS = 5

# Save outputs
SAVE_MODEL_PATH = "..\\..\\ProcessedData\\Models\\Second_network"
