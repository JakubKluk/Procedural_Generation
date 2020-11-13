from pathlib import Path
import numpy as np


JUMP_SIZE = 140
MAP_HEIGHT = 720
MAP_WIDTH = 1280

DATA_PATH = Path("../../OriginalData")
PROCESSED_PATH = Path("../../ProcessedData/")

MIN_LAND_PERCENTAGE = 30

COLOR_MAP = np.arange(256 * 3).reshape(256, 3)
COLOR_MAP[0, :] = [0, 51, 204]
COLOR_MAP[1:120, :] = [0, 179, 0]
COLOR_MAP[121:220, :] = [68, 102, 0]
COLOR_MAP[221:, :] = [68, 68, 34]
