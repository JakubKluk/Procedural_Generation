from pathlib import Path

import numpy as np
from PIL import Image

from Scripts.Util.configFile import MAP_WIDTH, MAP_HEIGHT, JUMP_SIZE, DATA_PATH


def crop(im: Image, height: int, width: int, jump_size=1) -> Image:
    img_width, img_height = im.size
    rows = np.int((img_height - height) / jump_size) + 1
    cols = np.int((img_width - width) / jump_size) + 1
    if rows < 2 or cols < 2:
        raise ValueError("Improper crop parameters. Either size of sub-images is too big or the size of the jump is too"
                         " big. Original picture size is width: {0}, height: {1}. Used jump_size: {2}"
                         "Trying to cut pictures of size: width: {3}, height: {4}".format(img_width, img_height,
                                                                                          jump_size, width, height))
    for i in range(rows):
        for j in range(cols):
            box = (j * jump_size, i * jump_size, width + j * jump_size, height + i * jump_size)
            yield im.crop(box)


def create_sub_images(im: Image, height: int, width: int, jump_size: int, save_path: Path, image_name: str) -> None:
    for k, piece in enumerate(crop(im, height, width, jump_size)):
        img = Image.new('L', (width, height), 255)
        img.paste(piece)
        path = save_path / "{0}_{1}.tif".format(image_name, k)
        img.save(path)


if __name__ == "__main__":

    cropped_data = Path("../ProcessedData/cropped_h{0}_w{1}_j{2}".format(MAP_HEIGHT, MAP_WIDTH, JUMP_SIZE))

    Path.mkdir(cropped_data, exist_ok=True)
    for f in DATA_PATH.iterdir():
        im = Image.open(f)
        create_sub_images(im, MAP_HEIGHT, MAP_WIDTH, JUMP_SIZE, cropped_data, f.name)
