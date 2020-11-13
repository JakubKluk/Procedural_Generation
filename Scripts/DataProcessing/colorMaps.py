from pathlib import Path
import numpy as np
from PIL import Image
from Scripts.Util.configFile import PROCESSED_PATH, COLOR_MAP


def map_gray_to_color(img: np.array) -> np.array:
    return COLOR_MAP[img].astype(np.uint8)


def color_maps(filtered_files: Path) -> None:
    for file in filtered_files.iterdir():
        im = Image.open(file)
        ar = np.array(im)
        color_arr = map_gray_to_color(ar)
        im = Image.fromarray(color_arr, 'RGB')
        dir_name = filtered_files.name.replace("filtered", "colored")
        save_path = PROCESSED_PATH / dir_name
        try:
            im.save(save_path / file.name)
        except FileNotFoundError:
            Path.mkdir(save_path)
            im.save(save_path / file.name)


if __name__ == "__main__":
    path = Path(PROCESSED_PATH / "filtered_h720_w1280_j140_m30")
    color_maps(path)
