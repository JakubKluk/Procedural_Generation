from pathlib import Path
import numpy as np
from PIL import Image
from Scripts.Util.ConfigFile import MIN_LAND_PERCENTAGE, PROCESSED_PATH


def filter_maps(min_perc: int, cropped_files: Path) -> None:
    for file in cropped_files.iterdir():
        im = Image.open(file)
        arr = np.array(im)
        if (100 * np.round(np.count_nonzero(arr) / (arr.shape[0] * arr.shape[1]), 2)) > min_perc:
            dir_name = cropped_files.name.replace("cropped", "filtered") + "_m" + str(min_perc)
            # redundant directory "subfolder" us used by torch libraries to load data
            save_path = PROCESSED_PATH / dir_name / "subfolder"
            try:
                im.save(save_path / file.name)
            except FileNotFoundError:
                Path.mkdir(save_path, parents=True)
                im.save(save_path / file.name)


if __name__ == "__main__":
    path = Path(PROCESSED_PATH / "cropped_h720_w1280_j140")
    filter_maps(MIN_LAND_PERCENTAGE, path)
