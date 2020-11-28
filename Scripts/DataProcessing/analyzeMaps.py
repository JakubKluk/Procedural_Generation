from bokeh.plotting import save, output_file, figure
from PIL import Image
from pathlib import Path
import numpy as np
from collections import Counter
from Scripts.Util.ConfigFile import PROCESSED_PATH
from tqdm import tqdm


def height_histogram(cropped_files: Path) -> None:
    # variable that holds count of each brightness value
    result = Counter()
    # counting brightness through all the maps
    for file in tqdm(cropped_files.iterdir()):
        im = Image.open(file)
        im = np.array(im)
        unique, counts = np.unique(im, return_counts=True)
        result += Counter(dict(zip(unique, counts)))
    # creating a histogram
    left = [i for i in range(256)]
    right = [i + 0.8 for i in range(256)]
    top = [result[i] for i in range(256)]
    output_file(Path(f"../processed_data/plots/{cropped_files.name}_height_histogram.html"))
    p = figure(title="Histogram of brightness for all maps", background_fill_color="#fafafa")
    p.quad(top=top, left=left, right=right, bottom=0, fill_color="navy", line_color="white", alpha=0.5)
    p.xaxis.axis_label = 'Brightness'
    p.yaxis.axis_label = 'Brightness frequency'
    save(p)


def land_percentage_histogram(cropped_files: Path) -> None:
    # variable that holds count of each brightness value
    result = Counter()
    # counting brightness through all the maps
    for file in tqdm(cropped_files.iterdir()):
        im = Image.open(file)
        im = np.array(im)
        percent = 100 * np.round(np.count_nonzero(im) / (im.shape[0] * im.shape[1]), 2)
        result[percent] += 1
    # creating a histogram
    left = [i for i in range(101)]
    right = [i + 0.8 for i in range(101)]
    top = [result[i] for i in range(101)]
    output_file(Path(f"../processed_data/plots/{cropped_files.name}_land_histogram.html"))
    p = figure(title="Histogram of land percentage for all maps", background_fill_color="#fafafa")
    p.quad(top=top, left=left, right=right, bottom=0, fill_color="navy", line_color="white", alpha=0.5)
    p.xaxis.axis_label = 'Land percentage'
    p.yaxis.axis_label = 'Land percentage frequency'
    save(p)


if __name__ == "__main__":
    path = Path(PROCESSED_PATH / "cropped_h720_w1280_j140")
    height_histogram(path)
    land_percentage_histogram(path)
