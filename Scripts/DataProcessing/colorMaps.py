from pathlib import Path
from typing import List, Dict

import numpy as np
from PIL import Image

from Scripts.Util.ConfigFile import PROCESSED_PATH, DEFAULT_COLORING, MAP_HEIGHT, MAP_WIDTH, JUMP_SIZE


def color_map(img: np.array, break_colors: Dict[str, List] = DEFAULT_COLORING, interpolation: str = "linear"
              ) -> np.array:
    # def interpolate_color(color_1: List[int], color_2: List[int], rang: int) -> np.array:
    #     result = np.ones((rang, 3))
    #     result[0, :] = color_1
    #     color_step = np.array([(c2 - c1) / rang for c1, c2 in zip(color_1, color_2)])
    #     for i in range(1, rang):
    #         result[i, :] = result[i - 1, :] + color_step
    #     result[rang - 1, :] = color_2
    #     return result.astype(np.uint8)

    def interpolate_color_linear(color_1: List[int], color_2: List[int], rang: int) -> np.array:
        result = np.outer(np.ones(rang), color_1)

        diff = np.array([(c2 - c1) / (rang - 1) for c1, c2 in zip(color_1, color_2)])
        elems = np.linspace(0, rang-1, rang)

        result += np.outer(elems, diff)
        return result.astype(np.uint8)

    def interpolate_color_exponential(color_1: List[int], color_2: List[int], rang: int) -> np.array:
        result = np.outer(np.ones(rang), color_1)

        diff = np.array([(c2 - c1) / (rang - 1) for c1, c2 in zip(color_1, color_2)])
        elems = np.geomspace(0.00001, rang - 1, rang)

        result += np.outer(elems, diff)
        return result.astype(np.uint8)

    cmap = np.ones((256, 3))
    for ind in range(len(break_colors["breakpoints"]) - 1):
        first_break = break_colors["breakpoints"][ind]
        second_break = break_colors["breakpoints"][ind + 1]
        first_color = break_colors["colors"][ind]
        second_color = break_colors["colors"][ind + 1]
        rang = second_break - first_break + 1
        if interpolation == "linear":
            cmap[first_break:second_break + 1, :] = interpolate_color_linear(first_color, second_color, rang)
        else:
            cmap[first_break:second_break + 1, :] = interpolate_color_exponential(first_color, second_color, rang)

    return cmap[img].astype(np.uint8)


def heightmap_to_color(filtered_files: Path, interpolation: str = "linear") -> None:
    for file in filtered_files.iterdir():
        im = Image.open(file)
        ar = np.array(im)
        color_arr = color_map(ar, interpolation=interpolation)
        im = Image.fromarray(color_arr, 'RGB')
        dir_name = filtered_files.name.replace("filtered", "colored")
        save_path = PROCESSED_PATH / dir_name
        try:
            im.save(save_path / file.name)
        except FileNotFoundError:
            Path.mkdir(save_path)
            im.save(save_path / file.name)


if __name__ == "__main__":
    path = Path(PROCESSED_PATH / "filtered_h{0}_w{1}_j{2}".format(MAP_HEIGHT, MAP_WIDTH, JUMP_SIZE))
    heightmap_to_color(path, interpolation="expo")
    # ar = np.outer(np.arange(0, 255, 1), np.ones(10)).astype(np.uint8)
    # c_ar = color_map(ar, interpolation="linear")
    # img = Image.fromarray(c_ar)
    # img.show()
