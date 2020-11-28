import numpy as np
import noise
from random import sample, seed
from Scripts.InputGenerator.InputGenerator import InputGenerator
from Scripts.DataProcessing.colorMaps import color_map
from PIL import Image
from typing import Tuple
from copy import deepcopy


class RandomNoiseGenerator(InputGenerator):

    def generate_input_data(self, land_percentage: float = None) -> np.array:
        return np.random.randint(0, 255, self._size).astype(np.uint8)


class ControlledNoiseGenerator(InputGenerator):

    def generate_input_data(self, land_percentage: float) -> np.array:
        if not self.validate_percentage(land_percentage):
            raise ValueError("The land_percentage argument is supposed to be a float number in range [0, 1]!")
        result = np.random.randint(1, 255, self._size)
        indexes = [(i, j) for i in range(result.shape[0]) for j in range(result.shape[1])]
        indexes = sample(indexes, int((1 - land_percentage) * result.size))
        indexes = (np.array([i[0] for i in indexes]), np.array([i[1] for i in indexes]))
        result[indexes] = 0
        return result.astype(np.uint8)


class PerlinNoiseGenerator(InputGenerator):

    def __init__(self, size: Tuple[int, int], scale: float = 100.0, octaves: int = 6, persistence: float = 0.5,
                 lacunarity: float = 2.0, *args, **kwargs):
        super().__init__(size, *args, **kwargs)
        self._scale = scale
        self._octaves = octaves
        self._persistence = persistence
        self._lacunarity = lacunarity

    def generate_input_data(self, land_percentage: float) -> np.array:
        if not self.validate_percentage(land_percentage):
            raise ValueError("The land_percentage argument is supposed to be a float number in range [0, 1]!")

        # creating perlin's noise array
        world = np.zeros(self._size)
        for i in range(self._size[0]):
            for j in range(self._size[1]):
                world[i][j] = noise.pnoise2(i / self._scale,
                                            j / self._scale,
                                            octaves=self._octaves,
                                            persistence=self._persistence,
                                            lacunarity=self._lacunarity,
                                            repeatx=self._size[0],
                                            repeaty=self._size[1],
                                            base=0)
        # standardizing created array
        world -= np.min(world)
        world *= 255
        world = world.astype(np.uint8)

        # adjusting land percentage
        world_height = np.sort(world.reshape(1, -1))
        threshold = world_height[0, int((1 - land_percentage) * world_height.size)]
        world[np.where(world < threshold)] = 0
        indexes = world.nonzero()
        min_nonzero = world[indexes].min()
        max_nonzero = world[indexes].max()
        factor = max_nonzero / (max_nonzero - min_nonzero - 1)
        world[indexes] = (world[indexes] - min_nonzero) * factor + 1
        print(world.min(), world.max(), world[indexes].min())
        return world
                


if __name__ == "__main__":
    seed(7)
    # pictures = []
    # for octa in range(1, 7):
    #     pictures.append(PerlinNoiseGenerator((720, 1280), octaves=octa).generate_input_data(0.3))
    # for i in range(len(pictures)):
    #     img = Image.fromarray(pictures[i])
    #     img.save(f"../../ProcessedData/noises/perlin/different_noises_{i}.png")
    a = PerlinNoiseGenerator((720, 1280), octaves=8)
    arr = a.generate_input_data(0.4)
    img = Image.fromarray(arr)
    img.save("../../ProcessedData/noises/perlin/img7.png")
    c_arr = color_map(arr, interpolation="expo")
    img = Image.fromarray(c_arr, 'RGB')
    img.save("../../ProcessedData/noises/perlin/color_img7.png")
