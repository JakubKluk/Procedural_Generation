from typing import Tuple
from numpy import array


class InputGenerator:

    def __init__(self, size: Tuple[int, int], *args, **kwargs):
        self._size = size

    def generate_input_data(self, land_percentage: float) -> array:
        pass

    @staticmethod
    def validate_percentage(land_percentage: float) -> bool:
        return isinstance(land_percentage, float) and (0 <= land_percentage <= 1)
