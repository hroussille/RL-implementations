import numpy as np
from . import filter

class CircleFilter(filter.Filter):

    def __init__(self, center, size):
        super().__init__(center, size)

    def isIn(self, position):
        return np.sqrt(np.sum((self.center - position) ** 2)) < self.size

