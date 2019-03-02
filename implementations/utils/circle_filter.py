import numpy as np
from filter import Filter

class CircleFilter(Filter):

    def __init__(self, center, size):
        super(self, center, size)

    def isIn(self, position):
        return np.sqrt((self.center - position) ** 2)

