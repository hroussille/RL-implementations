class Filter():

    def __init__(self, center, size, acceleration=False):
        self.center = center
        self.size = size
        self.acceleration = acceleration
        self.dimension = len(center)

    def isIn(self, position):
        pass

    def isOut(self, position):
        return not self.isIn(position)

