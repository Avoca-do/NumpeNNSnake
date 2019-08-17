import enum


class Direction(enum.Enum):
    right = 1
    left = -1
    up = 2
    down = -2

class TileType(enum.Enum):
    food = 1
    wall = -1

class Tile(object):
    def __init__(self, x=0, y=0, v=0):
        self.x = x
        self.y = y
        self.value = v

class Cord2d:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
