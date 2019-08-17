from utility.globalStuff import Tile, Cord2d
import random


class FoodGenerator(object):
    def __init__(self, game, count=1):
        self.foodList = []
        self.game = game
        self.count = count
        self.addNewFood()

    def setFoodCount(self, count):
        self.count = count

    def foodEaten(self, food):
        self.foodList.remove(food)
        self.game.removeTile(food)
        self.addNewFood()

    def reset(self):
        for food in self.foodList:
            self.game.removeTile(food)
        num = len(self.foodList)
        self.foodList = []
        for i in range(num):
            self.addNewFood()

    def addNewFood(self):
        add = self.count - len(self.foodList)
        for i in range(add):
            randx = random.randint(1, self.game.sizeX - 2)
            randy = random.randint(1, self.game.sizeY - 2)
            for food in self.foodList:
                if(randx == food.x and randy == food.y):
                    self.addNewFood()
                    return
            for snake in self.game.snakeList:
                for cord in snake.body:
                    if(randx == cord.x and randy == cord.y):
                        self.addNewFood()
                        return

            food = Tile(randx, randy, 1)
            self.foodList.append(food)
            self.game.addTile(food)
