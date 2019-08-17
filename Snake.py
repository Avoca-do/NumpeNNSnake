import random
import time
import math

from utility.globalStuff import Cord2d, Direction, Tile


class SnakeObject(object):
    def __init__(self, x, y, game):
        self.x = x
        self.y = y
        self.direction = Direction.up
        self.game = game
        self.body = []
        self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        self.alive = True
        self.brain = None
        self.score = 0
        self.fd = 0
        self.moveScore = 0

        self.fieldOfView = self.game.fieldOfView
        self.offset = int(self.fieldOfView * .5)
        for i in range(self.game.snakeStartingSize):
            self.body.append(Cord2d(x, y))

    def updateNNScore(self):
        if(self.brain is not None):
            self.brain.alive = False
            self.brain.score = self.score
            self.brain.fitness *= self.game.moveScale
            self.brain.fitness += self.score * 2
            self.brain.fitness = (1.5 + self.game.moveScale) ** self.brain.fitness - 1

    def setNN(self, NN):
        self.brain = NN
        self.brain.fitness = 0
        self.moves = 0

    def directionFromOutput(self, output):
        r = 0
        for i in range(1, 4):
            if(output[i] > output[r]):
                r = i
        if(r == 0):
            self.direction = Direction.left
        elif(r == 1):
            self.direction = Direction.right
        elif(r == 2):
            self.direction = Direction.up
        elif(r == 3):
            self.direction = Direction.down

    def getNNInputs(self, map):
        inputs = [0] * (self.fieldOfView ** 2)
        foodInputs = [0] * 3
        rx = (self.x - self.offset)
        ry = (self.y - self.offset)
        foodl = []
        for tile in map:
            x = tile.x - rx
            y = tile.y - ry
            if(tile.value == 1):
                foodl.append(tile)
            elif(x >= 0 and x < self.fieldOfView and y >= 0 and y < self.fieldOfView):
                inputs[int(y * self.fieldOfView + x)] = 1
        for part in self.body:
            x = part.x - rx
            y = part.y - ry
            if(x >= 0 and y >= 0 and x < self.fieldOfView and y < self.fieldOfView):
                inputs[int(y * self.fieldOfView + x)] = 1

        saved = None
        distance = 10000
        for food in foodl:
            current_distance = (self.x - food.x) ** 2 + (self.y - food.y) ** 2
            if(current_distance < distance):
                distance = current_distance
                saved = food
        if(saved):
            dx = self.x - saved.x
            dy = self.y - saved.y
            distance = math.sqrt(distance)
            if(distance > 0):
                dx /= distance
                dy /= distance
            else:
                dx = 0
                dy = 0
            foodInputs[0] = dx
            foodInputs[1] = dy
            foodInputs[2] = distance / 40
        if(distance > 0):
            self.fd = 1 / distance
        return inputs + foodInputs

    def updateBody(self, map):
        if(self.brain is not None):
            inputs = self.getNNInputs(map)
            prediction = self.brain.predict(inputs)
            self.directionFromOutput(prediction)
            self.moves += 1
            self.brain.fitness *= 0.75
            self.brain.fitness += (self.fd * self.fd * self.fd) * 0.5
            #print("score : %.1f, move_score : %.3f" % (self.score, self.moves))
            if(self.game.maxSteps < self.moves):
                self.alive = False
                self.game.onSnakeDeath()
                return False

        #print(self.direction)
        if self.direction == Direction.up:
            self.y += 1
        elif self.direction == Direction.right:
            self.x += 1
        elif self.direction == Direction.down:
            self.y -= 1
        elif self.direction == Direction.left:
            self.x -= 1
        for tile in map:
            if(self.x == tile.x and self.y == tile.y):
                if(tile.value == -1):
                    self.alive = False
                    #self.brain.fitness -= (100 / self.moves)
                    self.game.onSnakeDeath()
                    return False
                elif(tile.value == 1):
                    self.game.foodG.foodEaten(tile)
                    self.score += 1
                    self.moves = 0
                    last = self.body[-1]
                    for i in range(3):
                        self.body.append(Cord2d(last.x, last.y))
                    break

        self.body.insert(0, Cord2d(self.x, self.y))
        del self.body[-1]
        for part in self.body[1:]:
            if(part.x == self.x and part.y == self.y):
                self.alive = False
                #self.brain.fitness -= (100 / self.moves)
                self.game.onSnakeDeath()
                return False
        return True

    def updateDirection(self, dir):
        if(self.brain is not None):
            return
        if(self.direction.value + dir.value != 0):
            self.direction = dir

    def drawSnake(self):
        for part in self.body:
            self.game.fillTile(self.color, part.x, part.y)
