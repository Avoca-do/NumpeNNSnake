import pygame
import threading
import time
from utility.globalStuff import Direction, Tile
from Snake import SnakeObject
from Food import FoodGenerator
from NNObjects.NNEvol import EvNeuralTrainer, EvNeuralNet
import utility.constants as CONSTANTS

class Game(object):
    def __init__(self, tilesX, tilesY, ScreenWidth, ScreenHeight, Title):
        load = input("Load saved (y/n) : ")
        loadID = -1
        loadNumber = 25
        highestScore = 1
        if(CONSTANTS.checkIfYes(load)):
            for str in CONSTANTS.getAllSaves():
                print(str)

            sin = input("Specify target Save-id(sid) : ")
            if(sin.isdigit()):
                print("Specifications for Loading... ( press Enter for default value )")

                loadID = int(sin)
                sin = input("Specify maximum generation's to load ( all / default ) : ")
                if(sin == 'all'):
                    loadNumber = 200

                sin = input("Specify starting highscore( used to determine max steps ) :")
                highestScore = CONSTANTS.StringToInt(sin, -1, 1, 99)
                print('')
                print('Loading NN specefication from file...')
            else:
                print('Invalid input - SaveID supposed to be an integer!!!!')

        self.fieldOfView = 5
        self.populationSize = 150
        self.hidden_nodes1 = 10
        self.hidden_nodes2 = 10
        self.memoryNodes = 10
        if(loadID == -1):
            print("")
            print('Skipping loading, Initializing as DEFAULT!')
            print('Enter blank to leave as default!')
            print("")
            sin = input("Do you wish to Initialize NN as default (y/n) : ")
            print(len(sin))
            print(sin == '')
            if(not CONSTANTS.checkIfYes(sin) and not sin == ''):
                print("Specifications for NN...")

                sin = input('Choose agent fieldOfView ( default - 5, min - 3, max - 40 ) : ')
                self.fieldOfView = CONSTANTS.StringToInt(sin, self.fieldOfView, 3, 40)

                print('Specify number of nodes in each layer ( default for each layer is 10 )...')
                sin = input('Hidden layer n1 nodes : ')
                self.hidden_nodes1 = CONSTANTS.StringToInt(sin, self.hidden_nodes1, 3, 9999)
                sin = input('Hidden layer n2 nodes : ')
                self.hidden_nodes2 = CONSTANTS.StringToInt(sin, self.hidden_nodes2, 3, 9999)
                sin = input('Memory nodes : ')
                self.memoryNodes = CONSTANTS.StringToInt(sin, self.memoryNodes, 3, 9999)
                print('initializing NN with given values...')
            else:
                print('initializing NN with default values...')

        print('')
        print('Next steps are to initialize the enviroment...')
        print('Note the default snake size is +-20, instead of 3.')
        print('this is to avoid training the snake while he is small, and its much easier.')
        print('it is possible that even bigger starting size would yield better overall results.')
        print('')
        sin = input('Enter population size (default 150) : ')
        self.populationSize = CONSTANTS.StringToInt(sin, self.populationSize, 10, 10000)
        sid = input('Choose snake starting size ( default - 21, min - 3 ) : ')
        self.snakeStartingSize = CONSTANTS.StringToInt(sid, 21, 3, 200)

        print('Settings confirmed starting...')

        self.tileX = (int)(ScreenWidth / tilesX)
        self.tileY = (int)(ScreenHeight / tilesY)
        self.sizeX = tilesX
        self.sizeY = tilesY
        self.centerX = tilesX * .5
        self.centerY = tilesY * .5
        self.moveScale = 1
        self.snakeList = []
        self.updateDirection = False
        self.direction = Direction.up
        self._tiles = []
        self.tileByPos = {}
        self.foodCount = 1
        self.foodG = FoodGenerator(self, self.foodCount)
        self.AiC = 1
        self.delay = 1
        self.maxspeed = False
        self.resetSpeed = False
        self.maxSteps = 75

        #initialize pygame and create window
        pygame.init()
        self.run = True
        self.win = pygame.display.set_mode((ScreenWidth, ScreenHeight))
        #initialize out neural trainer and create/load the first population
        self.EvN = EvNeuralTrainer(self, self.populationSize, (self.fieldOfView ** 2) + 3,
                                   self.hidden_nodes1, self.hidden_nodes2, self.memoryNodes,
                                   4, loadID = loadID, stepsLevel = highestScore, loadNumber = loadNumber)

        #self.snakeList.append(SnakeObject(self.centerX, self.centerY, self))
        #set brains to the starting snakes. Not self.AiC is used to train multiple networks at once,
        #note: multiply agents at once, might result in poor, performence, due to the fact that each Snake.
        #Might "steal" the food of the other one so they would have to "race" each other with might not be what we want.
        #it might be really simple to add different "foods" for each snake, by giving them unique id for each snake,
        #then we would need to "not show" the uninteractable food for each snake.
        for i in range(self.AiC):
            snake = SnakeObject(self.centerX, self.centerY, self)
            snake.setNN(self.EvN._population[i])
            self.snakeList.append(snake)

        #create borders
        for x in range(tilesX):
            self.addTile(Tile(x, 0, -1))
            self.addTile(Tile(x, tilesY - 1, -1))

        for y in range(tilesY):
            self.addTile(Tile(0, y, -1))
            self.addTile(Tile(tilesX - 1, y, -1))

        pygame.display.set_caption(Title)
        threading.Thread(target=self.main_loop).start()

    def reset(self):
        if(len(self.snakeList) <= 0):
            self.foodG.reset()
            self.EvN.newGeneration()
            for i in range(self.AiC):
                snake = SnakeObject(self.centerX, self.centerY, self)
                snake.setNN(self.EvN._population[i])
                self.snakeList.append(snake)

    def input_handle(self):
        #user interactible "thread" or "main thread" handles user inputs.
        #solver freezes in the run time.
        while self.run:
            pygame.time.delay(10)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.run = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_w:
                        self.updateDirection = True
                        self.direction = Direction.down
                    elif event.key == pygame.K_s:
                        self.updateDirection = True
                        self.direction = Direction.up
                    elif event.key == pygame.K_d:
                        self.updateDirection = True
                        self.direction = Direction.right
                    elif event.key == pygame.K_a:
                        self.updateDirection = True
                        self.direction = Direction.left
                    elif event.key == pygame.K_UP:
                        self.delay = int(self.delay * .5)
                        if(self.delay < 1):
                            self.delay = 1
                            self.maxspeed = True
                    elif event.key == pygame.K_DOWN:
                        if(self.maxspeed):
                            self.delay = 1
                            self.maxspeed = False
                        else:
                            self.delay *= 2
                    elif event.key == pygame.K_i:
                        self.AiC += 1
                    elif event.key == pygame.K_k:
                        self.AiC -= 1
                        if(self.AiC < 1):
                            self.AiC = 1
                    elif event.key == pygame.K_u:
                        self.foodCount += 1
                        self.foodG.setFoodCount(self.foodCount)
                    elif event.key == pygame.K_j:
                        self.foodCount -= 1
                        if(self.foodCount < 1):
                            self.foodCount = 1
                        self.foodG.setFoodCount(self.foodCount)

    def main_loop(self):
        while self.run:
            if(not self.maxspeed):
                pygame.time.delay(self.delay)
            self.win.fill((0, 0, 0))
            #self.showInputView()
            self.update()
            pygame.display.update()
        pygame.quit()

    def showInputView(self):
        if self.updateDirection:
            self.updateDirection = False
            for snake in self.snakeList:
                snake.updateDirection(self.direction)

        for snake in self.snakeList:
            snake.updateBody(self._tiles)
            map = snake.getNNInputs(self._tiles)
            for x in range(self.fieldOfView):
                for y in range(self.fieldOfView):
                    if(map[y * self.fieldOfView + x] == 1):
                        self.fillTile((100, 255, 100), x, y)
                    elif(map[y * self.fieldOfView + x] == 0.5):
                        self.fillTile((100, 100, 255), x, y)
                    elif(map[y * self.fieldOfView + x] == -1):
                        self.fillTile((255, 100, 100), x, y)

    def update(self):
        if self.updateDirection:
            self.updateDirection = False
            for snake in self.snakeList:
                snake.updateDirection(self.direction)

        if(not self.maxspeed):
            for tile in self._tiles:
                if(tile.value < 0):
                    self.fillTile((200, 50, 50), tile.x, tile.y)
                else:
                    self.fillTile((50, 255, 50), tile.x, tile.y)

        for snake in self.snakeList:
            snake.updateBody(self._tiles)
            if(not self.maxspeed):
                snake.drawSnake()

    def onSnakeDeath(self):
        #pygame.time.delay(20)
        delobject = []
        reset = False
        for i in range(len(self.snakeList)):
            if(not self.snakeList[i].alive):
                #the game was initially made for players.
                #so we need to cheack if its an agent, and respawn the snake with new brain.
                self.snakeList[i].updateNNScore()
                NN = self.EvN.getNewNN()
                if(NN is not None):
                    snake = SnakeObject(self.centerX, self.centerY, self)
                    snake.setNN(NN)
                    self.snakeList[i] = snake
                else:
                    reset = True
                    delobject.append(i)

        if(reset):
            for i in delobject:
                del self.snakeList[i]
            self.reset()

    def fillTile(self, color, x, y):
        pygame.draw.rect(self.win, color,
                         (self.tileX * x + 1, self.tileY * y + 1,
                          self.tileX - 2, self.tileY - 2))

    def addTile(self, tile):
        self._tiles.append(tile)

    def removeTile(self, tile):
        self._tiles.remove(tile)
