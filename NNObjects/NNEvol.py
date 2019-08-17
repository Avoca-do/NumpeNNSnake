import numpy as np
import time
import random
import json
import os
import utility.constants as CONSTANTS


mutationRate = 0.02
scale = .5
#random.seed(1)

def sigmoid(x):
    return 1 / (1 + (np.exp(-x)))

def tanh_prime(x):
    return 1 - np.tanh(x)**2

def relu(x):
    return np.multiply(x, (1 + x))

mutations = 0
genes = 0

def mixMetrix(A, B, rand):
    # randomly create a new metrix out of two existing with same shape,
    # with a slight chance of getting a new value.
    mix = np.copy(A)
    global mutations, genes
    for x in range(B.shape[0]):
        for y in range(B.shape[1]):
            genes += 1
            if(random.random() < rand):
                mt = random.randint(0, 5)
                mutations += 1
                if(mt == 0):
                    mix[x][y] = random.random() * scale * 2 - scale
                elif(mt == 1):
                    mix[x][y] += random.random() * scale
                elif(mt == 2):
                    mix[x][y] -= random.random() * scale
                elif(mt == 3):
                    mix[x][y] *= random.random()
                elif(mt == 4):
                    mix[x][y] *= 1 + random.random()
                elif(mt == 5):
                    mix[x][y] *= -1
            elif(random.random() > 0.5):
                mix[x][y] = B[x][y]
    return mix

def mix1d(A, B, rand):
    # mixes a 1 dimentional metrix (bias layer)
    mix = np.copy(A)
    for x in range(A.size):
        if(random.random() < rand):
            mix[x] = random.random() * scale
        elif(random.random() > 0.5):
            mix[x] = B[x]
    return mix

class EvNeuralNet(object):
    def __init__(self, inputs, hlayer1, hlayer2, memory, outputs, init = True):
        self.inputs = inputs
        self.memory = memory
        self.init_params = [inputs, hlayer1, hlayer2, memory, outputs]

        inputs += self.memory

        self.memoryData = np.zeros([self.memory])
        self.fitness = 0
        self.alive = True
        if(init):
            self.hlayer1 = np.random.normal(scale=scale, size=(inputs, hlayer1))
            self.memoryLayer = np.random.normal(scale=scale, size=(inputs, self.memory))
            #self.hlayer1 = relu(self.hlayer1)
            #print(self.hlayer1)
            self.hlayer2 = np.random.normal(scale=scale, size=(hlayer1, hlayer2))
            #self.hlayer2 = relu(self.hlayer2)
            self.outlayer = np.random.normal(scale=scale, size=(hlayer2, outputs))
            #self.outlayer = relu(self.outlayer)
            self.bl1 = np.random.normal(scale=0, size=hlayer1)
            self.bl2 = np.random.normal(scale=0, size=hlayer2)
            self.bo = np.random.normal(scale=0, size=outputs)
            self.memoryBios = np.zeros([self.memory])

    def createChild(parentA, parentB):
        #used to create a child out of two parents, with mixed values in layers, as well as a chance for mutation.
        ret = EvNeuralNet(*parentA.init_params, init = False)
        ret.hlayer1 = mixMetrix(parentA.hlayer1, parentB.hlayer1, mutationRate)
        #ret.hlayer1 = relu(ret.hlayer1)
        ret.hlayer2 = mixMetrix(parentA.hlayer2, parentB.hlayer2, mutationRate)
        #ret.hlayer2 = relu(ret.hlayer2)
        ret.outlayer = mixMetrix(parentA.outlayer, parentB.outlayer, mutationRate)
        #ret.outlayer = relu(ret.outlayer)
        ret.bl1 = mix1d(parentA.bl1, parentB.bl1, mutationRate)
        ret.bl2 = mix1d(parentA.bl2, parentB.bl2, mutationRate)
        ret.bo = mix1d(parentA.bo, parentB.bo, mutationRate)

        ret.memoryLayer = mixMetrix(parentA.memoryLayer, parentB.memoryLayer, mutationRate)
        ret.memoryBios = mix1d(parentA.memoryBios, parentB.memoryBios, mutationRate)
        return ret

    def predict(self, x):
        deltaX = np.concatenate([x, self.memoryData])
        A = np.dot(deltaX, self.hlayer1) + self.bl1
        self.memoryData = np.tanh(np.dot(deltaX, self.memoryLayer) + self.memoryBios)
        B = np.dot(np.tanh(A), self.hlayer2) + self.bl2
        C = np.dot(np.tanh(B), self.outlayer) + self.bo
        return (sigmoid(C))

    def save(self, name, dir = CONSTANTS.HOLDFOLDER):
        # save the brain with np.savez with we can easily load at any time.
        np.savez(dir + name + '.npz', hl1=self.hlayer1, hl2=self.hlayer2, ol=self.outlayer,
                 bl1=self.bl1, bl2=self.bl2, bo=self.bo, memoryLayer=self.memoryLayer,
                 memoryBios=self.memoryBios, init_params=self.init_params, fitness = self.fitness)
        print('Saved %s NN fitness: %.2f' % (name, self.fitness))

    def load(name, dir = CONSTANTS.HOLDFOLDER):
        data = np.load(dir + name + '.npz')
        loaded = EvNeuralNet(*data['init_params'], init=False)
        loaded.hlayer1 = data['hl1']
        loaded.hlayer2 = data['hl2']
        loaded.outlayer = data['ol']
        loaded.memoryLayer = data['memoryLayer']
        loaded.bl1 = data['bl1']
        loaded.bl2 = data['bl2']
        loaded.bo = data['bo']
        loaded.memoryBios = data['memoryBios']
        return loaded

def sortByScore(NN):
    return -NN.fitness
class EvNeuralTrainer(object):
    def __init__(self, game, population_num, input_nodes, hidden_nodes1,
                 hidden_nodes2, memory, output_nodes, loadID = None, stepsLevel = 1, loadNumber = 25):

        if(loadID < 0):
            loadID = None
        if(loadID):
            # if we loading an old simulation we need to first make sure,
            # we are using the right setting for out neural network
            dirName = CONSTANTS.FOLDER(loadID)
            loadData = np.load(dirName + CONSTANTS.SETTINGS)
            input_nodes = int(loadData['input_nodes'])
            hidden_nodes1 = int(loadData['hidden_nodes1'])
            hidden_nodes2 = int(loadData['hidden_nodes2'])
            memory = int(loadData['memory'])
            output_nodes = int(loadData['output_nodes'])

            try:
                game.fieldOfView = int(loadData['fieldOfView'])
            except KeyError:
                n = int(np.sqrt(input_nodes - 3))
                game.fieldOfView = n

        self.nnParams = [input_nodes, hidden_nodes1, hidden_nodes2, memory, output_nodes]
        print(self.nnParams)
        self.generation = 1
        self.population_num = population_num
        self._population = []
        self._used = []
        self.index = 0
        self.game = game
        self.highestFitness = 0
        self.top3Fitness = [0] * 3
        self.highestFitnesGeneration = -1
        self.stepsLevel = stepsLevel
        self.highestScore = -1
        self.level = 1
        self.nextSave = 0
        self.autoSave = -1

        self.game.maxSteps = 75 + self.stepsLevel * 35 + (1.1 ** self.stepsLevel) * 5
        self.gainStreak = 0
        self.time = time.clock()

        # here we creating the files that would store all the setting's regarding this run
        # so first of all we are making sure we have the main data file
        # then we initializing all the data files regarding this run.
        try:
            load = np.load(CONSTANTS.MAIN_DATA)
            self.tid = int(load['id'] + 1)
            np.savez(CONSTANTS.MAIN_DATA, id = self.tid)
        except FileNotFoundError:
            np.savez(CONSTANTS.MAIN_DATA, id = 0)
            self.tid = 0

        dirName = CONSTANTS.FOLDER(self.tid)
        dirName = dirName[:-1]
        if not os.path.exists(dirName):
            os.makedirs(dirName)
            print("Directory ", dirName, " Created - NN saves")
        else:
            print("Directory ", dirName, " already exists - NN saves !!Overrided!!")
        self.dirName = dirName + '/'
        np.savez(self.dirName + CONSTANTS.SETTINGS, input_nodes = input_nodes, hidden_nodes1 = hidden_nodes1,
                 hidden_nodes2 = hidden_nodes2, memory = memory, output_nodes = output_nodes,
                 lastID = 0, fieldOfView = self.game.fieldOfView)
        np.savez(self.dirName + CONSTANTS.NN_SAVE_DATA, generations = [],
                 maxFitness = 0, bestGeneration = -1, highestScore = self.highestScore, level = 1)

        if(loadID):
            # after we created all the important setting we just load out old, generations
            # as well as creating a few totally new networks.
            dirName = CONSTANTS.FOLDER(loadID)
            gen = np.load(dirName + CONSTANTS.NN_SAVE_DATA)

            r = len(gen['generations'])
            if(r > loadNumber):
                r = loadNumber
            print('loading %d saved NN\'s...' % r * 4)
            print('')
            for i in range(0, r):
                genID = gen['generations'][i]
                if(stepsLevel < 1):
                    self.stepsLevel = int(gen['highestScore'] * 0.5)
                    self.game.maxSteps = 75 + self.stepsLevel * 35 + (1.1 ** self.stepsLevel) * 5
                    self.game.moveScale = 1 / self.stepsLevel
                    self.level = self.stepsLevel

                print("Loading gen : %d" % genID)
                n0 = EvNeuralNet.load(CONSTANTS.FILENAME(genID, 0), dirName)
                n1 = EvNeuralNet.load(CONSTANTS.FILENAME(genID, 1), dirName)
                n2 = EvNeuralNet.load(CONSTANTS.FILENAME(genID, 2), dirName)
                n3 = EvNeuralNet.load(CONSTANTS.FILENAME(genID, 3), dirName)
                self._population.append(n0)
                self._population.append(n1)
                self._population.append(n2)
                self._population.append(n3)
            print('Generating %d new NN\'s...' % (population_num - r))
            print('')
            for i in range(population_num - r):
                self._population.append(EvNeuralNet(*self.nnParams))
        else:
            print('generating NN\'s...')
            for i in range(population_num):
                self._population.append(EvNeuralNet(*self.nnParams))

    def getNewNN(self):
        if(len(self._population) <= 0):
            #self.game.reset()
            return None

        nexte = self._population[-1]
        self._used.append(nexte)
        del self._population[-1]
        return nexte

    def pickRandomElement(self, n):
        # here we getting the a random agent but the higher his
        # fitness the higher the chance he will be picked
        # agent.probability = agent.fitness / Total_generation_fitness
        x = len(self._used)
        r = random.random()
        index = -1
        while(r > 0):
            index += 1
            r -= self._used[index].probability
        #print("index: %d, probability: %.5f" % (index, self._used[index].probability))
        return self._used[index]

    def newGeneration(self):
        global mutations, genes

        # here we creating a new generation saving the current if needed,
        # we also logging any changes and the stats of the current run.
        self._population = []
        totalScore = 0
        for nn in self._used:
            totalScore += nn.fitness

        for nn in self._used:
            nn.probability = nn.fitness / totalScore

        for i in range(self.population_num - 20):
            a = self.pickRandomElement(0)
            b = self.pickRandomElement(0)

            Net = EvNeuralNet.createChild(a, b)
            self._population.append(Net)

        for i in range(20):
            index = i * 0.25
            a = self._used[int(index)]
            net = EvNeuralNet.createChild(a, a)
            self._population.append(net)

        # here we sorting the generation members so we can pick the best performing one
        self._used.sort(key = sortByScore)
        print("-------------------------------------------------------------------------------------")
        print("Generation %d level %d stats --- max fitness: %.2f, min fitness: %.2f, Gen time: %.2f" %
              (self.generation, self.level, self._used[0].fitness, self._used[-1].fitness, time.clock() - self.time))
        print("Max gen score: %d" % self._used[0].score)
        print("")
        self.autoSave += 1
        hf = self._used[0].fitness
        sid = -1
        for i in range(0, 3):
            if(self.top3Fitness[i] < hf):
                print('ID - %d | oldValue = %.1f | newValue %.1f' % (i, self.top3Fitness[i], hf))
                print("")
                sid = i
                self.top3Fitness[i] = hf
                break

        if(self._used[0].fitness > self.highestFitness):
            print('new Highest fitness : %.1f, Broken fitness : %.1f' % (hf, self.highestFitness))
            print('Saving as best Current NN...')
            self.highestFitness = self._used[0].fitness
            self.highestFitnesGeneration = self.generation
            self.highestScore = self._used[0].score

            self._used[0].save(CONSTANTS.SAVEDABEST(0), self.dirName)
            self._used[1].save(CONSTANTS.SAVEDABEST(1), self.dirName)
            self._used[2].save(CONSTANTS.SAVEDABEST(2), self.dirName)
            self._used[3].save(CONSTANTS.SAVEDABEST(3), self.dirName)
            print("")

        if(sid + 1 > 0 or self.autoSave > 9):
            self.autoSave = -1
            sd = np.load(self.dirName + CONSTANTS.NN_SAVE_DATA)
            generations = sd['generations']
            generations = np.insert(generations, 0, self.generation)
            generations = generations.astype(int)
            if(sid + 1 > 0):
                print('Save - %d score has been broken!' % sid)
            else:
                print('Autosave auto saving each %d generation' & 10)
            print('saving current generation...')
            np.savez(self.dirName + CONSTANTS.NN_SAVE_DATA, generations = generations,
                     highestScore = self.highestScore, maxFitness = self.highestFitness,
                     bestGeneration = self.highestFitnesGeneration, level = self.level)
            self._used[0].save(CONSTANTS.FILENAME(self.generation, 0), self.dirName)
            self._used[1].save(CONSTANTS.FILENAME(self.generation, 1), self.dirName)
            self._used[2].save(CONSTANTS.FILENAME(self.generation, 2), self.dirName)
            self._used[3].save(CONSTANTS.FILENAME(self.generation, 3), self.dirName)

            gnt = ', '.join(map(str, generations))
            if(len(generations) > 15):
                gnt = ', '.join(map(str, generations[:15]))
                gnt += '...'
            print('Saved generations ID-list : [%s]' % gnt)
            print("")

        print("Overall stats --- Number of active Neural networks: %d, food count: %d" %
              (self.game.AiC, self.game.foodCount))
        print("")
        print("Total stats --- max fitness: %.2f, Highest fitness gen: %d, GenMax Steps: %d, GenLevel: %d" %
              (self.highestFitness, self.highestFitnesGeneration, self.game.maxSteps, self.level))
        print("Total genes %d, total mutations %d, percent mutations %.2f" % (genes, mutations, mutations / genes))
        cmax = self._used[0].score
        if(cmax >= self.stepsLevel):
            self.gainStreak += 1
            if(self.gainStreak > 2):
                if(self.gainStreak < 10):
                    self.game.moveScale = 1 / self.stepsLevel
                else:
                    self.game.moveScale = 0
                print('')
                print('Training level up, reseting all saving data!!')
                print('')
                self.top3Fitness = [0] * 3
                self.highestFitness = 0
                self.level = self.stepsLevel
                self.stepsLevel = cmax
                self.gainStreak = 0
                self.game.maxSteps = 75 + self.stepsLevel * 35 + (1.1 ** self.stepsLevel) * 5
        else:
            self.gainStreak = 0

        self.generation += 1
        self.time = time.clock()

        used = []
        for nn in self._used:
            if(nn.alive):
                used.append(nn)
        self._used = used
        mutations = 0
        genes = 0


'''
#test neural net init and save/load
A = EvNeuralNet(3, 4, 5, 4)
B = EvNeuralNet(3, 4, 5, 4)
C = EvNeuralNet.createChild(A, B)

print(A.hlayer1)
print('--------------------')
print(A.hlayer2)
print('--------------------')
np.savez('data_NN/test.npz', hlayer1=A.hlayer1, hlayer2=A.hlayer2, params=A.init_params)
data = np.load('data_NN/test.npz')
print(data.files)
print(data['hlayer1'])
print('--------------------')
print(data['hlayer2'])
print('--------------------')
print(data['params'])
print('--------------------')

np.savetxt('test.txt', A.init_params, fmt='%d')
get = np.loadtxt('test.txt', dtype=int)
print("loaded v - ")
print(get)

A.save('A')
D = EvNeuralNet.load('A')
print(A.init_params)
print(D.init_params)

print('-------------')
print(A.hlayer2)
print(D.hlayer2)
'''
