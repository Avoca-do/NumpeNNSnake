import os
import numpy as np

NN_BESTGEN_DATA = "BestGenData.npz"
NN_SAVE_DATA = "saveData.npz"
SETTINGS = "Settings.npz"
HOLDFOLDER = "data_NN/"

SAVEFOLDER = HOLDFOLDER + "Saved/"
MAIN_DATA = HOLDFOLDER + "data.npz"


def FILENAME(id, index):
    return "SNN_%d_%d" % (id, index)

def SAVEDABEST(index):
    return "DABESTSNN_%d" % index

def FOLDER(id):
    return SAVEFOLDER + "sid%d/" % id

def getAllSaves():
    saveNumbers = []
    for entry_name in os.listdir(SAVEFOLDER):
        entry_path = os.path.join(SAVEFOLDER, entry_name)
        if os.path.isdir(entry_path):
            sd = np.load(entry_path + '/' + NN_SAVE_DATA)
            generations = sd['generations']

            entry_name = ''.join(filter(str.isdigit, entry_name))
            try:
                maxFitness = sd['maxFitness']
                bestGeneration = sd['bestGeneration']
                try:
                    bestScore = sd['highestScore']
                    level = sd['level']
                    genLog = 'level %d | max fitness : %.1f | best score : %d | best gen : %d' % (
                             level, maxFitness, bestScore, bestGeneration)
                except KeyError:
                    genLog = 'max fitness : %.1f | best gen : %d | !!OtherData missing!!' % (
                             maxFitness, bestGeneration)
            except KeyError:
                genLog = 'Cannot load Generation\'s data'

            saveNumbers.append('Save id : %s, Generations saved : %d,\n  ( %s )'
                               % (entry_name, len(generations), genLog))
    saveNumbers.sort()

    return saveNumbers

def StringToInt(str, default, min, max):
    if(str.isdigit()):
        v = int(str)
        if(v > max):
            v = max
        elif(v < min):
            v = min
        return v
    else:
        return default

def checkIfYes(str):
    str = str.lower()
    if(str == 'y' or str == 'yes'):
        return True
    return False
