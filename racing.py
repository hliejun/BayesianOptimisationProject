# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from bayes_opt.bayesian_optimization import BayesianOptimization
from bayes_opt.helpers import UtilityFunction

__all__ = ["BayesianOptimization", "UtilityFunction"]

# read excel file
raceDataFrame = pd.read_excel('data.xlsx', sheetname='Sheet1')
raceData = np.array(raceDataFrame.values)
raceData[np.argsort(raceData[:, 12])]
numberOfRaces = int(raceData[len(raceData) - 1][12])
proportion = 1/3
separationPoint = int(proportion * (numberOfRaces - 1))

# init array containing all races
#(each race is a sequence of horse profile(X) and timing(Y), with timing as the first index)
races = []
for _ in range (0, numberOfRaces):
    races.append([])

# separate data into X and Y, indexed by races (each as an array of horse-in-race entry)
for i in range (0, len(raceData)):
    # group into races
    currentRow = raceData[i]
    raceNumber = int(currentRow[12])
    rowData = currentRow[0:12]
    races[raceNumber-1].append(rowData)

# convert races into np.arrays
races = list(map(lambda x: np.array(x), races))

# split data into different sets
np.random.shuffle(races)
historicalRaces = races[:separationPoint]
futureRaces = races[separationPoint:]

# all races ready as an array: races (randomised order)
# historical races are used to initialise the GP model within the Bayesian Optimiser
# future races are used to simulate the selection performed by the Bayesian Optimiser

parameter_bounds = {'win': (0, 1000),
                    'horse_rating': (0, 300),
                    'horse_weight': (300, 800),
                    'handicapped_weight': (0, 100),
                    'carried_weight': (0, 100),
                    'lane_number': (1, 18),
                    'running_one': (1, 18),
                    'running_two': (1, 18),
                    'placing': (1, 18),
                    'length_behind_winner': (0, 100),
                    'distance': (800, 2000)
                   }

# create and init a new bayesian optimiser that follows the param limits specified above
bayesianOptimiser = BayesianOptimization(parameter_bounds)

acquisitionFunctionFlag = 'poi' # poi (probability of improvement) | ucb (upper confidence bound) | ei (expected improvement)
kappa = 2.576
xi = 0.0
theta0 = 0.02
nugget = 0.01

bayesianOptimiser.setup(acquisitionFunctionFlag, kappa, xi)

# strip historical races
historicalHorses = []

for race in historicalRaces:
    for horse in race:
        historicalHorses.append(horse)

bayesianOptimiser.initialize(np.vstack(historicalHorses))

# can only select from n sets of races, each time picking 1 race to watch
# after watching n races, predict for a race and place a bet ... see if
# successful ...
splitRatio = 0.5
splitPoint = int(splitRatio * (len(futureRaces) - 1))
np.random.shuffle(futureRaces)
trainRaces = races[:splitPoint]
testRaces = races[splitPoint:]

# group training races into sets/batches
batchSize = 3
split_points = range(batchSize, len(trainRaces), batchSize)
batchedTrainRaces = np.split(trainRaces, split_points)

# call minimise to optimise
bayesianOptimiser.minimize(np.array(batchedTrainRaces))
