import os
import joblib
import numpy as np
import numpy.typing as npt

from typing import List
from joblib import Parallel, delayed
from const import *

class WrapperABC():
    def __init__(self, fitness, n_features, bees=2, iterations=4, lower=0, upper=1, threshold=0.5, limit=3):
        self.fitness_function = fitness
        self.features = n_features
        self.lb = lower
        self.ub = upper
        self.threshold = threshold
        self.limit = limit
        self.iterations = iterations

        self.bees = bees // 2
        self.solutions = np.zeros((self.bees, self.features))

        for solution in range(len(self.solutions)):
            for feature in range(len(self.solutions[solution])):
                self.solutions[solution, feature] = self.lb + (self.ub - self.lb) * np.random.rand()

        self.fitnesses = np.zeros(self.bees)
        self.accuracy = 0.0
        self.solution = None

        for bee in range(self.bees):
            self.fitnesses[bee] = self.fitness_function(np.where(self.solutions[bee,:] > self.threshold)[0])
            if self.fitnesses[bee] > self.accuracy:
                self.accuracy = self.fitnesses[bee]
                self.solution = self.solutions[bee]
        
        self.limits = np.zeros(self.bees)
        self.candidates = np.zeros((self.bees, self.features))

    # Get initial solutions
    def employedBee(self, bee):
        k = [x for x in range(0, bee)] + [x for x in range(bee + 1, self.bees)]
        k = k[np.random.randint(0, len(k))]
        for feature in range(self.features):
            phi = -1 + 2 * np.random.rand()
            self.candidates[bee, feature] = self.solutions[bee, feature] + phi * (self.solutions[bee, feature] - self.solutions[k, feature])
        boundary = self.candidates[bee,:]
        boundary = np.clip(boundary, self.lb, self.ub)
        self.candidates[bee,:] = boundary

    # Greedy search of bees
    def onlookerBees(self, bee, probability):
        if np.random.rand() < probability[bee]:
            k = [x for x in range(0, bee)] + [x for x in range(bee + 1, self.bees)]
            k = k[np.random.randint(0, len(k))]
            for feature in range(self.features):
                phi = -1 + 2 * np.random.rand()
                self.candidates[bee, feature] = self.solutions[bee, feature] + phi * (self.solutions[bee, feature] - self.solutions[k, feature])
            boundary = self.candidates[bee,:]
            boundary = np.clip(boundary, self.lb, self.ub)
            self.candidates[bee,:] = boundary
            
            fitness = self.fitness_function(np.where(self.candidates[bee,:] > self.threshold)[0])
            if fitness > self.fitnesses[bee]:
                self.solutions[bee,:] = self.candidates[bee,:]
                self.fitnesses[bee] = fitness
                self.limits[bee] = 0
                iteration_fit = 1 / (1 + self.fitnesses)
                probability = iteration_fit / np.sum(iteration_fit)
            else:
                self.limits[bee] += 1

    # Scout if solution of a bee becomes stagnant
    def scoutBees(self, bee):
        if self.limits[bee] >= self.limit:
            for feature in range(self.features):
                self.solutions[bee, feature] = self.lb + (self.ub - self.lb) * np.random.rand()
            self.limits[bee] = 0
            self.fitnesses[bee] = self.fitness_function(np.where(self.solutions[bee,:] > self.threshold)[0])
        if self.fitnesses[bee] > self.accuracy:
            self.accuracy = self.fitnesses[bee]
            self.solution = self.solutions[bee]

    def optimize(self):
        for iteration in range(self.iterations):
            for bee in range(self.bees):
                self.employedBee(bee)

            for bee in range(self.bees):
                fitness = self.fitness_function(np.where(self.candidates[bee,:] > self.threshold)[0])
                if fitness >= self.fitnesses[bee]:
                    self.solutions[bee,:] = self.candidates[bee,:]
                    self.fitnesses[bee] = fitness
                    self.limits[bee] = 0
                else:
                    self.limits[bee] += 1
            iteration_fit = 1 / (1 + self.fitnesses)
            probability = iteration_fit / np.sum(iteration_fit)

            for bee in range(self.bees):
                self.onlookerBees(bee, probability)

            for bee in range(self.bees):
                self.scoutBees(bee)

            print(np.where(self.solution > self.threshold)[0], (np.where(self.solution > self.threshold)[0]).shape[0], self.accuracy)
        return np.where(self.solution > self.threshold)[0]
