import os
import joblib
import numpy as np
import numpy.typing as npt

from typing import List
from joblib import Parallel, delayed
from const import *

class WrapperACO:
    def __init__(self, fitness, n_features, ants=20, iterations=50, alpha=1.0, beta=1.0, rho=0.1, Q=1.0, debug=False, parrallel=False, cores=0, accuracy=0.0):
        self.ants = ants
        self.iterations = iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q

        self.debug = debug
        self.parallelization = parrallel

        self.fitness = fitness
        self.features = n_features
        self.tau = np.ones((self.features, self.features))
        self.eta = np.ones((self.features, self.features))
        self.accuracy = accuracy
        self.solution = np.arange(0, self.features)

        if cores == 0 and self.parallelization:
            self.cores = os.cpu_count() // 2
            self.cores = self.cores if self.cores // 2 >= os.cpu_count() else self.cores + 1 
        else:
            self.cores = cores
        assert self.cores <= os.cpu_count()

        if self.debug:
            print(f"Settings:\nalpha={self.alpha} beta={self.beta} rho={self.rho} Q={self.Q}\nants={self.ants} iterations={self.iterations} features={self.features}")
            if self.parallelization:
                print(f"Parralelization={self.parallelization} cores={self.cores}")

    # Calculate transition probability and select a node based on the probability
    def transition(self, current: int, visited_nodes: List) -> int:
        N = (self.tau[current] ** self.alpha) * (self.eta[current] ** self.beta)
        M = N
        M[visited_nodes] = 0 # Visited Nodes is set to 0 Probability
        M[current] = 0 # Current Node is set to 0 Probability
        P = N / M.sum()
        node = np.random.choice(len(P), p=P)
        return node

    # Update delta tau for each path
    def delta_tau(self, solution: List, quality: float, delta_tau: npt.NDArray):
        for node in range(len(solution)-1):
            # Update edge(i, j) delta tau based on quality of solution for this path
            delta_tau[solution[node]][solution[node+1]] = self.Q / (1 - quality + 1) 
        return delta_tau
    
    # Update tau pheromone by applying evaporation and adding delta_tau to each tau edge explored
    def update_pheromone(self, delta_tau: npt.NDArray):
        return (1 - self.rho) * self.tau + delta_tau
    
    # Ants explore the graph
    def tour(self, ant):
        node = np.random.randint(self.features) # Select a random node
        subset_amount = np.random.randint(1, self.features) # Initialize a random number of features
        path = [node] # Start with arbitrary node

        while len(path) < subset_amount:
            node = self.transition(node, path) # Transitions from current node 'i' to the next node 'j'
            path.append(node) # Append the transitioned node
        
        solution = np.array(path) # Get the solution as an numpy array
        fitness = self.fitness(solution) # Evaluate subset solution

        if self.debug >= 5:
            print(f"{ant+1}: {solution} {fitness*100:.2f} {len(solution)}")

        return solution, fitness

    # Apply ACO as feature selector
    def optimize(self):
        for iteration in range(self.iterations):
            if self.debug:
                print(f"Iteration {iteration+1}", end=" ")

            local_solutions = [] # Store local solutions
            delta_tau = np.zeros_like(self.tau) # Initialize initial tau values

            if self.parallelization:
                local_solutions = Parallel(n_jobs=self.cores)(delayed(self.tour)(ant) for ant in range(self.ants))
            else:
                for ant in range(self.ants):
                    local_solution, local_accuracy = self.tour(ant)
                    local_solutions.append((local_solution, local_accuracy))

            for solution, accuracy in local_solutions:
                delta_tau = self.delta_tau(solution, accuracy, delta_tau) # Update delta tau for path(solution) found
                if accuracy > self.accuracy: # Obtain best local solution, if better local solution is the global solution
                    self.accuracy = accuracy
                    self.solution = solution

            self.tau = self.update_pheromone(delta_tau)

            if self.debug:
                print(f"Solution:\t {self.solution} {self.accuracy:02f} {len(self.solution)} {self.solution.shape[0]}")

        return self.solution, self.accuracy