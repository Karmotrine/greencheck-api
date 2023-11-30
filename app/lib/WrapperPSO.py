import numpy as np
from joblib import Parallel, delayed

from const import *

class WrapperPSO:
    def __init__(self, fitness, n_features, particles=5, iterations=10, c1=1.49618, c2=1.49618, w=0.7298, threshold=0.6):
        self.fitness = fitness
        self.features = n_features
        self.particles = particles
        self.iterations = iterations
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.threshold = threshold

        self.lb = [0] * self.features  # Lower bound for each feature (0: not selected, 1: selected)
        self.ub = [1] * self.features  # Upper bound for each feature

        # Initialize particle positions and velocities
        self.particle_pos = np.random.uniform(0, 1, (self.particles, self.features))
        self.velocities = np.zeros((self.particles, self.features))

        self.local_best_pos = self.particle_pos.copy()
        self.local_best_scores = np.ones(self.particles)
        self.global_best_index = np.argmin(self.local_best_scores)
        self.global_best_position = self.local_best_pos[self.global_best_index]
    
    def optimize(self):
        for _ in range(self.iterations):
            # Evaluate particles in parallel
            results = Parallel(n_jobs=CORES)(delayed(self.fitness)(np.where(self.particle_pos[i] > self.threshold)[0]) for i in range(self.particles))

            for i, (score, selected_features) in enumerate(results):
                # Update personal best position and score
                if score < self.local_best_scores[i]:
                    self.local_best_scores[i] = score
                    self.local_best_pos[i] = self.particle_pos[i]

                    # Update global best position
                    if score < self.local_best_scores[self.global_best_index]:
                        self.global_best_index = i
                        self.global_best_position = self.local_best_pos[i]

            for i in range(self.particles):
                # Update particle velocities
                r1, r2 = np.random.rand(2)
                self.velocities[i] = self.w * self.velocities[i] + self.c1 * r1 * (self.local_best_pos[i] - self.particle_pos[i]) + self.c2 * r2 * (
                            self.global_best_position - self.particle_pos[i])

                # Update particle positions
                self.particle_pos[i] = self.particle_pos[i] + self.velocities[i]

                # Clamp particle positions to the lower and upper bounds
                self.particle_pos[i] = np.clip(self.particle_pos[i], self.lb, self.ub)

        solution = np.where(self.global_best_position > self.threshold)[0]
        return solution