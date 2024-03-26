# Mutations
import numpy as np


class AdditiveGaussianMutation:
    def __init__(self, std=0.1):
        self.std = std

    def mutate(self, x):
        xp = x.copy()
        idx = np.random.randint(xp.shape[0])
        xp[idx] = np.inf
        while xp[idx] < 0 or xp[idx] > 1:
            z = np.random.randn() * self.std
            xp[idx] = x[idx] + z
        return xp


class Crossover:
    def __init__(self):
        pass

    def cross(self, parent1, parent2):
        D = parent1.shape[0]
        r = np.random.randint(D)
        soln = np.concatenate((parent1[:r], parent2[r:]))
        return soln
