import numpy as np
import os
os.system("cls")

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
    

    def fit(self, x):
        self.mean = np.mean(x, axis=0)
        x = x - self.mean
        cov_matrix = np.cov(x.T)
        eign_vectors, eign_values = np.linalg.eig(cov_matrix)
        eign_vectors = eign_vectors.T
        indexes = np.argsort(eign_values)[::-1]
        eign_vectors = eign_vectors[indexes]
        self.components = eign_vectors[:self.n_components]
    
    def tarnsform(self, x):
        x = x - self.mean
        return x.dot(self.components.T)
    
    def fit_transform(self, x):
        self.fit(x)
        self.tarnsform(x)