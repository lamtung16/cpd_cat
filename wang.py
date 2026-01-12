import pymanopt
import pymanopt.manifolds
import pymanopt.optimizers
import torch
import time
from copy import deepcopy
from pymanopt.optimizers.optimizer import Optimizer, OptimizerResult
from pymanopt.tools import printer
import numpy as np
from scipy.stats import wishart, matrix_normal

def generate_random_SPD_Wishart(df, scale_matrix):
    """ A function to generate a random SPD matrix from a Wischart distribution.
    Usage: matrix = generate_random_SPD_Wishart(df, scale_matrix)
    Inputs:
        * df: degrees of freedom
        * scale_matrix: a postive definite diagonal matrix
    Outputs:
        * matrix: a random SPD matrix."""
        
    matrix = wishart(df, scale_matrix).rvs()
    return matrix

def generate_random_SPD_mtx(temp, eigsv):
    """ A function to generate a SPD matrix with given eigenvectors and eigenvalues.
    Usage: matrix = generate_random_SPD_mtx(temp, eigsv)
    Inputs:
    * temp: a matrix to generate eigenvectors
    * eigsv: a vecter with positive eigenvalues
    Outputs:
    * matrix: a SPD matrix."""
    
    temp = np.linalg.svd(temp)[0]
    eigsv = eigsv / np.sum(eigsv) + 1e-6 # positive definite
    matrix = temp @ np.diag(eigsv) @ temp.T
    return matrix

def generate_random_mtx_normal(M, U, V):
    """ A function to generate a random matrix from a normal distribution.
    Usage: matrix = generate_random_mtx_normal(M, U, V)
    Inputs:
        * M: a matrix
        * U, V: two postive definite matrices
    Outputs:
        * matrix: a random matrix."""
        
    matrix = matrix_normal(M, U, V).rvs()
    return matrix

class StochasticGradientDescent(Optimizer):
    def __init__(self, step_size=None, num_iter=1, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if step_size is None:
            self._step_size = 1e-3
        else:
            self._step_size = step_size
        self._num_iter = num_iter 
    def run(
        self, problem, *, initial_point=None
    ) -> OptimizerResult:
        manifold = problem.manifold
        gradient = problem.riemannian_gradient
        if initial_point is None:
            x = manifold.random_point()
        else:
            x = initial_point

        start_time = time.time()

        for iteration in range(self._num_iter):
            grad = gradient(x)
            desc_dir = -grad
            x = manifold.retraction(x, self._step_size*desc_dir)

        return self._return_result(
            start_time=start_time,
            point=x,
            cost=None,
            iterations=iteration,
            stopping_criterion=None,
            step_size=self._step_size
        )

def cpd_spd_wang(manifold, X, lambda_0, lambda_1):
    # optimizer
    optimizer0 = StochasticGradientDescent(step_size = lambda_0, num_iter = 1)
    optimizer1 = StochasticGradientDescent(step_size = lambda_1, num_iter = 1)
    # online CPD on Riemannian manifolds
    stat = []
    for matrix in X:
        @pymanopt.function.pytorch(manifold)
        def cost(point):
            temp1 = torch.linalg.eig(torch.from_numpy(matrix))
            temp2 = temp1[0].real
            c = temp1[1].real @ torch.diag(torch.sqrt(1/(torch.where(temp2 > 0, temp2, torch.tensor(1e-6, dtype=torch.float64))))) @ temp1[1].real.T
            temp3 = c @ point @ c
            temp4 = torch.linalg.eig(temp3)[0].real
            temp5 = torch.log(torch.where(temp4 > 0, temp4, torch.tensor(1e-6, dtype=torch.float64)))
            return torch.norm(temp5)**2
        problem = pymanopt.Problem(manifold, cost)
        if np.all(matrix == X[0]):
            result0 = optimizer0.run(problem, initial_point=matrix)
            result1 = optimizer1.run(problem, initial_point=matrix)
        else:
            result0 = optimizer0.run(problem, initial_point=result0.point)
            result1 = optimizer1.run(problem, initial_point=result1.point)
        stat.append(manifold.dist(result0.point, result1.point))
    return stat

def cpd_grassmann_wang(manifold, X, lambda_0, lambda_1):
    # optimizer
    optimizer0 = StochasticGradientDescent(step_size = lambda_0, num_iter = 1)
    optimizer1 = StochasticGradientDescent(step_size = lambda_1, num_iter = 1)
    # online CPD on Riemannian manifolds
    stat = []
    for matrix in X:
        @pymanopt.function.pytorch(manifold)
        def cost(point):
            temp1 = torch.from_numpy(matrix.transpose()) @ point
            temp2 = torch.linalg.svd(temp1)[1]
            temp3 = torch.acos(torch.clamp(temp2, -1+1e-6, 1-1e-6))
            return torch.norm(temp3)**2
        problem = pymanopt.Problem(manifold, cost)
        if np.all(matrix == X[0]):
            result0 = optimizer0.run(problem, initial_point=matrix)
            result1 = optimizer1.run(problem, initial_point=matrix)
        else:
            result0 = optimizer0.run(problem, initial_point=result0.point)
            result1 = optimizer1.run(problem, initial_point=result1.point)
        stat.append(manifold.dist(result0.point, result1.point))
    return stat


def adaptive_threshold(stats, alpha = 0.005, a = 1.64):
    mean = 0.0
    variance = 0.0
    threshold = []
    for stat in stats:
        mean = (1-alpha) * mean + alpha*stat
        variance = (1-alpha) * variance + alpha*stat**2
        sigma = np.sqrt(variance - mean**2)
        threshold.append(mean + a*sigma)
    return threshold