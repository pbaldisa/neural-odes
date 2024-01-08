import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import application_examples.helpers.training as train
import timeit

from torchdiffeq import odeint_adjoint
from torchdiffeq import odeint

device = torch.device('cuda:'+str(0) if torch.cuda.is_available() else 'cpu')

args = {
    'method': 'dopri5',
    'data_size': 5000,
    'batch_time': 20,
    'batch_size': 2500,
    'niters': 100000,
    'test_freq': 1000,
    'terminal_time': 25.,
    'learning_rate': 1e-4,
    'eps': 1e-2,
    'tol': 1
}