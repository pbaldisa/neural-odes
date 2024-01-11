# Import libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint

import experiments.helpers.training as train


class ODEnnR(nn.Module):

    def __init__(self):
        super(ODEnnR, self).__init__()

        self.net = nn.Sequential(
            # ODE-Net's neural network architecture: 1 hidden layer with 50 neurons, tanh activation, 2 output neurons
            nn.Linear(2, 50),
            nn.ReLU(),
            nn.Linear(50, 2),
        )

        # Initialize the weights and biases of the network for better convergence
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    # Define the forward pass of the network
    def forward(self, t, y):
        return self.net(y)


def define_and_train(args, adjoint=True):
    # Define the same model for both cases
    model = ODEnnR()

    # Define the true dynamics of the system
    t = torch.linspace(0., args['terminal_time'], args['data_size'])
    true_y0 = torch.tensor([[2., 0.]])
    true_A = torch.tensor([[-0.1, 3.0], [-3.0, -0.1]])
    true_vector_field = lambda t, y: torch.mm(y, true_A)
    with torch.no_grad():
        true_y = odeint(true_vector_field, true_y0, t, method='dopri5')

    # Define the optimizer
    optimiser = optim.RMSprop(model.parameters(), lr=args['learning_rate'])
    # Train the model using the adjoint method
    losses = train.train_model(model, optimiser, true_y0, true_y, t, args, adjoint=adjoint)
    del model
    del optimiser
    return losses[-1]
