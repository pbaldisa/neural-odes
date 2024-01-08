# General imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import application_examples.helpers.training as train

from torchdiffeq import odeint

# Define the same hyperparameters for both cases
args = {
    'method': 'dopri5',
    'data_size': 5000,
    'batch_time': 20,
    'batch_size': 2500,
    'niters': 25000,
    'test_freq': 1,
    'terminal_time': 25.,
    'learning_rate': 1e-4,
    'eps': 1e-2,
    'tol': 1,
    'verbose': True
}


# Define the same model for both cases
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


def get_batch(true_y, t, batch_time, data_size, batch_size):
    s = torch.from_numpy(np.random.choice(np.arange(data_size - batch_time, dtype=np.int64), batch_size, replace=False))
    batch_y0 = true_y[s]  # (M, D)
    batch_t = t[:batch_time]  # (T)
    batch_y = torch.stack([true_y[s + i] for i in range(batch_time)], dim=0)  # (T, M, D)
    return batch_y0, batch_t, batch_y


def train_with_adjoint(adjoint):
    from torchdiffeq import odeint
    from torchdiffeq import odeint_adjoint

    device = torch.device('cuda:'+str(0) if torch.cuda.is_available() else 'cpu')
    print("Using device: ", device)

    # Define the true dynamics of the system
    t = torch.linspace(0., args['terminal_time'], args['data_size']).to(device)
    true_y0 = torch.tensor([[2., 0.]]).to(device)
    true_A = torch.tensor([[-0.1, 3.0], [-3.0, -0.1]]).to(device)
    true_vector_field = lambda t, y: torch.mm(y, true_A).to(device)
    with torch.no_grad():
        true_y = odeint(true_vector_field, true_y0, t, method='dopri5').to(device)

    # Define the model
    model = ODEnnR().to(device)

    # Define the optimizer
    optimiser = optim.RMSprop(model.parameters(), lr=args['learning_rate'])

    # Train the model using the adjoint method
    niters = args.get('niters', 25000)
    verbose = args.get('verbose', False)
    method = args.get('method', 'dopri5')
    loss_fn = args.get('loss_fn', lambda x, y: torch.norm(x - y))
    tol = args.get('tol', 5e-1)
    eps = args.get('eps', 1e-2)

    # Initial error
    with torch.no_grad():
        pred_y = odeint_adjoint(model, true_y0, t, method=method)
        loss = loss_fn(pred_y, true_y)
        if verbose:
            print('Iter {:04d} | Total Loss {:.6f}'.format(0, loss.item()))
        prev_loss = loss.item()

    # Training loop
    for itr in range(1, niters + 1):
        optimiser.zero_grad()
        batch_y0, batch_t, batch_y = get_batch(true_y, t, args['batch_time'], args['data_size'], args['batch_size'])
        pred_y = odeint_adjoint(model, batch_y0, batch_t, method=method)
        loss = loss_fn(pred_y, batch_y)
        loss.backward()
        optimiser.step()

        if itr % args['test_freq'] == 0:
            with torch.no_grad():
                pred_y = odeint_adjoint(model, true_y0, t, method=method)
                loss = loss_fn(pred_y, true_y)
                if verbose:
                    print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                if abs(loss.item() - prev_loss) < eps and loss.item() < tol:
                    break
                prev_loss = loss.item()

    # Measure the loss of the trained model
    with torch.no_grad():
        pred_y = odeint(model, true_y0, t, method='dopri5')
        loss = torch.norm(pred_y - true_y)
        return loss.item()
