import os
import argparse
import imageio
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim


# Define the arguments to be passed to the script
parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=1000)
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
parser.add_argument('--terminal_time', type=float, default=25.)
parser.add_argument('--dirname', type=str, default='linearODE_learningProcess')
args = parser.parse_args()

""" Startup settings """
# Import the correct version depending on arguments (usage of the adjoint method)
if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

# Set the device to use
device = torch.device(
    # TODO change first 'cpu' to mps if possible
    'cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu' if torch.backends.mps.is_available() else 'cpu')

# If the visualization flag is set, create the directory to save the plots and define a figure
if args.viz:
    if not os.path.exists(args.dirname):
        os.makedirs(args.dirname)
    else:
        # Remove all files
        for filename in os.listdir(args.dirname):
            file_path = os.path.join(args.dirname, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax_traj = fig.add_subplot(131, frameon=False)
    ax_phase = fig.add_subplot(132, frameon=False)
    ax_vecfield = fig.add_subplot(133, frameon=False)
    plt.show(block=False)


# Define a Neural Network to learn the vector field
class ODEnn(nn.Module):

    def __init__(self):
        super(ODEnn, self).__init__()

        self.net = nn.Sequential(
            # ODE-Net's neural network architecture: 1 hidden layer with 50 neurons, tanh activation, 2 output neurons
            nn.Linear(2, 50),
            nn.Tanh(),
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


""" Auxiliary functions """
def get_batch():
    s = torch.from_numpy(np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False))
    batch_y0 = true_y[s]  # (M, D)
    batch_t = t[:args.batch_time]  # (T)
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
    return batch_y0.to(device), batch_t.to(device), batch_y.to(device)


# This function plots the results of the simulation
def visualize(true_y, pred_y, odefunc, itr):

    if args.viz:
        ax_traj.cla()
        ax_traj.set_title('Trajectories')
        ax_traj.set_xlabel('t')
        ax_traj.set_ylabel('x,y')
        ax_traj.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 0], 'g-', label="True x")
        ax_traj.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 1], 'b-', label="True y")
        ax_traj.plot(t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 0], 'y--', label="Predicted x")
        ax_traj.plot(t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 1], 'c--', label="Predicted y")
        ax_traj.set_xlim(t.cpu().min(), t.cpu().max())
        ax_traj.set_ylim(-2, 2)
        ax_traj.legend()

        ax_phase.cla()
        ax_phase.set_title('Phase Portrait')
        ax_phase.set_xlabel('x')
        ax_phase.set_ylabel('y')
        ax_phase.plot(true_y.cpu().numpy()[:, 0, 0], true_y.cpu().numpy()[:, 0, 1], 'g-', label="True")
        ax_phase.plot(pred_y.cpu().numpy()[:, 0, 0], pred_y.cpu().numpy()[:, 0, 1], 'y--', label="Predicted")
        ax_phase.set_xlim(-2, 2)
        ax_phase.set_ylim(-2, 2)
        ax_phase.legend()

        ax_vecfield.cla()
        ax_vecfield.set_title('Learned Vector Field')
        ax_vecfield.set_xlabel('x')
        ax_vecfield.set_ylabel('y')

        y, x = np.mgrid[-2:2:21j, -2:2:21j]
        dydt = odefunc(0, torch.Tensor(np.stack([x, y], -1).reshape(21 * 21, 2)).to(device)).cpu().detach().numpy()
        mag = np.sqrt(dydt[:, 0] ** 2 + dydt[:, 1] ** 2).reshape(-1, 1)
        dydt = (dydt / mag)  # el fa unitari
        dydt = dydt.reshape(21, 21, 2)

        ax_vecfield.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")
        ax_vecfield.set_xlim(-2, 2)
        ax_vecfield.set_ylim(-2, 2)

        fig.tight_layout()
        plt.savefig(args.dirname + '/{:03d}'.format(itr))
        plt.draw()
        plt.pause(0.001)


def plot_learning_curve(losses):
    plt.figure(figsize=(8, 6))
    time_intervals = list(range(0, args.niters, args.test_freq))
    plt.plot(time_intervals, losses, marker='o', linestyle='-')
    plt.title('Loss Over Time')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.draw()
    plt.pause(2)
    plt.savefig(args.dirname + '/learning_curve')


def create_gif(gif_filename, fps=2):
    images = []
    for filename in sorted(os.listdir(args.dirname)):
        if filename.endswith(".png"):
            file_path = os.path.join(args.dirname, filename)
            images.append(imageio.v2.imread(file_path))
    imageio.mimsave(gif_filename, images, fps=fps)

if __name__ == '__main__':
    # Define the time points at which the system will be simulated
    t = torch.linspace(0., args.terminal_time, args.data_size).to(device)

    """ Define the system to simulate """
    # Define the initial condition and the matrix A of the system to simulate
    true_y0 = torch.tensor([[2., 0.]]).to(device)
    true_A = torch.tensor([[-0.1, -3.0], [3.0, -0.1]]).to(device)

    # This defines the vector field of the system to simulate
    lambda_func = lambda t, y: torch.mm(y, true_A)

    # Compute the real solution of the system using a numerical solver
    with torch.no_grad():
        true_y = odeint(lambda_func, true_y0, t, method=args.method)


    """ Define the Neural Network to learn the vector field """
    func = ODEnn().to(device)

    optimizer = optim.RMSprop(func.parameters(), lr=1e-3)

    img_counter = 0
    losses = []

    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = get_batch()
        pred_y = odeint(func, batch_y0, batch_t, method=args.method).to(device) # Forward pass amb el batch, resolent IVP amb el model de la ODE-Net
        loss = torch.mean(torch.abs(pred_y - batch_y))  # Calcula la loss
        loss.backward() # Backward pass
        optimizer.step()

        if itr % args.test_freq == 0:
            with torch.no_grad():
                pred_y = odeint(func, true_y0, t, method=args.method)
                loss = torch.mean(torch.abs(pred_y - true_y))
                losses.append(loss.item())
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                visualize(true_y, pred_y, func, img_counter)
                img_counter += 1

    create_gif(args.dirname + '/linearODE.gif')
    plot_learning_curve(losses)
