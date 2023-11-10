import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_learning_curve(losses, test_freq):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    time_intervals = [i*test_freq for i in range(len(losses))]
    ax.plot(time_intervals, losses, marker='o', linestyle='-')
    ax.set_title('Loss Over Time')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Loss')
    ax.grid(True)
    plt.show()
    fig.clear()


def plot_ode(y, odefunc):
    max_coord = torch.max(y).item() + 0.1

    fig = plt.figure(figsize=(12, 4), facecolor="white")
    fig.suptitle("ODE's dynamics")
    ax_phase = fig.add_subplot(131)
    ax_vecfield = fig.add_subplot(132)

    ax_phase.set_title("Phase Portrait")
    ax_phase.set_xlabel("x")
    ax_phase.set_ylabel("y")
    ax_phase.plot(y.cpu().numpy()[:,0, 0], y.cpu().numpy()[:,0, 1], 'g-', label="Actual solution")
    ax_phase.set_xlim(-max_coord, max_coord)
    ax_phase.set_ylim(-max_coord, max_coord)
    ax_phase.legend()

    ax_vecfield.set_title("Vector field")
    y, x = np.mgrid[-max_coord:max_coord:21j, -max_coord:max_coord:21j]
    dydt = odefunc(0, torch.Tensor(np.stack([x, y], -1).reshape(21 * 21, 2))).detach().numpy()
    norm = np.sqrt(dydt[:, 0] ** 2 + dydt[:, 1] ** 2).reshape(-1, 1)
    zero_magnitude_rows = np.where(norm == 0)[0]
    norm[zero_magnitude_rows] = 1.0
    dydt = (dydt / norm)
    dydt = dydt.reshape(21, 21, 2)

    ax_vecfield.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")
    ax_vecfield.set_xlim(-max_coord, max_coord)
    ax_vecfield.set_ylim(-max_coord, max_coord)

    fig.tight_layout()
    plt.show()
    fig.clear()


def plot_linearODE_result(true_y, pred_y, t, odefunc, losses = [], test_freq = 0):
    max_coord = torch.max(true_y).item() # This works in this example because it is contractive

    if len(losses) == 0:
        fig = plt.figure(figsize=(12, 4), facecolor='white')
        ax_traj = fig.add_subplot(131, frameon=False)
        ax_phase = fig.add_subplot(132, frameon=False)
        ax_vecfield = fig.add_subplot(133, frameon=False)
    else:
        fig = plt.figure(figsize=(12, 8), facecolor='white')
        ax_traj = fig.add_subplot(221, frameon=False)
        ax_phase = fig.add_subplot(222, frameon=False)
        ax_vecfield = fig.add_subplot(223, frameon=False)
        ax_learncurve = fig.add_subplot(224, frameon=False)

    ax_traj.set_title('Trajectories')
    ax_traj.set_xlabel('t')
    ax_traj.set_ylabel('x,y')
    ax_traj.plot(t.numpy(), true_y.numpy()[:, 0, 0], 'g-', label="True x")
    ax_traj.plot(t.numpy(), true_y.numpy()[:, 0, 1], 'b-', label="True y")
    ax_traj.plot(t.numpy(), pred_y.detach().numpy()[:, 0, 0], 'y--', label="Predicted x")
    ax_traj.plot(t.numpy(), pred_y.detach().numpy()[:, 0, 1], 'c--', label="Predicted y")
    ax_traj.set_xlim(t.min(), t.max())
    ax_traj.set_ylim(-max_coord, max_coord)
    ax_traj.legend()

    ax_phase.set_title('Phase Portrait')
    ax_phase.set_xlabel('x')
    ax_phase.set_ylabel('y')
    ax_phase.plot(true_y.detach().numpy()[:, 0, 0], true_y.detach().numpy()[:, 0, 1], 'g-', label="True")
    ax_phase.plot(pred_y.detach().numpy()[:, 0, 0], pred_y.detach().numpy()[:, 0, 1], 'y--', label="Predicted")
    ax_phase.set_xlim(-max_coord, max_coord)
    ax_phase.set_ylim(-max_coord, max_coord)
    ax_phase.legend()

    ax_vecfield.set_title('Learned Vector Field')
    ax_vecfield.set_xlabel('x')
    ax_vecfield.set_ylabel('y')

    y, x = np.mgrid[-max_coord:max_coord:21j, -max_coord:max_coord:21j]
    dydt = odefunc(0, torch.Tensor(np.stack([x, y], -1).reshape(21 * 21, 2))).detach().numpy()
    mag = np.sqrt(dydt[:, 0] ** 2 + dydt[:, 1] ** 2).reshape(-1, 1)
    dydt = (dydt / mag)  # el fa unitari
    dydt = dydt.reshape(21, 21, 2)

    ax_vecfield.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")
    ax_vecfield.set_xlim(-max_coord, max_coord)
    ax_vecfield.set_ylim(-max_coord, max_coord)

    if len(losses) != 0:
        time_intervals = [i * test_freq for i in range(len(losses))]
        ax_learncurve.plot(time_intervals, losses, marker='o', linestyle='-')
        ax_learncurve.plot(time_intervals, losses, marker='o', linestyle='-')
        ax_learncurve.set_title('Loss Over Time')
        ax_learncurve.set_xlabel('Iterations')
        ax_learncurve.set_ylabel('Loss')
        ax_learncurve.grid(True)

    fig.tight_layout()
    plt.show()
    fig.clear()