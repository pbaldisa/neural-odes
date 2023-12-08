from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from glob import glob
from IPython.display import display, HTML
from matplotlib import gridspec


def show_gif(images, results_dir, duration=250, gif_width=1000):
    img, *imgs = [Image.open(img) for img in images]
    gif_path = os.path.join(results_dir, "cnf-viz.gif")
    img.save(fp=gif_path, format='GIF', append_images=imgs, save_all=True, duration=duration, loop=0)
    display(HTML(f'<img src="{gif_path}" width="{gif_width}">'))
    img.close()
    for im in imgs:
        im.close()


def plot_flow(get_batch, results_dir, odeint, func, p_z0, t0, t1, device):
    viz_samples = 30000
    viz_timesteps = 41
    target_sample, _ = get_batch(viz_samples)

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    with torch.no_grad():
        # Generate evolution of samples
        z_t0 = p_z0.sample([viz_samples]).to(device)
        logp_diff_t0 = torch.zeros(viz_samples, 1).type(torch.float32).to(device)

        z_t_samples, _ = odeint(
            func,
            (z_t0, logp_diff_t0),
            torch.tensor(np.linspace(t0, t1, viz_timesteps)).to(device),
            atol=1e-5,
            rtol=1e-5,
            method='dopri5'
        )

        # Generate evolution of density
        x = np.linspace(-1.5, 1.5, 100)
        y = np.linspace(-1.5, 1.5, 100)
        points = np.vstack(np.meshgrid(x, y)).reshape([2, -1]).T

        z_t1 = torch.tensor(points).type(torch.float32).to(device)
        logp_diff_t1 = torch.zeros(z_t1.shape[0], 1).type(torch.float32).to(device)

        z_t_density, logp_diff_t = odeint(
            func,
            (z_t1, logp_diff_t1),
            torch.tensor(np.linspace(t1, t0, viz_timesteps)).to(device),
            atol=1e-5,
            rtol=1e-5,
            method='dopri5',
        )

        # Create plots for each timestep
        '''
        for (t, z_sample, z_density, logp_diff) in zip(
                np.linspace(t0, t1, viz_timesteps),
                z_t_samples, z_t_density, logp_diff_t
        ):
            fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=200)
            plt.suptitle(f'{t:.2f}s')
            plt.tight_layout()

            axes[0].set_title('Target')
            axes[0].hist2d(*target_sample.detach().cpu().numpy().T, bins=300, density=True,
                           range=[[-1.5, 1.5], [-1.5, 1.5]])

            axes[1].set_title('Samples')
            axes[1].hist2d(*z_sample.detach().cpu().numpy().T, bins=300, density=True, range=[[-1.5, 1.5], [-1.5, 1.5]])

            logp = p_z0.log_prob(z_density) - logp_diff.view(-1)
            axes[2].set_title('Log Probability')
            axes[2].tricontourf(*z_t1.detach().cpu().numpy().T, np.exp(logp.detach().cpu().numpy()), 200)

            plt.savefig(os.path.join(results_dir, f"cnf-viz-{int(t * 1000):05d}.jpg"),
                        pad_inches=0.2, bbox_inches='tight')
            plt.close()
        '''
        # Create plots for each timestep
        for (t, z_sample, z_density, logp_diff) in zip(
                np.linspace(t0, t1, viz_timesteps),
                z_t_samples, z_t_density, logp_diff_t
        ):
            fig = plt.figure(figsize=(18, 6), dpi=200)
            gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 1])

            # Plot Target
            ax0 = plt.subplot(gs[0])
            ax0.set_title('Target')
            ax0.hist2d(*target_sample.detach().cpu().numpy().T, bins=300, density=True,
                       range=[[-1.5, 1.5], [-1.5, 1.5]])

            # Plot Samples
            ax1 = plt.subplot(gs[1])
            ax1.set_title('Samples')
            ax1.hist2d(*z_sample.detach().cpu().numpy().T, bins=300, density=True, range=[[-1.5, 1.5], [-1.5, 1.5]])

            # Plot Log Probability
            ax2 = plt.subplot(gs[2])
            ax2.set_title('Log Probability')
            logp = p_z0.log_prob(z_density) - logp_diff.view(-1)
            ax2.tricontourf(*z_t1.detach().cpu().numpy().T, np.exp(logp.detach().cpu().numpy()), 200)

            # Plot Vector Field
            ax3 = plt.subplot(gs[3])
            ax3.set_title('Vector Field')

            # Reshape t to have two dimensions before passing to HyperNetwork
            t_for_hypernet = torch.tensor([[t]]).to(device)

            # Calculate vector field values at points
            func_values = func(t_for_hypernet, [z_density, logp_diff] )
            ax3.quiver(z_t1[:, 0].detach().cpu().numpy(), z_t1[:, 1].detach().cpu().numpy(),
                       func_values[:, 0].detach().cpu().numpy(), func_values[:, 1].detach().cpu().numpy(),
                       color='black', scale=20, scale_units='xy', angles='xy')

            plt.suptitle(f'{t:.2f}s')
            plt.tight_layout()

            plt.savefig(os.path.join(results_dir, f"cnf-viz-{int(t * 1000):05d}.jpg"),
                        pad_inches=0.2, bbox_inches='tight')
            plt.close()
        # Display GIF
        gif_images = sorted(glob(os.path.join(results_dir, f"cnf-viz-*.jpg")))
        show_gif(gif_images, results_dir)
