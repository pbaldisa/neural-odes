import numpy as np
import torch


def import_solver(adjoint):
    if adjoint:
        from torchdiffeq import odeint_adjoint as odeint
    else:
        from torchdiffeq import odeint
    return odeint


def get_batch(true_y, t, batch_time, data_size, batch_size):
    s = torch.from_numpy(np.random.choice(np.arange(data_size - batch_time, dtype=np.int64), batch_size, replace=False))
    batch_y0 = true_y[s]  # (M, D)
    batch_t = t[:batch_time]  # (T)
    batch_y = torch.stack([true_y[s + i] for i in range(batch_time)], dim=0)  # (T, M, D)
    return batch_y0, batch_t, batch_y


def train_model(model, optimiser, true_y0, true_y, t, args):
    odeint = import_solver(args.get('adjoint', False))

    losses = []
    niters = args.get('niters', 25000)
    verbose = args.get('verbose', True)
    method = args.get('method', 'dopri5')
    loss_fn = args.get('loss_fn', lambda x, y: torch.norm(x - y))
    tol = args.get('tol', 5e-1)
    eps = args.get('eps', 1e-2)

    # Initial error
    with torch.no_grad():
        pred_y = odeint(model, true_y0, t, method=method)
        loss = loss_fn(pred_y, true_y)
        losses.append(loss.item())
        if verbose:
            print('Iter {:04d} | Total Loss {:.6f}'.format(0, loss.item()))
        prev_loss = loss.item()

    # Training loop
    for itr in range(1, niters + 1):
        optimiser.zero_grad()
        batch_y0, batch_t, batch_y = get_batch(true_y, t, args['batch_time'], args['data_size'], args['batch_size'])
        pred_y = odeint(model, batch_y0, batch_t, method=method)
        loss = loss_fn(pred_y, batch_y)
        loss.backward()
        optimiser.step()

        if itr % args['test_freq'] == 0:
            with torch.no_grad():
                pred_y = odeint(model, true_y0, t, method=method)
                loss = loss_fn(pred_y, true_y)
                losses.append(loss.item())
                if verbose:
                    print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                if abs(loss.item() - prev_loss) < eps and loss.item() < tol:
                    break
                prev_loss = loss.item()

    pred_y = odeint(model, true_y0, t, method=method)
    loss = loss_fn(pred_y, true_y)
    losses.append(loss.item())
    print('Final Loss {:.6f}'.format(loss.item()))

    return losses


'''
def train_model(model, optimiser, true_y0, true_y, t, batch_time, data_size, batch_size, niters, test_freq,
                loss_fn=lambda pred_y, batch_y: torch.norm(pred_y - batch_y), method='dopri5', eps=1e-2,
                tol=5e-1, adjoint=False, verbose=True):
    odeint = import_solver(adjoint)

    losses = []

    # Initial error
    with torch.no_grad():
        pred_y = odeint(model, true_y0, t, method=method)
        loss = loss_fn(pred_y, true_y)
        losses.append(loss.item())
        if verbose:
            print('Iter {:04d} | Total Loss {:.6f}'.format(0, loss.item()))
        prev_loss = loss.item()

    for itr in range(1, niters + 1):
        optimiser.zero_grad()
        batch_y0, batch_t, batch_y = get_batch(true_y, t, batch_time, data_size, batch_size)
        pred_y = odeint(model, batch_y0, batch_t, method=method)
        loss = loss_fn(pred_y, batch_y)
        loss.backward()
        optimiser.step()

        if itr % test_freq == 0:
            with torch.no_grad():
                pred_y = odeint(model, true_y0, t, method=method)
                loss = loss_fn(pred_y, true_y)
                losses.append(loss.item())
                if verbose:
                    print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                if abs(loss.item() - prev_loss) < eps and loss.item() < tol:
                    break
                prev_loss = loss.item()

    pred_y = odeint(model, true_y0, t, method=method)
    loss = loss_fn(pred_y, true_y)
    losses.append(loss.item())
    print('Final Loss {:.6f}'.format(loss.item()))

    return losses
'''
