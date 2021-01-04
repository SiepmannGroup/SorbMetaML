import numpy as np
import pandas as pd
import torch
import sys, os
import argparse
from torch.utils.data import DataLoader
from models import MLModel, wrapper
from utils import fetch_models
from evaluate import get_latent_vectors


def geteos(p1):
    if p1 <= 30:
        return lambda p, V, T: p * V / R / T
    elif p1 == 54.6:
        return lambda p, V, T: p * V / R / T / (0.9934 + 21.25/T - 1440/T**2)
    else:
        print("Warning: no hyrdogen EOS available for this pressure, use ideal gas law.")
        return lambda p, V, T: p * V / R / T
R = 0.08314


def parse():
    parser = argparse.ArgumentParser(description='Calculate the optimal temperatures of zeolites')
    parser.add_argument('model', type=str, default='jobs/final/output/', help='path to trained models')
    parser.add_argument('dataset', type=str, help='path to dataset NPY file')
    parser.add_argument('-p', type=float, default=30, help='Adsorption pressure')
    parser.add_argument('-p0', type=float, default=2.71, help='Depletion pressure')
    parser.add_argument('-f', type=float, default=0.7, help='Volume fraction of zeolite in the tank')
    parser.add_argument('-s', type=float, default=0, help='Temperature swing between adsorption and desorption')
    args = parser.parse_args()
    return args

def get_optimal_temps(net, latents, norms, args):
    predict = wrapper(net, latents, norms)
    optimal_temps_all = []
    optimal_capacities_all = []
    t_swing = args.s
    frac = args.f
    p_deliver = np.array([args.p0, args.p])
    eos = geteos(args.p)
    temps = np.arange(45, 300 - t_swing, 0.1)
    size = len(temps)
    xs = np.zeros((len(p_deliver), size, 2), dtype=np.float32)
    xs[:, :, 1] = 1000 / (temps - t_swing)
    xs[0, :, 1] = 1000 / temps
    xs[:, :, 0] = np.log(p_deliver.reshape(-1, 1))
    dn_h2 = eos(p_deliver.reshape(-1, 1), 1 - frac, temps.reshape(1, -1) - t_swing) - eos(p_deliver[0], 1 - frac, temps.reshape(1, -1))
    for i in range(len(zeolites)):
        n_zeo = predict(xs.reshape(-1, 2), i).reshape(-1, size) * frac
        n_work = (n_zeo - n_zeo[0, :] + dn_h2) * 2.0158
        # first search maximum > 77 K, otherwise extrapolate
        tid = np.argmax(n_work[:, temps >= 77.0], axis=1)
        if tid[1] == 0:
            tid = np.argmax(n_work, axis=1)
        else:
            tid[1] += np.where(abs(temps - 77.0) < 0.01)[0][0]
        optimal_temps_all.append([temps[x] for x in tid])
        optimal_capacities_all.append([n_work[i, x] for i, x in enumerate(tid)])
    optimal_temps_all = np.array(optimal_temps_all)[:, 1:]
    optimal_capacities_all = np.array(optimal_capacities_all)[:, 1:]

    optimal_temps = optimal_temps_all[:, 0]
    optimal_capacities = optimal_capacities_all[:, 0]
    excess_capacities = [optimal_capacities[i] - \
                            (eos(p_deliver[1], 2.0158, optimal_temps[i] - t_swing)- eos(args.p0, 2.0158, optimal_temps[i]) ) \
                        for i in range(len(zeolites))]
    return optimal_temps, optimal_capacities, excess_capacities


if __name__ == '__main__':
    t_all = []
    n_all = []
    e_all = []

    args = parse()

    # whether use one model or all models in a directory
    if '.pt' in args.model:
        names = [args.model.strip('.pt')]
    else:
        names = [os.path.join(args.model, n).rstrip('.pt') for n in os.listdir(args.model) if '.pt' in n]

    zeolites = np.genfromtxt(os.path.join(os.getcwd(), 'names.csv'), dtype=str)
    norms = np.loadtxt(os.path.join(os.getcwd(), 'norms.csv'), delimiter=',', dtype=np.float32)
    if zeolites.shape == ():
        zeolites = [zeolites.tolist()]
        norms = [norms.tolist()]

    print(names)
    data = np.load(args.dataset).astype(np.float32)
    for name in names:
        net = MLModel(5)
        net.load_state_dict(torch.load(name + '.pt'))
        latents = get_latent_vectors(data, net, device='cpu')[:, :-1].astype(np.float32)
        t_cur, n_cur, e_cur = get_optimal_temps(net, latents, norms, args)
        t_all.append(t_cur)
        n_all.append(n_cur)
        e_all.append(e_cur)
    
    t_mean, t_std = np.mean(t_all, axis=0), np.std(t_all, axis=0)
    n_mean, n_std = np.mean(n_all, axis=0), np.std(n_all, axis=0)
    e_mean, e_std = np.mean(e_all, axis=0), np.std(e_all, axis=0)

    df = pd.DataFrame({'Zeolite': zeolites, 
                    'Optimal temperature (K)': t_mean, 
                    'Optimal temperature error (K)': t_std, 
                    'Tank capacity (mol)': n_mean,
                    'Tank capacity error (mol)': n_std,
                    'Excess capacity (mol)': e_mean,
                    'Excess capacity error (mol)': e_std
                    },)
    df.sort_values('Optimal temperature (K)', ascending=False)
    df.to_csv('temps-p0%s-p%s-f%.1f-swing%d.csv' % (args.p0, args.p, args.f, args.s))
