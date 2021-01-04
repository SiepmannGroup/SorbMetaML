import numpy as np
import pandas as pd
import torch
import sys, os
import argparse
from torch.utils.data import DataLoader
from models import MLModel, wrapper
from utils import fetch_models

def parse():
    parser = argparse.ArgumentParser(description='Calculate the adsorption isotherms of a zeolite')
    parser.add_argument('zeolite', type=str, help='zeolite to calculate')
    parser.add_argument('model', type=str, default='jobs/final/output/', help='path to trained models')
    parser.add_argument('dataset', type=str, help='path to dataset NPY file')
    parser.add_argument('temperatures', type=float, nargs='+')
    parser.add_argument('-p0', type=float, default=1, help='Starting pressure in bar')
    parser.add_argument('-p1', type=float, default=403.4, help='End pressure in bar')
    parser.add_argument('-i', type=float, default=0.1, help='pressure interval')
    args = parser.parse_args()
    return args

def get_isotherms(model, keys, latents, norms, args):
    temperatures = np.array(args.temperatures)
    predict = wrapper(model, latents, norms)
    zid = np.where(keys == args.zeolite)[0][0]
    pressures = np.arange(args.p0, args.p1, args.i)
    xs = np.zeros((len(pressures), len(temperatures), 2), dtype=np.float32)
    xs[:, :, 1] = 1000 / temperatures
    xs[0, :, 1] = 1000 / temperatures
    xs[:, :, 0] = np.log(pressures).reshape(-1, 1)
    n_zeo = predict(xs.reshape(-1, 2), zid).reshape(len(pressures), -1)
    return pressures, n_zeo


if __name__ == '__main__':

    args = parse()
    names, zeolites, norms, data = fetch_models(args.zeolite, args.model, args.dataset)
    if len(np.shape(names)) == 0:
        names = np.array([names])
    if len(np.shape(norms)) == 0:
        norms = np.array([norms])

    n_all = []
    for name in names:
        net = MLModel(5)
        net.load_state_dict(torch.load(name + '.pt'))
        latents, _ = net.encoder(data)
        pressure, n_zeo = get_isotherms(net, zeolites, latents, norms, args)
        n_all.append(n_zeo)
        print(n_zeo.shape)

    n_all = np.array(n_all)

    n_mean, n_std = np.mean(n_all, axis=0), np.std(n_all, axis=0)
    out = np.zeros((n_mean.shape[0], 2 * n_mean.shape[1] + 1))
    out[:, 0] = pressure
    out[:, 1::2] = n_mean
    out[:, 2::2] = n_std
    

    np.savetxt('isotherms/%s.txt' % args.zeolite, out)
    
