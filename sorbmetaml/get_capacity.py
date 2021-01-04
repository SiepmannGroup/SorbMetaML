import numpy as np
import pandas as pd
import torch
import sys, os
import argparse
from torch.utils.data import DataLoader
from models import MLModel, wrapper
from utils import fetch_models

eos = lambda p, V, T: p * V / R / T
R = 0.08314



def parse():
    parser = argparse.ArgumentParser(description='Calculate the deliverable capacity as a function of temperature')
    parser.add_argument('zeolite', type=str, help='zeolite to calculate')
    parser.add_argument('model', type=str, default='jobs/final/output/', help='path to trained models')
    parser.add_argument('dataset', type=str, help='path to dataset NPY file')
    parser.add_argument('-p', type=float, default=30, help='Adsorption pressure')
    parser.add_argument('-f', type=float, default=0.7, help='Volume fraction of zeolite in the tank')
    parser.add_argument('-t0', type=float, default=77, help='Starting temperature')
    parser.add_argument('-t1', type=float, default=279, help='End temperature')
    parser.add_argument('-i', type=float, default=0.1, help='temperature interval')
    parser.add_argument('-O', help='calculate and print optimal temperature', action='store_true')
    parser.add_argument('-L', help='output loading values', action='store_true')
    args = parser.parse_args()
    return args

def get_capacity_curve(model, keys, latents, norms, args):
    predict = wrapper(model, latents, norms)
    zid = np.where(keys == args.zeolite)[0][0]
    frac = args.f
    p_deliver = np.array([2.71, args.p])
    temps = np.arange(args.t0, args.t1, args.i)
    size = len(temps)
    xs = np.zeros((len(p_deliver), size, 2), dtype=np.float32)
    xs[:, :, 1] = 1000 / temps
    xs[0, :, 1] = 1000 / (temps)
    xs[:, :, 0] = np.log(p_deliver.reshape(-1, 1))
    dn_h2 = eos(p_deliver.reshape(-1, 1), 1 - frac, temps.reshape(1, -1)) - eos(p_deliver[0], 1 - frac, temps.reshape(1, -1))
    n_zeo = predict(xs.reshape(-1, 2), zid).reshape(-1, size) * frac
    n_work = (n_zeo - n_zeo[0, :] + dn_h2) * 2.0158
    return temps, n_work[1, :], n_zeo[0, :], n_zeo[1, :]

def calc_optimal_tempeature(temps, n_work_all):
    indices = np.argmax(n_work_all, axis=1)
    optimal_t = temps[indices]
    return np.mean(optimal_t), np.std(optimal_t)


if __name__ == '__main__':

    args = parse()
    names, zeolites, norms, data = fetch_models(args.zeolite, args.model, args.dataset)
    
    if len(np.shape(names)) == 0:
        names = np.array([names])
    if len(np.shape(norms)) == 0:
        norms = np.array([norms])
    n_work_all = []
    n_full_all = []
    n_empty_all = []
    for name in names:
        net = MLModel(5)
        net.load_state_dict(torch.load(name + '.pt'))
        latents, indices = net.encoder(data)
        temps, n_work, n_empty, n_full = get_capacity_curve(net, zeolites, latents, norms, args)
        n_work_all.append(n_work)
        n_empty_all.append(n_empty)
        n_full_all.append(n_full)
    n_work_all = np.vstack(n_work_all)
    n_empty_all = np.vstack(n_empty_all)
    n_full_all = np.vstack(n_full_all)

    n_mean, n_std = np.mean(n_work_all, axis=0), np.std(n_work_all, axis=0)
    n_empty_mean, n_empty_std = np.mean(n_empty_all, axis=0), np.std(n_empty_all, axis=0)
    n_full_mean, n_full_std = np.mean(n_full_all, axis=0), np.std(n_full_all, axis=0)
   
    if args.L:
        out = np.vstack([temps, n_mean, n_std, n_empty_mean, n_empty_std, n_full_mean, n_full_std]).T
    else:
        out = np.vstack([temps, n_mean, n_std]).T
    

    np.savetxt('capacity/%s_capacity.txt' % args.zeolite, out)
    
    if args.O:
        optimal_t = calc_optimal_tempeature(temps, n_work_all)
        print("Optimal temperature: %.1f +/- %.1f K" % optimal_t)
