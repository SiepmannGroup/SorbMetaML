import argparse
import configparser
import hashlib
import time
import os, random
import numpy as np
import torch


def parse_train():

    parser = argparse.ArgumentParser(description='Train a meta-learning autoencoder.')

    parser.add_argument('data', type=str, help='Dataset to load; should be numpy .npy format')
    parser.add_argument('-e', '--epochs', type=int, default=500, help='Epochs to train')
    parser.add_argument('-s', '--seed', type=int, default=random.randint(0, 255), help='Random seed')
    parser.add_argument('-d', '--device', type=str, default='cuda:0', help='Device which runs training')
    parser.add_argument('--unsup', action='store_true', help='Train in unsupervised mode')
    parser.add_argument('--config', type=str, default='config.ini', help='Configuration file')
    parser.add_argument('--checkpoint', type=str, default=None, help='Model checkpoint to load')

    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.config)
    return args, config



def get_runname(args):
    hash = hashlib.sha1()
    hash.update(str(time.time()).encode('utf-8'))
    hashtag = hash.hexdigest()[:6]
    return hashtag + '-' + str(args.seed)



def parse_eval():

    parser = argparse.ArgumentParser(description='Evaluate models using the given dataset; \
                            outputs latent vectors and reconstruction loss for each task')

    parser.add_argument('data', type=str, help='Dataset to load (base-training); should be numpy .npy format')
    parser.add_argument('model', type=str, help='Model to load; should be PyTorch state_dict')
    parser.add_argument('-z', type=int, default=5, 
                        help='Length of latent vector; should be consistent with the model')
    parser.add_argument('--test', type=str, default='', help='Base-test set to load, use the base training set by default')
    args = parser.parse_args()
    return args


def fetch_models(key, model_path, custom_data_path):
    # whether use one model or all models in a directory
    if '.pt' in model_path:
        names = [model_path.rstrip('.pt')]
    else:
        names = [os.path.join(model_path, n).rstrip('.pt') for n in os.listdir(model_path) if '.pt' in n]

    # whether use training data or custom data
    if custom_data_path[-4:] == '.npy':
        fpath = os.getcwd()
        zeolites = np.genfromtxt(os.path.join(fpath, 'names.csv'), dtype=str)
        norms = np.loadtxt(os.path.join(fpath, 'norms.csv'), delimiter=',', dtype=np.float32)
        data = torch.from_numpy(np.load(custom_data_path).astype(np.float32))
    else:
        zeolites = np.array([key])
        data = np.loadtxt(custom_data_path, dtype=np.float32, skiprows=1)
        #data[:, 1] = (data[:, 1] - 3.049796) / 1.8759118
        #data[:, 2] = (data[:, 2] - 7.4749565) / 3.0644488 
        data = torch.from_numpy(data).t().unsqueeze(0)
        norms = None
        with open(custom_data_path, 'r') as f:
            line = f.readline()
            norms = [np.float32(line.strip())]
    return names, zeolites, norms, data

