import os, argparse
import pandas as pd
import torch
import numpy as np
from models import MLModel, wrapper
from evaluate import get_latent_vectors

def decoder_gradient(decoder, z, x, output_y=False):
    device = 'cuda:0' if torch.cuda.is_available else 'cpu'
    z = torch.tensor(z).to(device)
    x = torch.tensor(x).to(device)
    decoder.to(device)
    x.requires_grad = True
    y_pred = decoder(z, x)
    torch.sum(y_pred).backward()
    if output_y:
        return x.grad.cpu().numpy(), y_pred.detach().cpu().numpy().reshape(z.shape[0], -1)
    else:
        return x.grad.cpu().numpy()

def parse():
    parser = argparse.ArgumentParser(description='Calculate heat of adsorption from the neural network')
    parser.add_argument('model', type=str, help='path to the trained model')
    parser.add_argument('dataset', type=str, help='path to dataset NPY file')
    parser.add_argument('p', type=float, help='Pressure in bar')
    parser.add_argument('t', type=float, help='Temperature in K')
    parser.add_argument('-r', action='store_true', help='Remove highest and lowest outliers in models')
    args = parser.parse_args()
    return args

def get_heat_of_adsorption(net, latents, p, t):
    #ts = 1000 / np.linspace(77, 275.9, args.r, dtype=np.float32)
    #ps = np.linspace(0, 6, args.r, dtype=np.float32)
    x = np.tile(np.array([np.log(p), 1000/t]).reshape(2, -1), [latents.shape[0], 1, 1]).astype(np.float32)
    grad, y_pred = decoder_gradient(net.decoder, latents, x, output_y=True)
    q_st = 8.314 * grad[:, 1, :] / grad[:, 0, :]
    return q_st.ravel()
    

if __name__ == "__main__":
    args = parse()
    
    # whether use one model or all models in a directory
    if '.pt' in args.model:
        names = [args.model.rstrip('.pt')]
    else:
        names = [os.path.join(args.model, n).rstrip('.pt') for n in os.listdir(args.model) if '.pt' in n]

    zeolites = np.genfromtxt(os.path.join(os.getcwd(), 'names.csv'), dtype=str)

    data = np.load(args.dataset).astype(np.float32)
    q_all = []
    for name in names:
        net = MLModel(5)
        net.load_state_dict(torch.load(name + '.pt'))
        latents = get_latent_vectors(data, net, device='cuda:0')[:, :-1].astype(np.float32)
        q_st = get_heat_of_adsorption(net, latents, args.p, args.t)
        q_all.append(q_st)
        
    q_all = np.array(q_all)
    if args.r:
        mean_over_z = np.mean(q_all, axis=1)
        max_idx = mean_over_z != np.max(mean_over_z, axis=0)
        min_idx = mean_over_z != np.min(mean_over_z, axis=0)
        q_all = q_all[min_idx & max_idx]
    q_mean, q_std = np.mean(q_all, axis=0), np.std(q_all, axis=0)
    print(q_all.shape, q_mean.shape, q_std.shape)
    df = pd.DataFrame({'Zeolite': zeolites, 
                    'Heat of adsorption (%s K, %s bar)' % (args.t, args.p): q_mean, 
                    'Heat of adsorption error (%s K, %s bar)' % (args.t, args.p): q_std, 
                    },)
    df.to_csv('heat-of-adsorption-p%s-t%s.csv' % (args.p, args.t))