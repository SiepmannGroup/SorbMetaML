import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class MetaDataset(Dataset):
    def __init__(self, data, sample=True, samples_per_task=16, task_size=32):             
        if sample:
            samples = []
            for i in range(data.shape[0]):
                for _ in range(samples_per_task):
                    rand_ind = np.random.permutation(data.shape[2])[:task_size]
                    samples.append(data[i:i+1, :, rand_ind])
            self.data = np.concatenate(samples, axis=0)
        else:
            self.data = data
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, index):
        return self.data[index, :, :]


class Encoder(nn.Module):
    activation = F.elu

    def __init__(self, nz):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv1d(3, 16, 1)
        self.conv2 = nn.Conv1d(16, 32, 1)
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, nz)
    
    def forward(self, x):
        x = Encoder.activation(self.conv1(x))
        x = Encoder.activation(self.conv2(x))
        x, argx = torch.max(x, 2)
        z = Encoder.activation(self.fc1(x))
        z = self.fc2(z)
        z[:, 0] = torch.exp(z[:, 0])
        z[:, 1:] = torch.tanh(z[:, 1:])
        return z, argx
        
    
class Decoder(nn.Module):
    activation = F.elu

    def __init__(self, nz, nx=2):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(nz + nx - 1, 12)
        self.fc2 = nn.Linear(12, 6)
        self.fc3 = nn.Linear(6, 1)
    
    def forward(self, z, x):
        z = z.view(z.shape[0], z.shape[1], 1).repeat(1, 1, x.shape[-1])
        x = torch.cat([z, x], 1)
        x = torch.t(x.transpose(0, 1).flatten(1))
        norm = x[:, 0:1]
        x = x[:, 1:]
        x = Decoder.activation(self.fc1(x))
        x = Decoder.activation(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return norm * x
    
class Decoder_weights(nn.Module):
    activation = F.softplus

    def __init__(self, nz, nx=2, layers=[8, 4]):
        super(Decoder_weights, self).__init__()
        self.layers = layers
        self.nz = nz
        self.nx = nx
        self.fc_w = nn.ModuleList([])
        self.fc_b = nn.ModuleList([])
        self.layers.append(1)
        n_prev = nx
        for n in self.layers:
            self.fc_w.append(nn.Linear(nz - 1, n_prev * n))
            self.fc_b.append(nn.Linear(nz - 1, n))
            n_prev = n
    
    def forward(self, z, x):
        bs = x.shape[0]
        norm = z[:, 0:1].view(bs, 1, 1)
        z = z[:, 1:]
        n_prev = self.nx
        for i, n in enumerate(self.layers):
            w = self.fc_w[i](z).view(bs, n, n_prev)
            b = self.fc_b[i](z).view(bs, n, 1)
            x = torch.bmm(w, x) + b
            if i != len(self.layers) - 1:
                x = Decoder_weights.activation(x)
            else:
                x = torch.sigmoid(x)
            n_prev = n
        y = norm * x
        return y.transpose(2, 1).view(-1, 1)

class Decoder_bottom(nn.Module):
    activation = F.softplus

    def __init__(self, nz, nx=2, hidden_gen=8):
        super(Decoder_bottom, self).__init__()
        self.nz = nz
        self.nx = nx

        self.fc2 = nn.Linear(8, 4)
        self.fc3 = nn.Linear(4, 1)
        
        self.fc_gen1 = nn.Linear(nz - 1, 8)
        self.fc_w = nn.Linear(8, 2 * 8)
        self.fc_b = nn.Linear(8, 8)

   
    def forward(self, z, x):
        bs = x.shape[0]
        norm = z[:, 0:1].view(-1, 1).repeat(1, x.shape[-1]).view(-1, 1)
        z = z[:, 1:]
        z = Decoder_bottom.activation(self.fc_gen1(z))
       
        w = self.fc_w(z).view(bs, 8, 2)
        b = self.fc_b(z).view(bs, 8, 1)
        x = Decoder_bottom.activation(torch.bmm(w, x) + b)
        x = torch.t(x.transpose(0, 1).flatten(1))
        x = Decoder_bottom.activation(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        y = norm * x
        return y


class MLModel(nn.Module):

    def __init__(self, nz, ntrain_task=24, encoder=Encoder, decoder=Decoder, 
                  uncorrelated=False, unsupervised=False, **kwargs):
        super(MLModel, self).__init__()
        self.encoder = encoder(nz)
        self.decoder = decoder(nz, **kwargs)
        self.nz = nz
        self.ntrain_task = ntrain_task
        self.uncorrelated = uncorrelated
        self.unsupervised = unsupervised

    def forward(self, x):
        if self.unsupervised:
            z, _ = self.encoder(x[:, :, :])
        else:
            z, _ = self.encoder(x[:, :, :self.ntrain_task])
        y_pred = self.decoder(z, x[:, 1:, :])
        if self.uncorrelated:
            return y_pred, correlation_matrix(z)
        else:    
            return y_pred


def correlation_matrix(z):
    z_centered = z - z.mean(dim=0).view(1, -1)
    return torch.mm(z_centered.t(), z_centered) / z.shape[0]

def wrapper(model, latents, norms):
    if type(latents) != torch.Tensor:
        latents = torch.tensor(latents)
    with torch.no_grad():
        def decode(x, i):  
            x = torch.t(torch.tensor(x)).unsqueeze(0)
            z = latents[i: i+1, :]
            y_pred = model.decoder(z, x)
            return y_pred.detach().numpy().ravel() * norms[i]
    return decode


