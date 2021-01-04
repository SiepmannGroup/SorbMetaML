import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models import MetaDataset, MLModel, Decoder_bottom
from utils import parse_train, get_runname
from evaluate import get_latent_vectors

args, config = parse_train()
seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)

device = torch.device(args.device)

data = np.load(args.data)
bs = int(config['DATASET']['batch_size'])
ntrain = int(config['DATASET']['n_train'])
nval = int(config['DATASET']['n_val'])
ntrain_task = int(config['DATASET']['ntrain_task'])

data_train = MetaDataset(data[:ntrain, :, :], sample=True, task_size=int(ntrain_task*1.334))
data_val = MetaDataset(data[ntrain:ntrain + nval, :, :], sample=True, task_size=int(ntrain_task*1.334))
trainloader = DataLoader(data_train, batch_size=bs, num_workers=8, shuffle=True, pin_memory=True)
validationloader = DataLoader(data_val, batch_size=len(data_val), num_workers=8, shuffle=False, pin_memory=True)

nz = int(config['TRAINING']['n_latent_vec'])
lr_init = float(config['TRAINING']['initial_lr'])

lr_decay = float(config['TRAINING']['lr_decay'])
l_reg = float(config['TRAINING']['regularization'])

net = MLModel(nz, ntrain_task, uncorrelated=True, unsupervised=args.unsup)
net.to(device)
opt = optim.Adam(net.parameters(), lr=lr_init)
scheduler = optim.lr_scheduler.ExponentialLR(opt, gamma=lr_decay)

if args.checkpoint:
    net.load_state_dict(torch.load(args.checkpoint))

if not os.path.exists('output/'):
    os.makedirs('output')

runname = get_runname(args)

def train(epochs, displevel, checkpoint_level, loss_func=torch.nn.MSELoss(), bs=bs):
    train_loss = []
    val_loss = []
    for epoch in range(epochs):
        running_loss = 0.0
        running_cov_loss = 0.0

        for i, batch in enumerate(trainloader, 0):
            opt.zero_grad()
            batch = batch.to(device)
            y_true = torch.t(batch[:, 0:1, :].transpose(0, 1).flatten(1))
            y_pred, cov_mat = net(batch)
            cov_loss = torch.sum((cov_mat - cov_mat * torch.eye(nz).to(device)) ** 2)
            loss = loss_func(y_pred, y_true) + l_reg * cov_loss
            loss.backward()
            opt.step()
            running_loss += loss.item()
            running_cov_loss += cov_loss.item()

        running_loss /= i + 1
        running_cov_loss /= i + 1
        train_loss.append(running_loss)
        with torch.no_grad():
            for batch in validationloader:   
                batch = batch.to(device)
                y_true = torch.t(batch[:, 0:1, :].transpose(0, 1).flatten(1))
                y_pred, cov_mat = net(batch)
                val_cov_loss = torch.sum((cov_mat - cov_mat * torch.eye(nz).to(device)) ** 2)
        val_loss.append(loss_func(y_pred, y_true).item())

        if epoch % displevel == 0:
            print('Epoch %d' % epoch)
            print('Meta-training loss: %e, %e' % (running_loss, running_cov_loss))
            print('Meta-validation loss: %e %e' % (val_loss[-1], val_cov_loss.item()))
        scheduler.step()
        np.savetxt('output/%s-loss.txt' % runname, np.vstack([train_loss, val_loss]).T)

        if epoch % checkpoint_level == 0:
            if not os.path.exists('checkpoints'):
                os.makedirs('checkpoints')
            torch.save(net.state_dict(), 'checkpoints/%s-model-%d.pt' % (runname, epoch))

    return train_loss, val_loss

epochs = args.epochs
displevel = int(config['OUTPUT']['display_interval'])
checkpoint_level = int(config['OUTPUT']['checkpoint_interval'])

print('model hash:', runname)

train_loss, val_loss = train(epochs, displevel, checkpoint_level)

torch.save(net.state_dict(), 'output/%s.pt' % runname)

eval_data = get_latent_vectors(data, net)

np.savetxt('output/%s-latents.csv' % runname, eval_data, delimiter=',')


