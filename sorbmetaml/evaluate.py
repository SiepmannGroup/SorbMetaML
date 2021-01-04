import numpy as np
import torch
import matplotlib
from torch.utils.data import DataLoader
from models import MetaDataset, MLModel
from utils import parse_eval

def get_latent_vectors(data, model, loss_func=torch.nn.MSELoss(), data_test=None, device='cuda:0'):
    data_test = data_test if data_test is not None else data
    dataloader = DataLoader(data, batch_size=1, num_workers=4, shuffle=False, pin_memory=True)
    testloader = DataLoader(data_test, batch_size=1, num_workers=4, shuffle=False, pin_memory=True)
    device = torch.device(device)
    model.to(device)
    with torch.no_grad():
        z_list = []
        loss_list = []
        for x, x_test in zip(dataloader, testloader):
            x = x.to(device)
            x_test = x_test.to(device)
            z, _ = model.encoder(x)
            y_pred = model.decoder(z, x_test[:, 1:, :])
            y_true = torch.t(x_test[:, 0:1, :].transpose(0, 1).flatten(1))
            loss = loss_func(y_pred, y_true)
            z_list.append(z.cpu().numpy())
            loss_list.append([loss.item()])
    return np.append(np.vstack(z_list), loss_list, axis=1)


if __name__ == '__main__':
    args = parse_eval()
    data = np.load(args.data).astype(np.float32)
    data_test = np.load(args.test).astype(np.float32) if args.test != '' else data
    net = MLModel(args.z, data.shape[2])
    net.load_state_dict(torch.load(args.model))
    eval_data = get_latent_vectors(data, net, data_test=data_test)
    np.savetxt('nn-latents.csv', eval_data, delimiter=',')
