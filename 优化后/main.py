print('hello!')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import csv
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

tr_path = 'covid.train.csv'
tt_path = 'covid.test.csv'


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def plot_learning_curve(loss_record, title='', filename='final_learning_curve.png'):
    total_steps = len(loss_record['train'])
    x_1 = range(total_steps)
    x_2 = x_1[::len(loss_record['train']) // len(loss_record['dev'])]
    figure(figsize=(6, 4))
    plt.plot(x_1, loss_record['train'], c='tab:red', label='train')
    plt.plot(x_2, loss_record['dev'], c='tab:cyan', label='dev')
    plt.xlabel('Training steps')
    plt.ylabel('MSE loss')
    plt.title('Learning curve of {}'.format(title))
    plt.legend()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_pred(dv_set, model, device, lim=45., preds=None, targets=None,
              filename='final_prediction.png'):
    if preds is None or targets is None:
        model.eval()
        preds, targets = [], []
        for x, y in dv_set:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                preds.append(pred.detach().cpu())
                targets.append(y.detach().cpu())
        preds = torch.cat(preds, dim=0).numpy()
        targets = torch.cat(targets, dim=0).numpy()

    figure(figsize=(5, 5))
    plt.scatter(targets, preds, c='r', alpha=0.5)
    plt.plot([-0.2, lim], [-0.2, lim], c='b')
    plt.xlim(-0.2, lim)
    plt.ylim(-0.2, lim)
    plt.xlabel('ground truth value')
    plt.ylabel('predicted value')
    plt.title('Ground Truth v.s. Prediction')
    plt.savefig(filename, dpi=150)
    plt.close()


class COVID19Dataset(Dataset):
    def __init__(self, path, mode='train', target_only=False,
                 target_mean=None, target_std=None):
        self.mode = mode

        with open(path, 'r') as fp:
            data = list(csv.reader(fp))
            data = np.array(data[1:])[:, 1:].astype(float)

        if not target_only:
            feats = list(range(93))
        else:
            feats = list(range(40)) + [57, 75]

        if mode == 'test':
            data = data[:, feats]
            self.data = torch.FloatTensor(data)
        else:
            target = data[:, -1]
            data = data[:, feats]

            if mode == 'train':
                indices = [i for i in range(len(data)) if i % 5 != 0]
            elif mode == 'dev':
                indices = [i for i in range(len(data)) if i % 5 == 0]

            self.data = torch.FloatTensor(data[indices])
            target_tensor = torch.FloatTensor(target[indices])

            if mode == 'train':
                self.target_mean = target_tensor.mean()
                self.target_std = target_tensor.std()
            else:
                self.target_mean = target_mean
                self.target_std = target_std

            self.target = (target_tensor - self.target_mean) / (self.target_std + 1e-8)

        self.data[:, 40:] = \
            (self.data[:, 40:] - self.data[:, 40:].mean(dim=0, keepdim=True)) \
            / (self.data[:, 40:].std(dim=0, keepdim=True) + 1e-8)

        self.dim = self.data.shape[1]

        print('Finished reading the {} set of COVID19 Dataset ({} samples found, each dim = {})'
              .format(mode, len(self.data), self.dim))

    def __getitem__(self, index):
        if self.mode in ['train', 'dev']:
            return self.data[index], self.target[index]
        else:
            return self.data[index]

    def __len__(self):
        return len(self.data)


def prep_dataloader(path, mode, batch_size, n_jobs=0, target_only=False,
                    target_mean=None, target_std=None):
    dataset = COVID19Dataset(path, mode=mode, target_only=target_only,
                             target_mean=target_mean, target_std=target_std)
    dataloader = DataLoader(
        dataset, batch_size,
        shuffle=(mode == 'train'), drop_last=False,
        num_workers=n_jobs, pin_memory=True)
    return dataloader


class NeuralNet(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNet, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(32, 1)
        )
        self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, x):
        return self.net(x).squeeze(1)

    def cal_loss(self, pred, target):
        return self.criterion(pred, target)


def train(tr_set, dv_set, model, config, device):
    n_epochs = config['n_epochs']

    optimizer = getattr(torch.optim, config['optimizer'])(
        model.parameters(), **config['optim_hparas'])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=80, T_mult=2, eta_min=1e-7)

    min_mse = 1000.
    loss_record = {'train': [], 'dev': []}
    early_stop_cnt = 0
    epoch = 0
    while epoch < n_epochs:
        model.train()
        for x, y in tr_set:
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = model.cal_loss(pred, y)
            loss.backward()
            optimizer.step()
            loss_record['train'].append(loss.detach().cpu().item())

        dev_mse = dev(dv_set, model, device)
        if dev_mse < min_mse:
            min_mse = dev_mse
            print('Saving model (epoch = {:4d}, loss = {:.4f})'
                .format(epoch + 1, min_mse))
            torch.save(model.state_dict(), config['save_path'])
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        epoch += 1
        loss_record['dev'].append(dev_mse)
        scheduler.step(epoch)
        if early_stop_cnt > config['early_stop']:
            break

    print('Finished training after {} epochs'.format(epoch))
    return min_mse, loss_record


def dev(dv_set, model, device):
    model.eval()
    total_loss = 0
    target_mean = dv_set.dataset.target_mean
    target_std = dv_set.dataset.target_std
    for x, y in dv_set:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            pred = model(x)
            pred_orig = pred * target_std + target_mean
            y_orig = y * target_std + target_mean
            mse_loss = nn.MSELoss(reduction='mean')(pred_orig, y_orig)
        total_loss += mse_loss.detach().cpu().item() * len(x)
    return total_loss / len(dv_set.dataset)


def test(tt_set, model, device, target_mean=None, target_std=None):
    model.eval()
    preds = []
    for x in tt_set:
        x = x.to(device)
        with torch.no_grad():
            pred = model(x)
            preds.append(pred.detach().cpu())
    preds = torch.cat(preds, dim=0)
    if target_mean is not None and target_std is not None:
        preds = preds * target_std + target_mean
    return preds.numpy()


device = get_device()
os.makedirs('models', exist_ok=True)
target_only = False

config = {
    'n_epochs': 3000,
    'batch_size': 128,
    'optimizer': 'AdamW',
    'optim_hparas': {
        'lr': 0.001,
        'weight_decay': 1e-3,
    },
    'early_stop': 400,
}

tr_set = prep_dataloader(tr_path, 'train', config['batch_size'], target_only=target_only)
target_mean = tr_set.dataset.target_mean
target_std = tr_set.dataset.target_std
dv_set = prep_dataloader(tr_path, 'dev', config['batch_size'], target_only=target_only,
                         target_mean=target_mean, target_std=target_std)
tt_set = prep_dataloader(tt_path, 'test', config['batch_size'], target_only=target_only)

# === 集成训练：3 个不同种子 ===
ensemble_seeds = [42069, 12345, 67890]
n_ensemble = len(ensemble_seeds)
all_preds = []
best_mses = []
all_loss_records = []

for i, seed in enumerate(ensemble_seeds):
    print(f'\n=== Training ensemble model {i+1}/{n_ensemble} (seed={seed}) ===')
    set_seed(seed)
    config['save_path'] = f'models/model_seed{seed}.pth'

    model = NeuralNet(tr_set.dataset.dim).to(device)
    model_loss, loss_record = train(tr_set, dv_set, model, config, device)
    best_mses.append(model_loss)
    all_loss_records.append(loss_record)

    del model
    model = NeuralNet(tr_set.dataset.dim).to(device)
    ckpt = torch.load(config['save_path'], map_location='cpu')
    model.load_state_dict(ckpt)

    preds = test(tt_set, model, device, target_mean=target_mean, target_std=target_std)
    all_preds.append(preds)
    print(f'Model {i+1} best dev MSE: {model_loss:.4f}')

ensemble_preds = np.mean(all_preds, axis=0)
print(f'\nEnsemble complete! Individual MSEs: {[f"{m:.4f}" for m in best_mses]}')
print(f'Best single MSE: {min(best_mses):.4f}')

# 绘制第一个模型的学习曲线和预测图
set_seed(ensemble_seeds[0])
config['save_path'] = f'models/model_seed{ensemble_seeds[0]}.pth'
model = NeuralNet(tr_set.dataset.dim).to(device)
ckpt = torch.load(config['save_path'], map_location='cpu')
model.load_state_dict(ckpt)
plot_learning_curve(all_loss_records[0], title='Best ensemble model',
                    filename='best_learning_curve.png')
plot_pred(dv_set, model, device, filename='best_prediction.png')


def save_pred(preds, file):
    print('Saving results to {}'.format(file))
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])

save_pred(ensemble_preds, 'best_submission.csv')
print('Final ensemble submission saved to best_submission.csv')
