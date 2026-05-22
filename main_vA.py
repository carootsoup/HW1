"""
Version A: 博客方向优化版
基于 Round 2 (Kaggle 1.16) 的改进:
- 更深网络: 64→32→16→1 (3层 BN+Dropout+LeakyReLU)
- SelectKBest 更多特征: 40 states + 35 numerical = 75 维
- T_0=6 温和重启 (比 T_0=2 更稳定)
- 3模型集成
- SGD+momentum + RMSE
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import csv
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.feature_selection import SelectKBest, f_regression

myseed = 42069
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

tr_path = 'covid.train.csv'
tt_path = 'covid.test.csv'


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def plot_learning_curve(loss_record, title='', filename='learning_curve.png'):
    total_steps = len(loss_record['train'])
    x_1 = range(total_steps)
    x_2 = x_1[::len(loss_record['train']) // len(loss_record['dev'])]
    figure(figsize=(6, 4))
    plt.plot(x_1, loss_record['train'], c='tab:red', label='train')
    plt.plot(x_2, loss_record['dev'], c='tab:cyan', label='dev')
    plt.ylim(0.0, 5.)
    plt.xlabel('Training steps')
    plt.ylabel('RMSE loss')
    plt.title('Learning curve of {}'.format(title))
    plt.legend()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_pred(dv_set, model, device, lim=35., preds=None, targets=None,
              filename='prediction.png'):
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


def save_pred(preds, file):
    print('Saving results to {}'.format(file))
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])


def get_feature_indices(train_path, k=35):
    """SelectKBest 从数值特征中选择 top-k，始终保留40个州特征"""
    with open(train_path, 'r') as fp:
        data = list(csv.reader(fp))
        data = np.array(data[1:])[:, 1:].astype(float)

    train_indices = [i for i in range(len(data)) if i % 5 != 0]
    train_data = data[train_indices]
    X_numerical = train_data[:, 40:93]
    y = train_data[:, -1]

    selector = SelectKBest(score_func=f_regression, k=k)
    selector.fit(X_numerical, y)
    selected_numerical = np.sort(np.argsort(selector.scores_)[::-1][:k])
    selected_features = list(range(40)) + list(40 + selected_numerical)
    print(f'SelectKBest: 40 states + {k} numerical = {len(selected_features)} features')
    return selected_features


class COVID19Dataset(Dataset):
    """80/20 划分"""
    def __init__(self, path, mode='train', feats=None):
        self.mode = mode

        with open(path, 'r') as fp:
            data = list(csv.reader(fp))
            data = np.array(data[1:])[:, 1:].astype(float)

        if feats is None:
            feats = list(range(93))

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
            self.target = torch.FloatTensor(target[indices])

        num_state = min(40, self.data.shape[1])
        if self.data.shape[1] > num_state:
            self.data[:, num_state:] = \
                (self.data[:, num_state:] - self.data[:, num_state:].mean(dim=0, keepdim=True)) \
                / (self.data[:, num_state:].std(dim=0, keepdim=True) + 1e-8)

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


def prep_dataloader(path, mode, batch_size, n_jobs=0, feats=None):
    dataset = COVID19Dataset(path, mode=mode, feats=feats)
    dataloader = DataLoader(
        dataset, batch_size,
        shuffle=(mode == 'train'), drop_last=False,
        num_workers=n_jobs, pin_memory=True)
    return dataloader


class NeuralNet(nn.Module):
    """更深网络: 3层 BN+Dropout+LeakyReLU (64→32→16→1)"""
    def __init__(self, input_dim):
        super(NeuralNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.2),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.Dropout(0.15),
            nn.LeakyReLU(0.2),

            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.Dropout(0.1),
            nn.LeakyReLU(0.2),

            nn.Linear(16, 1)
        )
        self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, x):
        return self.net(x).squeeze(1)

    def cal_loss(self, pred, target):
        return torch.sqrt(self.criterion(pred, target))


def dev(dv_set, model, device):
    model.eval()
    total_loss = 0
    for x, y in dv_set:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            pred = model(x)
            mse_loss = model.cal_loss(pred, y)
        total_loss += mse_loss.detach().cpu().item() * len(x)
    return total_loss / len(dv_set.dataset)


def test(tt_set, model, device):
    model.eval()
    preds = []
    for x in tt_set:
        x = x.to(device)
        with torch.no_grad():
            pred = model(x)
            preds.append(pred.detach().cpu())
    return torch.cat(preds, dim=0).numpy()


def train(tr_set, dv_set, model, config, device):
    n_epochs = config['n_epochs']
    optimizer = getattr(torch.optim, config['optimizer'])(
        model.parameters(), **config['optim_hparas'])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=config['T_0'], T_mult=2,
        eta_min=config['optim_hparas']['lr'] / 50)

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
            mse_loss = model.cal_loss(pred, y)
            mse_loss.backward()
            optimizer.step()
            loss_record['train'].append(mse_loss.detach().cpu().item())

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
        scheduler.step()
        if early_stop_cnt > config['early_stop']:
            break

    print('Finished training after {} epochs'.format(epoch))
    return min_mse, loss_record


# ============================================================
# 主程序
# ============================================================

device = get_device()
os.makedirs('models', exist_ok=True)

config = {
    'n_epochs': 3000,
    'batch_size': 270,
    'optimizer': 'SGD',
    'optim_hparas': {
        'lr': 0.001,
        'momentum': 0.9,
        'weight_decay': 1e-5,
    },
    'early_stop': 200,
    'T_0': 6,
}

# SelectKBest: 从53个数值特征中选35个 → 40+35=75维
feats = get_feature_indices(tr_path, k=35)

# 3模型集成
n_models = 3
all_preds = []
best_losses = []

for i in range(n_models):
    seed = myseed + i * 100
    set_seed(seed)
    print(f'\n=== Model {i+1}/3 (seed={seed}) ===')

    config['save_path'] = f'models/model_vA_{i}.pth'

    tr_set = prep_dataloader(tr_path, 'train', config['batch_size'], feats=feats)
    dv_set = prep_dataloader(tr_path, 'dev', config['batch_size'], feats=feats)
    tt_set = prep_dataloader(tt_path, 'test', config['batch_size'], feats=feats)

    model = NeuralNet(tr_set.dataset.dim).to(device)
    model_loss, loss_record = train(tr_set, dv_set, model, config, device)
    best_losses.append(model_loss)

    if i == 0:
        plot_learning_curve(loss_record,
                            title='Version A: Deeper BN+Dropout+LeakyReLU + SelectKBest + CosWR(T0=6)',
                            filename='vA_learning_curve.png')

    del model
    model = NeuralNet(tr_set.dataset.dim).to(device)
    ckpt = torch.load(config['save_path'], map_location='cpu')
    model.load_state_dict(ckpt)

    if i == 0:
        plot_pred(dv_set, model, device, filename='vA_prediction.png')

    preds = test(tt_set, model, device)
    all_preds.append(preds)
    print(f'Model {i+1} Dev RMSE: {model_loss:.4f}')

# 集成预测
ensemble_preds = np.mean(all_preds, axis=0)
save_pred(ensemble_preds, 'vA_pred.csv')
print(f'\nVersion A Ensemble: losses={[f"{l:.4f}" for l in best_losses]}')
print(f'Version A Best Dev RMSE: {min(best_losses):.4f}')
