"""
Version B: 严格博客方法版
严格遵循博客 Boss Baseline 的所有推荐:
- SelectKBest f_regression k=18 (40 states + 18 numerical = 58维)
- BN + Dropout + LeakyReLU (64→16→1, 博客标准架构)
- CosineAnnealingWarmRestarts(T_0=2, T_mult=2, eta_min=lr/50)
- AdamW 优化器 (博客推荐的 Adam + weight_decay)
- 目标值 z-score 归一化 + 反变换
- RMSE 损失, batch_size=270, 80/20 split
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
              filename='prediction.png', target_mean=0, target_std=1):
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
        # 反归一化（确保是 float 而非 tensor）
        preds = preds * float(target_std) + float(target_mean)
        targets = targets * float(target_std) + float(target_mean)

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


def get_feature_indices(train_path, k=18):
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
    """80/20 划分 + 目标值 z-score 归一化（博客推荐）"""
    def __init__(self, path, mode='train', feats=None,
                 target_mean=None, target_std=None):
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
            target_tensor = torch.FloatTensor(target[indices])

            # 博客推荐: 目标值 z-score 归一化
            if mode == 'train':
                self.target_mean = target_tensor.mean()
                self.target_std = target_tensor.std()
            else:
                self.target_mean = target_mean
                self.target_std = target_std

            self.target = (target_tensor - self.target_mean) / (self.target_std + 1e-8)

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


def prep_dataloader(path, mode, batch_size, n_jobs=0, feats=None,
                    target_mean=None, target_std=None):
    dataset = COVID19Dataset(path, mode=mode, feats=feats,
                             target_mean=target_mean, target_std=target_std)
    dataloader = DataLoader(
        dataset, batch_size,
        shuffle=(mode == 'train'), drop_last=False,
        num_workers=n_jobs, pin_memory=True)
    return dataloader


class NeuralNet(nn.Module):
    """博客标准架构: BN + Dropout + LeakyReLU (64→16→1)"""
    def __init__(self, input_dim):
        super(NeuralNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.2),

            nn.Linear(64, 16),
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


def test(tt_set, model, device, target_mean=0, target_std=1):
    model.eval()
    preds = []
    for x in tt_set:
        x = x.to(device)
        with torch.no_grad():
            pred = model(x)
            preds.append(pred.detach().cpu())
    preds = torch.cat(preds, dim=0).numpy()
    return preds * float(target_std) + float(target_mean)


def train(tr_set, dv_set, model, config, device):
    n_epochs = config['n_epochs']
    optimizer = getattr(torch.optim, config['optimizer'])(
        model.parameters(), **config['optim_hparas'])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=2, T_mult=2,
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
set_seed(myseed)

# 博客 SelectKBest: 40 states + 18 numerical = 58 features
feats = get_feature_indices(tr_path, k=18)

config = {
    'n_epochs': 3000,
    'batch_size': 270,
    'optimizer': 'AdamW',
    'optim_hparas': {
        'lr': 0.001,
        'weight_decay': 1e-3,
    },
    'early_stop': 200,
    'save_path': 'models/model_vB.pth'
}

# 加载训练集获取 target_mean/std（用于归一化）
tr_set = prep_dataloader(tr_path, 'train', config['batch_size'], feats=feats)
target_mean = tr_set.dataset.target_mean
target_std = tr_set.dataset.target_std
print(f'Target normalization: mean={target_mean:.4f}, std={target_std:.4f}')

dv_set = prep_dataloader(tr_path, 'dev', config['batch_size'], feats=feats,
                         target_mean=target_mean, target_std=target_std)
tt_set = prep_dataloader(tt_path, 'test', config['batch_size'], feats=feats)

model = NeuralNet(tr_set.dataset.dim).to(device)
model_loss, loss_record = train(tr_set, dv_set, model, config, device)

plot_learning_curve(loss_record,
                    title='Version B: SelectKBest + AdamW + TargetNorm + CosWR(T0=2)',
                    filename='vB_learning_curve.png')

del model
model = NeuralNet(tr_set.dataset.dim).to(device)
ckpt = torch.load(config['save_path'], map_location='cpu')
model.load_state_dict(ckpt)
plot_pred(dv_set, model, device, filename='vB_prediction.png',
          target_mean=target_mean, target_std=target_std)

preds = test(tt_set, model, device, target_mean=target_mean, target_std=target_std)
save_pred(preds, 'vB_pred.csv')
print(f'Version B best dev RMSE (normalized): {model_loss:.4f}')
