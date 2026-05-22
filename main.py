import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import csv
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

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
    plt.ylabel('MSE loss')
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


class COVID19Dataset(Dataset):
    def __init__(self, path, mode='train', target_only=False, feats=None):
        self.mode = mode

        with open(path, 'r') as fp:
            data = list(csv.reader(fp))
            data = np.array(data[1:])[:, 1:].astype(float)

        if feats is not None:
            pass
        elif not target_only:
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
                indices = [i for i in range(len(data)) if i % 10 != 0]
            elif mode == 'dev':
                indices = [i for i in range(len(data)) if i % 10 == 0]

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


def prep_dataloader(path, mode, batch_size, n_jobs=0, target_only=False, feats=None):
    dataset = COVID19Dataset(path, mode=mode, target_only=target_only, feats=feats)
    dataloader = DataLoader(
        dataset, batch_size,
        shuffle=(mode == 'train'), drop_last=False,
        num_workers=n_jobs, pin_memory=True)
    return dataloader


def save_pred(preds, file):
    print('Saving results to {}'.format(file))
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])


# ============================================================
# Round 1: R1 改进基线模型
# 架构: 93 → 128 → 64 → 1 (ReLU)
# 优化器: SGD(lr=0.001, momentum=0.9, weight_decay=1e-4)
# 损失: MSE, batch_size=32
# 集成: 3个不同种子的模型取平均预测
# ============================================================

class NeuralNet_R1(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNet_R1, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
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
        if early_stop_cnt > config['early_stop']:
            break

    print('Finished training after {} epochs'.format(epoch))
    return min_mse, loss_record


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


device = get_device()
os.makedirs('models', exist_ok=True)

# ---- Round 1: R1 集成 ----
print('='*60)
print('Round 1: R1 改进基线 (3模型集成)')
print('='*60)

config_r1 = {
    'n_epochs': 3000,
    'batch_size': 32,
    'optimizer': 'SGD',
    'optim_hparas': {
        'lr': 0.001,
        'momentum': 0.9,
        'weight_decay': 1e-4,
    },
    'early_stop': 300,
    'save_path': 'models/model_r1.pth'
}

n_models = 3
all_preds_r1 = []
best_losses_r1 = []

for i in range(n_models):
    seed = myseed + i * 100
    set_seed(seed)
    print(f'\n--- R1 Model {i+1}/3 (seed={seed}) ---')

    tr_set = prep_dataloader(tr_path, 'train', config_r1['batch_size'])
    dv_set = prep_dataloader(tr_path, 'dev', config_r1['batch_size'])
    tt_set = prep_dataloader(tt_path, 'test', config_r1['batch_size'])

    model = NeuralNet_R1(tr_set.dataset.dim).to(device)
    model_loss, model_loss_record = train(tr_set, dv_set, model, config_r1, device)
    best_losses_r1.append(model_loss)

    if i == 0:
        plot_learning_curve(model_loss_record,
                            title='Round 1: R1 Baseline (SGD+ReLU+bs32+MSE)',
                            filename='round1_learning_curve.png')

    del model
    model = NeuralNet_R1(tr_set.dataset.dim).to(device)
    ckpt = torch.load(config_r1['save_path'], map_location='cpu')
    model.load_state_dict(ckpt)

    if i == 0:
        plot_pred(dv_set, model, device, filename='round1_prediction.png')

    preds = test(tt_set, model, device)
    all_preds_r1.append(preds)

ensemble_r1 = np.mean(all_preds_r1, axis=0)
save_pred(ensemble_r1, 'round1_pred.csv')
print(f'\nR1 Individual losses: {[f"{l:.4f}" for l in best_losses_r1]}')
print(f'R1 Best Dev MSE: {min(best_losses_r1):.4f}')
print(f'R1 Mean Dev MSE: {np.mean(best_losses_r1):.4f}')


# ============================================================
# Round 2: 博客 Strong Baseline
# 架构: BN + Dropout(0.2) + LeakyReLU(0.2)
# 优化器: SGD(lr=0.001, momentum=0.9, weight_decay=1e-5)
# 调度器: CosineAnnealingWarmRestarts(T_0=2, T_mult=2)
# 损失: RMSE, batch_size=270, 80/20 split
# ============================================================

print('\n' + '='*60)
print('Round 2: 博客 Strong Baseline')
print('='*60)


class COVID19Dataset_R2(Dataset):
    """博客推荐的 80/20 划分"""
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


def prep_dataloader_r2(path, mode, batch_size, n_jobs=0, feats=None):
    dataset = COVID19Dataset_R2(path, mode=mode, feats=feats)
    dataloader = DataLoader(
        dataset, batch_size,
        shuffle=(mode == 'train'), drop_last=False,
        num_workers=n_jobs, pin_memory=True)
    return dataloader


class NeuralNet_R2(nn.Module):
    """博客 Strong Baseline: BN + Dropout + LeakyReLU"""
    def __init__(self, input_dim):
        super(NeuralNet_R2, self).__init__()
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
        # RMSE 损失（与 Kaggle 评分指标一致）
        return torch.sqrt(self.criterion(pred, target))


def train_r2(tr_set, dv_set, model, config, device):
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


set_seed(myseed)

# 博客推荐特征: 40 states + CLI(40-43,58-61,76-79) + tested_positive(57,75)
blog_feats = list(range(40)) + list(range(40,44)) + [57] + list(range(58,62)) + [75] + list(range(76,80))

config_r2 = {
    'n_epochs': 3000,
    'batch_size': 270,
    'optimizer': 'SGD',
    'optim_hparas': {
        'lr': 0.001,
        'momentum': 0.9,
        'weight_decay': 1e-5,
    },
    'early_stop': 200,
    'save_path': 'models/model_r2.pth'
}

tr_set_r2 = prep_dataloader_r2(tr_path, 'train', config_r2['batch_size'], feats=blog_feats)
dv_set_r2 = prep_dataloader_r2(tr_path, 'dev', config_r2['batch_size'], feats=blog_feats)
tt_set_r2 = prep_dataloader_r2(tt_path, 'test', config_r2['batch_size'], feats=blog_feats)

model_r2 = NeuralNet_R2(tr_set_r2.dataset.dim).to(device)
loss_r2, loss_record_r2 = train_r2(tr_set_r2, dv_set_r2, model_r2, config_r2, device)

plot_learning_curve(loss_record_r2,
                    title='Round 2: Blog Strong Baseline (BN+Dropout+LeakyReLU+CosWR)',
                    filename='round2_learning_curve.png')

del model_r2
model_r2 = NeuralNet_R2(tr_set_r2.dataset.dim).to(device)
ckpt_r2 = torch.load(config_r2['save_path'], map_location='cpu')
model_r2.load_state_dict(ckpt_r2)
plot_pred(dv_set_r2, model_r2, device, filename='round2_prediction.png')

preds_r2 = test(tt_set_r2, model_r2, device)
save_pred(preds_r2, 'round2_pred.csv')
print(f'\nRound 2 best dev RMSE: {loss_r2:.4f}')

print('\n' + '='*60)
print('优化完成！文件输出:')
print('  round1_pred.csv - R1 改进基线预测 (3模型集成)')
print('  round1_learning_curve.png - R1 学习曲线')
print('  round1_prediction.png - R1 预测 vs 真实值')
print('  round2_pred.csv - 博客 Strong Baseline 预测')
print('  round2_learning_curve.png - Round 2 学习曲线')
print('  round2_prediction.png - Round 2 预测 vs 真实值')
print('='*60)
