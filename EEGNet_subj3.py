import torch
from torch import nn, optim, autograd
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torch.nn import functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import os.path
import errno
from functools import reduce
from operator import __add__

file = open("Output_subj3.txt", "w")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
file.write(f"DEVICE = {DEVICE}\n")

np.random.seed(42)
torch.manual_seed(42)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)

class DataNShot():
    def __init__(self, root, data_npy, train_subj, test_subj, batchsz, n_way, k_shot, k_query, eeg_shape, num_trials, batchst=1):
        self.num_trials = num_trials
        self.batchsz = {"train": batchsz, "test": batchst}
        self.n_way = n_way
        self.k_shot = k_shot
        self.k_query = k_query
        self.eeg_shape = eeg_shape
        self.root = root
        self.data_npy = data_npy
        self.train_subj = train_subj
        self.test_subj = test_subj
        self.subj = {"train": self.train_subj, "test": self.test_subj}

    def get_data(self, mode, subj, cls, trial):
        return np.load(os.path.join(self.root, self.data_npy+'_'+'train', f"s{subj}_c{cls}_t{trial}.npy"))
        # return raw - raw.mean(axis=0)

    def get_batch(self, mode):
        setsz = self.k_shot * self.n_way
        querysz = self.k_query * self.n_way
        num_subj = self.subj[mode]

        support_x = np.zeros((self.batchsz[mode], setsz) + self.eeg_shape)
        support_y = np.zeros((self.batchsz[mode], setsz), dtype=int)
        query_x = np.zeros((self.batchsz[mode], querysz) + self.eeg_shape)
        query_y = np.zeros((self.batchsz[mode], querysz), dtype=int)

        selected_tasks = np.random.choice(num_subj, self.batchsz[mode], False)
        for i, cur_task in enumerate(selected_tasks):
            shuffle_idx = np.arange(self.n_way)
            np.random.shuffle(shuffle_idx)
            shuffle_idx_test = np.arange(self.n_way)
            np.random.shuffle(shuffle_idx_test)

            for j in range(self.n_way):
                selected_data = np.random.choice(self.num_trials, self.k_shot + self.k_query, False)

                for offset, eeg in enumerate(selected_data[:self.k_shot]):
                    support_x[i, shuffle_idx[j] * self.k_shot + offset, ...] = self.get_data(mode, cur_task, j, eeg)
                    support_y[i, shuffle_idx[j] * self.k_shot + offset] = j

                for offset, eeg in enumerate(selected_data[self.k_shot:]):
                    query_x[i, shuffle_idx_test[j] * self.k_query + offset, ...] = self.get_data(mode, cur_task, j, eeg)
                    query_y[i, shuffle_idx_test[j] * self.k_query + offset] = j

        return support_x, support_y, query_x, query_y

class CustomDataset(Dataset):
    def __init__(self, db, mode, k_shot, k_query) -> None:
        super().__init__()
        # self.data = data
        self.db = db
        self.mode = mode
        self.shape = (len(db.subj[mode]), db.n_way, db.num_trials) + db.eeg_shape
        self.n_way = self.shape[1]
        self.k_shot = k_shot
        self.k_query = k_query
        self.out_shape = (self.n_way * self.k_shot,) + self.shape[-2:]
        self.out_shape_query = (self.n_way * self.k_query,) + self.shape[-2:]
        self.shuffle_idx = np.zeros(self.shape[:3], dtype=int)
        for p in range(self.shape[0]):
            for q in range(self.shape[1]):
                idx_range = np.arange(self.shape[2])
                np.random.shuffle(idx_range)
                self.shuffle_idx[p, q, ...] = idx_range

    def __len__(self):
        return self.shape[0] * (self.shape[2] // (self.k_shot + self.k_query))

    def __getitem__(self, idx):
        idx2 = (self.k_shot + self.k_query) * (idx // self.shape[0])
        idx0 = idx % self.shape[0]

        support_x = np.zeros(self.out_shape)
        support_y = np.zeros(self.out_shape[:1], dtype=int)
        query_x = np.zeros(self.out_shape_query)
        query_y = np.zeros(self.out_shape_query[:1], dtype=int)

        for j in range(self.n_way):
            # support_x[(j*self.k_shot):((j+1)*self.k_shot), ...] = self.data[idx0][j][self.shuffle_idx[idx0, j, idx2:idx2+self.k_shot]]
            for v in range(self.k_shot):
                support_x[(j*self.k_shot) + v, ...] = self.db.get_data(self.mode, self.db.subj[self.mode][idx0], j, self.shuffle_idx[idx0, j, idx2+v])
            support_y[(j*self.k_shot):((j+1)*self.k_shot)] = j

            # query_x[(j*self.k_query):((j+1)*self.k_query), ...] = self.data[idx0][j][self.shuffle_idx[idx0, j, idx2+self.k_shot:idx2+self.k_shot+self.k_query]]
            for v in range(self.k_query):
                query_x[(j*self.k_query) + v, ...] = self.db.get_data(self.mode, self.db.subj[self.mode][idx0], j, self.shuffle_idx[idx0, j, idx2+self.k_shot+v])
            query_y[(j*self.k_query):((j+1)*self.k_query)] = j

        return support_x, support_y, query_x, query_y

class ZeroDataset(Dataset):
    def __init__(self, db, mode, k_shot, k_query, subj) -> None:
        super().__init__()
        # self.data = data
        self.db = db
        self.mode = mode
        self.subj = subj
        self.shape = (1, db.n_way, db.num_trials) + db.eeg_shape
        self.n_way = self.shape[1]
        self.k_shot = k_shot
        self.k_query = k_query
        self.out_shape = (self.n_way * self.k_shot,) + self.shape[-2:]
        self.out_shape_query = (self.n_way * self.k_query,) + self.shape[-2:]
        self.shuffle_idx = np.zeros(self.shape[:3], dtype=int)
        for p in range(self.shape[0]):
            for q in range(self.shape[1]):
                idx_range = np.arange(self.shape[2])
                np.random.shuffle(idx_range)
                self.shuffle_idx[p, q, ...] = idx_range

    def __len__(self):
        return self.shape[0] * (self.shape[2] // (self.k_shot + self.k_query))

    def __getitem__(self, idx):
        idx2 = (self.k_shot + self.k_query) * (idx // self.shape[0])
        idx0 = idx % self.shape[0]

        support_x = np.zeros(self.out_shape)
        support_y = np.zeros(self.out_shape[:1], dtype=int)
        query_x = np.zeros(self.out_shape_query)
        query_y = np.zeros(self.out_shape_query[:1], dtype=int)

        for j in range(self.n_way):
            # support_x[(j*self.k_shot):((j+1)*self.k_shot), ...] = self.data[idx0][j][self.shuffle_idx[idx0, j, idx2:idx2+self.k_shot]]
            for v in range(self.k_shot):
                support_x[(j*self.k_shot) + v, ...] = self.db.get_data(self.mode, self.subj, j, self.shuffle_idx[idx0, j, idx2+v])
            support_y[(j*self.k_shot):((j+1)*self.k_shot)] = j

            # query_x[(j*self.k_query):((j+1)*self.k_query), ...] = self.data[idx0][j][self.shuffle_idx[idx0, j, idx2+self.k_shot:idx2+self.k_shot+self.k_query]]
            for v in range(self.k_query):
                query_x[(j*self.k_query) + v, ...] = self.db.get_data(self.mode, self.subj, j, self.shuffle_idx[idx0, j, idx2+self.k_shot+v])
            query_y[(j*self.k_query):((j+1)*self.k_query)] = j

        return support_x, support_y, query_x, query_y

class Learner(nn.Module):
    '''
    It stores a specific nn.Module class
    '''

    def __init__(self, net_class, *args) -> None:
        '''
        net_class is a class, not an instance
        args: the parameters for net_class
        '''
        super(Learner, self).__init__()
        assert net_class.__class__ == type

        self.net = net_class(*args).to(DEVICE)
        self.net_pi = net_class(*args).to(DEVICE)
        self.learner_lr = 0.1
        self.optimizer = optim.SGD(self.net_pi.parameters(), self.learner_lr)

    def parameters(self):
        '''
        ignore self.net_pi.parameters()
        '''
        return self.net.parameters()

    def update_pi(self):
        for m_from, m_to in zip(self.net.modules(), self.net_pi.modules()):
            if isinstance(m_to, nn.Linear) or isinstance(m_to, nn.Conv2d) or isinstance(m_to, nn.BatchNorm2d):
                m_to.weight.data = m_from.weight.data.clone()
                if m_to.bias is not None:
                    m_to.bias.data = m_from.bias.data.clone()

    def forward(self, support_x, support_y, query_x, query_y, num_updates, testing=False):
        self.update_pi()
        if testing:
            self.net_pi.freeze()
        for i in range(num_updates):
            loss, pred = self.net_pi(support_x, support_y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        if testing:
            self.net_pi.unfreeze()
        loss, pred = self.net_pi(query_x, query_y)
        indices = torch.argmax(pred, dim=1)
        correct = torch.eq(indices, query_y).sum().item()
        acc = correct / query_y.size(0)

        grads_pi = autograd.grad(loss, self.net_pi.parameters(), create_graph=True)
        return loss, grads_pi, acc

    def net_forward(self, support_x, support_y):
        loss, pred = self.net(support_x, support_y)
        return loss, pred

class MetaLearner(nn.Module):
    def __init__(self, net_class, net_class_args, n_way, k_shot, meta_batchesz, beta, num_updates, num_updates_test) -> None:
        super(MetaLearner, self).__init__()
        self.n_way = n_way
        self.k_shot = k_shot
        self.meta_batchesz = meta_batchesz
        self.beta = beta
        self.num_updates = num_updates
        self.num_updates_test = num_updates_test

        self.learner = Learner(net_class, *net_class_args)
        self.optimizer = optim.Adam(self.learner.parameters(), lr=beta)

    def write_grads(self, dummy_loss, sum_grads_pi):
        hooks = []
        for i, v in enumerate(self.learner.parameters()):
            def closure():
                ii = i
                return lambda grad : sum_grads_pi[ii]
            # h = v.register_hook(closure())
            hooks.append(v.register_hook(closure()))

        self.optimizer.zero_grad()
        dummy_loss.backward()
        self.optimizer.step()

        for h in hooks:
            h.remove()

    def forward(self, support_x, support_y, query_x, query_y):
        sum_grads_pi = None
        meta_batchesz = support_y.size(0)

        accs = []
        for i in range(meta_batchesz):
            _, grad_pi, episode_acc = self.learner(support_x[i], support_y[i], query_x[i], query_y[i], self.num_updates)
            accs.append(episode_acc)
            if sum_grads_pi is None:
                sum_grads_pi = grad_pi
            else:
                sum_grads_pi = [torch.add(p, q) for p, q in zip(sum_grads_pi, grad_pi)]
        dummy_loss, _ = self.learner.net_forward(support_x[0], support_y[0])
        self.write_grads(dummy_loss, sum_grads_pi)

        return accs

    def pred(self, support_x, support_y, query_x, query_y):
        meta_batchesz = support_y.size(0)
        accs = []
        for i in range(meta_batchesz):
            _, _, episode_acc = self.learner(support_x[i], support_y[i], query_x[i], query_y[i], self.num_updates_test, testing=True)
            accs.append(episode_acc)
        return np.array(accs).mean()

class Conv2dSamePadding(nn.Conv2d):
    def __init__(self,*args,**kwargs):
        super(Conv2dSamePadding, self).__init__(*args, **kwargs)
        self.zero_pad_2d = nn.ZeroPad2d(reduce(__add__,
            [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in self.kernel_size[::-1]]))

    def forward(self, input):
        return  self._conv_forward(self.zero_pad_2d(input), self.weight, self.bias)

class EEGNet(nn.Module):

    def __init__(self, nb_classes, Chans = 22, Samples = 1001,
                 dropoutRate = 0.5, kernLength = 64, F1 = 8,
                 D = 2, F2 = 16, norm_rate = 0.25) -> None:
        super().__init__()
        self.device = DEVICE

        self.block1 = nn.Sequential(
            Conv2dSamePadding(1, F1, (1, kernLength), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (Chans, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropoutRate)
        ).to(DEVICE)

        self.block2 = nn.Sequential(
            Conv2dSamePadding(F2, F2, (1, 16), bias=False),
            nn.Conv2d(F2, F2, 1, padding=0, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropoutRate)
        ).to(DEVICE)

        self.block_weights = ['block1.0.weight', 'block1.1.weight', 'block1.1.bias', 'block1.2.weight', 'block1.3.weight', 'block1.3.bias', 'block2.0.weight', 'block2.1.weight', 'block2.2.weight', 'block2.2.bias']

        self.classifier_input = 16 * ((Samples // 4) // 8)
        self.classifier_hidden = int((self.classifier_input * nb_classes) ** 0.5)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.classifier_input, self.classifier_hidden),
            nn.Linear(self.classifier_hidden, nb_classes)
        ).to(DEVICE)

        self.criterion = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, x, target=None):
        x = self.block1(torch.unsqueeze(x, 1))
        x = self.block2(x)
        pred = self.classifier(x)

        loss = self.criterion(pred, target)
        return loss, pred

    def freeze(self):
        for name, param in self.named_parameters():
            if name in self.block_weights:
                param.requires_grad = False

    def unfreeze(self):
        for name, param in self.named_parameters():
            if name in self.block_weights:
                param.requires_grad = True

def meta_train(db, meta, iterations):
    file.write("\nReptile\n")
    for episode_num in range(iterations):
        support_x, support_y, query_x, query_y = db.get_batch('train')
        support_x = Variable( torch.from_numpy(support_x).float()).to(DEVICE)
        query_x = Variable( torch.from_numpy(query_x).float()).to(DEVICE)
        support_y = Variable(torch.from_numpy(support_y).long()).to(DEVICE)
        query_y = Variable(torch.from_numpy(query_y).long()).to(DEVICE)

        accs = meta(support_x, support_y, query_x, query_y)
        train_acc = 100 * np.array(accs).mean()

        if episode_num % 50 == 0:
            test_accs = []
            for i in range(min(episode_num // 5000 + 3, 10)):
                support_x, support_y, query_x, query_y = db.get_batch('test')
                support_x = Variable( torch.from_numpy(support_x).float()).to(DEVICE)
                query_x = Variable( torch.from_numpy(query_x).float()).to(DEVICE)
                support_y = Variable(torch.from_numpy(support_y).long()).to(DEVICE)
                query_y = Variable(torch.from_numpy(query_y).long()).to(DEVICE)

                test_acc = meta.pred(support_x, support_y, query_x, query_y)
                test_accs.append(test_acc)

            test_acc = 100 * np.array(test_accs).mean()
            file.write(f"episode: {episode_num}\tfinetune acc: {train_acc:.5f}\t\ttest acc: {test_acc:.5f}\n")

def train(net, train_loader, epochs):
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    train_log = []
    val_log = []
    for epoch in range(epochs):
        accs = []
        train_loss = []
        val_loss = []
        for support_x, support_y, query_x, query_y in tqdm(train_loader):
            support_x = Variable(support_x[0].float()).to(DEVICE)
            query_x = Variable(query_x[0].float()).to(DEVICE)
            support_y = Variable(support_y[0].long()).to(DEVICE)
            query_y = Variable(query_y[0].long()).to(DEVICE)

            net.train()
            loss, pred = net(support_x, support_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

            net.eval()
            loss, pred = net(query_x, query_y)
            val_loss.append(loss.item())
            indices = torch.argmax(pred, dim=1)
            correct = torch.eq(indices, query_y).sum().item()
            acc = correct / query_y.size(0)
            accs.append(acc)
        train_loss = np.array(train_loss).mean()
        train_log.append(train_loss)
        val_loss = np.array(val_loss).mean()
        val_log.append(val_loss)
        accuracy = 100 * np.array(accs).mean()
        file.write(f"Epoch {epoch+1}: \tvalidation acc: {accuracy:.5f}\tvalidation loss: {val_loss:.6f}\ttrain loss: {train_loss:.6f}\n")
    plt.plot(train_log)
    plt.plot(val_log)
    plt.show()

def evaluate(learner, db, min_updates_test=5, max_updates_test=150):
    file.write("\nEvaluate:\n")
    test_loader = DataLoader(
        dataset = CustomDataset(db, "test", k_shot=0, k_query=4),
        batch_size = 1
    )
    accs = []
    for support_x, support_y, query_x, query_y in test_loader:
        support_x = Variable(support_x[0].float()).to(DEVICE)
        query_x = Variable(query_x[0].float()).to(DEVICE)
        support_y = Variable(support_y[0].long()).to(DEVICE)
        query_y = Variable(query_y[0].long()).to(DEVICE)
        _, _, episode_acc = learner(support_x, support_y, query_x, query_y, 0, testing=True)
        accs.append(episode_acc)
    accs = 100 * np.array(accs)

    results = [{"mean":accs.mean(), "std":accs.std(), "num_upd":0}]
    file.write(f"{0}-shot accuracy: \tmean: {results[0]['mean']:.6f}{'%'}\tstd: {results[0]['std']:.6f}{'%'}\n")

    for K in range(1, 11):
        dct = {"mean":0, "std":0, "num_upd":0}
        for num_updates_test in tqdm(range(min_updates_test, max_updates_test, 2)):
            test_loader = DataLoader(
                dataset = CustomDataset(db, "test", k_shot=K, k_query=4),
                batch_size = 1
            )
            accs = []
            for support_x, support_y, query_x, query_y in test_loader:
                support_x = Variable(support_x[0].float()).to(DEVICE)
                query_x = Variable(query_x[0].float()).to(DEVICE)
                support_y = Variable(support_y[0].long()).to(DEVICE)
                query_y = Variable(query_y[0].long()).to(DEVICE)
                _, _, episode_acc = learner(support_x, support_y, query_x, query_y, num_updates_test, testing=True)
                accs.append(episode_acc)
            accs = 100 * np.array(accs)
            if accs.mean() > dct["mean"]:
                dct["mean"] = accs.mean()
                dct["std"] = accs.std()
                dct["num_upd"] = num_updates_test

        results.append(dct)
        file.write(f"{K}-shot accuracy: \tmean: {dct['mean']:.6f}{'%'}\tstd: {dct['std']:.6f}{'%'}\tafter {dct['num_upd']} updates\n")

    return results

def evaluate0(learner, db):
    file.write("\nEvaluate on train subjects:\n")
    for subj in db.subj['train']:
        test_loader = DataLoader(
            dataset = ZeroDataset(db, "train", k_shot=0, k_query=4, subj=subj),
            batch_size = 1
        )
        accs = []
        for support_x, support_y, query_x, query_y in test_loader:
            support_x = Variable(support_x[0].float()).to(DEVICE)
            query_x = Variable(query_x[0].float()).to(DEVICE)
            support_y = Variable(support_y[0].long()).to(DEVICE)
            query_y = Variable(query_y[0].long()).to(DEVICE)
            _, _, episode_acc = learner(support_x, support_y, query_x, query_y, 0, testing=True)
            accs.append(episode_acc)
        accs = 100 * np.array(accs)

        results = [{"mean":accs.mean(), "std":accs.std(), "num_upd":0}]
        file.write(f"{0}-shot accuracy on subject {subj}: \tmean: {results[0]['mean']:.6f}{'%'}\tstd: {results[0]['std']:.6f}{'%'}\n")

def bciiv2a(model, iterations=30000, epochs=100, Reptile=True):
    root = ''
    data_npy = 'bciiv2a'
    # dataset = BNCI2014001()
    meta_batchsz = 5
    n_way = 4
    k_shot = 4
    k_query = k_shot

    meta_lr = 1e-3
    num_updates = 2
    num_updates_test = 10

    # fmin, fmax = 4, 32
    # raw = dataset.get_data(subjects=[1])[1]['session_T']['run_1']
    # dataset_channels = raw.pick_types(eeg=True).ch_names
    # sfreq = 250.
    # prgm_MI_classes = MotorImagery(n_classes=4, channels=dataset_channels, resample=sfreq, fmin=fmin, fmax=fmax)

    i = 3
    train_subj = list(range(1, i)) + list(range(i + 1, 10))
    test_subj = [i]

    db = DataNShot(root, data_npy, train_subj, test_subj, meta_batchsz, n_way, k_shot, k_query, (22, 1001), 144)

    if Reptile:
        meta = MetaLearner(model, (4, 22, 1001), n_way=n_way, k_shot=k_shot, meta_batchesz=meta_batchsz,
                           beta=meta_lr, num_updates=num_updates, num_updates_test=num_updates_test).to(DEVICE)
        meta_train(db, meta, iterations)
        return meta.learner, db
    else:
        net = model(4, 22, 1001).to(DEVICE)

        g = torch.Generator()
        g.manual_seed(42)
        train_loader = DataLoader(
            dataset=CustomDataset(db, "train", k_shot=k_shot, k_query=1),
            batch_size=1,
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=g
        )
        train(net, train_loader, epochs)
        net_trained = Learner(model, 4, 22, 1001)
        net_trained.net = net
        return net_trained, db

bciiv2a_EEGNet, bciiv2a_data = bciiv2a(EEGNet, epochs=49, Reptile=False)
evaluate0(bciiv2a_EEGNet, bciiv2a_data)
results_bciiv2a_EEGNet = evaluate(bciiv2a_EEGNet, bciiv2a_data)

bciiv2a_EEGNet_Reptile, bciiv2a_data = bciiv2a(EEGNet, iterations=20000, Reptile=True)
evaluate0(bciiv2a_EEGNet_Reptile, bciiv2a_data)
results_bciiv2a_EEGNet_Reptile = evaluate(bciiv2a_EEGNet_Reptile, bciiv2a_data)
