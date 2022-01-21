import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
import signatory
from time import process_time
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import multiprocessing as mp
import argparse
from matplotlib import font_manager

curwd = 'C:/study/research/code/sigpen'
with open(curwd+'/alphabet_3926', 'rb') as f:
    chardic = pickle.load(f)
chinese = list(chardic.keys())

## visualize the j-th hand-written character i
fontP = font_manager.FontProperties()
fontP.set_family('SimHei')
fontP.set_size(14)
def draw(i, j):
    setpath = osp.join(curwd + '/Pot11TrainPath/' + str(i).zfill(5))
    files = os.listdir(setpath)

    with open(osp.join(setpath + '/', files[j]), 'rb') as f:
        path = np.load(f)
    l = len(path)
    n_stroke = int(path[-1][2])
    for k in range(n_stroke):
        plt.plot([path[idx][0] for idx in range(l) if path[idx][2] == k+1],
                 [path[idx][1] for idx in range(l) if path[idx][2] == k+1], '*-')
    plt.title(chinese[i], fontproperties=fontP)
    plt.savefig('pic.png')
    plt.show()


def readTrain(i):#169, 3924
    setpath = osp.join(curwd+'/Pot11TrainPath/' + str(i).zfill(5))
    files = os.listdir(setpath)
    print(chinese[i])
    sigtrain = torch.tensor([])
    for j in range(len(files)):
        with open(osp.join(setpath + '/', files[j]), 'rb') as f:
            path = np.load(f)
        l = len(path)
        for k in range(l-1):
            path[k][2] = 0 if path[k][2] == path[k+1][2] else 1
        path[l-1][2] = 1
        path = path.tolist()
        if l < 30:
            path += [[0.0,0.0,0.0]] * (30-l)
            l = 30

        lenseg = int(l / 10)
        sigpath = torch.tensor([])
        for k in range(9):
            sig = signatory.signature(torch.tensor([path[(k*lenseg):((k+1)*lenseg)]]), 4)
            sigpath = torch.cat((sigpath, torch.cat((torch.tensor([path[k*lenseg][:2]]), sig), 1)), 0)
        sig = signatory.signature(torch.tensor([path[(9 * lenseg-1):l]]), 4)
        sigpath = torch.cat((sigpath, torch.cat((torch.tensor([path[9 * lenseg-1][:2]]), sig), 1)), 0)

        sigtrain = torch.cat((sigtrain, sigpath.unsqueeze(0)), 0)
        del path
        del sigpath
    trainlabel = [i] * len(files)
    return sigtrain, trainlabel


def readTest(i):
    setpath = osp.join('./Pot11TestPath/' + str(i).zfill(5))
    files = os.listdir(setpath)
    print(chinese[i])
    sigtest = torch.tensor([])
    for j in range(len(files)):
        with open(osp.join(setpath + '/', files[j]), 'rb') as f:
            path = np.load(f)
        l = len(path);
        for k in range(l-1):
            path[k][2] = 0 if path[k][2] == path[k+1][2] else 1
        path[l-1][2] = 1
        path = path.tolist()
        if l < 30:
            path += [[0.0,0.0,0.0]] * (30-l)
            l = 30

        lenseg = int(l / 10)
        sigpath = torch.tensor([])
        for k in range(9):
            sig = signatory.signature(torch.tensor([path[(k*lenseg):((k+1)*lenseg)]]), 4)
            sigpath = torch.cat((sigpath, torch.cat((torch.tensor([path[k*lenseg][:2]]), sig), 1)), 0)
        sig = signatory.signature(torch.tensor([path[(9 * lenseg-1):l]]), 4)
        sigpath = torch.cat((sigpath, torch.cat((torch.tensor([path[9 * lenseg-1][:2]]), sig), 1)), 0)

        sigtest = torch.cat((sigtest, sigpath.unsqueeze(0)), 0)
        del path
        del sigpath
    testlabel = [i] * len(files)
    return sigtest, testlabel

is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = self.fc(self.relu(out[:, -1]))
        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        return hidden


class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, h):
        out, h = self.lstm(x, h)
        out = self.fc(self.relu(out[:, -1]))
        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        return hidden

def train(train_loader, output_dim, learn_rate = 0.001, n_layers = 2, hidden_dim=256, EPOCHS=5, model_type="GRU"):
    # Setting common hyperparameters
    input_dim = next(iter(train_loader))[0].shape[2]

    # Instantiating the models
    if model_type == "GRU":
        model = GRUNet(input_dim, hidden_dim, output_dim, n_layers)
    else:
        model = LSTMNet(input_dim, hidden_dim, output_dim, n_layers)
    model.to(device)

    # Defining loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    model.train()
    print("Starting Training of {} model".format(model_type))
    epoch_times = []
    # Start training loop
    for epoch in range(1, EPOCHS + 1):
        start_time = process_time()
        h = model.init_hidden(batch_size)
        avg_loss = 0.
        counter = 0
        for x, label in train_loader:
            counter += 1
            if model_type == "GRU":
                h = h.data
            else:
                h = tuple([e.data for e in h])
            model.zero_grad()

            out, h = model(x.to(device), h)
            loss = criterion(out, label.to(device).float())
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            if counter % 200 == 0:
                print("Epoch {}......Step: {}/{}....... Average Loss for Epoch: {}".format(epoch, counter,
                                                                                           len(train_loader),
                                                                                           avg_loss / counter))
        current_time = process_time()
        print("Epoch {}/{} Done, Total Loss: {}".format(epoch, EPOCHS, avg_loss / len(train_loader)))
        print("Total Time Elapsed: {} seconds".format(str(current_time - start_time)))
        epoch_times.append(current_time - start_time)
    traintime = sum(epoch_times)
    print("Total Training Time: {} seconds".format(str(traintime)))
    return model, traintime

def evaluate(model, testset, testlabel):
    model.eval()
    start_time = process_time()

    h = model.init_hidden(testset.shape[0])
    out, h = model(testset.to(device).float(), h)
    _, predlabel = torch.max(torch.log_softmax(out.cpu().detach(), dim = 1), dim = 1)
    print("Evaluation Time: {}".format(str(process_time()  - start_time)))

    predrate = sum(predlabel.numpy() == testlabel)/len(testlabel) * 100
    print("prediction accuracy rate: {}%".format(predrate))
    return predrate

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--idstart', type=int, default=169)
    parser.add_argument('--idend', type=int, default=3924)
    args = parser.parse_args()

    idstart = args.idstart
    idend = args.idend
    pool = mp.Pool(mp.cpu_count())
    result = pool.map(readTrain, range(idstart, idend))
    sigtrain = torch.tensor([])
    trainlabel = []
    for i in range(idend - idstart):
        path = torch.tensor(result[i][0])
        print(path.shape)
        sigtrain = torch.cat((sigtrain, path), 0)
        trainlabel += result[i][1]
    print(sigtrain.shape, len(trainlabel))
    pool.close()

    pool = mp.Pool(mp.cpu_count())
    result = pool.map(readTest, range(idstart, idend))
    sigtest = torch.tensor([])
    testlabel = []
    for i in range(idend - idstart):
        path = torch.tensor(result[i][0])
        print(path.shape)
        sigtest = torch.cat((sigtest, path), 0)
        tl = result[i][1]
        testlabel += [i - idstart for i in tl]
    print(sigtest.shape, len(testlabel))
    pool.close()

    lr = 0.0005
    nBatch = [128, 256, 512]
    trainlabel = [tl - idstart for tl in trainlabel]
    trainlabel = np.eye(idend - idstart, dtype='uint8')[trainlabel]
    train_data = TensorDataset(sigtrain, torch.from_numpy(np.array(trainlabel)))
    text_file = open("Output.txt", "w")

    for i in range(len(nBatch)):
        batch_size = nBatch[i]
        train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
        gru_model, traintime = train(train_loader, idend - idstart, lr, 2, 1024, 10, model_type="GRU")
        predrate = evaluate(gru_model, sigtest, testlabel)
        text_file.write("Total Training Time: {} seconds\n".format(str(traintime)))
        text_file.write("prediction accuracy rate: {}%\n".format(predrate))
        del train_loader
        del gru_model
    text_file.close()