import math
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import signatory
import copy
import os
import numpy as np
from itertools import groupby
import re
import pandas as pd
import random
import time

import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm_notebook
from sklearn.preprocessing import MinMaxScaler

## Define a function to transform original data and save as numpy array
def GetPenDigit(Path='pendigits-orig.tra.txt'):
    """
    Load the original pen-digit data
    """
    path = Path
    penf = open(path)
    pen = penf.read()
    penf.close()
    penseq = pen.split('\n')
    penseq = [x for x in penseq if '.SEGMENT' not in x]
    label = []
    for i in range(len(penseq)):
        if '.COMMENT' in penseq[i]:
            tmplabel = re.split(r' ', penseq[i])
            tmplabel = int(tmplabel[1])
            label.append(tmplabel)
            penseq[i] = '.'
    penseq = [list(g) for k, g in groupby(penseq, lambda x: x == '.') if not k]
    penseq = penseq[1:]
    for i in range(len(penseq)):
        penseq[i] = [x for x in penseq[i] if '.PEN' not in x]
        penseq[i] = [x for x in penseq[i] if '.DT' not in x]
        penseq[i] = [x for x in penseq[i] if x != '']
        for j in range(len(penseq[i])):
            penseq[i][j] = re.split(r' ', penseq[i][j])
            penseq[i][j] = [int(x) for x in penseq[i][j] if x != '']

    return penseq, label
trainseq, trainlabel = GetPenDigit()
newtrain = copy.deepcopy(trainseq)

## delete too short or too long pen-digits
length = np.array([len(x) for x in trainseq])
plt.hist(length, bins=30)
lindex = np.where(length >= 80)[0]
xdig = [k[0] for k in trainseq[lindex[2]]]
ydig = [k[1] for k in trainseq[lindex[2]]]
plt.plot(xdig, ydig, '*b')

sindex = np.where(length <= 15)[0]
xdig = [k[0] for k in trainseq[sindex[3]]]
ydig = [k[1] for k in trainseq[sindex[3]]]
plt.plot(xdig, ydig, '*b')

bad = np.concatenate([lindex, sindex]).tolist()
trainseq = np.delete(trainseq, bad)
trainlabel = np.delete(trainlabel, bad)
#
# dist = [0] * len(trainseq)
# for i in range(len(trainseq)):
#     maxdist = 0
#     for j in range(len(trainseq[i])-1):
#         maxdist = max(maxdist, math.sqrt((trainseq[i][j][0] - trainseq[i][j+1][0])**2+
#                                    (trainseq[i][j][1] - trainseq[i][j+1][1])**2))
#     dist[i] = maxdist
# plt.hist(dist, bins = 30)
# plt.xlabel('max_distance', fontsize=18)
# plt.ylabel('counts', fontsize=18)

## scale and augment train data
l = len(trainseq)
for i in range(l):
    trainseq[i][0].append(0)
    for j in range(1,len(trainseq[i])):
        if math.sqrt((trainseq[i][j][0] - trainseq[i][j-1][0])**2+\
                     (trainseq[i][j][1] - trainseq[i][j-1][1])**2) < 30:
            trainseq[i][j].append(trainseq[i][j-1][2] + 0.01)
        else:
            trainseq[i][j].append(trainseq[i][j-1][2])

    tmpx = [k[0] for k in trainseq[i]]
    tmpy = [k[1] for k in trainseq[i]]
    minx = min(tmpx)
    maxx = max(tmpx)
    miny = min(tmpy)
    maxy = max(tmpy)
    scale = float(2) / max([(maxx - minx), (maxy - miny)])
    list_x = [(x - (maxx + minx) / 2) * scale for x in tmpx]
    list_y = [(y - (maxy + miny) / 2) * scale for y in tmpy]
    for j in range(len(trainseq[i])):
        trainseq[i][j][0] = list_x[j]
        trainseq[i][j][1] = list_y[j]
        newtrain[i][j][0] = list_x[j]
        newtrain[i][j][1] = list_y[j]

## Another augmentation
# for i in range(l):
#     tmpx = [k[0] for k in trainseq[i]]
#     tmpy = [k[1] for k in trainseq[i]]
#     minx = min(tmpx)
#     maxx = max(tmpx)
#     miny = min(tmpy)
#     maxy = max(tmpy)
#     scale = float(2) / max([(maxx - minx), (maxy - miny)])
#     list_x = [(x - (maxx + minx) / 2) * scale for x in tmpx]
#     list_y = [(y - (maxy + miny) / 2) * scale for y in tmpy]
#     for j in range(len(trainseq[i])):
#         trainseq[i][j][0] = list_x[j]
#         trainseq[i][j][1] = list_y[j]
#         trainseq[i][j] += [0, 0]
#         newtrain[i][j][0] = list_x[j]
#         newtrain[i][j][1] = list_y[j]
#
#     for j in range(len(trainseq[i])-1):
#         trainseq[i][j][2] = trainseq[i][j+1][0] - trainseq[i][j][0]
#         trainseq[i][j][3] = trainseq[i][j+1][1] - trainseq[i][j][1]
#     trainseq[i] = trainseq[i][:-1]

## signature of training data
sigdf = pd.DataFrame(index=range(l), columns=["Sig" + str(i+1) for i in range(30)])
logsigdf = pd.DataFrame(index=range(l), columns=["logSig" + str(i+1) for i in range(8)])
sigmd = pd.DataFrame(index=range(l), columns=["Sig" + str(i+1) for i in range(120)])
logsigmd = pd.DataFrame(index=range(l), columns=["logSig" + str(i+1) for i in range(32)])

for i in range(l):
    penseqs = torch.tensor([newtrain[i]])
    sigdf.loc[i, :] = signatory.signature(penseqs, 4)
    logsigdf.loc[i, :] = signatory.logsignature(penseqs, 4)

    misspen = torch.tensor([trainseq[i]])
    sigmd.loc[i, :] = signatory.signature(misspen, 4)
    logsigmd.loc[i, :] = signatory.logsignature(misspen, 4)

## process test data
testseq, testlabel = GetPenDigit('pendigits-orig.tes.txt')
newtest = copy.deepcopy(testseq)
testl = len(testseq)

for i in range(testl):
    testseq[i][0].append(0)
    for j in range(1,len(testseq[i])):
        if math.sqrt((testseq[i][j][0] - testseq[i][j-1][0])**2+\
                     (testseq[i][j][1] - testseq[i][j-1][1])**2) < 30:
            testseq[i][j].append(testseq[i][j-1][2] + 0.01)
        else:
            testseq[i][j].append(testseq[i][j-1][2])

    tmpx = [k[0] for k in testseq[i]]
    tmpy = [k[1] for k in testseq[i]]
    minx = min(tmpx)
    maxx = max(tmpx)
    miny = min(tmpy)
    maxy = max(tmpy)
    scale = float(2) / max([(maxx - minx), (maxy - miny)])
    list_x = [(x - (maxx + minx) / 2) * scale for x in tmpx]
    list_y = [(y - (maxy + miny) / 2) * scale for y in tmpy]
    for j in range(len(testseq[i])):
        testseq[i][j][0] = list_x[j]
        testseq[i][j][1] = list_y[j]
        newtest[i][j][0] = list_x[j]
        newtest[i][j][1] = list_y[j]

sigtest = pd.DataFrame(index=range(testl), columns=["Sig" + str(i+1) for i in range(30)])
logsigtest = pd.DataFrame(index=range(testl), columns=["logSig" + str(i+1) for i in range(8)])
sigmdtest = pd.DataFrame(index=range(testl), columns=["Sig" + str(i+1) for i in range(120)])
logsigmdtest = pd.DataFrame(index=range(testl), columns=["logSig" + str(i+1) for i in range(32)])

for i in range(testl):
    penseqs = torch.tensor([newtest[i]])
    sigtest.loc[i, :] = signatory.signature(penseqs, 4)
    logsigtest.loc[i, :] = signatory.logsignature(penseqs, 4)

    misspen = torch.tensor([testseq[i]])
    sigmdtest.loc[i, :] = signatory.signature(misspen, 4)
    logsigmdtest.loc[i, :] = signatory.logsignature(misspen, 4)

## logsitic regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

logis1 = LogisticRegression(penalty='l1', solver='liblinear',max_iter = 2000)
logis1.fit(sigdf.to_numpy(), trainlabel)
yPred1 = logis1.predict(sigtest.to_numpy())
confmt1 = confusion_matrix(testlabel, yPred1)
sns.heatmap(confmt1, annot=True)
corr1 = sum([confmt1[i][i] for i in range(10)])/testl

logis2 = LogisticRegression(penalty='l1', solver='liblinear',max_iter = 2000)
logis2.fit(logsigdf.to_numpy(), trainlabel)
yPred2 = logis2.predict(logsigtest.to_numpy())
confmt2 = confusion_matrix(testlabel, yPred2)
sns.heatmap(confmt2, annot=True)
corr2 = sum([confmt2[i][i] for i in range(10)])/testl

logis3 = LogisticRegression(penalty='l1', solver='liblinear',max_iter = 2000)
logis3.fit(sigmd.to_numpy(), trainlabel)
yPred3 = logis3.predict(sigmdtest.to_numpy())
confmt3 = confusion_matrix(testlabel, yPred3)
sns.heatmap(confmt3, annot=True)
corr3 = sum([confmt3[i][i] for i in range(10)])/testl

logis4 = LogisticRegression(penalty='l1', solver='liblinear',max_iter = 2000)
logis4.fit(logsigmd.to_numpy(), trainlabel)
yPred4 = logis4.predict(logsigmdtest.to_numpy())
confmt4 = confusion_matrix(testlabel, yPred4)
sns.heatmap(confmt4, annot=True)
corr4 = sum([confmt4[i][i] for i in range(10)])/testl

xdig = [k[0] for k in newtrain[6]]
ydig = [k[1] for k in newtrain[6]]
gapid = 0; l = len(xdig)
for i in range(l-1):
    if math.sqrt((xdig[i]-xdig[i+1])**2 +(ydig[i]-ydig[i+1])**2) <30:
        continue
    else:
        plt.plot(xdig[gapid:(i+1)],ydig[gapid:(i+1)],'*b-')
        gapid = i+1
plt.plot(xdig[gapid:l],ydig[gapid:l],'*b-')
xdig = [k[0] for k in newtrain[2]]
ydig = [k[1] for k in newtrain[2]]
plt.plot(xdig,ydig,'*')

## Digit prediction
# def digitrecognition(input):
#     inlen = len(input)
#     sigdf = pd.DataFrame(index=range(l), columns=["Sig" + str(i+1) for i in range(30)])
#     logsigdf = pd.DataFrame(index=range(l), columns=["logSig" + str(i+1) for i in range(8)])
#     sigmd = pd.DataFrame(index=range(l), columns=["Sig" + str(i+1) for i in range(120)])
#     logsigmd = pd.DataFrame(index=range(l), columns=["logSig" + str(i+1) for i in range(32)])
#
#     for i in range(l):
#         if len(newtrain[i]) <= inlen:
#             penseqs = torch.tensor([newtrain[i]])
#             misspen = torch.tensor([trainseq[i]])
#         else:
#             penseqs = torch.tensor([newtrain[i][0:inlen]])
#             misspen = torch.tensor([trainseq[i][0:inlen]])
#         sigdf.loc[i, :] = signatory.signature(penseqs, 4)
#         logsigdf.loc[i, :] = signatory.logsignature(penseqs, 4)
#         sigmd.loc[i, :] = signatory.signature(misspen, 4)
#         logsigmd.loc[i, :] = signatory.logsignature(misspen, 4)
#
#     auginput = copy.deepcopy(input)
#     auginput[0].append(0)
#     for j in range(1, inlen):
#         if math.sqrt((auginput[j][0] - auginput[j - 1][0])**2 + \
#                 (auginput[j][1] - auginput[j - 1][1])**2) < 30:
#             auginput[j].append(auginput[j - 1][2] + 0.01)
#         else:
#             auginput[j].append(auginput[j - 1][2])
#
#     tmpx = [k[0] for k in input]
#     tmpy = [k[1] for k in input]
#     minx = min(tmpx)
#     maxx = max(tmpx)
#     miny = min(tmpy)
#     maxy = max(tmpy)
#     scale = float(2) / max([(maxx - minx), (maxy - miny)])
#     list_x = [(x - (maxx + minx) / 2) * scale for x in tmpx]
#     list_y = [(y - (maxy + miny) / 2) * scale for y in tmpy]
#     for j in range(inlen):
#         input[j][0] = list_x[j]
#         input[j][1] = list_y[j]
#         auginput[j][0] = list_x[j]
#         auginput[j][1] = list_y[j]
#
#     logis1 = LogisticRegression(penalty='l1', solver='liblinear', max_iter=2000)
#     logis1.fit(sigdf.to_numpy(), trainlabel)
#     yPred1 = logis1.predict_proba(signatory.signature(torch.tensor([input]), 4))
#
#     logis2 = LogisticRegression(penalty='l1', solver='liblinear', max_iter=2000)
#     logis2.fit(logsigdf.to_numpy(), trainlabel)
#     yPred2 = logis2.predict_proba(signatory.logsignature(torch.tensor([input]), 4))
#
#     logis3 = LogisticRegression(penalty='l1', solver='liblinear', max_iter=2000)
#     logis3.fit(sigmd.to_numpy(), trainlabel)
#     yPred3 = logis3.predict_proba(signatory.signature(torch.tensor([auginput]), 4))
#
#     logis4 = LogisticRegression(penalty='l1', solver='liblinear', max_iter=2000)
#     logis4.fit(logsigmd.to_numpy(), trainlabel)
#     yPred4 = logis4.predict_proba(signatory.logsignature(torch.tensor([auginput]), 4))
#
#     pred_prob = yPred1 + yPred2 + yPred3 + yPred4
#     labels = np.argsort(pred_prob[0])[::-1]
#     xdig = [k[0] for k in input]
#     ydig = [k[1] for k in input]
#     plt.plot(xdig, ydig, '*')
#     print("The most possible number given by the input is: %d, %d, %d"
#           %(labels[0], labels[1], labels[2]))
#
# id = input("Please give me the index of the test data:")
# inlen = input("Please gice me the number of points in the hand-written data:")
# input = testseq[int(id)]
# iplen = min(len(input), int(inlen))
# digitrecognition(input[0:iplen])


## Dynamic binary
# trainlabel = np.array(trainlabel)
# id1 = np.where(trainlabel==8)[0]
# id2 = np.where(trainlabel==9)[0]
# binid = np.concatenate((id1,id2)).tolist()
# subtrainlabel = trainlabel[binid]
# subseq = np.array(trainseq)[binid]
#
# lsub = len(subseq)
# sigmd = pd.DataFrame(index=range(lsub), columns=["Sig" + str(i+1) for i in range(120)])
# for i in range(lsub):
#     sigmd.loc[i, :] = signatory.signature(torch.tensor([subseq[i][0:10]]), 4)
#     subtrainlabel[i] -= 8
#
# logis = LogisticRegression(penalty='l1', solver='liblinear',max_iter = 2000)
# logis.fit(sigmd.to_numpy(), subtrainlabel)
#
# def simple_plot(id, beta0 = logis.coef_[0], C0 = np.eye(120,120)):
#     l = len(subseq[id])
#     xdig = [k[0] for k in subseq[id]]
#     ydig = [k[1] for k in subseq[id]]
#     plt.figure(figsize=(8, 6), dpi=80)
#     plt.ion()
#
#     for index in range(l-9):
#         xt = signatory.signature(torch.tensor([subseq[id][0:(10+index)]]), 4)[0].numpy()
#         logit = np.dot(beta0, xt)
#         prob = np.exp(logit)/(1+np.exp(logit))
#         plt.cla()
#         plt.title("Prediction probability: (%f, %f)" %(prob, 1- prob))
#
#         plt.xlim(-1, 1)
#         plt.ylim(-1.0, 1.0)
#         plt.plot(xdig[:(index+10)], ydig[:(index+10)], '*')
#         plt.pause(0.5)
#
#         dl = (subtrainlabel[id] - prob)* xt
#         Rt = C0/0.99
#         xprod = np.matmul(np.reshape(xt,(120,1)), np.reshape(xt,(1,120)))
#         ddl = np.linalg.inv(Rt) + prob*(1- prob)*xprod
#         C0 = np.linalg.inv(-ddl)
#         beta0 += np.dot(C0, dl)
#
#     plt.ioff()
#     plt.show()
#     return

# logsigmd = pd.DataFrame(index=range(l), columns=["logSig" + str(i+1) for i in range(32)])
# for i in range(l):
#     logsigmd.loc[i, :] = signatory.logsignature(torch.tensor([trainseq[i][0:10]]), 4)
# logis = LogisticRegression(penalty='l1', solver='liblinear',max_iter = 2000)
# logis.fit(logsigmd.to_numpy(), trainlabel)
#
# def simple_plot(id, beta0 = logis.coef_, C0 = [np.eye(32,32) for i in range(10)]):
#     path = trainseq[id]
#     xdig = [k[0] for k in path]
#     ydig = [k[1] for k in path]
#     plt.figure(figsize=(8, 6), dpi=80)
#     plt.ion()
#
#     for index in range(len(path)-9):
#         xt = signatory.logsignature(torch.tensor([path[0:(10+index)]]), 4)[0].numpy()
#         xprod = np.matmul(np.reshape(xt, (32, 1)), np.reshape(xt, (1, 32)))
#         logit = np.exp(np.dot(beta0, xt))
#         prob = logit/sum(logit)
#         number = np.argsort(prob)[::-1]
#
#         plt.cla()
#         plt.title("The most possible numbers are: (%d, %d, %d)"
#                   %(number[0], number[1], number[2]))
#         plt.xlim(-1, 1)
#         plt.ylim(-1.0, 1.0)
#         plt.plot(xdig[:(index+10)], ydig[:(index+10)], '*')
#         plt.pause(0.5)
#
#         for k in range(10):
#             yt = 1 if k in number[:3] else 0
#             dl = (yt - prob[k])* xt
#             Rt = C0[k]/0.99
#             ddl = np.linalg.inv(Rt) + prob[k]*(1- prob[k])*xprod
#             C0[k] = np.linalg.inv(-ddl)
#             beta0[k] += np.dot(C0[k], dl)
#         plt.savefig('fig_%d.jpg' %index)
#     plt.ioff()
#     plt.show()
#     return
#
#
# import imageio
# import os
#
# def generate_gif(image_paths, gif_path, duration=0.35):
#     frames = []
#     for image_path in image_paths:
#         frames.append(imageio.imread(image_path))
#     imageio.mimsave(gif_path, frames, 'GIF', duration=duration)

# image_folder = "images"
# image_paths = []
# files = os.listdir(image_folder)
# for file in files:
#     image_path = os.path.join(image_folder, file)
#     image_paths.append(image_path)

# image_paths = []
# for i in range(36):
#     image_paths.append('images/fig_%d.jpg' %i)
# gif_path = "images/sample.gif"
# duration = 0.5
# generate_gif(image_paths, gif_path, duration)

l = len(trainseq)
for i in range(l):
    penlen = len(trainseq[i])
    # trainseq[i][0].append(0)
    # for j in range(1,penlen):
    #     if math.sqrt((trainseq[i][j][0] - trainseq[i][j-1][0])**2+\
    #                  (trainseq[i][j][1] - trainseq[i][j-1][1])**2) < 30:
    #         trainseq[i][j].append(trainseq[i][j-1][2] + 0.01)
    #     else:
    #         trainseq[i][j].append(trainseq[i][j-1][2])
    tmpx = [k[0] for k in trainseq[i]]
    tmpy = [k[1] for k in trainseq[i]]
    minx = min(tmpx)
    maxx = max(tmpx)
    miny = min(tmpy)
    maxy = max(tmpy)
    scale = float(2) / max([(maxx - minx), (maxy - miny)])
    list_x = [(x - (maxx + minx) / 2) * scale for x in tmpx]
    list_y = [(y - (maxy + miny) / 2) * scale for y in tmpy]
    for j in range(penlen):
        trainseq[i][j][0] = list_x[j]
        trainseq[i][j][1] = list_y[j]
    if penlen > 50:
        dropid = random.sample(range(penlen), penlen-50)
        trainseq[i] = np.delete(trainseq[i], dropid, 0).tolist()
    else:
        trainseq[i] += [[0.0, 0.0] for k in range(50 - penlen)]

sigtrain = torch.tensor([])

for i in range(l):
    sig1 = signatory.logsignature(torch.tensor([trainseq[i][0:17]]), 4)
    sig1 = torch.cat((torch.tensor([trainseq[i][0][:2]]), sig1), 1)
    sig2 = signatory.logsignature(torch.tensor([trainseq[i][17:34]]), 4)
    sig2 = torch.cat((torch.tensor([trainseq[i][17][:2]]), sig2), 1)
    sig3 = signatory.logsignature(torch.tensor([trainseq[i][34:50]]), 4)
    sig3 = torch.cat((torch.tensor([trainseq[i][34][:2]]), sig3), 1)

    sigpath = torch.cat((sig1, sig2, sig3),0)
    sigtrain = torch.cat((sigtrain, sigpath.unsqueeze(0)),0)

testseq, testlabel = GetPenDigit('pendigits-orig.tes.txt')
testl = len(testseq)

for i in range(testl):
    penlen = len(testseq[i])
    testseq[i][0].append(0)
    for j in range(1,len(testseq[i])):
        if math.sqrt((testseq[i][j][0] - testseq[i][j-1][0])**2+\
                     (testseq[i][j][1] - testseq[i][j-1][1])**2) < 30:
            testseq[i][j].append(testseq[i][j-1][2] + 0.01)
        else:
            testseq[i][j].append(testseq[i][j-1][2])
    tmpx = [k[0] for k in testseq[i]]
    tmpy = [k[1] for k in testseq[i]]
    minx = min(tmpx)
    maxx = max(tmpx)
    miny = min(tmpy)
    maxy = max(tmpy)
    scale = float(2) / max([(maxx - minx), (maxy - miny)])
    list_x = [(x - (maxx + minx) / 2) * scale for x in tmpx]
    list_y = [(y - (maxy + miny) / 2) * scale for y in tmpy]

    for j in range(penlen):
        testseq[i][j][0] = list_x[j]
        testseq[i][j][1] = list_y[j]
    if penlen > 50:
        dropid = random.sample(range(penlen), penlen-50)
        testseq[i] = np.delete(testseq[i], dropid, 0).tolist()
    else:
        testseq[i] += [[0.0, 0.0, 0.0] for k in range(50 - penlen)]


sigtest = torch.tensor([])

for i in range(len(testseq)):
    sig1 = signatory.logsignature(torch.tensor([testseq[i][0:17]]), 4)
    sig1 = torch.cat((torch.tensor([testseq[i][0][:2]]), sig1), 1)
    sig2 = signatory.logsignature(torch.tensor([testseq[i][17:34]]), 4)
    sig2 = torch.cat((torch.tensor([testseq[i][17][:2]]), sig2), 1)
    sig3 = signatory.logsignature(torch.tensor([testseq[i][34:50]]), 4)
    sig3 = torch.cat((torch.tensor([testseq[i][34][:2]]), sig3), 1)

    sigpath = torch.cat((sig1, sig2, sig3), 0)
    sigtest = torch.cat((sigtest, sigpath.unsqueeze(0)), 0)


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


def train(train_loader, learn_rate = 0.001, n_layers = 2, hidden_dim=256, EPOCHS=5, model_type="GRU"):
    # Setting common hyperparameters
    input_dim = next(iter(train_loader))[0].shape[2]
    output_dim = 10

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
        start_time = time.clock()
        h = model.init_hidden(train_loader.batch_size)
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
        current_time = time.clock()
        print("Epoch {}/{} Done, Total Loss: {}".format(epoch, EPOCHS, avg_loss / len(train_loader)))
        print("Total Time Elapsed: {} seconds".format(str(current_time - start_time)))
        epoch_times.append(current_time - start_time)
    traintime = sum(epoch_times)
    print("Total Training Time: {} seconds".format(str(traintime)))
    return model, traintime

# def evaluate(model, test_x, test_y, label_scalers):
#     model.eval()
#     outputs = []
#     targets = []
#     start_time = time.clock()
#     for i in test_x.keys():
#         inp = torch.from_numpy(np.array(test_x[i]))
#         labs = torch.from_numpy(np.array(test_y[i]))
#         h = model.init_hidden(inp.shape[0])
#         out, h = model(inp.to(device).float(), h)
#         outputs.append(label_scalers[i].inverse_transform(out.cpu().detach().numpy()).reshape(-1))
#         targets.append(label_scalers[i].inverse_transform(labs.numpy()).reshape(-1))
#     print("Evaluation Time: {}".format(str(time.clock()-start_time)))
#     sMAPE = 0
#     for i in range(len(outputs)):
#         sMAPE += np.mean(abs(outputs[i]-targets[i])/(targets[i]+outputs[i])/2)/len(outputs)
#     print("sMAPE: {}%".format(sMAPE*100))
#     return outputs, targets, sMAPE

def evaluate(model, testset, testlabel):
    model.eval()
    start_time = time.clock()

    h = model.init_hidden(testset.shape[0])
    out, h = model(testset.to(device).float(), h)
    _, predlabel = torch.max(torch.log_softmax(out.cpu().detach(), dim = 1), dim = 1)
    print("Evaluation Time: {}".format(str(time.clock() - start_time)))

    predrate = sum(predlabel.numpy() == testlabel)/len(testlabel) * 100
    print("prediction accuracy rate: {}%".format(predrate))
    return predrate

is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
lr = 0.0005

nBatch = 128
trainlabel = np.eye(10, dtype='uint8')[trainlabel]
train_data1 = TensorDataset(sigtrain, torch.from_numpy(trainlabel))
train_data2 = TensorDataset(torch.tensor(trainseq), torch.from_numpy(trainlabel))
text_file = open("Output.txt", "w")


train_loader = DataLoader(train_data2, shuffle=True, batch_size=nBatch, drop_last=True)
gru_model, traintime = train(train_loader, lr, 2, 1024, 10, model_type="GRU")
predrate = evaluate(gru_model, testseq, testlabel)
text_file.write("TSGRU Total Training Time: {} seconds\n".format(str(traintime)))
text_file.write("prediction accuracy rate: {}%\n".format(predrate))

train_loader = DataLoader(train_data1, shuffle=True, batch_size=nBatch, drop_last=True)
gru_model, traintime = train(train_loader, lr, 2, 1024, 10, model_type="GRU")
predrate = evaluate(gru_model, sigtest, testlabel)
text_file.write("SigGRU Total Training Time: {} seconds\n".format(str(traintime)))
text_file.write("prediction accuracy rate: {}%\n".format(predrate))
text_file.close()

