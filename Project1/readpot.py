import os
import os.path as osp
import numpy as np
import cv2
from PIL import Image
import struct
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

dataset_name = 'Pot11'
root = './'
train_dir = osp.join(root, dataset_name + 'Train')
test_dir = osp.join(root, dataset_name + 'Test')
train_dataset = os.listdir(train_dir)
test_dataset = os.listdir(test_dir)


def coordinate(pts):
    path = []
    pt_length = len(pts)
    nstroke = 1

    for i in range(pt_length):
        if pts[i][0] == -1 and pts[i][1] == 0:
            nstroke += 1
            continue
        else:
            path.append(np.append(pts[i], [nstroke]).tolist())

    tmpx = [k[0] for k in path]
    tmpy = [-k[1] for k in path]
    minx = min(tmpx)
    maxx = max(tmpx)
    miny = min(tmpy)
    maxy = max(tmpy)
    scale = float(2) / max([(maxx - minx), (maxy - miny)])
    list_x = [(x - (maxx + minx) / 2) * scale for x in tmpx]
    list_y = [(y - (maxy + miny) / 2) * scale for y in tmpy]

    for i in range(len(path)):
        path[i][0] = list_x[i]
        path[i][1] = list_y[i]
    return path


def read_from_pot_dir(pot_dir):
    def one_file(f):
        while True:
            # 文件头，交代了该sample所占的字节数以及label以及笔画数
            header = np.fromfile(f, dtype='uint8', count=8)
            if not header.size: break
            sample_size = header[0] + (header[1] << 8)
            tagcode = header[2] + (header[3] << 8) + (header[4] << 16) + (header[5] << 24)
            stroke_num = header[6] + (header[7] << 8)

            # 以下是参考官方POTView的C++源码View部分的Python解析代码
            traj = []
            for i in range(stroke_num):
                while True:
                    header = np.fromfile(f, dtype='int16', count=2)
                    x, y = header[0], header[1]
                    traj.append([x, y])

                    if x == -1 and y == 0:
                        break

            header = np.fromfile(f, dtype='int16', count=2)

            # 根据得到的采样点重构出样本
            pts = np.array(traj)
            path = coordinate(pts)
            yield path, tagcode

    for file_name in os.listdir(pot_dir):
        if file_name.endswith('.pot'):
            file_path = os.path.join(pot_dir, file_name)
            with open(file_path, 'rb') as f:
                for path, tagcode in one_file(f):
                    yield path, tagcode

# n = 0
# for image, tagcode in tqdm(read_from_pot_dir(pot_dir=test_dir)):
#     n += 1
#     if n >500:
#         path = image; code = tagcode
#         break
# totalpt = len(path); id = 0
# istroke = 1; x = []; y = []
# while id < totalpt:
#     if path[id][-1] == istroke:
#         x.append(path[id][0])
#         y.append(path[id][1])
#         id += 1
#     else:
#         plt.plot(x,y)
#         x = []
#         y = []
#         istroke += 1
#         continue
# plt.plot(x,y)
# struct.pack('>H', code).decode('gb18030')


# 解析字母表
char_set = set()
for _, tagcode in tqdm(read_from_pot_dir(pot_dir=test_dir)):
    tagcode_unicode = struct.pack('>H', tagcode).decode('gb18030')
    char_set.add(tagcode_unicode)
char_list = list(char_set)
char_dict = dict(zip(sorted(char_list), range(len(char_list))))
chinese_dict = {key:val for key, val in char_dict.items() if val>168 and val<3924}
alphabet_length = len(chinese_dict)

alphabet_path = osp.join(root, 'alphabet_' + str(alphabet_length))
with open(alphabet_path, 'wb') as f:
    pickle.dump(char_dict, f)

print('alphabet length: ', alphabet_length)



# 输出到文件夹
train_counter = 0
test_counter = 0

train_parse_dir = osp.join(root, dataset_name + 'TrainPath/')
if not os.path.exists(train_parse_dir):
    os.mkdir(train_parse_dir)
test_parse_dir = osp.join(root, dataset_name + 'TestPath/')
if not os.path.exists(test_parse_dir):
    os.mkdir(test_parse_dir)

for path, tagcode in tqdm(read_from_pot_dir(pot_dir=train_dir)):
    tagcode_unicode = struct.pack('>H', tagcode).decode('gb18030')
    chineseID = char_dict[tagcode_unicode]
    dir_name = train_parse_dir + '%0.5d' % chineseID
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    with open(dir_name + '/' + str(train_counter) + '.npy', 'wb') as f:
        np.save(f, path)
    train_counter += 1


for path, tagcode in tqdm(read_from_pot_dir(pot_dir=test_dir)):
    tagcode_unicode = struct.pack('>H', tagcode).decode('gb18030')
    chineseID = char_dict[tagcode_unicode]
    dir_name = test_parse_dir + '%0.5d' % chineseID
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    with open(dir_name + '/' + str(test_counter) + '.npy', 'wb') as f:
        np.save(f, path)
    test_counter += 1

