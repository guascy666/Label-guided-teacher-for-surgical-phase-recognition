import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.nn import DataParallel
from torch.utils.data import Sampler
from PIL import Image, ImageOps
import torchvision.transforms.functional as TF
import time
import pickle
import numpy as np
import argparse
import copy
import random
import numbers
from sklearn import metrics
from swin_transformer_v2 import SwinTransformerV2
import logging
from tqdm import tqdm
import os

parser = argparse.ArgumentParser(description='lstm training')
parser.add_argument('-mpath', default="./results/cholec80_20231120.1046/cholec80swinv2b_epoch_20_lr_0.0002_batch_120_train_9906_val_8619_test_8126.pth", type=str, help='swin model path')



args = parser.parse_args()

gpu_usg = True
sequence_length = 1
train_batch_size = 100
val_batch_size = 100
learning_rate = 0
num_gpu = torch.cuda.device_count()
use_gpu = (torch.cuda.is_available() and gpu_usg)
device = torch.device("cuda:0" if use_gpu else "cpu")
model_path=args.mpath
save_dir="/data2/guanjl/Swin_32_8_40/features_before_normal/40-40"
p=model_path
print(p)

folder=p.split('_')[-13].split('/')[-1]+"_"+p.split('_')[-12]+p.split('_')[-11]+'_train_'+p.split('_')[-5].split('.')[0]+'_val_'+p.split('_')[-3].split('.')[0]+'_test_'+p.split('_')[-1].split('.')[0]
print("folder:",folder)
if not os.path.exists(save_dir+"/"+folder):
    os.mkdir(save_dir+"/"+folder)




train_save_path="g_LFB_swin_train0"
val_save_path="g_LFB_swin_val0"
test_save_path="g_LFB_swin_test0"

print('number of gpu   : {:6d}'.format(num_gpu))
print('sequence length : {:6d}'.format(sequence_length))
print('train batch size: {:6d}'.format(train_batch_size))
print('valid batch size: {:6d}'.format(val_batch_size))
print('learning rate   : {:.4f}'.format(learning_rate))
print('swin_model_path : ',str(model_path))

#
def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

#
class RandomCrop(object):

    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.count = 0

    def __call__(self, img):

        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)

        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img

        random.seed(self.count // sequence_length)
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        # print(self.count, x1, y1)
        self.count += 1
        return img.crop((x1, y1, x1 + tw, y1 + th))

#
class RandomHorizontalFlip(object):
    def __init__(self):
        self.count = 0

    def __call__(self, img):
        seed = self.count // sequence_length
        random.seed(seed)
        prob = random.random()
        self.count += 1
        # print(self.count, seed, prob)
        if prob < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

#
class RandomRotation(object):
    def __init__(self, degrees):
        self.degrees = degrees
        self.count = 0

    def __call__(self, img):
        seed = self.count // sequence_length
        random.seed(seed)
        self.count += 1
        angle = random.randint(-self.degrees, self.degrees)
        return TF.rotate(img, angle)

#
class ColorJitter(object):
    def __init__(self, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.count = 0

    def __call__(self, img):
        seed = self.count // sequence_length
        random.seed(seed)
        self.count += 1
        brightness_factor = random.uniform(1 - self.brightness, 1 + self.brightness)
        contrast_factor = random.uniform(1 - self.contrast, 1 + self.contrast)
        saturation_factor = random.uniform(1 - self.saturation, 1 + self.saturation)
        hue_factor = random.uniform(- self.hue, self.hue)

        img_ = TF.adjust_brightness(img, brightness_factor)
        img_ = TF.adjust_contrast(img_, contrast_factor)
        img_ = TF.adjust_saturation(img_, saturation_factor)
        img_ = TF.adjust_hue(img_, hue_factor)

        return img_


class CholecDataset(Dataset):
    def __init__(self, file_paths, file_labels, transform=None,
                 loader=pil_loader):
        self.file_paths = file_paths
        self.file_labels_phase = file_labels[:, -1]
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        img_names = self.file_paths[index]
        labels_phase = self.file_labels_phase[index]
        imgs = self.loader(img_names)
        if self.transform is not None:
            imgs = self.transform(imgs)

        return imgs, labels_phase, index

    def __len__(self):
        return len(self.file_paths)

swin_trained_path = "/data2/guanjl/Swin_32_8_40/swinv2_base_patch4_window12_192_22k.pth"

class SwinV2(torch.nn.Module):
    def __init__(self):
        super(SwinV2, self).__init__()
        
        self.swin = SwinTransformerV2()

        if swin_trained_path is not None:
            state = torch.load(swin_trained_path)
            newdict = {}
            for k, v in state['model'].items():
                if k in self.swin.state_dict().keys():
                    newdict[k] = v
            self.swin.load_state_dict(newdict)
            
        self.fc = nn.Sequential(nn.Linear(1024, 512),
                                nn.ReLU(),
                                nn.Linear(512, 7))
        self.fc_tool = nn.Sequential(nn.Linear(1024, 512),
                                    nn.ReLU(),
                                    nn.Linear(512, 7))

    def forward(self, x):
        x = x.view(-1, 3, 192, 192)
        x = self.swin.forward(x)
        x = x.view(-1, 1024)
        return x


#
def get_useful_start_idx(sequence_length, list_each_length):
    count = 0
    idx = []
    for i in range(len(list_each_length)):
        for j in range(count, count + (list_each_length[i] + 1 - sequence_length)):
            idx.append(j)
        count += list_each_length[i]
    return idx


def get_data(data_path):
    
    with open(data_path, 'rb') as f:
        train_test_paths_labels = pickle.load(f)
    
    train_paths = train_test_paths_labels[0]
    test_paths = train_test_paths_labels[1]

    train_labels = train_test_paths_labels[2]
    test_labels = train_test_paths_labels[3]

    train_num_each = train_test_paths_labels[4]
    test_num_each = train_test_paths_labels[5]

    print('train_paths  : {:6d}'.format(len(train_paths)))
    print('train_labels : {:6d}'.format(len(train_labels)))
    print('test_paths   : {:6d}'.format(len(test_paths)))
    print('test_labels  : {:6d}'.format(len(test_labels)))

    train_labels = np.asarray(train_labels, dtype=np.int64)
    test_labels = np.asarray(test_labels, dtype=np.int64)


    train_transforms = transforms.Compose([
        transforms.Resize((200, 200)),
        transforms.RandomCrop(192),
        ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        RandomHorizontalFlip(),
        RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize([0.41757566, 0.26098573, 0.25888634], [0.21938758, 0.1983, 0.19342837])
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((200, 200)),
        transforms.CenterCrop(192),
        transforms.ToTensor(),
        transforms.Normalize([0.41757566, 0.26098573, 0.25888634], [0.21938758, 0.1983, 0.19342837])
    ])


    train_dataset = CholecDataset(train_paths, train_labels, train_transforms)
    test_dataset = CholecDataset(test_paths, test_labels, test_transforms)

    return train_dataset, train_num_each,test_dataset, test_num_each



# 序列采样sampler
class SeqSampler(Sampler):
    def __init__(self, data_source, idx):
        super().__init__(data_source)
        self.data_source = data_source
        self.idx = idx

    def __iter__(self):
        return iter(self.idx)

    def __len__(self):
        return len(self.idx)



# Long Term Feature bank

g_LFB_train = np.zeros(shape=(0, 1024))
g_LFB_test = np.zeros(shape=(0, 1024))


def train_model(train_dataset, train_num_each, test_dataset,test_num_each):
    num_train = len(train_dataset)
    num_test=len(test_dataset)


    train_useful_start_idx = get_useful_start_idx(sequence_length, train_num_each)
    test_useful_start_idx=get_useful_start_idx(sequence_length,test_num_each)

    num_train_we_use = len(train_useful_start_idx)
    num_test_we_use = len(test_useful_start_idx)

    train_we_use_start_idx = train_useful_start_idx[0:num_train_we_use]
    test_we_use_start_idx = test_useful_start_idx[0:num_test_we_use]

    train_idx = []
    for i in range(num_train_we_use):
        for j in range(sequence_length):
            train_idx.append(train_we_use_start_idx[i] + j)
    
            
    test_idx = []
    for i in range(num_test_we_use):
        for j in range(sequence_length):
            test_idx.append(test_we_use_start_idx[i] + j)
            
    num_train_all = len(train_idx)
    num_test_all =len(test_idx)

    print('num train start idx : {:6d}'.format(len(train_useful_start_idx)))
    print('last idx train start: {:6d}'.format(train_useful_start_idx[-1]))
    print('num of train dataset: {:6d}'.format(num_train))
    print('num of train we use : {:6d}'.format(num_train_we_use))
    print('num of all train use: {:6d}'.format(num_train_all))

    print('num test start idx : {:6d}'.format(len(test_useful_start_idx)))
    print('last idx test start: {:6d}'.format(test_useful_start_idx[-1]))
    print('num of test dataset: {:6d}'.format(num_test))
    print('num of test we use : {:6d}'.format(num_test_we_use))
    print('num of all test use: {:6d}'.format(num_test_all))


    global g_LFB_train
    global g_LFB_test
    print("loading features!>.........")

    train_feature_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        sampler=SeqSampler(train_dataset, train_idx),
        num_workers=8,
        pin_memory=False
    )

    test_feature_loader = DataLoader(
        test_dataset,
        batch_size=val_batch_size,
        sampler=SeqSampler(test_dataset, test_idx),
        num_workers=8,
        pin_memory=False
    )

    model_LFB = SwinV2()

    model_LFB.load_state_dict(torch.load(model_path))
    
    def get_parameter_number(net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print('Total: {}, Trainable: {}'.format(total_num, trainable_num))
        return trainable_num

    total_papa_num = 0
    total_papa_num += get_parameter_number(model_LFB)

    if use_gpu:
        model_LFB = DataParallel(model_LFB)
        model_LFB.to(device)

    for params in model_LFB.parameters():
        params.requires_grad = False

    model_LFB.eval()

    with torch.no_grad():

        for data in train_feature_loader:
            if use_gpu:
                inputs, labels_phase = data[0].to(device), data[1].to(device)
            else:
                inputs, labels_phase = data[0], data[1]

            inputs = inputs.view(-1, sequence_length, 3, 192, 192)
            outputs_feature = model_LFB.forward(inputs).data.cpu().numpy()

            g_LFB_train = np.concatenate((g_LFB_train, outputs_feature), axis=0)
            
            print("train feature length:", len(g_LFB_train))
        
        

        for data in test_feature_loader:
            if use_gpu:
                inputs, labels_phase = data[0].to(device), data[1].to(device)
            else:
                inputs, labels_phase = data[0], data[1]

            inputs = inputs.view(-1, sequence_length, 3, 192, 192)
            outputs_feature = model_LFB.forward(inputs).data.cpu().numpy()

            g_LFB_test = np.concatenate((g_LFB_test, outputs_feature), axis=0)

            print("test feature length:", len(g_LFB_test))

        print("finish!")
        g_LFB_train = np.array(g_LFB_train)
        g_LFB_test = np.array(g_LFB_test)
        print("train:",g_LFB_train.shape)
        print("test:", g_LFB_test.shape)

        #'''
        with open(save_dir+"/"+folder+"/"+train_save_path+".pkl", 'wb') as f:
            pickle.dump(g_LFB_train, f)
        

        with open(save_dir+"/"+folder+"/"+test_save_path+".pkl", 'wb') as f:
            pickle.dump(g_LFB_test, f)


def main():
    train_dataset, train_num_each, test_dataset, test_num_each = get_data("./train_val_test_paths_labels_40_40.pkl")
    train_model(train_dataset, train_num_each, test_dataset, test_num_each)


if __name__ == "__main__":
    main()

print('Done')
print()