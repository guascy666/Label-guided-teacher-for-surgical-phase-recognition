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

task_name = 'cholec80'

parser = argparse.ArgumentParser(description='feature extraction with SwinBV2')
parser.add_argument('-g', '--gpu', default=True, type=bool, help='gpu use, default True')
parser.add_argument('-s', '--seq', default=1, type=int, help='sequence length, default 10')
parser.add_argument('-t', '--train', default=120, type=int, help='train batch size, default 400')
parser.add_argument('-v', '--val', default=120, type=int, help='valid batch size, default 10')
parser.add_argument('-e', '--epo', default=50, type=int, help='epochs to train and val, default 25')
parser.add_argument('-l', '--lr', default=2e-4, type=float, help='learning rate for optimizer, default 5e-5')

def get_log(file_name):
    logger = logging.getLogger('*')
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fh = logging.FileHandler(file_name, mode='a')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

args = parser.parse_args()

gpu_usg = args.gpu
sequence_length = args.seq
train_batch_size = args.train
val_batch_size = args.val
epochs = args.epo
learning_rate = args.lr
num_gpu = torch.cuda.device_count()
use_gpu = (torch.cuda.is_available() and gpu_usg)
device = torch.device("cuda:0" if use_gpu else "cpu")
save_epoch=10

print('number of gpu   : {:6d}'.format(num_gpu))
print('sequence length : {:6d}'.format(sequence_length))
print('train batch size: {:6d}'.format(train_batch_size))
print('valid batch size: {:6d}'.format(val_batch_size))
print('num of epochs   : {:6d}'.format(epochs))
print('learning rate   :',str(learning_rate))
print('save epoch   : {:6d}'.format(save_epoch))

seed = 1
print("Random Seed: ", seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


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
    def __init__(self, file_paths, file_labels, transform=None, loader=pil_loader):
        self.file_paths = file_paths
        self.file_labels_1 = file_labels[:, range(7)]
        self.file_labels_2 = file_labels[:, -1]
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        img_names = self.file_paths[index]
        labels_1 = self.file_labels_1[index]
        labels_2 = self.file_labels_2[index]
        imgs = self.loader(img_names)
        if self.transform is not None:
            imgs = self.transform(imgs)
        return imgs, labels_1, labels_2

    def __len__(self):
        return len(self.file_paths)


swin_trained_path = "./swinv2_base_patch4_window12_192_22k.pth"

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
        phase = self.fc(x)
        tool = self.fc_tool(x)
        return tool, phase


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
    val_paths = train_test_paths_labels[1]
    test_paths = train_test_paths_labels[2]

    train_labels = train_test_paths_labels[3]
    val_labels=train_test_paths_labels[4]
    test_labels = train_test_paths_labels[5]

    train_num_each = train_test_paths_labels[6]
    val_num_each=train_test_paths_labels[7]
    test_num_each = train_test_paths_labels[8]

    print('train_paths  : {:6d}'.format(len(train_paths)))
    print('train_labels : {:6d}'.format(len(train_labels)))
    print('val_paths  : {:6d}'.format(len(val_paths)))
    print('val_labels : {:6d}'.format(len(val_labels)))
    print('test_paths   : {:6d}'.format(len(test_paths)))
    print('test_labels  : {:6d}'.format(len(test_labels)))

    train_labels = np.asarray(train_labels, dtype=np.int64)
    val_labels = np.asarray(val_labels, dtype=np.int64)
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
    val_dataset = CholecDataset(val_paths, val_labels, test_transforms)
    test_dataset = CholecDataset(test_paths, test_labels, test_transforms)

    return train_dataset, train_num_each,val_dataset,val_num_each, test_dataset, test_num_each


class SeqSampler(Sampler):
    def __init__(self, data_source, idx):
        super().__init__(data_source)
        self.data_source = data_source
        self.idx = idx

    def __iter__(self):
        return iter(self.idx)

    def __len__(self):
        return len(self.idx)


def train_model(train_dataset, train_num_each, val_dataset,val_num_each,test_dataset,test_num_each):
    
    time_cur = time.strftime("%Y%m%d.%H%M", time.localtime(time.time()))
    #logger = get_log('log/' + task_name + '_' + str(time_cur) + '.txt')

    save_folder_name = "./results/" + task_name + '_' + str(time_cur)
    import os
    # if not os.path.exists(save_folder_name):
    #     os.mkdir(save_folder_name)
            

    num_train = len(train_dataset)
    num_val = len(val_dataset)
    num_test=len(test_dataset)

    train_useful_start_idx = get_useful_start_idx(sequence_length, train_num_each)
    val_useful_start_idx = get_useful_start_idx(sequence_length,val_num_each)
    test_useful_start_idx=get_useful_start_idx(sequence_length,test_num_each)

    num_train_we_use = len(train_useful_start_idx)
    num_val_we_use = len(val_useful_start_idx)
    num_test_we_use = len(test_useful_start_idx)

    train_we_use_start_idx = train_useful_start_idx[0:num_train_we_use]
    val_we_use_start_idx = val_useful_start_idx[0:num_val_we_use]
    test_we_use_start_idx = test_useful_start_idx[0:num_test_we_use]

    train_idx = []
    for i in range(num_train_we_use):
        for j in range(sequence_length):
            train_idx.append(train_we_use_start_idx[i] + j)
    
    val_idx = []
    for i in range(num_val_we_use):
        for j in range(sequence_length):
            val_idx.append(val_we_use_start_idx[i] + j)
            
    test_idx = []
    for i in range(num_test_we_use):
        for j in range(sequence_length):
            test_idx.append(test_we_use_start_idx[i] + j)
            
    num_train_all = len(train_idx)
    num_val_all = len(val_idx)
    num_test_all =len(test_idx)

    print('num train start idx : {:6d}'.format(len(train_useful_start_idx)))
    print('last idx train start: {:6d}'.format(train_useful_start_idx[-1]))
    print('num of train dataset: {:6d}'.format(num_train))
    print('num of train we use : {:6d}'.format(num_train_we_use))
    print('num of all train use: {:6d}'.format(num_train_all))

    print('num val start idx : {:6d}'.format(len(val_useful_start_idx)))
    print('last idx val start: {:6d}'.format(val_useful_start_idx[-1]))
    print('num of val dataset: {:6d}'.format(num_val))
    print('num of val we use : {:6d}'.format(num_val_we_use))
    print('num of all val use: {:6d}'.format(num_val_all))

    print('num test start idx : {:6d}'.format(len(test_useful_start_idx)))
    print('last idx test start: {:6d}'.format(test_useful_start_idx[-1]))
    print('num of test dataset: {:6d}'.format(num_test))
    print('num of test we use : {:6d}'.format(num_test_we_use))
    print('num of all test use: {:6d}'.format(num_test_all))

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        sampler=SeqSampler(train_dataset, train_idx),
        num_workers=8,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        sampler=SeqSampler(val_dataset, val_idx),
        num_workers=8,
        pin_memory=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=val_batch_size,
        sampler=SeqSampler(test_dataset, test_idx),
        num_workers=8,
        pin_memory=False
    )

    model = SwinV2()
    if use_gpu:
        model = DataParallel(model)
        model.to(device)
    
    weights_train = np.asarray([1.6411019141231247,
                                0.19090963801041133,
                                1.0,
                                0.2502662616859295,
                                1.9176363911137977,
                                0.9840248158200853,
                                2.174635818337618, ])
    
    criterion_2 = nn.CrossEntropyLoss(weight=torch.from_numpy(weights_train).float().to(device), size_average=False)
    criterion_1 = nn.BCEWithLogitsLoss(size_average=False)

    optimizer = torch.optim.AdamW([
        {'params': model.module.swin.parameters()},
        {'params': model.module.fc.parameters(), 'lr': learning_rate},
        {'params': model.module.fc_tool.parameters(), 'lr': learning_rate},
    ], lr=learning_rate / 10, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-3)
    
    best_model_wts = copy.deepcopy(model.module.state_dict())
    best_val_accuracy_2 = 0.0  # judge by accu2
    correspond_train_acc_2 = 0.0
    best_epoch=0

    

    for epoch in range(epochs):
        torch.cuda.empty_cache()
        np.random.shuffle(train_we_use_start_idx)
        train_idx = []
        for i in range(num_train_we_use):
            for j in range(sequence_length):
                train_idx.append(train_we_use_start_idx[i] + j)

        train_loader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            sampler=SeqSampler(train_dataset, train_idx),
            num_workers=8,
            pin_memory=False
        )

        model.train()
        
        train_loss_phase= 0.0
        train_loss_tool=0.0
        
        train_correct_phase=0
        train_start_time = time.time()
    
        for data in tqdm(train_loader):
            
            optimizer.zero_grad()
            
            inputs, labels_tool, labels_phase = data

            inputs = inputs.cuda()
            labels_tool = labels_tool.float()
            labels_tool = labels_tool.cuda()
            labels_phase = labels_phase.cuda()
            
            inputs=inputs.view(-1,sequence_length,3,192,192)
            
            outputs1,outputs2 = model.forward(inputs)

            _, preds = torch.max(outputs2.data, -1)
            #outputs=outputs.squeeze(1)

            #loss1=criterion_1(outputs1,labels_tool)
            loss2 = criterion_2(outputs2, labels_phase)
            
            loss=loss2
            loss.backward()
            optimizer.step()
            
            #train_loss_tool+=loss1.item()
            train_loss_phase+=loss2.item()
            train_correct_phase+= torch.sum(preds == labels_phase.data)


        train_elapsed_time = time.time() - train_start_time
        train_accuracy_2 = float(train_correct_phase) / float(num_train_all) * sequence_length
        train_average_loss_1 = train_loss_tool/ num_train_all / 7
        train_average_loss_2 = train_loss_phase / num_train_all

        if True:
            model.eval()
            with torch.no_grad():
                val_correct_phase = 0
                val_start_time = time.time()
                val_all_preds_phase = []
                val_all_labels_phase = []
            
                
                for data in tqdm(val_loader):
                    inputs, labels_tool, labels_phase = data
                    inputs = inputs.cuda()
                    labels_tool = labels_tool.float()
                    labels_tool = labels_tool.cuda()
                    labels_phase = labels_phase.cuda()
                    inputs = inputs.view(-1, sequence_length, 3, 192, 192)
                    outputs1, outputs2 = model.forward(inputs)
                    _, preds = torch.max(outputs2.data, -1)
                    # outputs=outputs.squeeze(1)
                    val_correct_phase += torch.sum(preds == labels_phase.data)

                    for i in range(len(preds)):
                        val_all_preds_phase.append(int(preds.data.cpu()[i]))
                    for i in range(len(labels_phase)):
                        val_all_labels_phase.append(int(labels_phase.data.cpu()[i]))


            val_elapsed_time = time.time() - val_start_time
            val_accuracy_2 = float(val_correct_phase) / float(num_val_all)

            val_recall_phase = metrics.recall_score(val_all_labels_phase,val_all_preds_phase, average='macro')
            val_precision_phase = metrics.precision_score(val_all_labels_phase, val_all_preds_phase, average='macro')
            val_jaccard_phase = metrics.jaccard_score(val_all_labels_phase, val_all_preds_phase, average='macro')
            
            model.eval()
        
            with torch.no_grad():
                
                test_correct_phase = 0
                test_start_time = time.time()
                test_all_preds_phase = []
                test_all_labels_phase = []
            
                
                for data in tqdm(test_loader):
                    inputs, labels_tool, labels_phase = data
                    inputs = inputs.cuda()
                    labels_tool = labels_tool.float()
                    labels_tool = labels_tool.cuda()
                    labels_phase = labels_phase.cuda()
                    inputs = inputs.view(-1, sequence_length, 3, 192, 192)
                    outputs1, outputs2 = model.forward(inputs)
                    _, preds = torch.max(outputs2.data, -1)
                    # outputs=outputs.squeeze(1)
                    test_correct_phase += torch.sum(preds == labels_phase.data)

                    for i in range(len(preds)):
                        test_all_preds_phase.append(int(preds.data.cpu()[i]))
                    for i in range(len(labels_phase)):
                        test_all_labels_phase.append(int(labels_phase.data.cpu()[i]))


            test_elapsed_time = time.time() - test_start_time
            test_accuracy_2 = float(test_correct_phase) / float(num_test_all)

            test_recall_phase = metrics.recall_score(test_all_labels_phase,test_all_preds_phase, average='macro')
            test_precision_phase = metrics.precision_score(test_all_labels_phase, test_all_preds_phase, average='macro')
            test_jaccard_phase = metrics.jaccard_score(test_all_labels_phase, test_all_preds_phase, average='macro')


            

            print('epoch: {:4d}'
                ' train loss_phase: {:4.4f}'
                ' train accu_phase: {:.4f}'
                ' val accu_phase: {:.4f}'
                ' test accu_phase: {:.4f}'
                .format(epoch,
                        train_average_loss_2,
                        train_accuracy_2,
                        val_accuracy_2,
                        test_accuracy_2))
            
            if val_accuracy_2 > best_val_accuracy_2:#test_accuracy is phase_acc
                best_epoch = epoch
                best_val_acc=val_accuracy_2
                best_val_accuracy_2 = val_accuracy_2
                correspond_train_acc_2 = train_accuracy_2
                best_model_wts = copy.deepcopy(model.module.state_dict())
            elif val_accuracy_2 == best_val_accuracy_2:
                if train_accuracy_2 > correspond_train_acc_2:
                    best_valt_acc=val_accuracy_2
                    best_epoch = epoch
                    correspond_train_acc_2 = train_accuracy_2
                    best_model_wts = copy.deepcopy(model.module.state_dict())
            print("best_epoch:", best_epoch)

            
            
            logger.info('Epoch: %d/%d (%d h %d m %d s)' % (epoch, epochs, int(train_elapsed_time/3600), int(np.mod(train_elapsed_time,3600)/60), int(np.mod(np.mod(train_elapsed_time,3600),60))) + time.strftime("%d.%m.-%H:%M:%S", time.localtime()))
            logger.info('validation time: %d h %d m %d s' % (int((test_elapsed_time+val_elapsed_time)/3600), int(np.mod(test_elapsed_time,3600)/60), int(np.mod(np.mod(test_elapsed_time,3600),60))) + time.strftime("%d.%m.-%H:%M:%S", time.localtime()))
            
            logger.info("train accuracy phase: %6f" % (train_accuracy_2))
            logger.info("train average loss phase: %6f" % (train_average_loss_2))
            logger.info("train average loss tool: %6f" % (train_average_loss_1))

            logger.info("val accuracy phase: %6f" % (val_accuracy_2))
            logger.info("val_recall_phase: %6f" % (val_recall_phase))
            logger.info("val_precision_phase: %6f" % (val_precision_phase))
            logger.info("val_jaccard_phase: %6f" % (val_jaccard_phase))

            logger.info("test accuracy phase: %6f" % (test_accuracy_2))
            logger.info("test_recall_phase: %6f" % (test_recall_phase))
            logger.info("test_precision_phase: %6f" % (test_precision_phase))
            logger.info("test_jaccard_phase: %6f" % (test_jaccard_phase))

            logger.info("best_epoch: %6f" % (best_epoch))

            logger.info('\n')
            
        
            base_name = task_name + "swinv2b" \
                        + "_epoch_" + str(epoch) \
                        + "_lr_" + str(learning_rate) \
                        + "_batch_" + str(train_batch_size) \
                        + "_train_" + str(int(train_accuracy_2*10000)) \
                        + "_val_" + str(int(val_accuracy_2*10000)) \
                        + "_test_" + str(int(test_accuracy_2*10000))
            
          
            now_model_wts=copy.deepcopy(model.module.state_dict())
            if True:
                torch.save(now_model_wts, save_folder_name +'/' + base_name + '.pth')
            
            if epoch==-1:
                torch.save(best_model_wts, save_folder_name +'/' + "best_model" + '.pth')


        
        

def main():
    train_dataset, train_num_each,val_dataset,val_num_each,test_dataset,test_num_each = get_data("./train_val_test_paths_labels.pkl")
    train_model(train_dataset, train_num_each,val_dataset,val_num_each,test_dataset,test_num_each)


if __name__ == "__main__":
    main()

print('Done')
print()