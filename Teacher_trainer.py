import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import pickle
import numpy as np
import copy
import random
import logging

from sklearn.metrics import average_precision_score
from sklearn import metrics
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import warnings
import os
import cv2
from model_all import MM_model,SupConLoss
warnings.filterwarnings("ignore")

import argparse
parser = argparse.ArgumentParser(description='lstm training')
parser.add_argument('-pd', default="", type=str, help='save pkl dir')
parser.add_argument('-bw', default=1000, type=int, help='bandwidth')
parser.add_argument('-dm', default=1024, type=int, help='dmodel')
parser.add_argument('-dr', default=4, type=int, help='down rate')
parser.add_argument('-ld', default=0.1, type=float, help='lamda for CL')
parser.add_argument('-act',default=1,type=int,help='whether use tanh 1 for true 0 for false')

args = parser.parse_args()
save_pkl_dir = args.pd
band_width = args.bw
d_model = args.dm
down_rate = args.dr
lamda = args.ld
active_tanh = bool(args.act)
data_split_pkl_path = './train_val_test_paths_labels_40_40.pkl'
print("file_dir_name: ", save_pkl_dir)
print("band_width:",band_width)
print("down_rate:",down_rate)
print("lamda for CL:",lamda)
print("whether use tanh:",active_tanh)

save_dir=save_pkl_dir.split('/')[-1].split('.')[0]

def plot_maps(plot_list, save_path):
    pred = np.array(plot_list)
    w1 = 400
    vis = np.zeros((w1, 8000))
    ori_pred = np.tile(np.array(pred+1).transpose(),(w1, 1)).transpose()
    interp_pred = cv2.resize(ori_pred, (ori_pred.shape[1], 8000), interpolation=cv2.INTER_NEAREST).transpose()
    vis[0:w1, :] = interp_pred

    cmap = ListedColormap(["darkorange", "gold", "lawngreen", "lightseagreen", '#7fc97f', '#ac25e2',
                           '#386cb0', '#f0027f'])

    plt.figure(dpi=300,figsize=(20,6))
    plt.xticks([])
    plt.yticks([])
    plt.imshow(vis,cmap=cmap)
    plt.savefig(save_path,bbox_inches='tight', pad_inches = -0.1)
    plt.clf()
    return


from sklearn.metrics import accuracy_score
def get_pr_re_acc_f1_onehot(one_hot_input, one_hot_label, input, label):
    precision, recall, f1_score, jaccard = [], [], [], []
    for l in range(one_hot_input.shape[1]):
        tmp_gt = one_hot_label[:, l]
        tmp_pred = one_hot_input[:, l]
        if (tmp_pred.sum() == 0) or (tmp_gt.sum() == 0):
            continue
        sum = (tmp_pred * tmp_gt).sum()
        pr = sum/tmp_pred.sum()
        re = sum/tmp_gt.sum()
        f1 = 2*(pr*re)/(pr+re)
        jc = sum/(tmp_pred.sum() + tmp_gt.sum()-sum)
        precision.append(pr)
        recall.append(re)
        f1_score.append(f1)
        jaccard.append(jc)
        
    precision, recall, f1_score, jaccard = np.mean(np.array(precision)), np.mean(np.array(recall)), np.mean(np.array(f1_score)),np.mean(np.array(jaccard))
    accuracy = accuracy_score(label, input)
    return accuracy, precision, recall, f1_score, jaccard


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

    train_start_vidx = []
    count = 0
    for i in range(len(train_num_each)):
        train_start_vidx.append(count)
        count += train_num_each[i]


    test_start_vidx = []
    count = 0
    for i in range(len(test_num_each)):
        test_start_vidx.append(count)
        count += test_num_each[i]
    
    print("get data finish!")

    return train_labels, train_num_each, train_start_vidx,\
        test_labels, test_num_each, test_start_vidx

def get_long_feature(start_index, lfb, LFB_length):
    long_feature = []
    long_feature_each = []
    # 上一个存在feature的index
    for k in range(LFB_length):
        LFB_index = (start_index + k)
        LFB_index = int(LFB_index)
        long_feature_each.append(lfb[LFB_index])
    long_feature.append(long_feature_each)
    return long_feature


train_labels_80, train_num_each_80, train_start_vidx,\
    test_labels_80, test_num_each_80, test_start_vidx = get_data(data_split_pkl_path)

train_pkl_path = save_pkl_dir+"/"+"g_LFB_swin_train0.pkl"
test_pkl_path = save_pkl_dir+"/"+"g_LFB_swin_test0.pkl"

task_name = save_pkl_dir.split('/')[-1].split('.')[0]
print("loading pkl name: ", task_name)

learning_rate=5e-5
weight_decay=1e-5
max_epoch=50

print('Learning rate:',learning_rate)
print('Weight decay:',weight_decay)
print('Max epoch:',max_epoch)

seed = 1
print("Random Seed: ", seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

""" load pkl """
with open(train_pkl_path, 'rb') as f:
    g_LFB_train = pickle.load(f)

with open(test_pkl_path, 'rb') as f:
    g_LFB_test = pickle.load(f)

print("load completed")
print("g_LFB_train shape:", g_LFB_train.shape)

print("g_LFB_test shape:", g_LFB_test.shape)

#####################################################################

model=MM_model(mask_length=band_width,dmodel=d_model,sample_rate=down_rate,Enc_active=active_tanh)
model=model.cuda()

###########################################################################

optimizer=optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=weight_decay)

#############################################################################
weights_train = np.asarray([1.6411019141231247,
            0.19090963801041133,
            1.0,
            0.2502662616859295,
            1.9176363911137977,
            0.9840248158200853,
            2.174635818337618,])

CELoss = nn.CrossEntropyLoss(weight=torch.from_numpy(weights_train).float().cuda())
cri_sup=SupConLoss()
cri_sup=cri_sup.cuda()

#############################################################################
train_we_use_start_idx_80 = [x for x in range(40)]
test_we_use_start_idx_80 = [x for x in range(40)]

#############################################################################
best_model_wts = copy.deepcopy(model.state_dict())
best_test_accuracy_phase = 0.0
best_val_accuracy_phase = 0.0
correspond_train_acc_phase=0.0
best_test_acc_phase_video=0
best_val_acc_phase_video=0
best_epoch=-1
best_train_loss=10000000000

if not os.path.exists('teacher_results'):
    os.mkdir('teacher_results')

time_cur = time.strftime("%Y%m%d.%H%M", time.localtime(time.time()))
results_folder = task_name

if not os.path.exists('teacher_results/'+save_dir):
    os.mkdir('teacher_results/'+save_dir)

small_dir='_dm' + str(d_model) + '_dr' + str(down_rate) + '_bw' + str(band_width) + \
                    '_ld'+str(lamda) +'_act_'+str(active_tanh) + '_lr' + str(learning_rate)

if not os.path.exists('teacher_results/'+save_dir+'/'+small_dir):
    os.mkdir('teacher_results/'+save_dir+'/'+small_dir)

logger = get_log('teacher_results/'+save_dir+'/'+small_dir + '/log' + \
                   '_dm' + str(d_model) + '_dr' + str(down_rate) + '_bw' + str(band_width) + \
                    '_ld'+str(lamda) +'_act_'+str(active_tanh) + '_lr' + str(learning_rate) + '.txt')

plot_path = 'teacher_results/'+save_dir+'/'+small_dir+ '/plot' +'_dm' + str(d_model) + \
       '_dr' + str(down_rate) + '_bw' + str(band_width)+'_ld'+str(lamda) +'_act_'+str(active_tanh) + \
           '_lr' + str(learning_rate) + '/'

if not os.path.exists(plot_path):
    os.mkdir(plot_path)

for epoch in range(max_epoch):

    torch.cuda.empty_cache()
    train_start_time = time.time()
    random.shuffle(train_we_use_start_idx_80)
    
    model.train()
    train_loss_phase=0.0
    train_correct_phase=0
    train_loss_CL=0.0

    print("train_working>>>>>>>>>>>>>>>>>>>>>>>>")

    for i in train_we_use_start_idx_80:
        torch.cuda.empty_cache()
        optimizer.zero_grad()
        
        labels_phase = []
        for j in range(train_start_vidx[i], train_start_vidx[i] + train_num_each_80[i]):
            labels_phase.append(train_labels_80[j][-1])
        labels_phase = torch.LongTensor(np.array(labels_phase))
        labels_phase = labels_phase.cuda()
        
        long_feature = get_long_feature(start_index=train_start_vidx[i], lfb=g_LFB_train, LFB_length=train_num_each_80[i])

        long_feature = (torch.Tensor(np.array(long_feature))).cuda()

        latent_feature,_,out_phase = model(long_feature,labels_phase)
        out_phase = out_phase.squeeze(0)
        loss1 = CELoss(out_phase, labels_phase)
        loss2 = cri_sup(latent_feature,labels_phase)
        
        loss=loss1+lamda*loss2
        loss.backward()
        optimizer.step()
        train_loss_phase += loss1.item()
        train_loss_CL +=loss2.item() 
        _, preds_phase = torch.max(out_phase.data, 1)
        train_correct_phase += torch.sum(preds_phase == labels_phase.data)

    train_accuracy_phase = float(train_correct_phase) / len(train_labels_80)
    train_each_loss = train_loss_phase/40
    train_latent_loss = train_loss_CL/40

    

    """
    Test
    """
    test_accuracy_phase_list = []
    test_recall_phase_list = []
    test_precision_phase_list = []
    test_jaccard_phase_list = []
    test_f1_phase_list = []
    test_accuracy_phase_total=0
    test_correct_total=0
    
    pred_list_all = []
    
    model.eval()
    print("test_working>>>>>>>>>>>>>>>>>>>>>>>>")
    with torch.no_grad():
        for i in test_we_use_start_idx_80:
            torch.cuda.empty_cache()
            labels_phase = []
            for j in range(test_start_vidx[i], test_start_vidx[i] + test_num_each_80[i]):
                labels_phase.append(test_labels_80[j][-1])
            labels_phase = torch.LongTensor(np.array(labels_phase))
            labels_phase = labels_phase.cuda()
            long_feature = get_long_feature(start_index=test_start_vidx[i], lfb=g_LFB_test, LFB_length=test_num_each_80[i])

            long_feature = (torch.Tensor(np.array(long_feature))).cuda()

            _,_,out_phase= model(long_feature,labels_phase)
            out_phase = out_phase.squeeze(0)
            _, preds_phase = torch.max(out_phase.data, 1)
            test_correct_total+=torch.sum(preds_phase.cpu()==labels_phase.cpu())
            
            pred_list_all.append(list(preds_phase.cpu().numpy()))

            preds_phase_one_hot = np.identity(7)[preds_phase.cpu()]
            labels_phase_one_hot = np.identity(7)[labels_phase.cpu()]
            
            accuracy, precision, recall, f1_score, jaccard = get_pr_re_acc_f1_onehot(preds_phase_one_hot, labels_phase_one_hot, preds_phase.cpu().numpy(), labels_phase.cpu().numpy())

            test_accuracy_phase_list.append(accuracy)
            test_recall_phase_list.append(recall)
            test_precision_phase_list.append(precision)
            test_jaccard_phase_list.append(jaccard)
            test_f1_phase_list.append(f1_score)

    test_accuracy_phase_total=test_correct_total/len(test_labels_80)
            
    test_accuracy_phase_final = np.nanmean(test_accuracy_phase_list)   
    test_recall_phase_final = np.nanmean(test_recall_phase_list)   
    test_precision_phase_final = np.nanmean(test_precision_phase_list)   
    test_jaccard_phase_final = np.nanmean(test_jaccard_phase_list)   
    test_f1_phase_final = np.nanmean(test_f1_phase_list)

    now_model_wts=copy.deepcopy(model.state_dict())
    if test_accuracy_phase_total > best_test_accuracy_phase:
        best_epoch=epoch
        best_test_accuracy_phase = test_accuracy_phase_total
        correspond_train_acc_phase = train_accuracy_phase
        best_model_wts = copy.deepcopy(model.state_dict())
        
        # for idx in test_we_use_start_idx_80:
        #     save_path = plot_path + str(idx) + '.png'
        #     plot_maps(pred_list_all[idx], save_path)
            
    elif test_accuracy_phase_total == best_test_accuracy_phase:
        if train_accuracy_phase > correspond_train_acc_phase:
            best_epoch = epoch
            best_test_accuracy_phase=test_accuracy_phase_total
            correspond_train_acc_phase = train_accuracy_phase
            best_model_wts = copy.deepcopy(model.state_dict())
            
            # for idx in test_we_use_start_idx_80:
            #     save_path = plot_path + str(idx) + '.png'
            #     plot_maps(pred_list_all[idx], save_path)

    logger.info('Epoch: %d/%d Time: %s' % (epoch, max_epoch, time.strftime("%d.%m.-%H:%M:%S", time.localtime())))
    logger.info("train accuracy phase: %6f" % (train_accuracy_phase))
    logger.info("train average loss phase: %6f" % (train_each_loss))
    logger.info("train average loss CL: %6f" % (train_latent_loss))
    logger.info("test accuracy phase total: %6f" % (test_accuracy_phase_total))
    logger.info("test accuracy phase final: %6f" % (test_accuracy_phase_final))
    logger.info("test recall phase final: %6f" % (test_recall_phase_final))
    logger.info("test precision phase final: %6f" % (test_precision_phase_final))
    logger.info("test jaccard phase final: %6f" % (test_jaccard_phase_final))
    logger.info("test f1 phase final: %6f" % (test_f1_phase_final))
    logger.info("best_epoch: %6f" % (best_epoch))
    logger.info('\n')


    print("best_epoch:",best_epoch)
    print('epoch: {:4d}'
        ' train loss(phase_frame): {:4.4f}'
        ' train accuracy(phase_frame): {:.4f}'
        .format(epoch,
                train_each_loss,
                train_accuracy_phase,
                ))
    
    print('epoch: {:4d}'
        ' test accuracy(phase_total): {:.4f}'
        ' test accuracy(phase_video): {:.4f}'
        ' test recall(phase_video): {:.4f}'
        ' test precision(phase_video): {:.4f}'
        ' test jaccard(phase_video): {:.4f}'
        ' test f1(phase_video): {:.4f}'
        .format(epoch,
                test_accuracy_phase_total,
                test_accuracy_phase_final,
                test_recall_phase_final,
                test_precision_phase_final,
                test_jaccard_phase_final,
                test_f1_phase_final
                ))

    torch.save(best_model_wts,'teacher_results/'+save_dir+'/'+small_dir+'/checkpoint-best-epoch'+ '.pth')



