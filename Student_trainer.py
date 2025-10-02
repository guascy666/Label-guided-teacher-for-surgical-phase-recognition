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
import sys
from model_all import MM_model,Student,MM_classifier

warnings.filterwarnings("ignore")

import argparse
parser = argparse.ArgumentParser(description='lstm training')
parser.add_argument('-pd', default="", type=str, help='save pkl dir')
parser.add_argument('-bw', default=1000, type=int, help='bandwidth')
parser.add_argument('-dm', default=1024, type=int, help='dmodel')
parser.add_argument('-nl', default=2, type=int, help='num_layers for each student block')
parser.add_argument('-dr', default=4, type=int, help='down rate for teacher')
parser.add_argument('-dr_S', default=4, type=int, help='down rate for student')
parser.add_argument('-ld', default=0.25, type=float, help='lamda for teacher CL')
parser.add_argument('-act',default=1,type=int,help='whether teacher use tanh 1fortrue 0forfalse')
parser.add_argument('-act_S',default=1,type=int,help='whether student use tanh 1fortrue 0forfalse')
#parser.add_argument('-load_LFB', default=1, type=int, help='whether load exist teacher feature 1 for true 0 for false')


args = parser.parse_args()
save_pkl_dir = args.pd
band_width = args.bw
d_model = args.dm
lamda=args.ld
down_rate = args.dr
down_rate_S=args.dr_S
n_layers=args.nl
active_tanh = bool(args.act)
active_tanh_S=bool(args.act_S)
#load_LFB=bool(args.load_LFB)
save_dir=save_pkl_dir.split('/')[-1].split('.')[0]
small_teacher_dir='_dm' + str(d_model) + '_dr' + str(down_rate) + '_bw' + str(band_width) + \
                    '_ld'+str(lamda) +'_act_'+str(active_tanh) + '_lr' + str(5e-5)
teacher_path="./teacher_results/"+save_dir+'/'+small_teacher_dir+ '/checkpoint-best-epoch'+ '.pth'
task_setting='t_dr' + str(down_rate) + 't_ld'+str(lamda) +'t_act_'+str(active_tanh)+ 's_dr' + str(down_rate_S)+'s_actS_'+str(active_tanh_S)+'_bw'+str(band_width)


print("file_pkl_name: ", save_pkl_dir)
print("teacher model path: ", teacher_path)
print("band_width: ",band_width)
print("down_rate_teacher: ",down_rate)
print("down_rate_student: ",down_rate_S)
print("num_layers for each student block: ",n_layers)
print("whether teacher use tanh: ",active_tanh)
print("whether student use tanh: ",active_tanh_S)
#print("whether load exist teacher feature: ",load_LFB)



data_split_pkl_path = './train_val_test_paths_labels_40_40.pkl'



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

def STAGE_2_LOSS(embedding1,embedding2,p,A):
    N=embedding1.size(1)
    out=A*torch.abs(embedding1-embedding2)/(1+torch.exp(1-p*torch.abs(embedding1-embedding2)))
    loss=torch.sum(out)/N
    return loss


train_labels_80, train_num_each_80, train_start_vidx,\
    test_labels_80, test_num_each_80, test_start_vidx = get_data(data_split_pkl_path)

train_pkl_path = save_pkl_dir+"/"+"g_LFB_swin_train0.pkl"
test_pkl_path = save_pkl_dir+"/"+"g_LFB_swin_test0.pkl"

task_name = train_pkl_path.split('/')[-1].split('.')[0]
print("loading pkl name: ", task_name)


print("saving pkl name: ",save_pkl_dir)

learning_rate=2e-5
weight_decay=0
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

train_we_use_start_idx_80 = [x for x in range(40)]
test_we_use_start_idx_80 = [x for x in range(40)]



""" teacher feature """

print("Extracting teacher feature >>>>>>>>")
    

model_T=MM_model(mask_length=band_width,dmodel=d_model,sample_rate=down_rate,Enc_active=active_tanh)
model_T=model_T.cuda()
model_T.load_state_dict(torch.load(teacher_path))
model_T.eval()

g_LFB_train_teacher_final = np.zeros(shape=(0, 1024))
g_LFB_test_teacher_final = np.zeros(shape=(0, 1024))

g_LFB_train_teacher_middle=np.zeros(shape=(0,32))
g_LFB_test_teacher_middle=np.zeros(shape=(0,32))

train_start_time = time.time()
torch.cuda.empty_cache()

with torch.no_grad():
    for i in train_we_use_start_idx_80:
        torch.cuda.empty_cache()
        labels_phase = []
        for j in range(train_start_vidx[i], train_start_vidx[i] + train_num_each_80[i]):
            labels_phase.append(train_labels_80[j][-1])

        labels_phase = torch.LongTensor(np.array(labels_phase))
        labels_phase = labels_phase.cuda()
        long_feature = get_long_feature(start_index=train_start_vidx[i],
                                        lfb=g_LFB_train, LFB_length=train_num_each_80[i])

        long_feature = (torch.Tensor(np.array(long_feature))).cuda()
        middle, final_feature, out_phase_label = model_T(long_feature, labels_phase)
        outputs_feature = final_feature.squeeze(0).data.cpu().numpy()
        middle_feature = middle.squeeze(0).data.cpu().numpy()

        g_LFB_train_teacher_middle = np.concatenate((g_LFB_train_teacher_middle, middle_feature), axis=0)
        g_LFB_train_teacher_final = np.concatenate((g_LFB_train_teacher_final, outputs_feature), axis=0)

        print("teacher train feature length:", len(g_LFB_train_teacher_final))

with torch.no_grad():
    for i in test_we_use_start_idx_80:
        torch.cuda.empty_cache()
        labels_phase = []
        for j in range(test_start_vidx[i], test_start_vidx[i] + test_num_each_80[i]):
            labels_phase.append(test_labels_80[j][-1])
        labels_phase = torch.LongTensor(np.array(labels_phase))
        labels_phase = labels_phase.cuda()
        long_feature = get_long_feature(start_index=test_start_vidx[i],
                                        lfb=g_LFB_test, LFB_length=test_num_each_80[i])

        long_feature = (torch.Tensor(np.array(long_feature))).cuda()

        middle, final_feature, out_phase_label = model_T(long_feature, labels_phase)
        outputs_feature = final_feature.squeeze(0).data.cpu().numpy()
        middle_feature = middle.squeeze(0).data.cpu().numpy()

        g_LFB_test_teacher_middle = np.concatenate((g_LFB_test_teacher_middle, middle_feature), axis=0)
        g_LFB_test_teacher_final = np.concatenate((g_LFB_test_teacher_final, outputs_feature), axis=0)

        print("teacher test feature length:", len(g_LFB_test_teacher_final))
    
    
train_elapsed_time = time.time() - train_start_time
print("extract time:", train_elapsed_time)

print("Finishi extracting teacher feature")

g_LFB_trainF  = np.array(g_LFB_train_teacher_final)
g_LFB_testF   = np.array(g_LFB_test_teacher_final)
print("train_final:", g_LFB_trainF.shape)
print("test_final:", g_LFB_testF.shape)

g_LFB_trainM=np.array(g_LFB_train_teacher_middle)
g_LFB_testM = np.array(g_LFB_test_teacher_middle)
print("train_middle:",g_LFB_trainM.shape)
print("test_middle:",g_LFB_testM.shape)
    
    
""" load teacher feature """
print("Finish loading teacher feature")
print("g_LFB_teacher_middle_train shape:", g_LFB_trainM.shape)
print("g_LFB_teacher_middle_test shape:", g_LFB_testM.shape)

""" load teacher feature """
print("Finish loading teacher feature")
print("g_LFB_teacher_final_train shape:", g_LFB_trainF.shape)
print("g_LFB_teacher_final_test shape:", g_LFB_testF.shape)

model_test=MM_classifier(mask_length=band_width,dmodel=d_model,sample_rate=down_rate,Enc_active=active_tanh)
model_test=model_test.cuda()
model_test.load_state_dict(torch.load(teacher_path))
model_test.eval()

model=Student(mask_length=band_width,dmodel=d_model,sample_rate=down_rate_S,Enc_active=active_tanh_S,n_layers=n_layers)
model=model.cuda()

optimizer=optim.AdamW([{'params': model.parameters()},], lr=learning_rate,weight_decay=weight_decay)
cri_mse=nn.L1Loss()

best_model_wts = copy.deepcopy(model.state_dict())
best_test_accuracy_phase = 0.0
correspond_train_acc_phase=0.0
best_test_acc_phase_video=0
best_epoch=-1
best_train_loss=10000000000

if not os.path.exists('student_results'):
    os.mkdir('student_results')

time_cur = time.strftime("%Y%m%d.%H%M", time.localtime(time.time()))

teacher=teacher_path.split(os.sep)
teacher_folder=teacher[-2]
if not os.path.exists('./student_results/'+save_dir):
    os.mkdir('./student_results/'+save_dir)

small_dir='_dm' + str(d_model) + '_dr' + str(down_rate_S)+'_ld'+str(lamda)+'_nl'+str(n_layers) + '_bw' + str(band_width) +'_actS_'+str(active_tanh_S) + '_lr' \
                      + str(learning_rate)
if not os.path.exists('./student_results/'+save_dir+'/'+small_dir):
    os.mkdir('./student_results/'+save_dir+'/'+small_dir)

logger = get_log('./student_results/'+save_dir+'/'+small_dir + '/log' + '_dm' + str(d_model) \
                 + '_dr' + str(down_rate_S)+'_nl'+str(n_layers) + '_bw' + str(band_width) +'_actS_'+str(active_tanh_S) + '_lr' \
                      + str(learning_rate) + '.txt')

plot_path = './student_results/'+save_dir +'/'+small_dir+ '/plot_'  + '_dm' + str(d_model) \
      + '_dr' + str(down_rate_S)+'_nl'+str(n_layers) + '_bw' + str(band_width) +'_actS_'+str(active_tanh_S) + '_lr' + str(learning_rate) + '/'
if not os.path.exists(plot_path):
    os.mkdir(plot_path)


test_accuracy_final=0
test_recall_final=0
test_jaccard_final=0
test_f1_final=0
test_precision_final=0
for epoch in range(max_epoch):

    train_start_time = time.time()

    torch.cuda.empty_cache()
    random.shuffle(train_we_use_start_idx_80)
    train_idx_80 = []
    model.train()
    train_loss_phase=0.0
    train_correct_phase=0
    loss_latent=0.0
    loss_latent_total=0.0
    print("train_working>>>>>>>>>>>>>>>>>>>>>>>>")

    for i in train_we_use_start_idx_80:
        torch.cuda.empty_cache()
        optimizer.zero_grad()
        labels_phase = []
        for j in range(train_start_vidx[i], train_start_vidx[i] + train_num_each_80[i]):
            labels_phase.append(train_labels_80[j][-1])
        labels_phase = torch.LongTensor(np.array(labels_phase))
        labels_phase = labels_phase.cuda()
        long_feature = get_long_feature(start_index=train_start_vidx[i],
                                        lfb=g_LFB_train, LFB_length=train_num_each_80[i])
        middle_fea=get_long_feature(start_index=train_start_vidx[i],
                                        lfb=g_LFB_trainM, LFB_length=train_num_each_80[i])
        final_fea=get_long_feature(start_index=train_start_vidx[i],
                                        lfb=g_LFB_trainF, LFB_length=train_num_each_80[i])

        middle_fea=(torch.Tensor(np.array(middle_fea))).cuda().squeeze(0)#teacher的f=32的模型
        final_fea=(torch.Tensor(np.array(final_fea))).cuda().squeeze(0)#teacher的f=1024的模型

        long_feature = (torch.Tensor(np.array(long_feature))).cuda()

        latent1,latent2= model(long_feature)
    
        out_phase=model_test(latent2)
        
        latent1=latent1.squeeze(0)
        latent2=latent2.squeeze(0)
   
        out_phase=out_phase.squeeze(0)
        loss=cri_mse(latent1,middle_fea)+cri_mse(latent2,final_fea)
 
        loss.backward()
        optimizer.step()
        loss_latent+=loss.item()

        _, preds_phase = torch.max(out_phase.data, 1)
        train_correct_phase += torch.sum(preds_phase == labels_phase.data)


    train_accuracy_phase = float(train_correct_phase) / len(train_labels_80)
    train_each_loss=loss_latent/40


    test_corrects_phase = 0
    test_accuracy_phase_list = []
    test_recall_phase_list = []
    test_precision_phase_list = []
    test_jaccard_phase_list = []
    test_f1_phase_list = []
    pred_list_all=[]
    test_latent_loss=0
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
            long_feature = get_long_feature(start_index=test_start_vidx[i],
                                            lfb=g_LFB_test, LFB_length=test_num_each_80[i])
            
            middle_fea=get_long_feature(start_index=test_start_vidx[i],
                                        lfb=g_LFB_testM, LFB_length=test_num_each_80[i])
            final_fea=get_long_feature(start_index=test_start_vidx[i],
                                        lfb=g_LFB_testF, LFB_length=test_num_each_80[i])

            middle_fea=(torch.Tensor(np.array(middle_fea))).cuda().squeeze(0)#teacher的f=32的模型
            final_fea=(torch.Tensor(np.array(final_fea))).cuda().squeeze(0)#teacher的f=1024的模型


            long_feature = (torch.Tensor(np.array(long_feature))).cuda()

            latent1,latent2 = model(long_feature)
            out_phase = model_test(latent2)

            out_phase = out_phase.squeeze(0).detach()

            _, preds_phase = torch.max(out_phase.data, 1)

          

            test_latent_loss+=cri_mse(latent1,middle_fea)+cri_mse(latent2,final_fea)

            pred_list_all.append(list(preds_phase.cpu().numpy()))

            preds_phase_one_hot = np.identity(7)[preds_phase.cpu()]
            labels_phase_one_hot = np.identity(7)[labels_phase.cpu()]
            
            accuracy, precision, recall, f1_score, jaccard = get_pr_re_acc_f1_onehot(preds_phase_one_hot, labels_phase_one_hot, preds_phase.cpu().numpy(), labels_phase.cpu().numpy())

            test_accuracy_phase_list.append(accuracy)
            test_recall_phase_list.append(recall)
            test_precision_phase_list.append(precision)
            test_jaccard_phase_list.append(jaccard)
            test_f1_phase_list.append(f1_score)
    
    test_loss_each=test_latent_loss/40
    test_accuracy_phase_final = np.nanmean(test_accuracy_phase_list)   
    test_recall_phase_final = np.nanmean(test_recall_phase_list)   
    test_precision_phase_final = np.nanmean(test_precision_phase_list)   
    test_jaccard_phase_final = np.nanmean(test_jaccard_phase_list)   
    test_f1_phase_final = np.nanmean(test_f1_phase_list)
            
    train_elapsed_time = time.time() - train_start_time
    print("Training time:", train_elapsed_time)

    now_model_wts=copy.deepcopy(model.state_dict())
    if test_accuracy_phase_final > best_test_accuracy_phase:
        best_epoch=epoch
        best_test_accuracy_phase = test_accuracy_phase_final
        correspond_train_acc_phase = train_accuracy_phase
        best_model_wts = copy.deepcopy(model.state_dict())

        test_accuracy_final = test_accuracy_phase_final
        test_recall_final = test_recall_phase_final
        test_precision_final = test_precision_phase_final
        test_jaccard_final = test_jaccard_phase_final
        test_f1_final = test_f1_phase_final
        
        # for idx in test_we_use_start_idx_80:
        #     save_path = plot_path + str(idx) + '.png'
        #     plot_maps(pred_list_all[idx], save_path)
            
    elif test_accuracy_phase_final == best_test_accuracy_phase:
        if train_accuracy_phase > correspond_train_acc_phase:
            best_epoch = epoch
            best_test_acc_phase_video=test_accuracy_phase_final
            correspond_train_acc_phase = train_accuracy_phase
            best_model_wts = copy.deepcopy(model.state_dict())

            test_accuracy_final = test_accuracy_phase_final
            test_recall_final = test_recall_phase_final
            test_precision_final = test_precision_phase_final
            test_jaccard_final = test_jaccard_phase_final
            test_f1_final = test_f1_phase_final
            
            # for idx in test_we_use_start_idx_80:
            #     save_path = plot_path + str(idx) + '.png'
            #     plot_maps(pred_list_all[idx], save_path)

    logger.info('Epoch: %d/%d Time: %s' % (epoch, max_epoch, time.strftime("%d.%m.-%H:%M:%S", time.localtime())))
    logger.info("train accuracy phase: %6f" % (train_accuracy_phase))
    logger.info("train average loss phase: %6f" % (train_each_loss))
    logger.info("test accuracy phase final: %6f" % (test_accuracy_phase_final))
    logger.info("test recall phase final: %6f" % (test_recall_phase_final))
    logger.info("test precision phase final: %6f" % (test_precision_phase_final))
    logger.info("test jaccard phase final: %6f" % (test_jaccard_phase_final))
    logger.info("test f1 phase final: %6f" % (test_f1_phase_final))
    logger.info("test average loss : %6f" % (test_loss_each))
    logger.info("best_epoch: %6f" % (best_epoch))
    logger.info('\n')

    print("best_epoch:",best_epoch)
    print('epoch: {:4d}'
        ' train loss(phase_frame): {:4.4f}'
        ' train accuracy(phase_frame): {:.4f}'
        .format(epoch,
                train_loss_phase/40,
                train_accuracy_phase,
                ))

    print('epoch: {:4d}'
        ' test accuracy(phase_video): {:.4f}'
        ' test recall(phase_video): {:.4f}'
        ' test precision(phase_video): {:.4f}'
        ' test jaccard(phase_video): {:.4f}'
        ' test f1(phase_video): {:.4f}'
        .format(epoch,
                test_accuracy_phase_final,
                test_recall_phase_final,
                test_precision_phase_final,
                test_jaccard_phase_final,
                test_f1_phase_final
                ))

    torch.save(best_model_wts,'./student_results/'+save_dir+'/'+small_dir + '/checkpoint-best-epoch'+ '.pth')


total_log='./student_results/'+save_dir+'.txt'
logger_total=get_log(total_log)
logger_total.info(" Ablation setting name : %s" % (task_setting))
logger_total.info("test accuracy phase final: %6f" % (test_accuracy_final))
logger_total.info("test recall phase final: %6f" % (test_recall_final))
logger_total.info("test precision phase final: %6f" % (test_precision_final))
logger_total.info("test jaccard phase final: %6f" % (test_jaccard_final))
logger_total.info("test f1 phase final: %6f" % (test_f1_final))
logger_total.info('\n')