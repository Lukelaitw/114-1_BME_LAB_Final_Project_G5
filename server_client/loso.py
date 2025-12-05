"""
CTNet: A Convolution-Transformer Network for EEG-Based Motor Imagery Classification

author: zhaowei701@163.com

Due to memory constraints, the data augmentation method in LOSO classification was slightly optimized based on the approach used in subject-specific classification (main.py).


Cite this work
Zhao, W., Jiang, X., Zhang, B. et al. CTNet: a convolutional transformer network for EEG-based motor imagery classification. Sci Rep 14, 20237 (2024). https://doi.org/10.1038/s41598-024-71118-7


"""

import os
gpus = [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import numpy as np
import pandas as pd
import random
import datetime
import time
import math
from pandas import ExcelWriter
from torchsummary import summary
import torch
from torch.backends import cudnn
from utils import calMetrics
from utils import calculatePerClass
from utils import numberClassChannel

import warnings
warnings.filterwarnings("ignore")
cudnn.benchmark = False
cudnn.deterministic = True

import torch
from torch import nn
from torch import Tensor
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange, reduce, repeat
import torch.nn.functional as F

from utils import numberClassChannel
from utils import load_data_evaluate

import numpy as np
import pandas as pd
from torch.autograd import Variable
from sklearn.metrics import accuracy_score  
from sklearn.metrics import precision_score 
from sklearn.metrics import recall_score  
from sklearn.metrics import f1_score  
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


class PatchEmbeddingCNN(nn.Module):
    def __init__(self, f1=16, kernel_size=64, D=2, pooling_size1=8, pooling_size2=8, dropout_rate=0.3, number_channel=22, emb_size=40):
        super().__init__()
        f2 = D*f1
        self.cnn_module = nn.Sequential(
            # temporal conv kernel size 64=0.25fs
            nn.Conv2d(1, f1, (1, kernel_size), (1, 1), padding='same', bias=False), # [batch, 22, 1000] 
            nn.BatchNorm2d(f1),
            # channel depth-wise conv
            nn.Conv2d(f1, f2, (number_channel, 1), (1, 1), groups=f1, padding='valid', bias=False), # 
            nn.BatchNorm2d(f2),
            nn.ELU(),
            # average pooling 1
            nn.AvgPool2d((1, pooling_size1)),  # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            nn.Dropout(dropout_rate),
            # spatial conv
            nn.Conv2d(f2, f2, (1, 16), padding='same', bias=False), 
            nn.BatchNorm2d(f2),
            nn.ELU(),

            # average pooling 2 to adjust the length of feature into transformer encoder
            nn.AvgPool2d((1, pooling_size2)),
            nn.Dropout(dropout_rate),  
                    
        )

        self.projection = nn.Sequential(
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        
        
    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.cnn_module(x)
        x = self.projection(x)
        return x
    
    
  
    
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out
    


# PointWise FFN
class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )



class ClassificationHead(nn.Sequential):
    def __init__(self, flatten_number, n_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(flatten_number, n_classes)
        )

    def forward(self, x):
        out = self.fc(x)
        
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn, emb_size, drop_p):
        super().__init__()
        self.fn = fn
        self.drop = nn.Dropout(drop_p)
        self.layernorm = nn.LayerNorm(emb_size)

    def forward(self, x, **kwargs):
        x_input = x
        res = self.fn(x, **kwargs)
        
        out = self.layernorm(self.drop(res)+x_input)
        return out

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=4,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                MultiHeadAttention(emb_size, num_heads, drop_p),
                ), emb_size, drop_p),
            ResidualAdd(nn.Sequential(
                FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                ), emb_size, drop_p)
            
            )    
        
        
class TransformerEncoder(nn.Sequential):
    def __init__(self, heads, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size, heads) for _ in range(depth)])




class BranchEEGNetTransformer(nn.Sequential):
    def __init__(self, heads=4, 
                 depth=6, 
                 emb_size=40, 
                 number_channel=22,
                 f1 = 20,
                 kernel_size = 64,
                 D = 2,
                 pooling_size1 = 8,
                 pooling_size2 = 8,
                 dropout_rate = 0.3,
                 **kwargs):
        super().__init__(
            PatchEmbeddingCNN(f1=f1, 
                                 kernel_size=kernel_size,
                                 D=D, 
                                 pooling_size1=pooling_size1, 
                                 pooling_size2=pooling_size2, 
                                 dropout_rate=dropout_rate,
                                 number_channel=number_channel,
                                 emb_size=emb_size),

        )



    
        
        
class PositioinalEncoding(nn.Module):
    def __init__(self, embedding, length=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.encoding = nn.Parameter(torch.randn(1, length, embedding))
    def forward(self, x): # x-> [batch, embedding, length]
        # 使用設備感知的代碼，確保 encoding 與輸入 x 在同一設備上
        encoding = self.encoding[:, :x.shape[1], :].to(x.device)
        x = x + encoding
        return self.dropout(x)        
        
   
    
class EEGTransformer(nn.Module):
    def __init__(self, heads=4, 
                 emb_size=40,
                 depth=6, 
                 database_type='A', 
                 eeg1_f1 = 20,
                 eeg1_kernel_size = 64,
                 eeg1_D = 2,
                 eeg1_pooling_size1 = 8,
                 eeg1_pooling_size2 = 8,
                 eeg1_dropout_rate = 0.3,
                 eeg1_number_channel = 22,
                 flatten_eeg1 = 600,
                 **kwargs):
        super().__init__()
        self.number_class, self.number_channel = numberClassChannel(database_type)
        self.emb_size = emb_size
        self.flatten_eeg1 = flatten_eeg1
        self.flatten = nn.Flatten()
        # print('self.number_channel', self.number_channel)
        self.cnn = BranchEEGNetTransformer(heads, depth, emb_size, number_channel=self.number_channel,
                                              f1 = eeg1_f1,
                                              kernel_size = eeg1_kernel_size,
                                              D = eeg1_D,
                                              pooling_size1 = eeg1_pooling_size1,
                                              pooling_size2 = eeg1_pooling_size2,
                                              dropout_rate = eeg1_dropout_rate,
                                              )
        self.position = PositioinalEncoding(emb_size, dropout=0.1)
        self.trans = TransformerEncoder(heads, depth, emb_size)

        self.flatten = nn.Flatten()
        self.classification = ClassificationHead(self.flatten_eeg1 , self.number_class) # FLATTEN_EEGNet + FLATTEN_cnn_module
    def forward(self, x):
        cnn = self.cnn(x)
        cnn = cnn * math.sqrt(self.emb_size)
        cnn = self.position(cnn)
        trans = self.trans(cnn)
        
        features = cnn+trans
        out = self.classification(self.flatten(features))
        return features, out




class ExP():
    def __init__(self, nsub, data_dir, result_name, 
                 epochs=2000, 
                 number_aug=2,
                 number_seg=8, 
                 gpus=[0], 
                 evaluate_mode = 'subject-dependent',
                 heads=4, 
                 emb_size=40,
                 depth=6, 
                 dataset_type='A',
                 eeg1_f1 = 20,
                 eeg1_kernel_size = 64,
                 eeg1_D = 2,
                 eeg1_pooling_size1 = 8,
                 eeg1_pooling_size2 = 8,
                 eeg1_dropout_rate = 0.3,
                 flatten_eeg1 = 600, 
                 validate_ratio = 0.2,
                 learning_rate = 0.001,
                 batch_size = 72,  
                 ):
        
        super(ExP, self).__init__()
        self.dataset_type = dataset_type
        self.batch_size = batch_size
        self.lr = learning_rate
        self.b1 = 0.5
        self.b2 = 0.999
        self.n_epochs = epochs
        self.nSub = nsub
        self.number_augmentation = number_aug
        self.number_seg = number_seg
        self.root = data_dir
        self.heads=heads
        self.emb_size=emb_size
        self.depth=depth
        self.result_name = result_name
        self.evaluate_mode = evaluate_mode
        self.validate_ratio = validate_ratio

        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor
        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()

        self.number_class, self.number_channel = numberClassChannel(self.dataset_type)
        self.model = EEGTransformer(
             heads=self.heads, 
             emb_size=self.emb_size,
             depth=self.depth, 
            database_type=self.dataset_type, 
            eeg1_f1=eeg1_f1, 
            eeg1_D=eeg1_D,
            eeg1_kernel_size=eeg1_kernel_size,
            eeg1_pooling_size1 = eeg1_pooling_size1,
            eeg1_pooling_size2 = eeg1_pooling_size2,
            eeg1_dropout_rate = eeg1_dropout_rate,
            eeg1_number_channel = self.number_channel,
            flatten_eeg1 = flatten_eeg1,  
            ).cuda()
        #self.model = nn.DataParallel(self.model, device_ids=gpus)
        self.model = self.model.cuda()
        self.model_filename = self.result_name + '/model_{}.pth'.format(self.nSub)
        


    # Segmentation and Reconstruction (S&R) data augmentation
    def interaug(self, timg, label):  
        aug_data = []
        aug_label = []
        
        number_segmentation_points = 1000 // self.number_seg
        for clsAug in range(self.number_class):
            cls_idx = np.where(label == clsAug + 1)
            tmp_data = timg[cls_idx]
            number_records_by_augmentation = self.number_augmentation * tmp_data.shape[0]
            tmp_aug_data = np.zeros((number_records_by_augmentation, 1, self.number_channel, 1000))
            for ri in range(number_records_by_augmentation):
                for rj in range(self.number_seg):
                    rand_idx = np.random.randint(0, tmp_data.shape[0], self.number_seg)
                    tmp_aug_data[ri, :, :, rj * number_segmentation_points:(rj + 1) * number_segmentation_points] = \
                        tmp_data[rand_idx[rj], :, :, rj * number_segmentation_points:(rj + 1) * number_segmentation_points]

            aug_data.append(tmp_aug_data)
            aug_label.append([clsAug + 1]*number_records_by_augmentation)
        aug_data = np.concatenate(aug_data)
        aug_label = np.concatenate(aug_label)
        # aug_shuffle = np.random.permutation(len(aug_data))
        # aug_data = aug_data[aug_shuffle, :, :]
        # aug_label = aug_label[aug_shuffle]

        # aug_data = torch.from_numpy(aug_data).cuda()
        # aug_data = aug_data.float()
        # aug_label = torch.from_numpy(aug_label-1).cuda()
        # aug_label = aug_label.long()
        return aug_data, aug_label



    def get_source_data(self):
        (self.train_data,    # (batch, channel, length)
         self.train_label, 
         self.test_data, 
         self.test_label) = load_data_evaluate(self.root, self.dataset_type, self.nSub, mode_evaluate=self.evaluate_mode)

        self.train_data = np.expand_dims(self.train_data, axis=1)  # (288, 1, 22, 1000)
        self.train_label = np.transpose(self.train_label) 
        
        self.allData = self.train_data
        self.allLabel = self.train_label[0]  
        # split original allData into training and validate datasets

        train_data_list = []
        train_label_list = []
        validate_data_list = []
        validate_label_list = []
        for cls in range(self.number_class):
            # filter by class 
            cls_idx = np.where(self.allLabel == cls + 1)
            cat_data = self.allData[cls_idx]
            cat_label = self.allLabel[cls_idx]

            
            # each category split
            number_sample = cat_data.shape[0]
            number_validate = int(self.validate_ratio * number_sample)
            # shuffle index
            index_shuffle = np.random.permutation(len(cat_data))
            index_train = index_shuffle[:-number_validate]
            index_validate = index_shuffle[-number_validate:]
            
            train_data_list.append(cat_data[index_train])
            train_label_list.append(cat_label[index_train])
            
            validate_data_list.append(cat_data[index_validate])
            validate_label_list.append(cat_label[index_validate])
        
        # data augmentation
        aug_data, aug_label = self.interaug(self.allData, self.allLabel)
            
        train_data_list.append(aug_data)
        train_label_list.append(aug_label)
            
        self.trainData = np.concatenate(train_data_list)
        self.trainLabel = np.concatenate(train_label_list)
        self.validateData = np.concatenate(validate_data_list)
        self.validateLabel = np.concatenate(validate_label_list)
        
        # shuffle in all category
        shuffle_num = np.random.permutation(len(self.trainData))
        self.trainData = self.trainData[shuffle_num, :, :, :]  # (number of training sample, 1, 22, 1000)
        self.trainLabel = self.trainLabel[shuffle_num]

        # self.test_data = np.transpose(self.test_data, (2, 1, 0))
        self.test_data = np.expand_dims(self.test_data, axis=1)
        self.test_label = np.transpose(self.test_label)

        self.testData = self.test_data
        self.testLabel = self.test_label[0]

        # standardize
        target_mean = np.mean(self.allData)
        target_std = np.std(self.allData)
        # self.allData = (self.allData - target_mean) / target_std
        self.trainData = (self.trainData - target_mean) / target_std
        self.validateData = (self.validateData - target_mean) / target_std
        self.testData = (self.testData - target_mean) / target_std
        
        isSaveDataLabel = False #True
        if isSaveDataLabel:
            np.save("./gradm_data/train_data_{}.npy".format(self.nSub), self.allData)
            np.save("./gradm_data/train_lable_{}.npy".format(self.nSub), self.allLabel)
            np.save("./gradm_data/test_data_{}.npy".format(self.nSub), self.testData)
            np.save("./gradm_data/test_label_{}.npy".format(self.nSub), self.testLabel)
        print(self.trainData.shape, self.trainLabel.shape, self.validateData.shape, self.validateLabel.shape, self.testData.shape, self.testLabel.shape)
        # data shape: (trial, conv channel, electrode channel, time samples)
        return self.trainData, self.trainLabel, self.validateData, self.validateLabel, self.testData, self.testLabel



    def fit_test(self, model, loss_fn, testloader):
        y_list = []
        y_pred_list = []

        test_correct = 0
        test_total = 0
        test_running_loss = 0
        model.eval()  
        with torch.no_grad():
            for x, y in testloader:
                x = Variable(x.type(self.Tensor))
                y = Variable(y.type(self.LongTensor))
                
                features, y_pred = model(x)
                loss = loss_fn(y_pred, y)
                y_pred = torch.argmax(y_pred, dim=1)
                test_correct += (y_pred == y).sum().item()
                test_total += y.size(0)
                test_running_loss += loss.item()
                y_pred = y_pred.cpu().numpy()
                y = y.cpu().numpy()
                y_list.extend(y)  
                y_pred_list.extend(y_pred)  

        acc_score = accuracy_score(y_list, y_pred_list)
        epoch_test_loss = test_running_loss / len(testloader.dataset)

        return epoch_test_loss, acc_score, y_list, y_pred_list
    
    
    def fit_train(self, model, loss_fn, dataloader, optimizer, trainData, trainLabel):
        correct = 0
        total = 0
        running_loss = 0
        model.train()

        for train_data, train_label in dataloader:
            # real train dataset
            img = Variable(train_data.type(self.Tensor))
            label = Variable(train_label.type(self.LongTensor))


            # training model
            features, y_pred = model(img)
            # print("train outputs: ", outputs.shape, type(outputs))
            # print(features.size())
            loss = loss_fn(y_pred, label) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                y_pred = torch.argmax(y_pred, dim=1)
                correct += (y_pred == label).sum().item()
                total += label.size(0)
                running_loss += loss.item()

        epoch_train_loss = running_loss / len(dataloader.dataset)
        epoch_train_acc = correct / total

        return epoch_train_loss, epoch_train_acc
    
    
    def train(self):
        img, label, validate_data, validate_label, test_data, test_label = self.get_source_data()
        # train dataset
        img = torch.from_numpy(img)
        label = torch.from_numpy(label - 1)
        dataset = torch.utils.data.TensorDataset(img, label)
        self.dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)
        # validate dataset
        validate_data = torch.from_numpy(validate_data)
        validate_label = torch.from_numpy(validate_label - 1)
        validate_dataset = torch.utils.data.TensorDataset(validate_data, validate_label)

        self.validate_dataloader = torch.utils.data.DataLoader(dataset=validate_dataset, batch_size=288, shuffle=False)
        # test dataset
        test_data = torch.from_numpy(test_data)
        test_label = torch.from_numpy(test_label - 1)
        test_dataset = torch.utils.data.TensorDataset(test_data, test_label)
        self.test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=288, shuffle=False)

        # Optimizers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2))

        test_data = Variable(test_data.type(self.Tensor))
        test_label = Variable(test_label.type(self.LongTensor))
        best_epoch = 0
        num = 0
        min_loss = 100
        max_acc = 0
        # recording train_acc, train_loss, test_acc, test_loss
        result_process = []
        # Train the CTNet model
        for e in range(self.n_epochs):
            epoch_process = {}
            epoch_process['epoch'] = e
            # train model
            self.model.train()
            train_loss, train_acc = self.fit_train(self.model, self.criterion_cls, self.dataloader, self.optimizer, self.allData, self.allLabel)
            epoch_process['train_acc'] = train_acc
            epoch_process['train_loss'] = train_loss
            
            # validate model
            (validate_loss, 
             validate_acc, 
             y_list, 
             y_pred_list) = self.fit_test(self.model, self.criterion_cls, self.validate_dataloader)
            epoch_process['val_acc'] = validate_acc                
            epoch_process['val_loss'] = validate_loss

            # test model (每個epoch都計算test_acc)
            (test_loss, 
             test_acc, 
             y_list, 
             y_pred_list) = self.fit_test(self.model, self.criterion_cls, self.test_dataloader)
            epoch_process['test_acc'] = test_acc                
            epoch_process['test_loss'] = test_loss

#             train_pred = torch.max(outputs, 1)[1]
#             train_acc = float((train_pred == label).cpu().numpy().astype(int).sum()) / float(label.size(0))
            num = num + 1

            # 當test_acc達到新高時保存模型
            if test_acc > max_acc:
                max_acc = test_acc
                best_epoch = e
                torch.save(self.model, self.model_filename)
                print("{}_{} train_acc: {:.4f} train_loss: {:.6f}\tval_acc: {:.6f} val_loss: {:.9f}, test_acc:{:.6f} [BEST]".format(self.nSub,
                                                                                       epoch_process['epoch'],
                                                                                       epoch_process['train_acc'],
                                                                                       epoch_process['train_loss'],
                                                                                       epoch_process['val_acc'],
                                                                                       epoch_process['val_loss'],
                                                                                       epoch_process['test_acc']
                                                                                    ))
            else:
                print("{}_{} train_acc: {:.4f} train_loss: {:.6f}\tval_acc: {:.6f} val_loss: {:.9f}, test_acc:{:.6f}".format(self.nSub,
                                                                                       epoch_process['epoch'],
                                                                                       epoch_process['train_acc'],
                                                                                       epoch_process['train_loss'],
                                                                                       epoch_process['val_acc'],
                                                                                       epoch_process['val_loss'],
                                                                                       epoch_process['test_acc']
                                                                                    ))
            
                
            result_process.append(epoch_process)  
        
        # load model for test
        self.model.eval()
        self.model = torch.load(self.model_filename, weights_only=False).cuda()
        # test model
        (test_loss, 
         test_acc, 
         y_list, 
         y_pred_list) = self.fit_test(self.model, self.criterion_cls, self.test_dataloader)

        print("epoch: ", best_epoch, '\tThe test accuracy is:', test_acc)


        df_process = pd.DataFrame(result_process)

        return test_acc, torch.tensor(y_list), torch.tensor(y_pred_list), df_process, best_epoch
        # writer.close()
        

#======plot func=====
def plot_results(results):
    """Plot result charts"""
    if results is None:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('BCI Classifier (Raw Data) - LOSO Cross-Validation Results', fontsize=16)
    
    # 1. Accuracy distribution
    axes[0].bar(range(len(results['accuracies'])), results['accuracies'], 
                color=['green' if acc >= 0.7 else 'orange' if acc >= 0.6 else 'red' 
                       for acc in results['accuracies']])
    axes[0].set_title('Accuracy by Subject')
    axes[0].set_xlabel('Subject Index')
    axes[0].set_ylabel('Accuracy')
    axes[0].axhline(y=np.mean(results['accuracies']), color='r', linestyle='--', 
                    label=f'Mean: {np.mean(results["accuracies"]):.3f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1)
    
    # 2. Overall confusion matrix
    total_cm = np.sum(results['confusion_matrices'], axis=0)
    sns.heatmap(total_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Relax', 'Focus'],
                yticklabels=['Relax', 'Focus'],
                ax=axes[1])
    axes[1].set_title('Overall Confusion Matrix')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    
    # 3. Training loss curves
    valid_loss_curves = [lc for lc in results['loss_curves'] if len(lc) > 0]
    if valid_loss_curves:
        for i, loss_curve in enumerate(valid_loss_curves[:5]):  # Show first 5
            axes[2].plot(loss_curve, alpha=0.7, label=f'S{i+1}')
        axes[2].set_title('Training Loss Curves (First 5 Subjects)')
        axes[2].set_xlabel('Iteration')
        axes[2].set_ylabel('Loss')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, 'No loss curves available', 
                     horizontalalignment='center', verticalalignment='center',
                     transform=axes[2].transAxes)
        axes[2].set_title('Training Loss Curves')
    
    plt.tight_layout()
    plt.savefig('bci_results_data.png', dpi=300, bbox_inches='tight')
    #plt.show()


def main(dirs,                
         evaluate_mode = 'subject-dependent', # LOSO or not
         heads=8,             # heads of MHA
         emb_size=48,         # token embding dim
         depth=3,             # Transformer encoder depth
         dataset_type='A',    # A->'BCI IV2a', B->'BCI IV2b'
         eeg1_f1=20,          # features of temporal conv
         eeg1_kernel_size=64, # kernel size of temporal conv
         eeg1_D=2,            # depth-wise conv 
         eeg1_pooling_size1=8,# p1
         eeg1_pooling_size2=8,# p2
         eeg1_dropout_rate=0.3,
         flatten_eeg1=600,   
         validate_ratio = 0.2,
         batch_size = 72
         ):

    if not os.path.exists(dirs):
        os.makedirs(dirs)

    result_write_metric = ExcelWriter(dirs+"/result_metric.xlsx")
    
    result_metric_dict = {}
    y_true_pred_dict = { }

    process_write = ExcelWriter(dirs+"/process_train.xlsx")
    pred_true_write = ExcelWriter(dirs+"/pred_true.xlsx")
    subjects_result = []
    best_epochs = []
    
    # 收集繪圖數據
    all_accuracies = []
    all_confusion_matrices = []
    all_loss_curves = []
    













    
    for i in range(0, N_SUBJECT):      
        
        starttime = datetime.datetime.now()
        seed_n = np.random.randint(2024)
        print('seed is ' + str(seed_n))
        random.seed(seed_n)
        np.random.seed(seed_n)
        torch.manual_seed(seed_n)
        torch.cuda.manual_seed(seed_n)
        torch.cuda.manual_seed_all(seed_n)
        index_round =0
        print('Subject %d' % (i+1))
        exp = ExP(i + 1, DATA_DIR, dirs, EPOCHS, N_AUG, N_SEG, gpus, 
                  evaluate_mode = evaluate_mode,
                  heads=heads, 
                  emb_size=emb_size,
                  depth=depth, 
                  dataset_type=dataset_type,
                  eeg1_f1 = eeg1_f1,
                  eeg1_kernel_size = eeg1_kernel_size,
                  eeg1_D = eeg1_D,
                  eeg1_pooling_size1 = eeg1_pooling_size1,
                  eeg1_pooling_size2 = eeg1_pooling_size2,
                  eeg1_dropout_rate = eeg1_dropout_rate,
                  flatten_eeg1 = flatten_eeg1,  
                  validate_ratio = validate_ratio,
                  batch_size = batch_size 
                  )

        testAcc, Y_true, Y_pred, df_process, best_epoch = exp.train()
        true_cpu = Y_true.cpu().numpy().astype(int)
        pred_cpu = Y_pred.cpu().numpy().astype(int)
        df_pred_true = pd.DataFrame({'pred': pred_cpu, 'true': true_cpu})
        df_pred_true.to_excel(pred_true_write, sheet_name=str(i+1))
        y_true_pred_dict[i] = df_pred_true

        # 收集準確率
        all_accuracies.append(testAcc)
        
        # 計算混淆矩陣
        cm = confusion_matrix(true_cpu, pred_cpu)
        all_confusion_matrices.append(cm)
        
        # 收集訓練損失曲線（從 df_process 獲取）
        train_losses = df_process['train_loss'].values.tolist()
        all_loss_curves.append(train_losses)

        accuracy, precison, recall, f1, kappa = calMetrics(true_cpu, pred_cpu)
        subject_result = {'accuray': accuracy*100,
                          'precision': precison*100,
                          'recall': recall*100,
                          'f1': f1*100, 
                          'kappa': kappa*100
                          }
        subjects_result.append(subject_result)
        df_process.to_excel(process_write, sheet_name=str(i+1))
        best_epochs.append(best_epoch)
    
        print(' THE BEST ACCURACY IS ' + str(testAcc) + "\tkappa is " + str(kappa) )
    

        endtime = datetime.datetime.now()
        print('subject %d duration: '%(i+1) + str(endtime - starttime))

        if i == 0:
            yt = Y_true
            yp = Y_pred
        else:
            yt = torch.cat((yt, Y_true))
            yp = torch.cat((yp, Y_pred))
                
        df_result = pd.DataFrame(subjects_result)
    process_write.close()
    pred_true_write.close()


    print('**The average Best accuracy is: ' + str(df_result['accuray'].mean()) + "kappa is: " + str(df_result['kappa'].mean()) + "\n" )
    print("best epochs: ", best_epochs)
    #df_result.to_excel(result_write_metric, index=False)
    result_metric_dict = df_result

    mean = df_result.mean(axis=0)
    mean.name = 'mean'
    std = df_result.std(axis=0)
    std.name = 'std'
    df_result = pd.concat([df_result, pd.DataFrame(mean).T, pd.DataFrame(std).T])
    
    df_result.to_excel(result_write_metric, index=False)
    print('-'*9, ' all result ', '-'*9)
    print(df_result)
    
    print("*"*40)

    result_write_metric.close()

    # 繪製結果圖表
    results_dict = {
        'accuracies': all_accuracies,
        'confusion_matrices': all_confusion_matrices,
        'loss_curves': all_loss_curves
    }
    plot_results(results_dict)
    
    return result_metric_dict

if __name__ == "__main__":
    #----------------------------------------
    # 自定義數據集配置（單通道二分類：專注vs放鬆）
    DATA_DIR = r'./bci_dataset_113-2/'  # 自定義txt數據目錄
    EVALUATE_MODE = 'LOSO' # leaving one subject out subject-dependent  subject-indenpedent

    N_SUBJECT = 35     # 自定義數據集有18個受試者 (S01-S18)
    N_AUG = 3           # data augmentation times for benerating artificial training data set
    N_SEG = 50           # segmentation times for S&R

    EPOCHS = 1000
    EMB_DIM = 16
    HEADS = 2
    DEPTH = 8
    TYPE = 'C'          # 'C' 代表自定義數據集（2類，1通道）
    validate_ratio = 0.1 # split raw train dataset into real train dataset and validate dataset
    BATCH_SIZE = 512
    EEGNet1_F1 = 8
    EEGNet1_KERNEL_SIZE=64
    EEGNet1_D=2
    EEGNet1_POOL_SIZE1 = 8
    EEGNet1_POOL_SIZE2 = 8
    FLATTEN_EEGNet1 = 240  # 對於1通道：15 (時間patches) * 16 (f1*D = 8*2) = 240

    if EVALUATE_MODE!='LOSO':
        EEGNet1_DROPOUT_RATE = 0.5
    else:
        EEGNet1_DROPOUT_RATE = 0.25    

    
    parameters_list = [0]
    for i in parameters_list:
        number_class, number_channel = numberClassChannel(TYPE)
        RESULT_NAME = "Loso_{}_heads_{}_depth_{}_{}".format(TYPE, HEADS, DEPTH, i)
    
        sModel = EEGTransformer(
            heads=HEADS, 
            emb_size=EMB_DIM,
            depth=DEPTH, 
            database_type=TYPE,
            eeg1_f1=EEGNet1_F1, 
            eeg1_D=EEGNet1_D,
            eeg1_kernel_size=EEGNet1_KERNEL_SIZE,
            eeg1_pooling_size1 = EEGNet1_POOL_SIZE1,
            eeg1_pooling_size2 = EEGNet1_POOL_SIZE2,
            eeg1_dropout_rate = EEGNet1_DROPOUT_RATE,
            eeg1_number_channel = number_channel,
            flatten_eeg1 = FLATTEN_EEGNet1,  
            ).cuda()
        summary(sModel, (1, number_channel, 1000)) 
    
        print(time.asctime(time.localtime(time.time())))
        
        result = main(RESULT_NAME,
                        evaluate_mode = EVALUATE_MODE,
                        heads=HEADS, 
                        emb_size=EMB_DIM,
                        depth=DEPTH, 
                        dataset_type=TYPE,
                        eeg1_f1 = EEGNet1_F1,
                        eeg1_kernel_size = EEGNet1_KERNEL_SIZE,
                        eeg1_D = EEGNet1_D,
                        eeg1_pooling_size1 = EEGNet1_POOL_SIZE1,
                        eeg1_pooling_size2 = EEGNet1_POOL_SIZE2,
                        eeg1_dropout_rate = EEGNet1_DROPOUT_RATE,
                        flatten_eeg1 = FLATTEN_EEGNet1,
                        validate_ratio = validate_ratio,
                        batch_size = BATCH_SIZE
                      )
        print(time.asctime(time.localtime(time.time())))