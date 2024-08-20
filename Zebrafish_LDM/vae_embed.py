import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torchvision.utils import save_image
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
import numpy as np
import pandas as pd
#import seaborn as pd
import matplotlib.pyplot as plt
import os, sys, datetime, glob, argparse, copy, importlib, csv
from util import *
############################################### 0. Load the data ###############################################
#Argument Parser
config = SimpleNamespace(
    DATA_PATH = '/group/gquongrp/workspaces/rmvaldar/Zebrafish/files/',
    META_PATH = '/group/gquongrp/workspaces/rmvaldar/Zebrafish/Morphometric_data-2.xlsx',
    VAE_PATH = '/group/gquongrp/workspaces/rmvaldar/VAE_results/',
    MODEL_PATH = '/group/gquongrp/workspaces/hongruhu/fishDiffusion/48_results_mean_loss/',
    )
####################  Set attributes of configs  ################################
def parse_args(config):
    parser = argparse.ArgumentParser(description='Loading Datapaths')
    parser.add_argument('--DATA_PATH', type=str, default=config.DATA_PATH, help='Path to Folders Storing JSON/TIFF Data')
    parser.add_argument('--META_PATH', type = str, default= config.META_PATH, help = 'Path to .xlsx Meta Data')
    parser.add_argument('--VAE_PATH',type = str, default= config.VAE_PATH, help = 'Path to save VAE Results')
    parser.add_argument('--MODEL_PATH',type = str, default= config.MODEL_PATH, help = 'Path to saved VAE Model')
    args = vars(parser.parse_args())
    # update config with parsed args
    for k, v in args.items():
        setattr(config, k, v)

parse_args(config)
full_imgs = np.load(config.DATA_PATH)
full_csv =  pd.read_csv(config.META_PATH, index_col=0)

#Fetching indices
Train_indices = full_csv.index[full_csv['Train_Or_Test'] == 'TRAIN'].tolist()
Valid_indices = full_csv.index[full_csv['Train_Or_Test'] == 'VALID'].tolist()
Test_indices = full_csv.index[full_csv['Train_Or_Test'] == 'TEST'].tolist()

ymap = {'pdzk1KO': 0,
 'hydinKO': 1,
 'arhgap11KO': 2,
 'scrambled': 3,
 'uninjected': 4,
 'gpr89KO': 5,
 'npy4rKO': 6,
 'frmpd2KO': 7,
 'fam72KO': 8,
 'ptpn20KO': 9,
 'ncf1KO': 10,
 'arhgap11MO': 11,
 'controlMO': 12,
 'ARHGAP11A': 13,
 'ARHGAP11B': 14,
 'srgap2KO': 15,
 'SRGAP2A': 16,
 'SRGAP2C': 17,
 'eGFP': 18,
 'GPR89B': 19,
 'NPY4R': 20,
 'FAM72B': 21,
 'FRMPD2A': 22,
 'FRMPD2B': 23,
 'NCF1A': 24,
 'NCF1C': 25,
 'PDZK1': 26,
 'PDZK1P1': 27,
 'PTPN20': 28,
 'PTPN20CP': 29}

#Stacking sets
data_train = np.stack([full_imgs[i] for i in Train_indices],axis = 0).astype(np.uint8)
data_val = np.stack([full_imgs[i] for i in Valid_indices],axis = 0).astype(np.uint8)
data_test = np.stack([full_imgs[i] for i in Test_indices],axis = 0).astype(np.uint8)

meta_train = np.stack([ymap[full_csv.iloc[i , full_csv.columns.get_loc("Label")]] for i in Train_indices], axis=0)
meta_val = np.stack([ymap[full_csv.iloc[i , full_csv.columns.get_loc("Label")]] for i in Valid_indices], axis=0)

print(data_train.shape)
print(data_val.shape)

print(meta_train.shape)
print(meta_val.shape)

sample_train, height, width, channels = data_train.shape
sample_val, _, _, _ = data_val.shape
target_size = [203, 794]
pad_height = int(target_size[0]-height)
crop_width = int((width-target_size[1])/2)
print(pad_height, crop_width)
# 3 78
print(data_train[:,:,0:crop_width,:].shape, data_train[:,:,-crop_width:,:].shape)
# (2928, 200, 78, 3) (2928, 200, 78, 3)
print(data_train[:,:,0:crop_width,:].max(), data_train[:,:,-crop_width:,:].max())
# 0 0
data_train, data_val,data_test = data_train[:,:,crop_width:-crop_width,:], data_val[:,:,crop_width:-crop_width,:],data_test[:,:,crop_width:-crop_width,:]
print(data_train.shape, data_val.shape)
# (2928, 200, 794, 3) (192, 200, 794, 3)
sample_train, height, width, channels = data_train.shape
sample_val, _, _, _ = data_val.shape
sample_test, _, _, _ = data_test.shape
new_arr_train = np.zeros((sample_train, height+pad_height, width, channels), dtype = np.uint8)
new_arr_val = np.zeros((sample_val, height+pad_height, width, channels), dtype = np.uint8)
new_arr_test = np.zeros((sample_test, height+pad_height, width, channels), dtype = np.uint8)

print(new_arr_train.shape, new_arr_val.shape, new_arr_test.shape)
# (2928, 203, 794, 3) (192, 203, 794, 3)
for i in range(sample_train):
    new_arr_train[i, 0:height, :, :] = data_train[i]


for i in range(sample_val):
    new_arr_val[i, 0:height, :, :] = data_val[i]

for i in range(sample_test):
    new_arr_test[i, 0:height, :, :] = data_test[i]

print(new_arr_train.shape, new_arr_val.shape, new_arr_test.shape)
# (2928, 203, 794, 3) (192, 203, 794, 3)
print(new_arr_train[:,-pad_height:,:,:].max(), new_arr_val[:,-pad_height:,:,:].max(), new_arr_test[:,-pad_height:,:,:].max())
# 0 0
data_transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda t: (t * 2) - 1)]) #, transforms.Resize((128, 128))])
transformed_images_train = [data_transform(image) for image in new_arr_train]
transformed_images_val = [data_transform(image) for image in new_arr_val]
transformed_images_test = [data_transform(image) for image in new_arr_test]

dataset_train = torch.stack(transformed_images_train)
dataset_val = torch.stack(transformed_images_val)
dataset_test = torch.stack(transformed_images_test)
dataset_train.shape # torch.Size([2928, 3, 203, 794])
dataset_val.shape   # torch.Size([192, 3, 203, 794])

############################################### Generating Embeddings ###############################################
from model48 import *
rvae = ResVAE()
data_set = dataset_train[0:5,:,:,:].cpu()
reconstruction, posterior = rvae(data_set)


trainData = dataset_train.cuda()
trainLabel = torch.FloatTensor(meta_train).cuda()
valData = dataset_val.cuda()
testData = dataset_test.cuda()
train_dataset = TensorDataset(trainData,trainLabel)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rvae = ResVAE()
rvae.to(device)


rvae_ckpt = copy.deepcopy(rvae.cpu())
rvae_ckpt.eval()
data_path = config.VAE_PATH
rvae_ckpt = torch.load(os.path.join(config.MODEL_PATH,'rvae_ckpt_angle1_48.pth'))
with torch.no_grad():
            _, train_posterior = rvae_ckpt(trainData.cpu())
            _, val_posterior = rvae_ckpt(valData.cpu())
            _, test_posterior = rvae_ckpt(testData.cpu())


train_embedding = train_posterior.mode()
val_embedding = val_posterior.mode()
test_embedding = test_posterior.mode()
print(train_embedding.shape, val_embedding.shape,test_embedding.shape)
# torch.Size([2928, 3, 48, 48]) torch.Size([192, 3, 48, 48])
torch.save(train_embedding, os.path.join(data_path,'model_ckpt/train_embedding_angle1_48_new.pt'))
torch.save(val_embedding, os.path.join(data_path,'model_ckpt/val_embedding_angle1_48_new.pt'))
torch.save(test_embedding, os.path.join(data_path,'model_ckpt/test_embedding_angle1_48_new.pt'))