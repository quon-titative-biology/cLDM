# conda env create -f environment.yaml
# conda activate ldm
# pip install packaging==21.3
# pip install 'torchmetrics<0.8'

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
    DATA_PATH = '/group/gquongrp/workspaces/rmvaldar/Zebrafish/Zebrafish_Angle1_Imgs.npy',
    META_PATH = '/group/gquongrp/workspaces/rmvaldar/Zebrafish/full_train_test_TRAINANDTEST.csv',
    VAE_PATH = '/group/gquongrp/workspaces/rmvaldar/VAE_results/')
####################  Set attributes of configs  ################################
def parse_args(config):
    parser = argparse.ArgumentParser(description='Loading Datapaths')
    parser.add_argument('--DATA_PATH', type=str, default=config.DATA_PATH, help='Path to Folders Storing JSON/TIFF Data')
    parser.add_argument('--META_PATH', type = str, default= config.META_PATH, help = 'Path to .xlsx Meta Data')
    parser.add_argument('--VAE_PATH',type = str, default= config.VAE_PATH, help = 'Path to save VAE Results')
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
#Test_indices = full_csv.index[full_csv['Train_Or_Test'] == 'TEST'].tolist()

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
#data_test = np.stack([full_imgs[i] for i in Test_indices],axis = 0).astype(np.uint8)

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
data_train, data_val = data_train[:,:,crop_width:-crop_width,:], data_val[:,:,crop_width:-crop_width,:]#,data_test[:,:,crop_width:-crop_width,:]
print(data_train.shape, data_val.shape)
# (2928, 200, 794, 3) (192, 200, 794, 3)
sample_train, height, width, channels = data_train.shape
sample_val, _, _, _ = data_val.shape
#sample_test, _, _, _ = data_test.shape
new_arr_train = np.zeros((sample_train, height+pad_height, width, channels), dtype = np.uint8)
new_arr_val = np.zeros((sample_val, height+pad_height, width, channels), dtype = np.uint8)
#new_arr_test = np.zeros((sample_test, height+pad_height, width, channels), dtype = np.uint8)

print(new_arr_train.shape, new_arr_val.shape)#, new_arr_test.shape)
# (2928, 203, 794, 3) (192, 203, 794, 3)
for i in range(sample_train):
    new_arr_train[i, 0:height, :, :] = data_train[i]


for i in range(sample_val):
    new_arr_val[i, 0:height, :, :] = data_val[i]

#for i in range(sample_test):
#    new_arr_test[i, 0:height, :, :] = data_test[i]

print(new_arr_train.shape, new_arr_val.shape)#, new_arr_test.shape)
# (2928, 203, 794, 3) (192, 203, 794, 3)
print(new_arr_train[:,-pad_height:,:,:].max(), new_arr_val[:,-pad_height:,:,:].max())#, new_arr_test[:,-pad_height:,:,:].max())
# 0 0
data_transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda t: (t * 2) - 1)]) #, transforms.Resize((128, 128))])
transformed_images_train = [data_transform(image) for image in new_arr_train]
transformed_images_val = [data_transform(image) for image in new_arr_val]
#transformed_images_test = [data_transform(image) for image in new_arr_test]

dataset_train = torch.stack(transformed_images_train)
dataset_val = torch.stack(transformed_images_val)
#dataset_test = torch.stack(transformed_images_test)
dataset_train.shape # torch.Size([2928, 3, 203, 794])
dataset_val.shape   # torch.Size([192, 3, 203, 794])



############################################### Train the model ###############################################
from model48 import *
rvae = ResVAE()
data_set = dataset_train[0:5,:,:,:].cpu()
reconstruction, posterior = rvae(data_set)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cuda'
trainData = dataset_train.to(device)
trainLabel = torch.FloatTensor(meta_train).to(device)
valData = dataset_val.to(device)

train_dataset = TensorDataset(trainData,trainLabel)

rvae = ResVAE()
rvae.to(device)
learning_rate = 1e-3
batch_size = 32
epochs = 200
min_val_loss = np.Inf
epochs_no_improve = 0
early_stop = False
patience = 25
SEED=34
setup_seed(SEED)
from torch.utils.data import DataLoader
DataLoader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
# rvae.apply(weights_init)
optimizer = torch.optim.Adam(rvae.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.99, last_epoch=-1)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
train_log ={"mse":[], "kld":[], "total_loss":[], "epoch":[]}
val_log ={"mse":[], "kld":[], "total_loss":[], "epoch":[]}
for epoch in range(epochs):
        kl_weight = 1e-6 # min(1, beta_increasing_rate*epoch)
        vae_training_mode(rvae)
        for idx, (x, _) in enumerate(DataLoader):  
            out, posterior = rvae(x)
            kl_divergence = posterior.kl().mean() # 0.5 * torch.sum(-1 - posterior.logvar + posterior.mean.pow(2) + posterior.logvar.exp())
            reconstruction_loss = F.mse_loss(out, x, reduction='mean')
            total_loss = reconstruction_loss + kl_divergence * kl_weight
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        vae_eval_mode(rvae)
        with torch.no_grad():
            tr_out, tr_posterior = rvae(trainData)
            tr_kl_divergence = tr_posterior.kl().mean() # 0.5 * torch.sum(-1 - posterior.logvar + posterior.mean.pow(2) + posterior.logvar.exp())
            tr_reconstruction_loss = F.mse_loss(tr_out, trainData, reduction='mean')
            tr_loss = tr_reconstruction_loss + tr_kl_divergence * kl_weight
            train_log["mse"].append(tr_reconstruction_loss.item())
            train_log["kld"].append(tr_kl_divergence.item())
            train_log["total_loss"].append(tr_loss.item())
            train_log["epoch"].append(epoch)
            print("Epoch [{}/{}], Train     : Total Loss: {:.4f}, MSE: {:.4f}, KLD: {:.4f}, KL_weight: {:.9f}".format(
                   epoch+1, epochs, 
                   tr_loss.item(), 
                   tr_reconstruction_loss.item(),
                   tr_kl_divergence.item(),
                   kl_weight))
        vae_eval_mode(rvae)
        with torch.no_grad():
            val_out, val_posterior = rvae(valData)
            val_kl_divergence = val_posterior.kl().mean() # 0.5 * torch.sum(-1 - posterior.logvar + posterior.mean.pow(2) + posterior.logvar.exp())
            val_reconstruction_loss = F.mse_loss(val_out, valData, reduction='mean')
            val_loss = val_reconstruction_loss + val_kl_divergence * kl_weight
            val_log["mse"].append(val_reconstruction_loss.item())
            val_log["kld"].append(val_kl_divergence.item())
            val_log["total_loss"].append(val_loss.item())
            val_log["epoch"].append(epoch)
            print("Epoch [{}/{}], Validation : Total Loss: {:.4f}, MSE: {:.4f}, KLD: {:.4f}, KL_weight: {:.9f}".format(
                   epoch+1, epochs, 
                   val_loss.item(), 
                   val_reconstruction_loss.item(),
                   val_kl_divergence.item(),
                   kl_weight))
        if optimizer.param_groups[0]['lr'] > 1e-4:
           scheduler.step()
           print(scheduler.get_last_lr())
        if epoch > 1:
           if val_loss.item() < min_val_loss:
              epochs_no_improve = 0
              min_val_loss = val_loss.item() * 0.99
            #   rvae_ckpt = copy.deepcopy(rvae)
           else:
              epochs_no_improve += 1
              print("Early stopping triggered!", patience - epochs_no_improve)
        else:
           continue
        if epoch > 1 and epochs_no_improve == patience:
           print("Stopped!")
           early_stop = True
           break
        else:
           continue


rvae_ckpt = copy.deepcopy(rvae.cpu())
rvae_ckpt.eval()
data_path = config.VAE_PATH
torch.save(rvae_ckpt, data_path + 'model_ckpt/rvae_ckpt_angle1_48_new.pth')
with torch.no_grad():
            _, train_posterior = rvae_ckpt(trainData.cpu())
            _, val_posterior = rvae_ckpt(valData.cpu())
            _, test_posterior = rvae_ckpt(testData.cpu())


train_embedding = train_posterior.mode()
val_embedding = val_posterior.mode()
test_embedding = test_posterior.mode()
print(train_embedding.shape, val_embedding.shape,test_embedding.shape)
# torch.Size([2928, 3, 48, 48]) torch.Size([192, 3, 48, 48])
torch.save(train_embedding, data_path + 'model_ckpt/train_embedding_angle1_48_new.pt')
torch.save(val_embedding, data_path + 'model_ckpt/val_embedding_angle1_48_new.pt')
torch.save(test_embedding, data_path + 'model_ckpt/test_embedding_angle1_48_new.pt')

train_embedding_array = train_embedding.numpy()
val_embedding_array = val_embedding.numpy()
test_embedding_array = test_embedding.numpy()
print(train_embedding.shape, val_embedding.shape,test_embedding.shape)
# torch.Size([2928, 3, 48, 48]) torch.Size([192, 3, 48, 48])
np.save(data_path + 'model_ckpt/train_embedding_angle1_48_new.npy', train_embedding_array)
np.save(data_path + 'model_ckpt/val_embedding_angle1_48_new.npy', val_embedding_array)
np.save(data_path + 'model_ckpt/test_embedding_angle1_48_new.npy', test_embedding_array)

# # Plotting loss curve
fig = plt.figure(figsize=(10, 5))
plt.plot(train_log["epoch"], train_log["mse"], label="mse")
plt.plot(train_log["epoch"], train_log["kld"], label="kld")
plt.plot(train_log["epoch"], train_log["total_loss"], label="loss")
plt.legend()
plt.show()
fig.savefig(data_path + 'train_loss.png')
fig = plt.figure(figsize=(10, 5))
plt.plot(val_log["epoch"], val_log["mse"], label="mse")
plt.plot(val_log["epoch"], val_log["kld"], label="kld")
plt.plot(val_log["epoch"], val_log["total_loss"], label="loss")
plt.legend()
plt.show()
fig.savefig(data_path + 'val_loss.png')



##############################################################################################
trainData = trainData.cpu()
valData = valData.cpu()
data_path = VAE_PATH
torch.save(rvae_ckpt, data_path + 'rvae_ckpt_angle1_48_new.pth')
# model_path = '/group/gquongrp/workspaces/hongruhu/fishDiffusion/48_results_mean_loss/'
# rvae_ckpt = torch.load(model_path + 'rvae_ckpt_angle1_48.pth')
vae_eval_mode(rvae_ckpt)
with torch.no_grad():
            val_out, val_posterior = rvae_ckpt(valData)
            val_kl_divergence = val_posterior.kl().sum() # 0.5 * torch.sum(-1 - posterior.logvar + posterior.mean.pow(2) + posterior.logvar.exp())
            val_reconstruction_loss = F.mse_loss(val_out, valData, reduction='sum')
            val_loss = val_reconstruction_loss + val_kl_divergence * kl_weight
            tr_out, tr_posterior = rvae_ckpt(trainData)
            tr_kl_divergence = tr_posterior.kl().sum() # 0.5 * torch.sum(-1 - posterior.logvar + posterior.mean.pow(2) + posterior.logvar.exp())
            tr_reconstruction_loss = F.mse_loss(tr_out, trainData, reduction='sum')
            tr_loss = tr_reconstruction_loss + tr_kl_divergence * kl_weight


print(tr_loss.item()/trainData.shape[0], val_loss.item()/valData.shape[0])
print(tr_reconstruction_loss.item()/trainData.shape[0], val_reconstruction_loss.item()/valData.shape[0])
print(tr_kl_divergence.item()/trainData.shape[0], val_kl_divergence.item()/valData.shape[0])
# "mean" loss
# 1926.4129098360656 1869.5231119791667 | 1925.7942281420765 1869.4816080729167
# 1926.410519125683 1869.5208333333333  | 1925.791837431694 1869.4793294270833
# 2386.3584357923496 2357.12158203125   | 2386.3584357923496 2357.12158203125
# "sum" loss
# 1265.5669398907103 1219.4440104166667
# 1265.5443989071039 1219.421630859375
# 22536.008196721312 22396.005208333332

# Plotting reconstruction
rvae_ckpt.eval()
with torch.no_grad():
    out, posterior = rvae_ckpt(trainData[0:10,:,:,:])
    out = out.cpu()


reverse_transform = transforms.Compose([Lambda(lambda t: (t + 1) / 2),
                                        Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
                                        Lambda(lambda t: t * 255.),
                                        Lambda(lambda t: t.numpy().astype(np.uint8)),
                                        ToPILImage()])
for i in range(10):
    plt.figure(figsize=(9, 7))
    plt.subplot(2, 1, 1)
    plt.imshow(np.asarray(reverse_transform(dataset_train[i].squeeze())))
    plt.subplot(2, 1, 2)
    plt.imshow(np.asarray(reverse_transform(out[i].squeeze())))
    plt.show()
    plt.savefig(data_path + 'reconstruction_' + str(i) + '.png')


with torch.no_grad():
     out_ = rvae_ckpt.final_layer(rvae_ckpt.decoder(rvae_ckpt.conv_in(rvae_ckpt.post_quant_conv(posterior.mode()))))


out_.shape # torch.Size([10, 3, 203, 794])
for i in range(10):
    plt.figure(figsize=(9, 9))
    plt.subplot(3, 1, 1)
    plt.imshow(np.asarray(reverse_transform(dataset_train[i].squeeze())))
    plt.subplot(3, 1, 2)
    plt.imshow(np.asarray(reverse_transform(out[i].squeeze())))
    plt.subplot(3, 1, 3)
    plt.imshow(np.asarray(reverse_transform(out_[i].squeeze())))
    plt.show()
    plt.savefig(data_path + 'reconstruction_w_sample_' + str(i) + '.png')


latent = posterior.sample()
latent.shape # torch.Size([10, 3, 48, 48])
reverse_transform_latent = transforms.Compose([Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
                                               Lambda(lambda t: t.numpy().astype(np.uint8)),
                                               ToPILImage(),])
for i in range(10):
    plt.figure(figsize=(4, 4))
    plt.imshow(np.asarray(reverse_transform_latent(latent[i].squeeze())))
    plt.show()
    plt.tight_layout()
    plt.savefig(data_path + 'reconstruction_latent_' + str(i) + '.png')


# Plotting reconstruction (validation)
rvae.eval()
with torch.no_grad():
    out_val, posterior = rvae_ckpt(valData[-10:,:,:,:])


in_val = valData[-10:,:,:,:].cpu()
for i in range(10):
    plt.figure(figsize=(9, 7))
    plt.subplot(2, 1, 1)
    plt.imshow(np.asarray(reverse_transform(in_val[i].squeeze())))
    plt.subplot(2, 1, 2)
    plt.imshow(np.asarray(reverse_transform(out_val[i].squeeze())))
    plt.show()
    plt.tight_layout()
    plt.savefig(data_path + 'reconstruction_held_out_' + str(i) + '.png')


mean, logvar = posterior.mean, posterior.logvar
std = torch.exp(0.5 * logvar)
print(mean.shape, std.shape)
# torch.Size([10, 3, 48, 48]) torch.Size([10, 3, 48, 48])
zero_mean = torch.zeros(mean.shape)
unit_std = torch.ones(std.shape)
print(zero_mean.shape, unit_std.shape)
# torch.Size([10, 3, 48, 48]) torch.Size([10, 3, 48, 48])
random_samping = zero_mean + unit_std * torch.randn(zero_mean.shape)
random_samping.shape # torch.Size([10, 3, 48, 48])
with torch.no_grad():
     random = rvae_ckpt.final_layer(rvae_ckpt.decoder(rvae_ckpt.conv_in(rvae_ckpt.post_quant_conv(random_samping))))


random.shape # torch.Size([10, 3, 203, 794])
for i in range(10):
    plt.figure(figsize=(9, 9))
    plt.subplot(3, 1, 1)
    plt.imshow(np.asarray(reverse_transform(dataset_train[i].squeeze())))
    plt.subplot(3, 1, 2)
    plt.imshow(np.asarray(reverse_transform(out[i].squeeze())))
    plt.subplot(3, 1, 3)
    plt.imshow(np.asarray(reverse_transform(random[i].squeeze())))
    plt.show()
    plt.tight_layout()
    plt.savefig(data_path + 'purely_samping_' + str(i) + '.png')