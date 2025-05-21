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
from types import SimpleNamespace
#from util import *
############################################### 0. Load the data ###############################################
#Argument Parser
config = SimpleNamespace(
    DATA_PATH = './outputs/Example_images.npy',
    META_PATH = './outputs/Example_metadata.csv',
    VAE_PATH = './VAE_results/',
    MODEL_PATH = './models/',
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

#Loading in Processed Images
full_imgs = np.load(config.DATA_PATH)
full_csv =  pd.read_csv(config.META_PATH, index_col=0)


#Preprocessing Steps for the VAE
sample_count, height, width, channels = full_imgs.shape
target_size = [203, 794]
pad_height = int(target_size[0]-height)
crop_width = int((width-target_size[1])/2)


cropped_images= full_imgs[:,:,crop_width:-crop_width,:]
crppped_batch, cropped_height, cropped_width, channels = cropped_images.shape
new_arr = np.zeros((crppped_batch, cropped_height+pad_height, cropped_width, channels), dtype = np.uint8)


data_transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda t: (t * 2) - 1)]) #, transforms.Resize((128, 128))])
transformed_images_train = [data_transform(image) for image in new_arr]

image_dataset = torch.stack(transformed_images_train)


############################################### Generating Embeddings ###############################################
from model48 import *
rvae = ResVAE()
data_set = image_dataset[0:5,:,:,:].cpu()
reconstruction, posterior = rvae(data_set)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trainData = image_dataset.to(device)


rvae = ResVAE()
rvae.to(device)

rvae_ckpt = copy.deepcopy(rvae.cpu())
rvae_ckpt.eval()
data_path = config.VAE_PATH
rvae_ckpt = torch.load(os.path.join(config.MODEL_PATH,'rvae_1_ckpt_angle1_48.pth'))
with torch.no_grad():
            _, train_posterior = rvae_ckpt(trainData.cpu())


train_embedding = train_posterior.mode()
torch.save(train_embedding, os.path.join(data_path,'model_ckpt/embedding_48_new.pt'))
