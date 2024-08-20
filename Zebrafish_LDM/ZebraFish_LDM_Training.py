import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torchvision.utils import save_image
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from torch import optim
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse, os, sys, datetime, glob, importlib, csv
from torch import Tensor
import argparse
import torchvision
from tqdm import tqdm
from PIL import Image
import pandas as pd
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split
import datetime;
import gc
import argparse, logging, copy
from model48 import *
from Unet_modules_new import *
#from diff_modules import *
from util import *
from random import randint
from torch import linalg as LA

torch.cuda.empty_cache() 

IMG_SIZE = 48
BATCH_SIZE = 128

#Hyperparamater configs
config = SimpleNamespace(
    run_name = "LDM_NOGUIDE",
    epochs = 2000,
    noise_steps=350,
    seed = 42,
    batch_size = BATCH_SIZE,
    img_size = IMG_SIZE ,
    device = 'cuda',
    lr = 1e-4,
    loss_opt = 'mse',
    USE_GUIDE = False,
    DROP = 0.1,
    scheduler = 'linear',
    unet_depth = 4,
    class_embed = 'ADD',
    drop_stragegy = 'origin',
    num_heads = 4,
    CONVAE_PATH = '/group/gquongrp/workspaces/rmvaldar/Zebrafish/Vae_results/rvae_ckpt_angle1_48.pth',
    DATA_PATH= '/group/gquongrp/workspaces/rmvaldar/Zebrafish/Vae_results/model_ckpt',
    OUT_PATH = '/group/gquongrp/workspaces/rmvaldar/Zebrafish/LDM',
    META_PATH = '/group/gquongrp/workspaces/rmvaldar/Zebrafish/full_train_test.csv',
    lambda_ = 0.5)

def parse_args(config):
    parser = argparse.ArgumentParser(description='Process hyper-parameters')
    parser.add_argument('--run_name', type=str, default=config.run_name, help='name of the run')
    parser.add_argument('--noise_steps', type=int, default=config.noise_steps, help='T sampling in diffusion process')
    parser.add_argument('--epochs', type=int, default=config.epochs, help='number of epochs')
    parser.add_argument('--seed', type=int, default=config.seed, help='random seed')
    parser.add_argument('--batch_size', type=int, default=config.batch_size, help='batch size')
    parser.add_argument('--img_size', type=int, default=config.img_size, help='image size')
    parser.add_argument('--device', type=str, default=config.device, help='device')
    parser.add_argument('--lr', type=float, default=config.lr, help='learning rate')
    parser.add_argument('--loss_opt', type=str, default=config.loss_opt, help='MSE of Huber')
    parser.add_argument('--USE_GUIDE', type=str2bool, default=config.USE_GUIDE, help='Whether using guidance')
    parser.add_argument('--DROP', type=float, default=config.DROP, help='drop probability for unconditional training')
    parser.add_argument('--unet_depth', type = int, default=4, help='number of layers in Unet')
    parser.add_argument('--class_embed', type = str, default=config.class_embed, help='using additive or concatenation')
    parser.add_argument('--drop_stragegy', type = str, default=config.drop_stragegy, help='hybrid dropping mutants only')
    parser.add_argument('--num_heads',type = int, default=config.num_heads, help = 'number of heads for attention layer')
    parser.add_argument('--CONVAE_PATH', type=str, default=config.CONVAE_PATH, help='conVAE model path ')
    parser.add_argument('--DATA_PATH', type=str, default=config.DATA_PATH, help='path to dataset')
    parser.add_argument('--OUT_PATH', type = str, default= config.OUT_PATH, help = 'path to out directory' )
    parser.add_argument('--META_PATH',type = str, default= config.META_PATH, help = 'zebrafish metadata csv files')
    parser.add_argument('--LAMBDA',type = float, default= config.lambda_)
    args = vars(parser.parse_args())
    # update config with parsed args
    for k, v in args.items():
        setattr(config, k, v)

parse_args(config)
set_seed(config.seed)
############# To save memory, perform LDP on embedding from VAE ################
pd.set_option("display.max_rows", 999)
pd.set_option("display.max_columns", 999)

# Relative path
def extend_config(config):
    #config.model_id= '-'.join([f'{config.run_name}', 'T',f'{config.noise_steps}', 'BATCH',f'{config.batch_size}','LOSS',f'{config.loss_opt}'])
    #config.model_id= '-'.join([f'{config.run_name}', 'T',f'{config.noise_steps}', 'BATCH',f'{config.batch_size}','LOSS',f'{config.loss_opt}','drop',f'{config.DROP}'])
    config.model_id= '-'.join([f'{config.run_name}',
                               'T',f'{config.noise_steps}',
                               'BATCH',f'{config.batch_size}',
                               'LOSS',f'{config.loss_opt}',
                               'drop',f'{config.DROP}',
                               'schedule',f'{config.scheduler}',
                               'unet_depth',f'{config.unet_depth}']) # try without stragety
    config.MODEL_PATH = os.path.join("saved_model", config.run_name, config.model_id)
    config.imed_plot_save = os.path.join(config.OUT_PATH,"train_result", config.run_name, config.model_id)
    config.profile_epoch = 100
    if config.unet_depth == 3:
        config.unet_channels = [64, 128, 128]
    if config.unet_depth == 4:
        config.unet_channels = [64, 128, 256, 256]
    elif config.unet_depth == 5:
        config.unet_channels = [64, 128, 256, 512, 512]
    # elif config.unet_depth == 6:
    #     config.unet_channels = [64, 128, 256, 512, 1024, 1024]
    config.unet_id = '-'.join([str(i) for i in config.unet_channels])
    config.CONFIG_PATH = os.path.join(config.OUT_PATH, 'config')
    os.makedirs(config.MODEL_PATH, exist_ok = True)
    os.makedirs(config.CONFIG_PATH, exist_ok = True)
    os.makedirs(config.imed_plot_save,exist_ok=True)
    return(config)


config = extend_config(config)
config_dict = vars(config)
np.save( os.path.join(config.CONFIG_PATH, config.model_id +  '.npy'),config_dict)


# Load all data
################################################################################
#  1.  Meta data preprocessing
#############s###################################################################
# Image metadata
image_meta = pd.read_csv(config.META_PATH)


# transform age, mutants and plate to be dummy variable:
dummie_meta_key= ['Age', 'Date','Label']
dummie_val= [pd.get_dummies(pd.DataFrame(image_meta[meta],dtype="category")).values.argmax(1) for meta in dummie_meta_key]
meta_dict = dict(zip(dummie_meta_key,dummie_val));
meta_dict = {key:torch.tensor(meta_dict[key],dtype=torch.int64) for key in meta_dict}


# generate one to one mapping between onehot encoding and class label
dummy_to_meta = [] # has columns to be dummy and meta
for dummie, meta in zip(dummie_val, dummie_meta_key):
    dummy_meta = pd.DataFrame({'dummy': dummie, 'meta': image_meta[meta]})
    dummy_to_meta.append(dummy_meta[~dummy_meta.duplicated()])


# Making the concatenation as a whole
age_plate_mut = image_meta['Age'].astype('str')+ '_' + image_meta['Date'].astype('str') + '_' +  image_meta['Label'].astype('str')
qury_age_pl_mut = pd.DataFrame(age_plate_mut.value_counts()).reindex()

################################################################################
# 2. embeddin loading:
################################################################################
train_data = torch.load(os.path.join(config.DATA_PATH,'train_embedding_angle1_48.pt'))
val_data =  torch.load(os.path.join(config.DATA_PATH, 'val_embedding_angle1_48.pt'))
test_data = torch.load(os.path.join(config.DATA_PATH,'test_embedding_angle1_48.pt'))
################################################################################
# 3. Match the meta feature to training and validation data
################################################################################
te_tr_val =  image_meta.groupby(['Train_Or_Test']).indices
train_index, val_index  = te_tr_val['TRAIN'], te_tr_val['VALID']#, test_index  = te_tr_val['TRAIN'], te_tr_val['VALID'], te_tr_val['TEST']

#Adding the testing train and validation to the training and validation
#train_data = train_data[train_index]
#val_data = val_data[trgt_fam_val_mask]

# build dataset loader
index_data_match = zip([train_data, val_data],[train_index, val_index])
train_dataloader, val_dataloader = [
    DataLoader(
        TensorDataset(data , meta_dict['Age'][index],  meta_dict['Date'][index], meta_dict['Label'][index]),
        batch_size=config.batch_size,
        shuffle=True) 
     for data, index in index_data_match ]

################################################################################
#  4. VAE architecture: Encode and Decode
################################################################################
rvae = torch.load(config.CONVAE_PATH)
rvae.to(config.device)
rvae.eval()

################################################################################
# Set up diffusion class
################################################################################
class DiffusionCond:
    #Conditional diffusion with classifier free guidence 
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256,guide = True, device="cuda", scheduler = 'linear'):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.scheduler = scheduler
        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        self.img_size = img_size
        self.device = device
        self.guide = guide
    def prepare_noise_schedule(self):
        if self.scheduler == 'linear':
            return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
        elif self.scheduler == 'cosine':
            #cosine schedule as proposed in https://arxiv.org/abs/2102.09672
            s = 0.008
            timesteps = self.noise_steps
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0.0001, 0.9999)
        elif self.scheduler == 'sigmoid':
            timesteps = self.noise_steps
            betas = torch.linspace(-6, 6, timesteps)
            return torch.sigmoid(betas) * (self.beta_end - self.beta_start) + self.beta_start
    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        #torch.seed(t)
        epsilon = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon
    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))
    def sample(self, model, n, labels,e=0, cfg_scale=3, latent = False, X_noise =None, mode='generate', seed=6743):
        #logging.info(f"Sampling {n} new images....")
        model.eval()
        torch.manual_seed(seed);
        saved_step = 0
        with torch.no_grad():
            #Check x and X_noise and see how similar they are
            if mode == 'generate':
                x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            elif mode == 'recon':
                x = X_noise # diffused from training
            # sample after fix initial point
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                # if with guidance
                if self.guide:
                    predicted_noise = self.from_guide(model, x, t, labels, cfg_scale)
                else:
                    predicted_noise = model(x, t, labels) #determine
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        if not latent:
            x = (x.clamp(-1, 1) + 1) / 2
            x = (x * 255).type(torch.uint8)
        return x
    def from_guide(self, model, x, t, labels, cfg_scale):
        predicted_noise = model(x, t, labels)
        if cfg_scale > 0:
            if config.drop_stragegy == 'origin':
                uncond_predicted_noise = model(x, t, None)
            elif config.drop_stragegy == 'hybrid_mut':
                # drop mutants only
                uncond_predicted_noise = model(x, t, [labels[0],labels[1],None])
            elif config.drop_stragegy == 'hybrid_plate':
                # drop mutants only
                uncond_predicted_noise = model(x, t, [labels[0],None,labels[2]])
            # classifier free guidance: mix the conditional and unconditional score estimates
            # decreasing the unconditional likelihood with a negative score term
            predicted_noise = (1 + cfg_scale) * predicted_noise - cfg_scale * uncond_predicted_noise
        return( predicted_noise )
    def recon(self, model, images, labels, cfg_scale=3, latent = True, seed=1234):
        torch.manual_seed(1)
        T = torch.randint(low=self.noise_steps-1, high=self.noise_steps, size=(images.shape[0],))
        T = T.to(self.device)
        # diffuse to the start time
        X_noise,noise = self.noise_images(images, T)
        # sample from diffused image:
        recon_image = self.sample(model = model,
                              n = images.shape[0],
                              labels = labels,
                              X_noise =X_noise,
                              latent = latent,
                              mode = 'recon',
                              cfg_scale=cfg_scale,
                              seed = seed)
        return([X_noise,recon_image])
def val_loss(model, dataloader, diffusion_cond, encode_model = None):
    with torch.no_grad():
        pbar = tqdm(dataloader)
        avg_val_loss = 0
        for i, data in enumerate(pbar):
            images, age, plate, mutant = [i.to(device) for i in data]
            #labels = labels.to(device)
            labels = [age, plate, mutant]
            if encode_model is not None:
                images = encode(model = encode_model, input_ = images)
            t = diffusion_cond.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion_cond.noise_images(images, t)
            ################################################################################
            # If there is guidance
            ################################################################################
            if diffusion_cond.guide:
                if np.random.random() < config.DROP:
                    if config.drop_stragegy == 'origin':
                        labels = None
                    elif config.drop_stragegy == 'hybrid_mut':
                        # drop mutants only
                        labels[2] = None
                    elif config.drop_stragegy == 'hybrid_plate':
                        # drop mutants only
                        labels[1] = None
                predicted_noise = model(x_t, t, labels)
            else:
                predicted_noise = model(x_t, t, labels)
        loss = mse(noise, predicted_noise)
        avg_val_loss += loss.item()
        return avg_val_loss
################################################################################
# Set up optimizer, learning_rate scheduler
################################################################################
device = config.device
ct = datetime.datetime.now()
mse = choose_loss(loss_opt = config.loss_opt)

# number of labels per class
multi_class = [np.unique(value).shape[0] for key, value in zip(dummie_meta_key,dummie_val)]

#
# # # Model set up

model_saved_path = os.path.join(config.MODEL_PATH, f"ckpt.pt")
if os.path.exists(model_saved_path):
    model = torch.load(model_saved_path)
    model.to(device)
else:
    model = UNet(c_in=3, # input image channel
                 c_out=3, # output image channel
                 time_dim=256, # time dimension
                 down_factor = 2, # sample factor used for maxpooling
                 num_heads = 4, # number of attention heads
                 kernel_size = 3, # kernel size
                 channels_down = config.unet_channels,# increase channel depth in Unet
                 num_classes = None,
                 multi_class = multi_class,
                 #class_embed = config.class_embed,
                 device=config.device).to(device)

        
diffusion_cond = DiffusionCond(noise_steps = config.noise_steps,
                               img_size=config.img_size,
                               guide = config.USE_GUIDE,
                               scheduler = config.scheduler,
                               device = config.device)

### optimizer and others set up
ema = EMA(0.998)
ema_model = copy.deepcopy(model).eval().requires_grad_(False)
optimizer = optim.AdamW(model.parameters(), lr=config.lr * 0.05)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.lr,
                                         steps_per_epoch = len(train_dataloader), epochs=config.epochs)
log ={"train_loss":[],"val_loss":[], "epoch":[]}

mutant_label = '3_2022.11.08_FAM72B'#
#age_plate_mut[3665] == mutant_label
sel_mutant_index = np.where(age_plate_mut[train_index] == mutant_label)#np.where(age_plate_mut == mutant_label)#np.where(age_plate_mut[train_index]== mutant_label)
sel_image_tr = train_data[sel_mutant_index].to(device)

#determine mutant class
age_pl_mut = mutant_label.split('_') # label to image class, age, plate, mutant
sample_class = [meta[meta['meta'].astype('str') == val]['dummy'].values
                        for meta,val in zip(dummy_to_meta, age_pl_mut)]
vae_recon = decode(model = rvae, input_ = sel_image_tr)
#torchclear.save(vae_recon, '/group/gquongrp/workspaces/rmvaldar/Zebrafish/Sampled_3_2022.11.08_FAM72B/vae_recon.pt')
save_images(tensorToImage(vae_recon), os.path.join(config.imed_plot_save ,'-'.join([mutant_label,"conVAE_recon.jpg"])))

#Pepare number of sampled image
NUM_SAMPLED_IMG = sel_image_tr.shape[0]
sample_class_ = [torch.Tensor(np.repeat(i, NUM_SAMPLED_IMG)).long().to(device) for i in sample_class]

# Early stop trigger
patience = 3
min_val_loss = np.Inf
epochs_no_improve = 0
early_stop = False
## training:

for epoch in range(config.epochs):
    print(f"Starting epoch {epoch}:")
    pbar = tqdm(train_dataloader)
    avg_tr_loss = 0
    avg_lst_step_loss = 0
    for i, data in enumerate(pbar):
        images, age, plate, mutant = [i.to(device) for i in data]
        #print(plate)
        #labels = [age, plate, mutant]
        labels = [age, plate,mutant]
        # RITA: labels = [age, plate]
        # sample t steps:
        t = diffusion_cond.sample_timesteps(images.shape[0]).to(device)
        #print(t)
        #print(torch.Tensor(int(config.noise_steps-1)).shape)
        x_t, noise = diffusion_cond.noise_images(images, t)
        #x_bigt_1, _ = diffusion_cond.noise_images(images,torch.full((images.shape[0],),config.noise_steps-1).to(device))
        if diffusion_cond.guide:
            if np.random.random() < config.DROP:
                if config.drop_stragegy == 'origin':
                    labels = None
                elif config.drop_stragegy == 'hybrid_mut':
                    # drop mutants only
                    labels[2] = None
                elif config.drop_stragegy == 'hybrid_plate':
                    # drop mutants only
                    labels[1] = None
            predicted_noise = model(x_t, t, labels)
            #last_predicted_noise = model(x_bigt_1,torch.full((images.shape[0],),config.noise_steps-1).to(device), labels)
        else:
            predicted_noise = model(x_t, t, labels)
            #last_predicted_noise = model(x_bigt_1,torch.full((images.shape[0],),config.noise_steps-1).to(device), labels)
            #NEW PREDICTED NOISE FROM x_t - 1
        #zero = torch.zeros(last_predicted_noise.shape).to(device)
        #last_step = mse(last_predicted_noise, zero)
        loss = mse(noise, predicted_noise) 
        #last_step_loss = mse(predicted_noise, zero)# ADD TIME STEP HERE
        avg_tr_loss += loss.item()
        # training step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema.step_ema(ema_model, model)
        scheduler.step()
    log['train_loss'].append(avg_tr_loss)
    #log['LastStep_loss'].append(avg_lst_step_loss)
    valloss = val_loss(model, dataloader = val_dataloader,  diffusion_cond = diffusion_cond, encode_model =None)
    log['val_loss'].append(valloss)
    log['epoch'].append(epoch)
    print('epoch {e}  train loss: {l} val loss {v}'.format(e=epoch, l=np.round(avg_tr_loss,2), v = np.round(log['val_loss'][-1],2)))
    ############################################################################
    # Loss profile and model save
    #############################################################################
    if epoch % 20 == 0 and epoch > 10:
        # loss profile
        try:
            loss_df = pd.DataFrame(log)
            #plot_loss = loss_df[5:1100].reset_index()
            #fig, axs = plt.subplots(1, 3)
            #sns.lineplot(data=plot_loss, x="epoch", y="train_loss",ax = axs[0])
            #sns.lineplot(data=plot_loss, x="epoch", y="val_loss",ax = axs[1])
            #fig.savefig(os.path.join(config.MODEL_PATH, 'loss_curve.png'))
            loss_df.to_csv(os.path.join(config.MODEL_PATH, 'loss.record.csv'))
        except:
            print('all array is not of the same length')
        # save model-
        torch.save(model, os.path.join(config.MODEL_PATH, f"ckpt.pt"))
        torch.save(ema_model, os.path.join(config.MODEL_PATH, f"ema_ckpt.pt"))
        torch.save(optimizer.state_dict(), os.path.join(config.MODEL_PATH, f"optim.pt"))
    ############################################################################
    # Evaluating FID during training: Don't focus too much figure
    ##############################################################################
    if epoch % config.profile_epoch == 0:
        # reconstruction
        # RITA: CHANGE the sample_class_ to only have 2 variables
        diffuse, denoise = diffusion_cond.recon(model, sel_image_tr, sample_class_, cfg_scale=3, latent = True)
        sampled_images = decode(model = rvae, input_ = denoise)
        recon_FID = computeFID(sampled_images, vae_recon)
        save_images(tensorToImage(sampled_images), os.path.join(config.imed_plot_save ,'-'.join([f"epoch{epoch}_FID{np.round(recon_FID,2)}",mutant_label,"LDM_recon.jpg"])))

        sampled_hidden = diffusion_cond.sample(model, e=epoch, cfg_scale=3, n=NUM_SAMPLED_IMG, labels=sample_class_,latent = True)
        new_sampled_images = decode(model = rvae, input_ = sampled_hidden)
        sample_FID = computeFID(new_sampled_images, vae_recon)
        save_images(tensorToImage(new_sampled_images), os.path.join(config.imed_plot_save ,'-'.join([f"epoch{epoch}_FID{np.round(sample_FID,2)}",mutant_label,"LDM_sampled_.jpg"])))