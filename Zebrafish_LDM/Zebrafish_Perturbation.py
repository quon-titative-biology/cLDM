from torch.autograd import Variable
import torch.utils.data
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
from FID import *
#from diff_modules import *
from util import *
import scipy.cluster as cl
import scipy.spatial as sp
torch.cuda.empty_cache()
import colorcet as cc
from matplotlib.patches import Rectangle
from torch.autograd import Variable
import torch.utils.data
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
from FID import *
#from diff_modules import *
from util import *
import scipy.cluster as cl
import scipy.spatial as sp
torch.cuda.empty_cache()
import colorcet as cc
from matplotlib.patches import Rectangle
IMG_SIZE = 48
BATCH_SIZE = 128





#Hyperparamater configs
config = SimpleNamespace(
    run_name = "LDM_NOGUIDE_PTO_PERGEN_HYDIN_1202",
    epochs = 2000,
    noise_steps=350,
    seed = 42,
    batch_size = BATCH_SIZE,
    img_size = IMG_SIZE ,
    device = 'cuda',#torch.device("cuda" if torch.cuda.is_available() else "cpu"),
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
    lambda_ = 0.5,
    PERTURBATION_PLATE= '3_2022.12.02_eGFP')

####################  Set attributes of configs  ################################
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
    parser.add_argument('--PERTURBATION_PLATE',type = str, default= config.PERTURBATION_PLATE, help = 'zebrafish metadata csv files')
    args = vars(parser.parse_args())
    # update config with parsed args
    for k, v in args.items():
        setattr(config, k, v)

parse_args(config)
set_seed(config.seed)
############# To save memory, perform LDP on embedding from VAE ################
pd.set_option("display.max_rows", 999)
pd.set_option("display.max_columns", 999)


def make_colormap(map_arr, data,df,palette):
    pal = sns.color_palette(palette=palette, n_colors=len(map_arr))
    lut = dict(zip(map(str, map_arr), pal))
    colors = pd.Series(data, df.iloc[:,0:].columns).map(lut)
    return lut, colors


def make_colormap_labels(data,df,dct):
    #Making a palette for each family
    #luts = []
    #for i in range(0,len(map_arr)):
    #    pal = sns.color_palette(palette=palettes[i], n_colors=len(map_arr[i]))
    #    lut = dict(zip(map(str, map_arr[i]), pal))
    #    luts.append(lut)
    #dct = {}
    #for item in luts:
    #    dct.update(item)
    colors = pd.Series(data, df.iloc[:,0:].columns).map(dct)
    return dct, colors

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

#Subset to one family and controls only
Family_groups = [['pdzk1KO','PDZK1P1','PDZK1'],
                ['ARHGAP11A', 'ARHGAP11B', 'arhgap11KO', 'arhgap11MO', 'controlMO'],
                ['ncf1KO','NCF1A','NCF1C'],
                ['srgap2KO', 'SRGAP2A', 'SRGAP2C'],
                ['FAM72B', 'fam72KO'],
                ['FRMPD2A', 'FRMPD2B', 'frmpd2KO'],
                ['ptpn20KO','PTPN20', 'PTPN20CP'],
                ['NPY4R','npy4rKO'],
                ['GPR89B', 'gpr89KO'],
                ['scrambled'],
                ['eGFP'],
                ['uninjected'],
                ['hydinKO']]

Control_family = ['scrambled','eGFP','uninjected']


#Changing each control thats not on a plate containing a mutant to common/ unique
age_plate_mut = image_meta['Age'].astype('str')+ '_' + image_meta['Date'].astype('str') + '_' +  image_meta['Label'].astype('str')
mutant_plates = np.unique(image_meta[image_meta['Label'].isin(Family_groups[12])].Date).tolist()
age_plate_mut = age_plate_mut[image_meta['Label'].isin(Control_family) & ~(image_meta['Date'].isin(mutant_plates))]
uniq_lb, uniq_lb_count = np.unique(age_plate_mut , return_counts = True) # This can be tested from tr, test, val
count = dict(zip(uniq_lb,uniq_lb_count))


#Actual subset process to chance control label to unique control label
i = 1
for plate in count:
    if  (plate !=config.PERTURBATION_PLATE):
        subset = image_meta.loc[(image_meta['Label'] == plate.split('_')[2]) & (image_meta['Date'] == plate.split('_')[1]), 'Label']
        subset.iloc[:len(subset)//2] = plate.split('_')[2] + str(i)
        i+= 1
        image_meta.loc[(image_meta['Label'] == plate.split('_')[2]) & (image_meta['Date'] == plate.split('_')[1]), 'Label']= subset

#Changing the labels to unique classes
encoding =  pd.read_csv('/group/gquongrp/workspaces/rmvaldar/Zebrafish/encoding.csv')
for ind in image_meta.index:
    Comb = "'"+ str(image_meta['Age'][ind]) + '_' + str(image_meta['Date'][ind]) + '_' + image_meta['Label'][ind] + "'"
    #print(Comb)
    if Comb in encoding.Mutant.tolist():
        image_meta['Label'][ind] = image_meta['Label'][ind] + str(encoding.loc[encoding['Mutant'] == Comb ]['Encoding'].item())



image_meta = image_meta.reset_index()

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
train_index, val_index = te_tr_val['TRAIN'], te_tr_val['VALID']#, te_tr_val['TEST']


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
    """ Conditional diffusion with classifier free guidence """
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
            """
            cosine schedule as proposed in https://arxiv.org/abs/2102.09672
            """
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
            for i in tqdm(reversed(range(0, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                # if with guidance
                if self.guide:
                    predicted_noise = self.from_guide(model, x, t, labels, cfg_scale)
                else:
                    predicted_noise = model(x, t, labels) #determine
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 0:
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
#path = '/group/gquongrp/workspaces/rmvaldar/Zebrafish/saved_model/LDM_GUIDE_ONLY_CONDITIONAL_FULL/LDM_GUIDE_ONLY_CONDITIONAL_FULL-T-350-BATCH-128-LOSS-mse-drop-0.1-schedule-linear-unet_depth-4/'
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


#Funciton to Generate sns Heatmap 
def generate_heatmap(matrix,name,class_names,uniq_data, full_data, singular_paletes,label_palettes,plate_plattes, df,order,date_order):
    #Age Side Bar
    Age_lut, Age_colors = make_colormap(uniq_data[0],full_data[0],df,singular_paletes[0])
    Label_lut, Label_colors = make_colormap_labels(full_data[1],df,label_palettes)
    Plate_lut, Plate_colors = make_colormap_labels(full_data[2],df,plate_plattes)
    Count_lut, Count_colors = make_colormap(uniq_data[2],full_data[3],df,singular_paletes[1])
    dist_metric = 'euclidean'
    linkage_method = 'ward'
    matrix_for_hc = sp.distance.pdist(matrix, dist_metric);
    Z = cl.hierarchy.linkage(matrix_for_hc,linkage_method)
    leaf_order = cl.hierarchy.leaves_list(Z);
    plot_mat = pd.DataFrame(matrix, columns = class_names, index = class_names)

    hx = sns.clustermap(plot_mat,
               figsize = (18,18),
               method = 'ward',
               cmap='Reds',
               col_cluster=True,
               row_cluster = True,
               row_colors=[Age_colors,Label_colors,Count_colors],
               col_colors=[Age_colors,Label_colors,Count_colors],
               col_linkage = Z,
               row_linkage = Z,
               xticklabels=1, yticklabels=1)
    hx.fig.suptitle(name)
    for age in list(set(full_data[0])):
        hx.ax_col_dendrogram.bar(0, 0, color=Age_lut[age],
                            label=age, linewidth=0)
    hx.ax_col_dendrogram.legend(loc="center left", ncol=5)
    for labels in order:
        hx.ax_row_dendrogram.bar(0, 0, color=Label_lut[labels],
                            label=labels, linewidth=0)
    hx.ax_row_dendrogram.legend(loc="center", ncol=2)
    for counts in uniq_data[2]:
        hx.ax_col_dendrogram.bar(0, 0, color=Count_lut[counts],
                            label=counts, linewidth=0)
    hx.ax_col_dendrogram.legend(loc="center left", ncol=2)
    plt.savefig(name +'.png')
    return hx, plot_mat

def generate_Dendo_Plot(matrix,name,class_names,uniq_data, full_data, singular_paletes,label_palettes,plate_plattes, df,order,date_order):
    #Age Side Bar
    Age_lut, Age_colors = make_colormap(uniq_data[0],full_data[0],df,singular_paletes[0])
    Label_lut, Label_colors = make_colormap_labels(full_data[1],df,label_palettes)
    Plate_lut, Plate_colors = make_colormap_labels(full_data[2],df,plate_plattes)
    Count_lut, Count_colors = make_colormap(uniq_data[2],full_data[3],df,singular_paletes[1])
    
    dist_metric = 'euclidean'
    linkage_method = 'ward'
    matrix_for_hc = sp.distance.pdist(matrix, dist_metric);
    Z = cl.hierarchy.linkage(matrix_for_hc,linkage_method)
    leaf_order = cl.hierarchy.leaves_list(Z);
    plot_mat = pd.DataFrame(matrix, columns = class_names, index = class_names)
    hx = sns.clustermap(plot_mat,
               figsize = (18,18),
               method = 'ward',
               cmap='Reds',
               col_cluster=True,
               row_cluster = False,
               row_colors=[Label_colors],
               #col_colors=[Age_colors,Label_colors,Count_colors],
               col_linkage = Z,
               row_linkage = Z,
               xticklabels=1, yticklabels=1)
    hx.fig.suptitle(name)
    hx.ax_heatmap.remove() 
    hx.cax.remove()  # remove the color bar
    print('got here')
    for labels in order:
        hx.ax_row_dendrogram.bar(0, 0, color=Label_lut[labels],
                            label=labels, linewidth=0)
    plt.savefig(name +'.png')
    return hx

#################################################################################
# predefine class to define reconstruction
#################################################################################
# pick a label:

CONTROL_SEL = config.PERTURBATION_PLATE.split('_')[2]; DATE = config.PERTURBATION_PLATE.split('_')[1]; AGE = config.PERTURBATION_PLATE.split('_')[0]
mutant_label = config.PERTURBATION_PLATE

age_sel = dummy_to_meta[0][dummy_to_meta[0]['meta'] == AGE]['dummy'].values
date_sel = dummy_to_meta[1][dummy_to_meta[1]['meta'] == DATE]['dummy'].values[0]


age_plate_mut_train = age_plate_mut[train_index]# test age_plate_mut
uniq_lb, uniq_lb_count = np.unique(age_plate_mut_train , return_counts = True) # This can be tested from tr, test, val
count = dict(zip(uniq_lb,uniq_lb_count))

sel_mutant_index = np.where(age_plate_mut[train_index] == mutant_label)#np.where(age_plate_mut == mutant_label)#np.where(age_plate_mut[train_index]== mutant_label)
sel_image_tr = train_data[sel_mutant_index].to(device)

#determine mutant class
age_pl_mut = mutant_label.split('_') # label to image class, age, plate, mutant
sample_class = [meta[meta['meta'].astype('str') == val]['dummy'].values
                        for meta,val in zip(dummy_to_meta, age_pl_mut)]


#################################################################################
# perturbation loop
#################################################################################
mutant_encoding = [(mut, mutclass) for index, [mut, mutclass] in enumerate(dummy_to_meta[2].values)]
dict_variable = {mutclass:mut for index, [mut, mutclass] in enumerate(dummy_to_meta[2].values)}

recon_batchs = []
for label in tqdm(uniq_lb):
    #original Imagesimed_plot_save
    sel_mutant_index = np.where(age_plate_mut[train_index]== label)

    sel_image_tr = train_data[sel_mutant_index].to(device)
    NUM_SAMPLED_IMG = sel_image_tr.shape[0]
    #Peturbing to each mutant to plate 3_2022.12.02
    sample_class = [age_sel, date_sel, dict_variable[label.split('_')[2]]]
    sample_class_array = [np.repeat(i, NUM_SAMPLED_IMG) for i in sample_class]
    sample_class_ = [torch.Tensor(i).long().to(device) for i in sample_class_array]
    #reconstruct and decode perturbed image
    diffuse, denoise = diffusion_cond.recon(model, sel_image_tr, sample_class_, cfg_scale=3, latent = True)

    sampled_images = decode(model = rvae, input_ = denoise)
    mutant_label = str(AGE) + '_' + DATE + '_' + label.split('_')[2]
    recon_batchs.append((label,sampled_images))

    #save_images(tensorToImage(sampled_images), os.path.join(config.imed_plot_save ,'-'.join([f"0",label,"_Perturbed_to_",mutant_label,"LDM_perturbed_recon.jpg"])))


#################################################################################
# Computing FID
#################################################################################
FID_perturb_dict = {}
FID_Cache = {}
for batch1 in tqdm(recon_batchs):
    FID_BATCH = []
    for batch2 in recon_batchs:
        try:
            FID = computeFID(batch1[1],batch2[1])
            FID_Cache[(batch2[0],batch1[0])] = FID.item()
            FID_BATCH.append(FID.item())
        except ValueError:
            print('error')
    FID_perturb_dict[batch1[0]] = FID_BATCH

#IF ERROR OCCURS 
for batch in FID_perturb_dict:
    if FID_perturb_dict[batch] == []:
        print(batch)


FID_perturb_df = pd.DataFrame(FID_perturb_dict, columns=[batch for batch in list(FID_perturb_dict.keys())], index=[batch for batch in list(FID_perturb_dict.keys())])
FID_perturb_df.to_csv(os.path.join(config.OUT_PATH,'FID_Perturbed_Image_LDM.csv'))