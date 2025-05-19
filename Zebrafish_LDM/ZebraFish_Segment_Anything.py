#Main file that runs helpers to process images:
from Zebrafish_LDM.Zebrafish_Segment_Anything_utils import *
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor
import argparse, logging, copy
from types import SimpleNamespace

#Argument Parser
config = SimpleNamespace(
    DATA_PATH = '/group/gquongrp/workspaces/rmvaldar/ZebraFish-Diffusion-Model/Zebrafish_LDM/Data/',
    META_PATH = '/group/gquongrp/workspaces/rmvaldar/ZebraFish-Diffusion-Model/Zebrafish_LDM/example.xlsx',
    SAM_PATH = "/group/gquongrp/workspaces/rmvaldar/ZebraFish-Diffusion-Model/Zebrafish_LDM/models/sam_vit_h_4b8939.pth",
    OUT_PATH = '/group/gquongrp/workspaces/rmvaldar/ZebraFish-Diffusion-Model/Zebrafish_LDM/outputs/')

####################  Set attributes of configs  ################################
def parse_args(config):
    parser = argparse.ArgumentParser(description='Loading Datapaths')
    parser.add_argument('--DATA_PATH', type=str, default=config.DATA_PATH, help='Path to Folders Storing JSON/TIFF Data')
    parser.add_argument('--META_PATH', type = str, default= config.META_PATH, help = 'Path to .xlsx Meta Data')
    parser.add_argument('--SAM_PATH',type = str, default= config.SAM_PATH, help = 'Path to Segment Anything Model')
    parser.add_argument('--OUT_PATH',type = str, default= config.OUT_PATH, help = 'Path to output Directory')
    args = vars(parser.parse_args())
    # update config with parsed args
    for k, v in args.items():
        setattr(config, k, v)

parse_args(config)

#=====================================================#
#LOADING IMAGE AND MORPHOMETRIX .XLSX
#=====================================================#
BASE_DIR = config.DATA_PATH
xls= pd.ExcelFile(config.META_PATH)
df_ordered = pd.read_excel(xls)

IMG_DICT, JSON_DICT = Load_Dictionaries(BASE_DIR,df_ordered)

#=====================================================#
#LOADING SAM PREDICTOR
#=====================================================#
sam_checkpoint = config.SAM_PATH
model_type = "vit_h"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

PREDICTOR = SamPredictor(sam)

#=====================================================#
#SEGMENTATION ON IMAGES USING SAM
#=====================================================#
DIR_TO_MASK = {}
for dir in IMG_DICT.keys():
    print(dir)
    DIR_TO_MASK[dir] = Segment_images(PREDICTOR,IMG_DICT[dir], JSON_DICT[dir])


#=====================================================#
#CROPPING IMAGES USING SAM MASKS
#=====================================================#
DIR_TO_CROPPED_IMG = {}
for dir in DIR_TO_MASK.keys():
    DIR_TO_CROPPED_IMG[dir] = Full_crop(IMG_DICT[dir], DIR_TO_MASK[dir])

IMG_DICT.clear(),JSON_DICT.clear
del IMG_DICT, JSON_DICT
#==============================================================#
#GENERATING METADATA AND COMPILING DATA INTO ONE BLOACK OF DATA
#==============================================================#
DIR_TO_PREPROCESSED_IMG = {}
DIR_TO_ISSUE_IMG = {}
for dir in DIR_TO_CROPPED_IMG:
    if dir[-1] == '1': 
        x = np.stack([img for img in DIR_TO_CROPPED_IMG[dir].values()])
        modified_x,indices = Process_imgs(x)
        DIR_TO_PREPROCESSED_IMG[dir] = modified_x
        DIR_TO_ISSUE_IMG[dir] = indices


#==============================================================#
#Creating Dataset and numpy array for Angle1
#==============================================================#
DIR_TO_PREPROCESSED_IMG[dir]
x1 = np.stack(list(DIR_TO_PREPROCESSED_IMG[dir]))
big_df = pd.concat(
    [
        create_Dataset(DIR_TO_MASK[dir], DIR_TO_CROPPED_IMG[dir]) 
        for dir in DIR_TO_CROPPED_IMG.keys() 
        if dir.endswith("1")
    ], 
    ignore_index=True
)
#==============================================================#
#
#==============================================================#
np.save(os.path.join(config.OUT_PATH,'Example_images.npy'),x1)
big_df.to_csv(os.path.join(config.OUT_PATH,'Example_metadata.csv'))