# ZebraFish-Diffusion-Model
Repository for the Zebrafish LDM Pipeline

This repository contains an implementation of Denoising Diffusion Probabilistic Models (DDPM). 
Its applications were used in the study:  Gene expansions contributing to human brain evolution by Daniela C Soto et al.
The directories are intended for the 


Downloading Required Models:
```
Download SAM model weights  wget https://huggingface.co/HCMUE-Research/SAM-vit-h/resolve/main/sam_vit_h_4b8939.pth
Download LDM model weights wget
```
Data File Stucture:
```
root/
|-- Zebrafish Batch Directory/
|   |-- Batch_1/
|       |-- Zebrafish_img.tiff
|       |-- Zebrafish_measurements.json


```
Running Example Process:
```
conda env create -f ZebraFish_LDM.yaml
python Zebrafish_LDM/ZebraFish_Segment_Anything.py --DATA_PATH=./Data/ --META_PATH=./Zebrafish_LDM/example.xlsx --SAM_PATH=./models/sam_vit_h_4b8939.pth --OUT_PATH=./outputs
python Zebrafish_LDM/vae_embed.py --DATA_PATH=./outputs/Example_images.npy --META_PATH=./outputs/Example_metadata.csv --VAE_PATH=./VAE_results/ --MODEL_PATH=./models/
python Zebrafish_LDM/Zebrafish_Perturbation.py --run_name=LDM_NOGUIDE_TESTANDTRAIN --noise_steps=350 --epochs=2000 --device='cpu' --CONVAE_PATH=./models/rvae_1_ckpt_angle1_48.pth --DATA_PATH=./VAE_results/model_ckpt/embedding_48_new.pt --OUT_PATH=.Zebrafish_LDM/LDM --META_PATH=./outputs/Example_metadata.csv --PERTURBATION_PLATE='3_2021.11.15_hydinKO'
```


