# ZebraFish-Diffusion-Model
Repository for the Zebrafish LDM Pipeline

This is an instructional guide as to how to run the Zebrafish Diffusion model pipeline.
This guide will show you how to format and store your data in the correct directories to preprocess, train, and evaluate the model.
For starters you’ll need to set up a conda environment with the required packaged and python version to run the model.

Downloading Required Models:
```
Download SAM model weights  wget https://huggingface.co/HCMUE-Research/SAM-vit-h/resolve/main/sam_vit_h_4b8939.pth
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
conda env create -f 
python Zebrafish-LDM/ZebraFish_Segment_Anything.py --DATA_PATH= ./Data/ --META_PATH= ./Zebrafish_LDM/example.xlsx —SAM_PATH= ./models/sam_vit_h_4b8939.pth --OUT_PATH=./outputs
python Zebrafish-LDM/vae_embed.py --DATA_PATH=./outputs/Example_images.npy --META_PATH=./outputs/Example_metadata.csv --VAE_PATH=./VAE_results/ --MODEL_PATH=./models/
python Zebrafish-LDM/Zebrafish_Perturbation.py --run_name=LDM_NOGUIDE_TESTANDTRAIN --noise_steps=350 --epochs=2000 --device='cuda' --CONVAE_PATH=./models/rvae_1_ckpt_angle1_48.pth --DATA_PATH=./VAE_results/model_ckpt/embedding_48_new.pt --OUT_PATH=.Zebrafish_LDM/LDM --META_PATH=./outputs/Example_metadata.csv --PERTURBATION_PLATE='3_2021.11.15_hydinKO'
```


