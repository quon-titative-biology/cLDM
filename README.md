# ZebraFish-Diffusion-Model
Repository for the Zebrafish LDM Pipeline

This is an instructional guide as to how to run the Zebrafish Diffusion model pipeline.
This guide will show you how to format and store your data in the correct directories to preprocess, train, and evaluate the model.
For starters you’ll need to set up a conda environment with the required packaged and python version to run the model.

Downloading Required Data:
```
!wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth


```

1. Run the command in your terminal
```
conda env create -f ZebraFish_LDM.yaml
```
3. Run Preprocessing Python Script with your PATHS:
```
python Zebrafish_Segment_Anything --DATA_PATH= --META_PATH= —SAM_PATH= --OUT_PATH=
```
*This will output the cropped and centered (x, 3, 200,950) numpy file and associated .csv metadata for the following steps (Specifically for Angle 1)*

(optional) Train VAE:
```
python vae_training.py --DATA_PATH= --META_PATH= --VAE_PATH=
```
3. Generate Embeddings:
```
python vae_embed.py --DATA_PATH= --META_PATH= --VAE_PATH= --MODEL_PATH=
```
4. Train Diffusion Model:
```
python Zebrafish_LDM_Training.py --run_name= --noise_steps= --epochs= --device= --CONVAE_PATH= --DATA_PATH= --OUT_PATH= --META_PATH=
```
6. Perturb Images to selected plate:
```
python Zebrafish_Perturbation.py --run_name= --noise_steps= --epochs= --device= --CONVAE_PATH= --DATA_PATH= --OUT_PATH= --META_PATH= --PERTURBATION_PLATE=
```
7. Plot FID Heatmap
```
python Zebrafish_FID_plot.py --FID_PATH= --MUTANT_COLOR_MAP_PATH= --PLATE_COLOR_MAP_PATH=
```
THINGS TO NOTE:
The Plotting aspect of the pipeline still needs some updates to clean up the visualiation
Ill keep updating the code with any changes to improve runtime since this can take quite a while even on GPU's
