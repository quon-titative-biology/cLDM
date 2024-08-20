# ZebraFish-Diffusion-Model
Repository for the Zebrafish LDM Pipeline

This is an instructional guide as to how to run the Zebrafish Diffusion model pipeline.
This guide will show you how to format and store your data in the correct directories to preprocess, train, and evaluate the model.
For starters you’ll need to set up a conda environment with the required packaged and python version to run the model.

1. Run the command in your terminal
```
conda env create -f environment.yml
```
3. Run Preprocessing Python Script with your PATHS:
```
python Zebrafish_Segment_Anything --DATA_PATH= --META_PATH= —SAM_PATH= --OUT_PATH
```
*This will output the cropped and centered (x, 3, 200,950) numpy file and associated .csv metadata for the following steps (Specifically for Angle 1)*

(optional) Train VAE:
```

```
3. Generate Embeddings:
   
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
 
