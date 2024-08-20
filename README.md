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
Python Zebrafish_Segment_Anything_Preprocessing --DATA_PATH= --META_PATH= —SAM_PATH=
*This will output the cropped and centered (x, 3, 200,950) numpy file and associated .csv metadata for the following steps*

(optional) Train VAE:

3. Generate Embeddings:
   
4. Train Diffusion Model:

5. Perturb Images to selected plate:

6. Plot FID Heatmap

THINGS TO NOTE:
 
