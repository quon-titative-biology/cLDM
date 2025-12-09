# ZebraFish Model Training Guide
This guide will navigate you through training the the VAE and cLDM respectively and highlight components which can be tweaked for optimal training.

** Prior to training VAE steps: **
- All images need to be properly processed through the `Zebrafish_SegmentAnything.py` script
- `TRAIN`, `VALID`, and `TEST` labels should be appeneded to the output `.csv` from the `Zebrafish_SegmentAnything.py` script in a column labeled `Train_Or_Test`
  
# Training VAE:
To train the VAE from the command line the script `vae_training.py` can be ran. This takes the following paramaters:
- `DATA_PATH`: Path to the processed images `.npy` output from the preprocessing script
- `META_PATH`: Path to the metadata `.csv` output from the preprocessing script
- `VAE_PATH`: Output path for the model weights as well as training, validation, and testing embedding `.pt` files used in the cLDM training
- `LR`: Learning rate parameter by default set to `1e-3`
- `BATCH_SIZE`: Batch size parameter by default set to 100
- `EPOCHS`: Epoch parameter by default set to 200

During training the model will output the following reuslts per epoch:
- `Train Total_Loss:` Training Sum of the reconstruction loss (MSE) and KL divergence loss for epoch
- `Train MSE`: Training mean square error for current epoch
- `Train kld`: Training Kl divengece for current epoch 

- `Validation Total_loss`: Validation Sum of the reconstruction loss (MSE) and KL divergence loss for current epoch
- `Validation MSE`: Validation mean square error for current epoch
- `Validation kld`: Validation Kl divengece for current epoch

Example command line prompt to run training script:
```
python Zebrafish_LDM/ZebraFish_Segment_Anything.py --DATA_PATH=./Data/ --META_PATH=./Zebrafish_LDM/example.xlsx --SAM_PATH=./models/sam_vit_h_4b8939.pth --OUT_PATH=./outputs
python Zebrafish_LDM/vae_training.py --DATA_PATH=./outputs/Example_images.npy --META_PATH=./outputs/Example_metadata.csv --VAE_PATH=./VAE_results/ --LR=1e-1 --BATCH_SIZE=100 --EPOCHS=200
```

# Training cLDM:
To train the cLDM from the command line the script `ZebraFish_LDM_Training.py` can be ran. This takes the following paramaters:
- `run_name`: Name of the process run
- `noise_steps`: number of diffusion time steps set to `350` by default
- `epochs`: Number of training epochs set to `2000` by default
- `seed`: Random seed for reproduciblity set to `42` by default
- `batch_size`: number of embeddings per training steps set to `128` by default
- `img_size`: Emebedding dimmension set to `48`
- `device`: hardware device used for training set to `cuda` by default
- `lr`: learning rate for training set to `1e-4` by default
- `loss_opt`: Loss function for training the model using `MSE`
- `USE_GUIDE`: Whether to use classifier-free guidance set to `False` by default
- `DROP`: Drop probability fo classifier free guidance
- `unet_depth`: Number of downsample/upsample blocks for model structure set to `4` by default
- `class_embed`: Using additive or concatenation for the class embeddings set to `ADD` by default
- `drop_strategy`: Drop strategy for sampling set to origin for default
- `num_heads`: Number of attention heads in the transformer blocks set to `4` by default
- `CONVAE_PATH`: Path to the saved VAE model `.pth`
- `DATA_PATH`: Path to the training, validation, and testing embeddings output from the `training_vae.py` script
- `OUT_PATH`: Output path where the mode is saved
- `META_PATH`: Meta data path for the `.csv` output from the `Zebrafish_SegmentAnything.py` script
- `LAMBDA`: weight for loss set to `0.5` by default








