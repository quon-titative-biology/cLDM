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
