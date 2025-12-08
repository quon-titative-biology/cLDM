# ZebraFish Model Training Guide
This guide will navigate you through training the the VAE and cLDM respectively and highlight components which can be tweaked for optimal training.

**Before any training occurs it is important that all images have been properly processed through the `Zebrafish_SegmentAnyrhing.py` script**

# Training VAE:
To train the VAE from the command line the script `vae_training.py` can be ran. This takes the following paramaters:
- `DATA_PATH`: Path to the processed images `.npy` output from the preprocessing script
