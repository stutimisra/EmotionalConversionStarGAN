# EmotionalConversionStarGAN
This repository contains code to replicate results from the ICASSP 2020 paper "StarGAN for Emotional Speech Conversion: Validated by Data Augmentation of End-to-End Emotion Recognition".

The original paper used the IEMOCAP dataset: https://sail.usc.edu/iemocap/

We are using the Emotional Speech Dataset (ESD) instead, since it is newer and more comparable to recent baselines. https://kunzhou9646.github.io/controllable-evc/

# Preparing
**- Requirements:**
* python>3.7.0
* pytorch
* numpy
* argparse
* librosa
* scikit-learn
* tensorflow < 2.0 (just for logging)
* pyworld
* matplotlib
* yaml

If you are running on AWS, you will have to `conda activate pytorch_p37` and `pip install librosa pyworld tensorflow==1.15`.

**- Clone repository:**
```
git clone https://github.com/glam-imperial/EmotionalConversionStarGAN.git
cd EmotionalConversionStarGAN
```

**- Download ESD dataset from https://kunzhou9646.github.io/controllable-evc/**

Running the script **run_preprocessing.py** will prepare the data as needed for training the model. It assumes that ESD is already downloaded and is stored in an arbitrary directory <DIR> with this file structure
```
<DIR>
  |- Speaker 0001  
  |     |- Annotations (0001.txt) 
  |     |- Angry/
  |     |-    |- evaluation
  |     |-    |-    |- 0001_<recording id>.wav
  |     |-    |- test
  |     |-    |- train
  |     |- Happy/
  |     |- ...
  |- ...
  |- Speaker 0020
  |     |- Annotations (0020.txt) 
  |     |- Angry/
  |     |- Happy/
  |     |- ...
```
where Annotations is a text file holding the emotion labels for each recording
  
 To preprocess run
 ```
 python run_preprocessing.py --iemocap_dir <DIR> 
 ```
 which will move all audio files to ./procesed_data/audio as well as extract all WORLD features and labels needed for training. It will only extract these for samples of the correct emotions (angry, sad, happy) and under the certain hardocded length threshold (to speed up training time). it will also create dictionaries for F0 statistics which are used to alter the F0 of a sample when converting.
After running you should have a file structure:
```
./processed_data
 |- annotations
 |- audio
 |- f0
 |- labels
 |- world
 ```
 
`run_preprocessing.py` will take a few hours. We recommend saving `processed_data` somewhere. If you are in a rush and trying to test whether the model will work, you can interrupt the script after it has converted only a subset of the data.
 
 # Training EmotionStarGAN
 Main training script is **train_main.py**. However to automatically train a three emotion model (angry, sad, happy) as it was trained for "StarGAN for Emotional Speech Conversion: Validated by Data Augmentation of End-to-End Emotion Recognition", simply call:
 ```
 ./full_training_script.sh
 ```
 This script runs three steps:
 1. Runs classifier_train.py - Pretrains an auxiliary emotional classifier. Saves best checkpoint to ./checkpoints/cls_checkpoint.ckpt
 2. Runs main training for 200k iterations in --recon_only mode, meaning model learns to simply reconstruct the input audio.
 3. Trains model for a further 100k steps, introducing the pre-trained classifier.
 
 A full training run will take ~24 hours on a decent GPU. The auxiliary emotional classifier can also be trained independently using **classifier_train.py**.
 
 # Sample Conversion
 Once a model is trained you can convert the output audio samples using **convert.py**. Running
 ```
 python convert.py --checkpoint <path/to/model_checkpoint.ckpt> -o ./processed_data/converted
 ```
 will load a model checkpoint and convert all samples from the train set and test set into each emotion and save the converted samples in /processed_data/converted
