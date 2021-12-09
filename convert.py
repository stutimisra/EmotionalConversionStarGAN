'''
convert.py

Author - Max Elliott

Script to perform conversion of speech using fully trained StarGAN_emo_VC1
models. Model checkpoints must be saved in the "../checkpoints" directory.
Converted files will be saved in the ./samples directory in a folder named
"<--out_dir>_<--iteration>_converted"

Command line arguments:

    --model -m     : Model name for conversion (as given by its config.yaml file)
    --in_dir -n    : wav files to be converted (won't work in code archive)
    --out_dir -o   : out directory name
    --iteration -i : iteration number of the checkpoint being used
'''

import argparse
import torch
import torch.nn.functional as F
import yaml
import numpy as np
import random
import math
import os
import pickle
import shutil

import librosa
from librosa.util import find_files
import pyworld
from pyworld import decode_spectral_envelope, synthesize

from matplotlib import pyplot as plt

import stargan.solver as solver
import stargan.model as model
import stargan.my_dataset as my_dataset
from stargan.my_dataset import get_filenames
from utils import audio_utils
import utils.data_preprocessing_utils as pp
import utils.preprocess_world as pw
from run_preprocessing import MAX_LENGTH

if __name__=='__main__':

    # Parse args:
    #   model checkpoint
    #   directory of wav files to be converted
    #   save directory
    parser = argparse.ArgumentParser()
    # parser.add_argument('-m', '--model', type = str,
    #                     help = "Model to use for conversion.")
    parser.add_argument('-in', '--in_dir', type=str, default=None)
    parser.add_argument('-out', '--out_dir', type=str)
    # parser.add_argument('-i', '--iteration', type = str)
    parser.add_argument('-c', '--checkpoint', type=str, help='Checkpoint file of model')
    # parser.add_argument('-n', '--num_emotions', type = int, default = None)
    # parser.add_argument('-f', '--features'), type = str,
                        # help = "mel or world features.")

    args = parser.parse_args()
    config = yaml.safe_load(open('./config_extension.yaml', 'r'))

    # checkpoint_dir = '../checkpoints/' + args.model + '/' + args.iteration + '.ckpt'
    checkpoint_dir = args.checkpoint

    print("Loading model at ", checkpoint_dir)

    #fix seeds to get consistent results
    SEED = 42
    # torch.backend.cudnn.deterministic = True
    # torch.backend.cudnn.benchmark = False
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # Use GPU
    USE_GPU = True

    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.manual_seed_all(SEED)
        map_location='cuda'
    else:
        device = torch.device('cpu')
        map_location='cpu'

    # @eric-zhizu: Get average emotion embeddings. Shapes: (1, 768)
    avg_neutral_embedding = torch.load('finetuned_ser_embed/avg_neutral_embedding.pt')
    avg_happy_embedding = torch.load('finetuned_ser_embed/avg_happy_embedding.pt')
    avg_sad_embedding = torch.load('finetuned_ser_embed/avg_sad_embedding.pt')
    avg_angry_embedding = torch.load('finetuned_ser_embed/avg_angry_embedding.pt')

    # @eric-zhizu: self.avg_embeddings shape: (4, 768)
    avg_embeddings = torch.cat((
        avg_neutral_embedding,
        avg_angry_embedding,
        avg_happy_embedding,
        avg_sad_embedding), dim=0)

    # Load model
    model = model.StarGAN_emo_VC1(config, config['model']['name'])
    model.load(checkpoint_dir)
    config = model.config
    model.to_device(device = device)
    model.set_eval_mode()

    # Make emotion targets (using config file)
    # s = solver.Solver(None, None, config, load_dir = None)
    # targets =
    num_emos = config['model']['num_classes']
    emo_labels = torch.Tensor(range(0, num_emos)).long()
    emo_targets = F.one_hot(emo_labels, num_classes = num_emos).float().to(device = device)
    print(f"Number of emotions = {num_emos}")

    data_dir = config['data']['dataset_dir']
    annotations_dict = pp.read_annotations(os.path.join(data_dir, 'annotations'))

    if args.in_dir is not None:
        # Get all .wav files in directory in_dir
        data_dir = args.in_dir
        print("Converting all samples in", args.in_dir)
    else:
        # Convert only train and test samples
        data_dir = os.path.join(config['data']['dataset_dir'], 'audio')
        print("Converting train and test samples in", data_dir)

    ########################################
    #        WORLD CONVERSION LOOP         #
    ########################################
    train_npy_files, test_npy_files = my_dataset.get_train_test_split(
        os.path.join(config['data']['dataset_dir'], 'world'), config
    )

    def npy_to_wav(filename):
        """ Convert xxx.npy to xxx.wav.  """
        filefront = filename.split('.')[0]
        return filefront + ".wav"

    # Assuming that sample.npy in the world folder exists as sample.wav in the audio folder
    train_wav_files = [npy_to_wav(f) for f in train_npy_files]
    test_wav_files = [npy_to_wav(f) for f in test_npy_files]

    def convert_files(files, out_folder):
        """
        Params:
            - files: List of filenames
            - out_folder: Files get stored in <args.out_dir>/<out_folder>
        Returns nothing
        """
        for file_num, f in enumerate(files):
            file_id = os.path.basename(f)[:-4]
            f = file_id + ".wav"

            try:
                wav, labels = pp.get_wav_and_labels(f, config['data']['dataset_dir'], annotations_dict)
            except Exception as e:
                print("Warning:", e, "file", f)
                continue

            wav = np.array(wav, dtype = np.float64)
            labels = np.array(labels)

            # If wav file is not in original train / test input (and args.in_dir not specified), skip
            if not args.in_dir and (labels[0] == -1 or labels[0] >= emo_targets.size(0)):
                continue

            # Temporary: @eric-zhizu, if speaker is <= 10, it is Chinese so skip
            if not args.in_dir and labels[1] <= 10:
                continue

            f0_real, ap_real, sp, coded_sp = pw.cal_mcep(wav)

            # If in_dir not specified, and length of the recording >= MAX_LENGTH, skip
            if not args.in_dir and coded_sp.shape[1] >= MAX_LENGTH:
                continue

            # coded_sp_temp = np.copy(coded_sp).T
            # print(coded_sp_temp.shape)
            coded_sp_cpu = coded_sp.T
            coded_sp = torch.Tensor(coded_sp_cpu).unsqueeze(0).unsqueeze(0).to(device = device)

            with torch.no_grad():
                # print(emo_targets)
                for i in range (0, emo_targets.size(0)):
                    f0 = np.copy(f0_real)
                    ap = np.copy(ap_real)

                    f0 = audio_utils.f0_pitch_conversion(f0, (labels[0],labels[1]),
                                                             (i, labels[1]))

                    # @eric-zhizu: Add emotion embeddings to the dataset
                    emo_embeddings_path = os.path.join('processed_data/emo_embeddings', file_id + '.pt')
                    if not os.path.exists(emo_embeddings_path):
                        print("Path does not exist" , emo_embeddings_path)
                        continue
                    emo_embeddings_dict = torch.load(emo_embeddings_path)

                    # @eric-zhizu: Use this to convert emo_label ==> emo_index. So 'Neutral' maps to 0, 'Angry' to 1, etc.
                    emo_list = ('Neutral', 'Angry', 'Happy', 'Sad')

                    # @eric-zhizu: Lazily hard-coded 768, but can be found in pretrained_models/.../hyperparams.yaml
                    emo_embeddings_list = []
                    for emo_idx, emo_label in enumerate(emo_list):
                        if emo_label in emo_embeddings_dict:
                            emo_embeddings_list.append(torch.squeeze(emo_embeddings_dict[emo_label]).cpu())
                        else:
                            # If for some reason the ref audio doesn't exist in the other emotion,
                            # use the avg target emotion embedding
                            emo_embeddings_list.append(avg_embeddings[emo_idx, :].cpu())
                    emo_embeddings = torch.vstack(emo_embeddings_list)
                    emo_embedding_target = emo_embeddings[emo_labels[i], :]

                    fake = model.G(coded_sp, emo_embedding_target.unsqueeze(0))

                    print(f"Converting {f[0:-4]} to {i}.")
                    model_iteration_string = model.config['model']['name'] + '_' + os.path.basename(args.checkpoint).replace('.ckpt', '')
                    filename_wav = model_iteration_string + '_' + f[0:-4] + "_" + str(int(labels[0].item())) + "to" + \
                                str(i) + ".wav"
                    filename_wav = os.path.join(args.out_dir, out_folder, filename_wav)

                    fake = fake.squeeze()
                    # print("Sampled size = ",fake.size())
                    # f = fake.data()
                    converted_sp = fake.cpu().numpy()
                    converted_sp = np.array(converted_sp, dtype = np.float64)

                    sample_length = converted_sp.shape[0]
                    if sample_length != ap.shape[0]:
                        # coded_sp_temp_copy = np.ascontiguousarray(coded_sp_temp_copy[0:sample_length, :], dtype = np.float64)
                        ap = np.ascontiguousarray(ap[0:sample_length, :], dtype = np.float64)
                        f0 = np.ascontiguousarray(f0[0:sample_length], dtype = np.float64)

                    f0 = np.ascontiguousarray(f0[20:-20], dtype = np.float64)
                    ap = np.ascontiguousarray(ap[20:-20,:], dtype = np.float64)
                    converted_sp = np.ascontiguousarray(converted_sp[20:-20,:], dtype = np.float64)
                    # coded_sp_temp_copy = np.ascontiguousarray(coded_sp_temp_copy[40:-40,:], dtype = np.float64)

                    # print("ap shape = ", ap.shape)
                    # print("f0 shape = ", f0.shape)
                    # print(converted_sp.shape)
                    audio_utils.save_world_wav([f0,ap,sp,converted_sp], filename_wav)

                    # Copy original file to output directory for ease
                    src_filepath = os.path.join(data_dir, f[0:-4] + ".wav")
                    dest_filename = model_iteration_string + '_' + f[0:-4] + "_" + str(int(labels[0].item())) + "_orig.wav"
                    dest_filepath = os.path.join(args.out_dir, out_folder, dest_filename)
                    shutil.copy(src_filepath, dest_filepath)
            # print(f, " converted.")
            if (file_num+1) % 20 == 0:
                print(file_num+1, " done.")

    convert_files(train_wav_files, "train")
    convert_files(test_wav_files, "test")

    ########################################
    #         MEL CONVERSION LOOP          #
    ########################################
    ### NEVER IMPLEMENTED AS ENDED UP NOT USING MEL SPECTROGRAMS
    # Make .npy arrays
    # Make audio
    # Make spec plots

    # Save all to directory
