"""
data_preprocessing2.py

Author - Max Elliott

Functions for pre-processing the IEMOCAP dataset. Can make mel-specs, WORLD
features, and labels for each audio clip.
"""

import torch

from utils import audio_utils

import numpy as np
import os
from librosa.util import find_files


def get_speaker_from_filename(filename):
    code = filename.split('_')[0]

    label = int(code)

    return label


def get_emotion_from_label(category):
    category = category.strip()
    if category == 'Surprise' or category == '':
        return -1

    conversion = {'Neutral': 0, 'Happy': 2, 'Sad': 3, 'Angry': 1, '中立': 0, '快乐': 2, '伤心': 3, '生气': 1}

    if category not in conversion:
        return -1

    label = conversion[category]

    return label


def getOneHot(label, n_labels):
    """ Not used anywhere in code base? """
    onehot = np.zeros(n_labels)
    onehot[label] = 1

    return onehot


def cont2list(cont, binned=False):

    list = [0,0,0]
    list[0] = float(cont[1:6])
    list[1] = float(cont[9:14])
    list[2] = float(cont[17:22])

    #Option to make the values discrete: low(0), med(1) or high(2)
    if binned:
        for i, val in enumerate(list):
            if val <= 2:
                list[i] = 0
            elif val < 4:
                list[i] = 1
            else:
                list[i] = 2
        return list
    else:
        return list


def concatenate_labels(emo, speaker):

    num_labels = 2
    all_labels = torch.zeros(num_labels)

    all_labels[0] = emo
    all_labels[1] = speaker

    return all_labels


def get_wav_and_labels(filename, data_dir, annotations_dict):
    """
    Assumes the data is in the format as specified in run_preprocessing.py
    """
    wav_path = os.path.join(data_dir, "audio", filename)

    audio = audio_utils.load_wav(wav_path)
    audio = np.array(audio, dtype=np.float32)
    filefront = filename.split(".")[0]
    if filefront in annotations_dict:
        labels = annotations_dict[filefront]
    else:
        labels = np.array([-1, 0])

    return audio, labels


def get_samples_and_labels(filename, config):
    speaker = filename.split('_')[0]
    wav_path = config['data']['sample_set_dir'] + "/" + filename
    label_path = config['data']['dataset_dir'] + "/Annotations/" + speaker + ".txt"

    with open(label_path, 'r', encoding='utf-16') as label_file:

        category = None
        # dimensions = None
        speaker = None

        for row in label_file:
            split = row.split("\t")
            category = get_emotion_from_label(split[2])
            # dimensions = cont2list(split[3])
            # dimensions_dis = cont2list(split[3], binned = True)
            speaker = get_speaker_from_filename(filename)

    audio = audio_utils.load_wav(wav_path)
    audio = np.array(audio, dtype = np.float32)
    labels = concatenate_labels(category, speaker)

    return audio, labels


def get_filenames(data_dir):

    files = find_files(data_dir, ext = 'wav')
    filenames = []

    for f in files:
        f = os.path.basename(f)[:-4]
        filenames.append(f)

    return filenames


def read_annotations(dir):
    """
    Read all labels inside dir into a dictionary
    """
    annotations = {}

    for file in os.listdir(dir):
        if file.endswith(".txt"):
            file_path = os.path.join(dir, file)

            # Determine the encoding
            def try_encoding(path, encoding):
                stream = open(path, 'r', encoding=encoding)
                stream.readlines()
                stream.seek(0)
                return encoding

            try:
                encoding = try_encoding(file_path, 'utf-16')
            except (UnicodeDecodeError, UnicodeError):
                try:
                    encoding = try_encoding(file_path, 'gb2312')
                except (UnicodeDecodeError, UnicodeError):
                    encoding = try_encoding(file_path, 'unicode-escape')

            with open(file_path, 'r', encoding=encoding) as label_file:
                for row in label_file:
                    split = row.split("\t")
                    if len(split) >= 3:
                        category = get_emotion_from_label(split[2])
                        speaker = int(file.split(".")[0])
                        labels = concatenate_labels(category, speaker)
                        annotations[split[0]] = labels

    return annotations


if __name__ == '__main__':
    """
    I don't think this code is actually run. (Eric)
    """

    min_length = 0 # actual is 59
    max_length = 688

    data_dir = '/Users/Max/MScProject/data'
    annotations_dir = os.path.join(data_dir, "audio")
    files = find_files(annotations_dir, ext = 'wav')

    filenames = []
    for f in files:
        f = os.path.basename(f)
        filenames.append(f)



    ############################################
    #      Code for making mels and labels     #
    ############################################
    i = 0
    found = 0
    lengths = []
    longest_lensgth = 0
    longest_name = ""
    for f in filenames:
        if i > 10000:
            print(f)
        wav, labels = get_wav_and_labels(f, data_dir)
        # mel = audio_utils.wav2melspectrogram(wav)
        labels = np.array(labels)
        if labels[0] in range(0,4) and f[0:3] == 'Ses':

            length = wav.shape[0]/16000.
            lengths.append(length)
            # np.save(data_dir + "/mels/" + f[:-4] + ".npy", mel)
            # np.save(data_dir + "/labels/" + f[:-4] + ".npy", labels)
            found += 1

            if length > longest_length:
                longest_length = length
                longest_name = f

        i += 1
        if i % 100 == 0:
            print(i, " complete.")
            print(found, "found.")

    print(found, "found.")
    print(f"longest + {longest_name}")

    lengths.sort()
    lengths = lengths[:int(len(lengths)*0.9)]
    print("Total seconds =", np.sum(lengths))

    # n, bins, patches = plt.hist(lengths, bins = 32)
    # plt.xlabel('Sequence length / seconds')
    # plt.xlim(0, 32)
    # plt.ylabel('Count')
    # plt.title(r'Histogram of sequence lengths for 4 emotional categories')
    # plt.show()

    ############################################
    #      Loop through mels for analysis      #
    ############################################
    # files = find_files(data_dir + "/mels", ext = 'npy')
    # lengths = []
    # for f in files:
    #
    #     mel = np.load(f)
    #     lengths.append(mel.shape[1])
    #     # print(mel.shape)
    #
    # n, bins, patches = plt.hist(lengths, bins = 22)
    # plt.xlabel('Sequence length')
    # plt.ylabel('Count')
    # plt.title(r'New histogram of sequence lengths for 4 emotional categories')
    # plt.show()

    ############################################
    #     Loop through labels for analysis     #
    ############################################
    # files = find_files(data_dir + "/labels", ext = 'npy')
    # category_counts = np.zeros((4))
    # speaker_counts = np.zeros((10))
    # for f in files:
    #
    #     labels = np.load(f)
    #     cat = int(labels[0])
    #     speaker = int(labels[1])
    #     category_counts[cat] += 1
    #     speaker_counts[speaker] += 1
    #
    # print(category_counts)
    # print(speaker_counts)
    # #### RESULTS ####
    # # [ 549.  890.  996. 1605.] 4040 total
    # # [416. 425. 353. 364. 448. 480. 342. 370. 473. 369.]
    # #### # # # # ####
    #
    # def make_autopct(values):
    #
    #     def my_autopct(pct):
    #         total = sum(values)
    #         val = int(round(pct*total/100.0))
    #         return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
    #
    #     return my_autopct
    #
    # plt.pie(category_counts, labels = ['Happy','Sad','Angry','Neutral'],
    #         autopct =make_autopct(category_counts), shadow=False)
    # plt.show()
    #
    # plt.pie(speaker_counts, labels = ['Ses01F','Ses01M','Ses02F','Ses02M','Ses03F',
    #                                 'Ses03M','Ses04F','Ses04M','Ses05F','Ses05M'],
    #         autopct ='%1.1f%%', shadow=False)
    # plt.show()

    # 1.34591066837310


    ############################################
    #   Finding min and max intensity of mels  #
    ############################################
    # i = 0
    # mels_made = 0
    # mel_lengths = []
    #
    # max_intensity = 0
    # min_intensity = 99999999
    #
    # for f in filenames:
    #
    #     wav, labels = get_wav_and_labels(f, data_dir)
    #     mel = audio_utils.wav2melspectrogram(wav)
    #     labels = np.array(labels)
    #     if labels[0] != -1:
    #
    #         # mel_lengths.append(mel.shape[1])
    #         max_val = np.max(mel)
    #         min_val = np.min(mel)
    #
    #         if max_val > max_intensity:
    #             max_intensity = max_val
    #         if min_val < min_intensity:
    #             min_intensity = min_val
    #         mels_made += 1
    #
    #     i += 1
    #     if i % 100 == 0:
    #         # print(mel_lengths[mels_made-1])
    #         print(mel[:, 45])
    #         print(max_intensity, ", ", min_intensity)
    #         print(i, " complete.")
    #         print(mels_made, "mels made.")
    #
    # print("max = {}".format(max_intensity))
    # print("min = {}".format(min_intensity))
    #
    # np.save('./stats/all_mel_lengths', np.array(mel_lengths))
    #
    # n, bins, patches = plt.hist(mel_lengths, bins = 22)
    # plt.xlabel('Sequence length')
    # plt.ylabel('Count')
    # plt.title(r'Histogram of sequence lengths for 4 emotional categories')
    # plt.show()
    #
    # mel_lengths = sorted(mel_lengths)
    # print(mel_lengths[0:30])
    # split_index = int(len(mel_lengths)*0.9)
    # print(mel_lengths[split_index])  # IS MAX LENGTH OF mels
    # print(mel_lengths[0])  # IS MIN LENGTH OF mels
