"""
File: plot_emo_embeddings.py

Authors:
    - Eric Zhou

Date:
    - Created: Nov 29, 2021

Purpose:
    - Plot the vector embeddings of speech files from
    the Emotion Speech Dataset (https://kunzhou9646.github.io/controllable-evc/)
    given a pre-trained model from SpeechBrain (https://speechbrain.github.io/),
    an open-source library for speech processing
    - This is a standalone script

Emotion Speech Dataset Source:
    - Kun Zhou, Berrak Sisman, Mingyang Zhang and Haizhou Li,
    "Converting Anyone's Emotion: Towards Speaker-Independent Emotional Voice Conversion",
    in Proc. INTERSPEECH, Shanghai, China, October 2020.
    - https://kunzhou9646.github.io/controllable-evc/

List of Pre-Trained Models from SpeechBrain:
    - https://huggingface.co/speechbrain

Setup (install speechbrain):
    - git clone https://github.com/speechbrain/speechbrain.git
    - cd speechbrain
    - pip install -r requirements.txt
    - pip install --editable .

Args:
    --hf_model_id <Hugging Face Model ID>
        - Type: String
        - Description: The speech encoder to evaluate
        - Default value: "speechbrain/emotion-recognition-wav2vec2-IEMOCAP"
    --dataset <Path to Emotion Speech Dataset>
        - Type: String
        - Description: Path to emotion speech dataset

Usage:
    plot_emo_embeddings.py --hf_model_id <Hugging Face Model ID> --dataset <Path to Emotion Speech Dataset>
"""

import argparse
import numpy as np
import os
import torch
from utils import audio_utils

from speechbrain.pretrained.interfaces import foreign_class


def main(hf_model_id: str, dataset_path: str) -> None:
    """ Plot audio embeddings on a 2D plane using PCA

    Args:
        hf_model_id: [String] Hugging Face model ID
        dataset_path: [String] Path to dataset, assumes dataset directory will have the same
            file structure as the Emotion Speech Dataset mentioned in file docstring

    Returns:
        Nothing
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    classifier = foreign_class(source=hf_model_id,
                               pymodule_file="custom_interface.py",
                               classname="CustomEncoderWav2vec2Classifier",
                               run_opts={"device": device})

    for wav_form, emotion_label in audio_files(dataset_path):
        classifier.encode_batch(wav_form)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process args for plot_emo_embeddings.py.')

    parser.add_argument('--hf_model_id', type=str, default='speechbrain/emotion-recognition-wav2vec2-IEMOCAP',
                        help='Hugging Face model name, e.g., speechbrain/emotion-recognition-wav2vec2-IEMOCAP')
    parser.add_argument('--dataset', type=str, default='Emotional Speech Dataset (ESD)',
                        help='Path to dataset directory')

    args = parser.parse_args()
    main(args.hf_model_id, args.dataset)

"""
The emotions defined in the Emotion Speech Dataset
"""
EMOTIONS = {
    "Neutral": [],
    "Happy": [],
    "Sad": [],
    "Angry": [],
    "Surprise": []
}

def audio_files(dataset_path: str) -> np.ndarray:
    """ Yield the next audio file and emotion in the dataset

    File structure of dataset_path:
        - root
            - speaker ids
                - emotion labels
                    - evaluation / test / train
                        - .wav files

    Args:
        dataset_path: [String] Path to dataset, assumes dataset directory will have the same
            file structure as the Emotion Speech Dataset mentioned in file docstring

    Yields:
        wav_form: [np.ndarray] Size (seq_length,)
        emotion: [String] "Happy", "Sad", "Neutral", "Angry", "Surprise"
    """
    # Iterate through speaker subdirectories
    for speaker_subdir in os.scandir(dataset_path):
        if speaker_subdir.is_dir():
            # Iterate through emotion label subdirectories
            for emotion_label_subdir in os.scandir(speaker_subdir.path):
                if emotion_label_subdir.is_dir():
                    # Iterate through evaluation, test, and train subdirectories
                    for wav_dir in os.scandir(emotion_label_subdir.path):
                        assert wav_dir.is_dir()
                        for wav_file in os.listdir(wav_dir):
                            assert wav_file.endswith(".wav")
                            wav_path = os.path.join(wav_dir.path, wav_file)
                            wav_form = audio_utils.load_wav(wav_path)
                            yield wav_form, emotion_label_subdir.name