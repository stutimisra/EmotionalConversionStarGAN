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
    - pip install transformers

Args:
    --hf_model_id <Hugging Face Model ID>
        - Type: String
        - Description: The speech encoder to evaluate
        - Default value: "speechbrain/emotion-recognition-wav2vec2-IEMOCAP"

Usage:
    plot_emo_embeddings.py --hf_model_id <optional: Hugging Face Model ID>
"""

import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
import time
import torch
import torch.utils.data as data_utils
from tqdm import tqdm
import yaml

import stargan.my_dataset as my_dataset

from speechbrain.pretrained.interfaces import foreign_class


def main(hf_model_id: str) -> None:
    """ Plot audio embeddings on a 2D plane using PCA

    Args:
        hf_model_id: [String] Hugging Face model ID
        dataset_path: [String] Path to dataset, assumes dataset directory will have the same
            file structure as the Emotion Speech Dataset mentioned in file docstring

    Assumes:
        configs/config_step1.yaml exists and contains proper configurations for loading
        the MyDataset class

    Returns:
        Nothing
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    classifier = foreign_class(source=hf_model_id,
                               pymodule_file="custom_interface.py",
                               classname="CustomEncoderWav2vec2Classifier",
                               run_opts={"device": device})

    ###
    #   Create dataloader
    ###
    dataset_config = yaml.safe_load(open("configs/config_step1.yaml", 'r'))
    world_dir = os.path.join(dataset_config["data"]["dataset_dir"], "world")
    filenames = my_dataset.get_filenames(world_dir)

    dataset = my_dataset.MyDataset(dataset_config, filenames)

    data_loader = data_utils.DataLoader(dataset, batch_size=64,
                                        collate_fn=my_dataset.collate_length_order,
                                        num_workers=0, shuffle=False)

    ###
    #   Loop through audio files
    ###
    data_batches = []
    for _, (wav_forms, wav_lengths), labels in tqdm(data_loader):
        # encoded_batch: (batch_size, 768)
        encoded_batch = classifier.encode_batch(wav_forms, wav_lengths)

        # Only the first element is the emotion label. Second element is the speaker label.
        emotion_labels = labels[:, 0]

        # encoded_batch_with_labels: (batch_size, 769)
        encoded_batch_with_labels = torch.cat(
            (encoded_batch.detach().cpu(), torch.unsqueeze(emotion_labels, dim=1)),
            dim=1
        )

        data_batches.append(encoded_batch_with_labels)
    data = torch.vstack(data_batches)
    print("Finished reading data. Shape", data.shape)

    torch.save(data, "emo_embeddings.pt")
    print("Saved data to emo_embeddings.pt")

    ###
    #   Put data into a Pandas dataframe for ease of computation and plotting
    #   Credits: https://towardsdatascience.com/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b
    ###
    n_features = data.shape[1] - 1
    feature_cols = ['feature' + str(i) for i in range(n_features)]
    df = pd.DataFrame(data, columns=feature_cols + ['y'])
    print("Created data frame of length", len(df))

    # For debugging, save the dataframe to disk
    # df.to_pickle('emo_embeddings.pkl')
    # print("Saved data frame to emo_embeddings.pkl")

    # See pretrained_models/CustomEncoder.../label_encoder.ckpy for index => emotion mappings
    df['label'] = df['y'].apply(lambda index: classifier.hparams.label_encoder.ind2lab[index])

    ###
    #   Perform dimensionality reduction using t-SNE
    ###
    np.random.seed(42)
    rndperm = np.random.permutation(df.shape[0])

    # Plot only a random 5000 samples to reduce computation time
    N = 5000
    df_subset = df.loc[rndperm[:N], :].copy()
    data_subset = df_subset[feature_cols].values

    # Perform t-SNE
    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(data_subset)
    print(f't-SNE done! Time elapsed: {time.time() - time_start} seconds')

    # Add t-SNE results to the data frame to plot
    df_subset['tsne-2d-one'] = tsne_results[:, 0]
    df_subset['tsne-2d-two'] = tsne_results[:, 1]

    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x='tsne-2d-one', y='tsne-2d-two',
        hue='label',
        palette=sns.color_palette('hls', 4),
        data=df_subset,
        legend='full',
        alpha=0.3
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process args for plot_emo_embeddings.py.')

    parser.add_argument('--hf_model_id', type=str, default='speechbrain/emotion-recognition-wav2vec2-IEMOCAP',
                        help='Hugging Face model name, e.g., speechbrain/emotion-recognition-wav2vec2-IEMOCAP')

    args = parser.parse_args()
    main(args.hf_model_id)
